# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# PyTorch reimplementation of CLRS decoders.py with *identical* logic & API.
# Notes:
# - Uses torch + nn.LazyLinear to match Haiku's shape-late parameter init.
# - Function signatures, control flow, and tensor semantics mirror the JAX/Haiku
#   version so it can be a drop-in replacement from nets_torch.

from __future__ import annotations

import functools
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from . import probing
from . import specs

_Array = torch.Tensor
_DataPoint = probing.DataPoint
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Type = specs.Type


def _strip_algo_prefix(name: str) -> str:
  """Turn 'algo_{idx}_{rest}' into '{rest}', otherwise return as-is."""
  if name.startswith('algo_'):
    parts = name.split('_', 2)
    if len(parts) >= 3 and parts[1].isdigit():
      return parts[2]
  return name


def log_sinkhorn(x: _Array, steps: int, temperature: float, zero_diagonal: bool,
                 noise_rng_key: Optional[torch.Generator]) -> _Array:
  """Sinkhorn operator in log space, to postprocess permutation pointer logits.

  Args:
    x: input of shape [..., n, n], a batch of square matrices.
    steps: number of iterations.
    temperature: temperature parameter (as temperature approaches zero, the
      output approaches a permutation matrix).
    zero_diagonal: whether to force the diagonal logits towards -inf.
    noise_rng_key: generator to add Gumbel noise (None = no noise).

  Returns:
    Elementwise logarithm of a doubly-stochastic matrix (rows/cols sum to 1).
  """
  assert x.ndim >= 2
  assert x.shape[-1] == x.shape[-2]
  if noise_rng_key is not None:
    # Add standard Gumbel noise (see https://arxiv.org/abs/1802.08665)
    # g = -log(-log(U)), U~Uniform(0,1)
    U = torch.rand_like(x, generator=noise_rng_key)
    noise = -(torch.log(-torch.log(U + 1e-12) + 1e-12))
    x = x + noise
  x = x / temperature
  if zero_diagonal:
    n = x.shape[-1]
    eye = torch.eye(n, dtype=x.dtype, device=x.device)
    x = x - 1e6 * eye
  for _ in range(steps):
    x = F.log_softmax(x, dim=-1)
    x = F.log_softmax(x, dim=-2)
  return x


def construct_decoders(loc: str, t: str, hidden_dim: int, nb_dims: int,
                       name: str):
  """Constructs decoders (LazyLinear to mirror Haiku init semantics)."""
  linear = functools.partial(nn.LazyLinear, out_features=None)  # placeholder
  # We wrap to set out_features while keeping naming parity through caller.
  def L(out_features: int):
    layer = nn.LazyLinear(out_features)
    # attach a soft name tag; actual registration/naming happens in nets_torch
    layer.torch_name = f"{name}_dec_linear"
    return layer

  if loc == _Location.NODE:
    # Node decoders.
    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
      decs = (L(1),)
    elif t == _Type.CATEGORICAL:
      decs = (L(nb_dims),)
    elif t in [_Type.POINTER, _Type.PERMUTATION_POINTER]:
      decs = (L(hidden_dim), L(hidden_dim), L(hidden_dim), L(1))
    else:
      raise ValueError(f"Invalid Type {t}")

  elif loc == _Location.EDGE:
    # Edge decoders.
    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
      decs = (L(1), L(1), L(1))
    elif t == _Type.CATEGORICAL:
      decs = (L(nb_dims), L(nb_dims), L(nb_dims))
    elif t == _Type.POINTER:
      decs = (L(hidden_dim), L(hidden_dim), L(hidden_dim), L(hidden_dim), L(1))
    else:
      raise ValueError(f"Invalid Type {t}")

  elif loc == _Location.GRAPH:
    # Graph decoders.
    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
      decs = (L(1), L(1))
    elif t == _Type.CATEGORICAL:
      decs = (L(nb_dims), L(nb_dims))
    elif t == _Type.POINTER:
      decs = (L(1), L(1), L(1))
    else:
      raise ValueError(f"Invalid Type {t}")
  else:
    raise ValueError(f"Invalid Location {loc}")

  return decs


def construct_diff_decoders(name: str):
  """Constructs diff decoders (LazyLinear)."""
  def L(out_features: int):
    layer = nn.LazyLinear(out_features)
    layer.torch_name = f"{name}_diffdec_linear"
    return layer
  decoders = {}
  decoders[_Location.NODE] = L(1)
  decoders[_Location.EDGE] = (L(1), L(1), L(1))
  decoders[_Location.GRAPH] = (L(1), L(1))
  return decoders


def postprocess(spec: _Spec, preds: Dict[str, _Array],
                sinkhorn_temperature: float,
                sinkhorn_steps: int,
                hard: bool) -> Dict[str, _DataPoint]:
  """Postprocesses decoder output (hints/outputs) in hard/soft modes."""
  result = {}
  for name in preds.keys():
    _, loc, t = spec[name]
    new_t = t
    data = preds[name]
    if t == _Type.SCALAR:
      if hard:
        data = data.detach()
    elif t == _Type.MASK:
      if hard:
        data = (data > 0.0).to(data.dtype)
      else:
        data = torch.sigmoid(data)
    elif t in [_Type.MASK_ONE, _Type.CATEGORICAL]:
      cat_size = data.shape[-1]
      if hard:
        best = torch.argmax(data, dim=-1)
        data = F.one_hot(best, num_classes=cat_size).to(data.dtype)
      else:
        data = F.softmax(data, dim=-1)
    elif t == _Type.POINTER:
      if hard:
        data = torch.argmax(data, dim=-1).to(data.dtype)
      else:
        data = F.softmax(data, dim=-1)
        new_t = _Type.SOFT_POINTER
    elif t == _Type.PERMUTATION_POINTER:
      # Convert the matrix of logits to a doubly stochastic matrix.
      data = log_sinkhorn(
          x=data,
          steps=sinkhorn_steps,
          temperature=sinkhorn_temperature,
          zero_diagonal=True,
          noise_rng_key=None)
      data = torch.exp(data)
      if hard:
        data = F.one_hot(torch.argmax(data, dim=-1), data.shape[-1]).to(data.dtype)
    else:
      raise ValueError("Invalid type")
    result[name] = probing.DataPoint(
        name=name, location=loc, type_=new_t, data=data)

  return result


def decode_fts(
    decoders,
    spec: _Spec,
    h_t: _Array,
    adj_mat: _Array,
    edge_fts: _Array,
    graph_fts: _Array,
    inf_bias: bool,
    inf_bias_edge: bool,
    repred: bool,
):
  """Decodes node, edge and graph features."""
  output_preds: Dict[str, _Array] = {}
  hint_preds: Dict[str, _Array] = {}

  for name in decoders:
    decoder = decoders[name]
    # stage, loc, t = spec[name]
    name = _strip_algo_prefix(name)  # <=== 新增
    stage, loc, t = spec[name]  # <=== 用 base_name 查 spec

    if loc == _Location.NODE:
      preds = _decode_node_fts(decoder, t, h_t, edge_fts, adj_mat,
                               inf_bias, repred)
    elif loc == _Location.EDGE:
      preds = _decode_edge_fts(decoder, t, h_t, edge_fts, adj_mat,
                               inf_bias_edge)
    elif loc == _Location.GRAPH:
      preds = _decode_graph_fts(decoder, t, h_t, graph_fts)
    else:
      raise ValueError("Invalid output type")

    if stage == _Stage.OUTPUT:
      output_preds[name] = preds
    elif stage == _Stage.HINT:
      hint_preds[name] = preds
    else:
      raise ValueError(f"Found unexpected decoder {name}")

  return hint_preds, output_preds


def _decode_node_fts(decoders, t: str, h_t: _Array, edge_fts: _Array,
                     adj_mat: _Array, inf_bias: bool, repred: bool) -> _Array:
  """Decodes node features."""
  if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
    preds = decoders[0](h_t).squeeze(-1)
  elif t == _Type.CATEGORICAL:
    preds = decoders[0](h_t)
  elif t in [_Type.POINTER, _Type.PERMUTATION_POINTER]:
    p_1 = decoders[0](h_t)
    p_2 = decoders[1](h_t)
    p_3 = decoders[2](edge_fts)

    p_e = p_2.unsqueeze(-2) + p_3
    p_m = torch.maximum(p_1.unsqueeze(-2), p_e.transpose(1, 2))

    preds = decoders[3](p_m).squeeze(-1)

    if inf_bias:
      dims = list(range(1, preds.dim()))
      per_batch_min = torch.amin(preds, dim=dims, keepdim=True)
      preds = torch.where(
          adj_mat > 0.5,
          preds,
          torch.minimum(torch.tensor(-1.0, dtype=preds.dtype, device=preds.device), per_batch_min - 1.0),
      )
    if t == _Type.PERMUTATION_POINTER:
      if repred:  # testing or validation, no Gumbel noise
        preds = log_sinkhorn(
            x=preds, steps=10, temperature=0.1,
            zero_diagonal=True, noise_rng_key=None)
      else:  # training, add Gumbel noise (use default generator)
        preds = log_sinkhorn(
            x=preds, steps=10, temperature=0.1,
            zero_diagonal=True, noise_rng_key=torch.Generator(device=preds.device))
  else:
    raise ValueError("Invalid output type")

  return preds


def _decode_edge_fts(decoders, t: str, h_t: _Array, edge_fts: _Array,
                     adj_mat: _Array, inf_bias_edge: bool) -> _Array:
  """Decodes edge features."""
  pred_1 = decoders[0](h_t)
  pred_2 = decoders[1](h_t)
  pred_e = decoders[2](edge_fts)
  pred = (pred_1.unsqueeze(-2) + pred_2.unsqueeze(-3) + pred_e)
  if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
    preds = pred.squeeze(-1)
  elif t == _Type.CATEGORICAL:
    preds = pred
  elif t == _Type.POINTER:
    pred_2b = decoders[3](h_t)

    p_m = torch.maximum(pred.unsqueeze(-2), pred_2b.unsqueeze(-3).unsqueeze(-3))

    preds = decoders[4](p_m).squeeze(-1)
  else:
    raise ValueError("Invalid output type")
  if inf_bias_edge and t in [_Type.MASK, _Type.MASK_ONE]:
    dims = list(range(1, preds.dim()))
    per_batch_min = torch.amin(preds, dim=dims, keepdim=True)
    preds = torch.where(
        adj_mat > 0.5,
        preds,
        torch.minimum(torch.tensor(-1.0, dtype=preds.dtype, device=preds.device), per_batch_min - 1.0),
    )

  return preds


def _decode_graph_fts(decoders, t: str, h_t: _Array,
                      graph_fts: _Array) -> _Array:
  """Decodes graph features."""
  gr_emb = torch.amax(h_t, dim=-2)
  pred_n = decoders[0](gr_emb)
  pred_g = decoders[1](graph_fts)
  pred = pred_n + pred_g
  if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
    preds = pred.squeeze(-1)
  elif t == _Type.CATEGORICAL:
    preds = pred
  elif t == _Type.POINTER:
    pred_2 = decoders[2](h_t)
    ptr_p = pred.unsqueeze(1) + pred_2.transpose(1, 2)
    preds = ptr_p.squeeze(1)
  else:
    raise ValueError("Invalid output type")

  return preds


def maybe_decode_diffs(
    diff_decoders,
    h_t: _Array,
    edge_fts: _Array,
    graph_fts: _Array,
    decode_diffs: bool,
) -> Optional[Dict[str, _Array]]:
  """Optionally decodes node, edge and graph diffs."""
  if decode_diffs:
    preds: Dict[str, _Array] = {}
    node = _Location.NODE
    edge = _Location.EDGE
    graph = _Location.GRAPH
    preds[node] = _decode_node_diffs(diff_decoders[node], h_t)
    preds[edge] = _decode_edge_diffs(diff_decoders[edge], h_t, edge_fts)
    preds[graph] = _decode_graph_diffs(diff_decoders[graph], h_t, graph_fts)
  else:
    preds = None
  return preds


def _decode_node_diffs(decoders, h_t: _Array) -> _Array:
  """Decodes node diffs."""
  return decoders(h_t).squeeze(-1)


def _decode_edge_diffs(decoders, h_t: _Array, edge_fts: _Array) -> _Array:
  """Decodes edge diffs."""
  e_pred_1 = decoders[0](h_t)
  e_pred_2 = decoders[1](h_t)
  e_pred_e = decoders[2](edge_fts)
  preds = (e_pred_1.unsqueeze(-1) + e_pred_2.unsqueeze(-2) + e_pred_e).squeeze(-1)
  return preds


def _decode_graph_diffs(decoders, h_t: _Array, graph_fts: _Array) -> _Array:
  """Decodes graph diffs."""
  gr_emb = torch.amax(h_t, dim=-2)
  g_pred_n = decoders[0](gr_emb)
  g_pred_g = decoders[1](graph_fts)
  preds = (g_pred_n + g_pred_g).squeeze(-1)
  return preds
