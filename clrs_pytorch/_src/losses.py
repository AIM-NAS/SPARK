# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# PyTorch reimplementation of CLRS losses.py with *identical* API & logic.
# - Keeps function names, signatures, and control flow aligned with original.
# - Replaces JAX/Haiku ops with torch equivalents.
#
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from . import probing
from . import specs

_Array = torch.Tensor
_DataPoint = probing.DataPoint
_Location = specs.Location
_OutputClass = specs.OutputClass
_PredTrajectory = Dict[str, _Array]
_PredTrajectories = List[_PredTrajectory]
_Type = specs.Type

EPS = 1e-12


def _ensure_tensor(x, device, dtype=None):
  if isinstance(x, torch.Tensor):
    t = x
  else:
    t = torch.as_tensor(x)
  if dtype is not None:
    t = t.to(dtype)
  return t.to(device)

def _expand_to(x: _Array, y: _Array) -> _Array:
  while y.dim() > x.dim():
    x = x.unsqueeze(-1)
  return x


def _expand_and_broadcast_to(x: _Array, y: _Array) -> _Array:
  return _expand_to(x, y).expand_as(y)


def output_loss_chunked(truth: _DataPoint, pred: _Array,
                        is_last: _Array, nb_nodes: int) -> _Array:
  """Output loss for time-chunked training."""
  mask = None
  device = pred.device

  # 关键：把 numpy -> torch
  truth_data = _ensure_tensor(truth.data, device=device)

  if truth.type_ == _Type.SCALAR:
    loss = (pred - truth_data)**2

  elif truth.type_ == _Type.MASK:
    loss = (
        torch.maximum(pred, torch.zeros_like(pred)) - pred * truth_data +
        torch.log1p(torch.exp(-torch.abs(pred)))
    )
    mask = (truth_data != _OutputClass.MASKED)

  elif truth.type_ in [_Type.MASK_ONE, _Type.CATEGORICAL]:
    mask = torch.any(truth_data == _OutputClass.POSITIVE, dim=-1)
    masked_truth = truth_data * (truth_data != _OutputClass.MASKED).to(pred.dtype)
    loss = -torch.sum(masked_truth * F.log_softmax(pred, dim=-1), dim=-1)

  elif truth.type_ == _Type.POINTER:
    td_long = truth_data.to(torch.long)
    loss = -torch.sum(
        F.one_hot(td_long, nb_nodes).to(pred.dtype) * F.log_softmax(pred, dim=-1), dim=-1)

  elif truth.type_ == _Type.PERMUTATION_POINTER:
    loss = -torch.sum(truth_data * pred, dim=-1)

  is_last = _ensure_tensor(is_last, device=device, dtype=loss.dtype)
  if mask is not None:
    mask = mask * _expand_and_broadcast_to(is_last, loss)
  else:
    mask = _expand_and_broadcast_to(is_last, loss)

  total_mask = torch.maximum(torch.sum(mask.to(loss.dtype)),
                             torch.tensor(EPS, dtype=loss.dtype, device=device))
  return torch.sum(torch.where(mask.bool(), loss, torch.zeros_like(loss))) / total_mask

def output_loss(truth: _DataPoint, pred: _Array, nb_nodes: int) -> _Array:
  """Output loss for full-sample training."""
  device = pred.device
  truth_data = _ensure_tensor(truth.data, device=device)

  if truth.type_ == _Type.SCALAR:
    total_loss = torch.mean((pred - truth_data)**2)

  elif truth.type_ == _Type.MASK:
    loss = (
        torch.maximum(pred, torch.zeros_like(pred)) - pred * truth_data +
        torch.log1p(torch.exp(-torch.abs(pred)))
    )
    mask = (truth_data != _OutputClass.MASKED).to(pred.dtype)
    total_loss = torch.sum(loss * mask) / torch.sum(mask)

  elif truth.type_ in [_Type.MASK_ONE, _Type.CATEGORICAL]:
    masked_truth = truth_data * (truth_data != _OutputClass.MASKED).to(pred.dtype)
    total_loss = (-torch.sum(masked_truth * F.log_softmax(pred, dim=-1)) /
                  torch.sum((truth_data == _OutputClass.POSITIVE)))

  elif truth.type_ == _Type.POINTER:
    td_long = truth_data.to(torch.long)
    total_loss = torch.mean(-torch.sum(
        F.one_hot(td_long, nb_nodes).to(pred.dtype) * F.log_softmax(pred, dim=-1), dim=-1))

  elif truth.type_ == _Type.PERMUTATION_POINTER:
    total_loss = torch.mean(-torch.sum(truth_data * pred, dim=-1))

  return total_loss

def hint_loss_chunked(
    truth: _DataPoint,
    pred: _Array,
    is_first: _Array,
    nb_nodes: int,
):
  device = pred.device
  is_first = _ensure_tensor(is_first, device=device)

  loss, mask = _hint_loss(
      truth_data=_ensure_tensor(truth.data, device=device),
      truth_type=truth.type_,
      pred=pred,
      nb_nodes=nb_nodes,
  )
  mask = mask * (1 - _expand_to(is_first, loss)).to(mask.dtype)
  denom = torch.maximum(torch.sum(mask.to(loss.dtype)),
                        torch.tensor(EPS, dtype=loss.dtype, device=device))
  return torch.sum(loss * mask) / denom


def hint_loss(
    truth: _DataPoint,
    preds: List[_Array],
    lengths: _Array,
    nb_nodes: int,
    verbose: bool = False,
):
  device = preds[0].device if len(preds) else torch.device('cpu')
  total_loss = torch.tensor(0., dtype=torch.float32, device=device)
  verbose_loss = {}

  truth_data_full = _ensure_tensor(truth.data, device=device)
  length = truth_data_full.shape[0] - 1

  loss, mask = _hint_loss(
      truth_data=truth_data_full[1:],   # 丢掉 t=0
      truth_type=truth.type_,
      pred=torch.stack(preds, dim=0),
      nb_nodes=nb_nodes,
  )
  lengths = _ensure_tensor(lengths, device=device)
  mask = mask * _is_not_done_broadcast(lengths, torch.arange(length, device=device)[:, None], loss)

  denom = torch.maximum(torch.sum(mask.to(loss.dtype)),
                        torch.tensor(EPS, dtype=loss.dtype, device=device))
  loss = torch.sum(loss * mask) / denom
  if verbose:
    verbose_loss['loss_' + truth.name] = loss
    return verbose_loss
  else:
    return total_loss + loss

def _hint_loss(
    truth_data: _Array,
    truth_type: str,
    pred: _Array,
    nb_nodes: int,
) -> Tuple[_Array, _Array]:
  device = pred.device
  truth_data = _ensure_tensor(truth_data, device=device)
  mask = None

  if truth_type == _Type.SCALAR:
    loss = (pred - truth_data)**2

  elif truth_type == _Type.MASK:
    loss = (torch.maximum(pred, torch.zeros_like(pred)) - pred * truth_data +
            torch.log1p(torch.exp(-torch.abs(pred))))
    mask = (truth_data != _OutputClass.MASKED).to(pred.dtype)

  elif truth_type == _Type.MASK_ONE:
    loss = -torch.sum(truth_data * F.log_softmax(pred, dim=-1), dim=-1, keepdim=True)

  elif truth_type == _Type.CATEGORICAL:
    loss = -torch.sum(truth_data * F.log_softmax(pred, dim=-1), dim=-1)
    mask = torch.any(truth_data == _OutputClass.POSITIVE, dim=-1).to(pred.dtype)

  elif truth_type == _Type.POINTER:
    td_long = truth_data.to(torch.long)
    loss = -torch.sum(
        F.one_hot(td_long, nb_nodes).to(pred.dtype) * F.log_softmax(pred, dim=-1), dim=-1)

  elif truth_type == _Type.PERMUTATION_POINTER:
    loss = -torch.sum(truth_data * pred, dim=-1)

  if mask is None:
    mask = torch.ones_like(loss, dtype=pred.dtype)
  return loss, mask


def _is_not_done_broadcast(lengths: _Array, i: _Array, tensor: _Array) -> _Array:
  is_not_done = (lengths > i + 1).to(tensor.dtype)
  while is_not_done.dim() < tensor.dim():
    is_not_done = is_not_done.unsqueeze(-1)
  return is_not_done
