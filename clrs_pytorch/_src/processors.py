# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# PyTorch reimplementation of CLRS processors.py with *identical* API & logic
# (Haiku/JAX → torch.nn/torch). Public class/function names, signatures and
# control flow are preserved so it can be used as a drop-in backend.
#
from __future__ import annotations

import abc
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_Array = torch.Tensor
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6
PROCESSOR_TAG = 'clrs_processor'


# ---------------------------------------------------------------------------
# Base processor
# ---------------------------------------------------------------------------
class Processor(nn.Module):
  """Processor abstract base class."""

  def __init__(self, name: str):
    super().__init__()
    if not name.endswith(PROCESSOR_TAG):
      name = name + '_' + PROCESSOR_TAG
    # Keep a readable name for param filtering (used upstream)
    self.torch_name = name

  @abc.abstractmethod
  def forward(
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **kwargs,
  ) -> Tuple[_Array, Optional[_Array]]:
    """Processor inference step (returns node, optional edge embeddings)."""
    raise NotImplementedError

  # Keep property names for optimizer masking logic
  @property
  def inf_bias(self):
    return False

  @property
  def inf_bias_edge(self):
    return False


# ---------------------------------------------------------------------------
# GAT (Veličković et al., ICLR'18)
# ---------------------------------------------------------------------------
class GAT(Processor):
  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      activation: Optional[_Fn] = F.relu,
      residual: bool = True,
      use_ln: bool = False,
      name: str = 'gat_aggr',
  ):
    super().__init__(name=name)
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    self.out_size = out_size
    self.nb_heads = nb_heads
    self.head_size = out_size // nb_heads
    self.activation = activation
    self.residual = residual
    self.use_ln = use_ln

    # Modules
    self.m = nn.Linear(out_size * 0 + 1, out_size, bias=True)  # placeholder reset later
    self.skip = nn.Linear(out_size * 0 + 1, out_size, bias=True)
    self.a_1 = nn.Linear(out_size * 0 + 1, nb_heads, bias=True)
    self.a_2 = nn.Linear(out_size * 0 + 1, nb_heads, bias=True)
    self.a_e = nn.Linear(out_size * 0 + 1, nb_heads, bias=True)
    self.a_g = nn.Linear(out_size * 0 + 1, nb_heads, bias=True)
    self._lazy = True  # emulate LazyLinear via first forward capture
    if use_ln:
      self.ln = nn.LayerNorm(out_size)

  def _maybe_lazy_init(self, z: _Array, edge_fts: _Array, graph_fts: _Array):
    if self._lazy:
      in_z = z.shape[-1]
      in_e = edge_fts.shape[-1]
      in_g = graph_fts.shape[-1]
      self.m = nn.Linear(in_z, self.out_size)
      self.skip = nn.Linear(in_z, self.out_size)
      self.a_1 = nn.Linear(in_z, self.nb_heads)
      self.a_2 = nn.Linear(in_z, self.nb_heads)
      self.a_e = nn.Linear(in_e, self.nb_heads)
      self.a_g = nn.Linear(in_g, self.nb_heads)
      self._lazy = False
      self.to(z.device)
  def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused):
    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = torch.cat([node_fts, hidden], dim=-1)
    self._maybe_lazy_init(z, edge_fts, graph_fts)

    # [B, N, H*F] -> [B, H, N, F]
    values = self.m(z).view(b, n, self.nb_heads, self.head_size).permute(0, 2, 1, 3)

    att_1 = self.a_1(z).unsqueeze(-1)
    att_2 = self.a_2(z).unsqueeze(-1)
    att_e = self.a_e(edge_fts)
    att_g = self.a_g(graph_fts).unsqueeze(-1)

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = bias_mat.unsqueeze(-1).expand(-1, -1, -1, self.nb_heads).permute(0, 3, 1, 2)  # [B,H,N,N]

    logits = (
        att_1.permute(0, 2, 1, 3) +   # [B,H,N,1]
        att_2.permute(0, 2, 3, 1) +   # [B,H,1,N]
        att_e.permute(0, 3, 1, 2) +   # [B,H,N,N]
        att_g.unsqueeze(-1)           # [B,H,1,1]
    )
    coefs = F.softmax(F.leaky_relu(logits) + bias_mat, dim=-1)
    ret = torch.matmul(coefs, values)                   # [B,H,N,F]
    ret = ret.permute(0, 2, 1, 3).reshape(b, n, self.out_size)  # [B,N,H*F]

    if self.residual:
      ret = ret + self.skip(z)
    if self.activation is not None:
      ret = self.activation(ret)
    if self.use_ln:
      ret = self.ln(ret)

    return ret, None


class GATFull(GAT):
  def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused):
    adj_mat = torch.ones_like(adj_mat)
    return super().forward(node_fts, edge_fts, graph_fts, adj_mat, hidden)


# ---------------------------------------------------------------------------
# GATv2 (Brody et al., ICLR'22)
# ---------------------------------------------------------------------------
class GATv2(Processor):
  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = F.relu,
      residual: bool = True,
      use_ln: bool = False,
      name: str = 'gatv2_aggr',
  ):
    super().__init__(name=name)
    self.out_size = out_size
    self.nb_heads = nb_heads
    self.mid_size = out_size if mid_size is None else mid_size
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    if self.mid_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the message!')
    self.head_size = out_size // nb_heads
    self.mid_head_size = self.mid_size // nb_heads
    self.activation = activation
    self.residual = residual
    self.use_ln = use_ln

    # Lazy layers
    self.m = None
    self.skip = None
    self.w_1 = None
    self.w_2 = None
    self.w_e = None
    self.w_g = None
    self.a_heads = nn.ModuleList([nn.Linear(1, 1) for _ in range(nb_heads)])  # real in dims set lazily
    self._lazy = True
    if use_ln:
      self.ln = nn.LayerNorm(out_size)

  def _maybe_lazy_init(self, z: _Array, edge_fts: _Array, graph_fts: _Array):
    if self._lazy:
      in_z = z.shape[-1]
      in_e = edge_fts.shape[-1]
      in_g = graph_fts.shape[-1]
      self.m = nn.Linear(in_z, self.out_size)
      self.skip = nn.Linear(in_z, self.out_size)
      self.w_1 = nn.Linear(in_z, self.mid_size)
      self.w_2 = nn.Linear(in_z, self.mid_size)
      self.w_e = nn.Linear(in_e, self.mid_size)
      self.w_g = nn.Linear(in_g, self.mid_size)
      self.a_heads = nn.ModuleList([nn.Linear(self.mid_head_size, 1) for _ in range(self.nb_heads)])
      self._lazy = False

      self.to(z.device)
  def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused):
    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = torch.cat([node_fts, hidden], dim=-1)
    self._maybe_lazy_init(z, edge_fts, graph_fts)

    values = self.m(z).view(b, n, self.nb_heads, self.head_size).permute(0, 2, 1, 3)  # [B,H,N,F]

    pre_att_1 = self.w_1(z)
    pre_att_2 = self.w_2(z)
    pre_att_e = self.w_e(edge_fts)
    pre_att_g = self.w_g(graph_fts)

    pre_att = (
        pre_att_1.unsqueeze(1) +            # [B,1,N,H*F]
        pre_att_2.unsqueeze(2) +            # [B,N,1,H*F]
        pre_att_e +                         # [B,N,N,H*F]
        pre_att_g.unsqueeze(1).unsqueeze(2) # [B,1,1,H*F]
    )

    pre_att = pre_att.view(b, n, n, self.nb_heads, self.mid_head_size).permute(0, 3, 1, 2, 4)  # [B,H,N,N,F]

    # Per-head scoring
    logit_heads = []
    for h in range(self.nb_heads):
      logit_heads.append(self.a_heads[h](F.leaky_relu(pre_att[:, h])).squeeze(-1))  # [B,N,N]
    logits = torch.stack(logit_heads, dim=1)  # [B,H,N,N]

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = bias_mat.unsqueeze(-1).expand(-1, -1, -1, self.nb_heads).permute(0, 3, 1, 2)

    coefs = F.softmax(logits + bias_mat, dim=-1)
    ret = torch.matmul(coefs, values)                   # [B,H,N,F]
    ret = ret.permute(0, 2, 1, 3).reshape(b, n, self.out_size)

    if self.residual:
      ret = ret + self.skip(z)
    if self.activation is not None:
      ret = self.activation(ret)
    if self.use_ln:
      ret = self.ln(ret)

    return ret, None


class GATv2FullD2(GATv2):
  def d2_forward(self,
                 node_fts: List[_Array],
                 edge_fts: List[_Array],
                 graph_fts: List[_Array],
                 adj_mat: _Array,
                 hidden: _Array,
                 **unused_kwargs) -> List[_Array]:
    num_d2_actions = 4
    d2_inverses = [0, 1, 2, 3]
    d2_multiply = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]
    assert len(node_fts) == len(edge_fts) == len(graph_fts) == num_d2_actions
    ret_nodes = []
    adj_mat = torch.ones_like(adj_mat)
    for g in range(num_d2_actions):
      emb_values = []
      for h in range(num_d2_actions):
        gh = d2_multiply[d2_inverses[g]][h]
        node_features = torch.cat([node_fts[g], node_fts[gh]], dim=-1)
        edge_features = torch.cat([edge_fts[g], edge_fts[gh]], dim=-1)
        graph_features = torch.cat([graph_fts[g], graph_fts[gh]], dim=-1)
        ret, _ = super().forward(node_features, edge_features, graph_features, adj_mat, hidden)
        emb_values.append(ret)
      ret_nodes.append(torch.mean(torch.stack(emb_values, dim=0), dim=0))
    return ret_nodes


class GATv2Full(GATv2):
  def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused):
    adj_mat = torch.ones_like(adj_mat)
    return super().forward(node_fts, edge_fts, graph_fts, adj_mat, hidden)


# ---------------------------------------------------------------------------
# PGN / MPNN family
# ---------------------------------------------------------------------------
class PGN(Processor):
  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = F.relu,
      reduction: str = 'max',  # 'mean' | 'max' | 'sum'
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      name: str = 'mpnn_aggr',
  ):
    super().__init__(name=name)
    self.out_size = out_size
    self.mid_size = out_size if mid_size is None else mid_size
    self.mid_act = mid_act
    self.activation = activation
    self.reduction = reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated

    # Lazy linear layers (depend on input dims)
    self.m_1 = None
    self.m_2 = None
    self.m_e = None
    self.m_g = None
    self.o1 = None
    self.o2 = None

    if self._msgs_mlp_sizes is not None:
      self.msg_mlp = None  # lazy MLP

    if use_ln:
      self.ln = nn.LayerNorm(out_size)

    if self.gated:
      self.gate1 = None
      self.gate2 = None
      self.gate3 = None
      # gate3 bias will be initialized lazily

    self._lazy = True

  def _maybe_lazy_init(self, z: _Array, edge_fts: _Array, graph_fts: _Array):
    """Create all learnable layers on first call, based on runtime dims."""
    if self._lazy:
      in_z = z.shape[-1]
      in_e = edge_fts.shape[-1]
      in_g = graph_fts.shape[-1]

      # Node/edge/global message path (ret channel)
      self.m_1 = nn.Linear(in_z, self.mid_size)
      self.m_2 = nn.Linear(in_z, self.mid_size)
      self.m_e = nn.Linear(in_e, self.mid_size)
      self.m_g = nn.Linear(in_g, self.mid_size)
      self.o1 = nn.Linear(in_z, self.out_size)
      self.o2 = nn.Linear(self.mid_size, self.out_size)

      # ===== BEGIN EVOLVE TRIPLET PROJECTIONS REGION =====
      if self.use_triplets:
        # Shared projection with directional split
        self.t_leg_shared = nn.Linear(in_e, self.nb_triplet_fts)
        self.t_direct = nn.Linear(in_e, self.nb_triplet_fts // 2)
        self.t_global = nn.Linear(in_g, self.nb_triplet_fts // 2)
        self.o3 = nn.Linear(self.nb_triplet_fts // 2, self.out_size)
      # ===== END EVOLVE TRIPLET PROJECTIONS REGION =====

      # Optional MLP on messages
      if self._msgs_mlp_sizes is not None:
        layers: List[nn.Module] = []
        d = self.mid_size
        for h in self._msgs_mlp_sizes:
          layers.append(nn.Linear(d, h))
          layers.append(nn.ReLU())
          d = h
        self.msg_mlp = nn.Sequential(*layers)

      # Optional gating
      if self.gated:
        self.gate1 = nn.Linear(in_z, self.out_size)
        self.gate2 = nn.Linear(self.mid_size, self.out_size)
        self.gate3 = nn.Linear(self.out_size, self.out_size)
        with torch.no_grad():
          self.gate3.bias.fill_(-3.0)

      self._lazy = False
      self.to(z.device)

  def _compute_tri_msgs(
      self,
      z: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
  ) -> Optional[_Array]:
    if not self.use_triplets:
      return None

    # ===== BEGIN EVOLVE TRI_MSGS REGION =====
    # Shared projection with directional split
    leg_proj = self.t_leg_shared(edge_fts)  # (B, N, N, H_t)
    leg1, leg2 = torch.chunk(leg_proj, 2, dim=-1)  # Each (B, N, N, H_t//2)
    # Asymmetric activation: only apply to leg2
    leg2 = F.relu(leg2)

    direct_edge = self.t_direct(edge_fts)  # (B, N, N, H_t//2)
    global_feat = self.t_global(graph_fts).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H_t//2)

    # Apply adjacency masks
    adj_exp = adj_mat.unsqueeze(-1)
    leg1 = leg1 * adj_exp
    leg2 = leg2 * adj_exp

    # Factorized aggregation: Σ_k [leg1(i,k) ⊙ leg2(k,j)]
    tri_feat = torch.einsum('bikh,bkjh->bijh', leg1, leg2)  # (B, N, N, H_t//2)

    # Combine components
    tri_feat = tri_feat + direct_edge + global_feat.expand_as(tri_feat)

    # Final projection
    tri_msgs = self.o3(tri_feat)
    if self.activation is not None:
      tri_msgs = self.activation(tri_msgs)
    # ===== END EVOLVE TRI_MSGS REGION =====

    return tri_msgs

  def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused):
    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    # Concatenate node features and hidden state as input z
    z = torch.cat([node_fts, hidden], dim=-1)
    self._maybe_lazy_init(z, edge_fts, graph_fts)

    # --- Node/edge/global message path for ret ---
    msg_1 = self.m_1(z)
    msg_2 = self.m_2(z)
    msg_e = self.m_e(edge_fts)
    msg_g = self.m_g(graph_fts)

    msgs = (
        msg_1.unsqueeze(1) +
        msg_2.unsqueeze(2) +
        msg_e +
        msg_g.unsqueeze(1).unsqueeze(2)
    )  # (B, N, N, mid_size)

    if self._msgs_mlp_sizes is not None:
      msgs = self.msg_mlp(F.relu(msgs))

    if self.mid_act is not None:
      msgs = self.mid_act(msgs)

    if self.reduction == 'mean':
      msgs = torch.sum(msgs * adj_mat.unsqueeze(-1), dim=1)
      msgs = msgs / (adj_mat.sum(dim=-1, keepdim=True) + 1e-9)
    elif self.reduction == 'max':
      maxarg = torch.where(
          adj_mat.unsqueeze(-1).bool(),
          msgs,
          torch.full_like(msgs, -BIG_NUMBER),
      )
      msgs = torch.amax(maxarg, dim=1)
    else:  # 'sum' or custom handled upstream
      msgs = torch.sum(msgs * adj_mat.unsqueeze(-1), dim=1)

    h_1 = self.o1(z)
    h_2 = self.o2(msgs)
    ret = h_1 + h_2

    if self.activation is not None:
      ret = self.activation(ret)
    if self.use_ln:
      ret = self.ln(ret)

    if self.gated:
      gate = torch.sigmoid(
          self.gate3(F.relu(self.gate1(z) + self.gate2(msgs)))
      )
      ret = ret * gate + hidden * (1.0 - gate)

    # --- Triplet / high-order structural channel (evolvable) ---
    tri_msgs = self._compute_tri_msgs(z, edge_fts, graph_fts, adj_mat)

    return ret, tri_msgs

# class PGN(Processor):
#   def __init__(
#       self,
#       out_size: int,
#       mid_size: Optional[int] = None,
#       mid_act: Optional[_Fn] = None,
#       activation: Optional[_Fn] = F.relu,
#       reduction: str = 'max',  # 'mean' | 'max' | 'sum'
#       msgs_mlp_sizes: Optional[List[int]] = None,
#       use_ln: bool = False,
#       use_triplets: bool = False,
#       nb_triplet_fts: int = 8,
#       gated: bool = False,
#       name: str = 'mpnn_aggr',
#   ):
#     super().__init__(name=name)
#     self.out_size = out_size
#     self.mid_size = out_size if mid_size is None else mid_size
#     self.mid_act = mid_act
#     self.activation = activation
#     self.reduction = reduction
#     self._msgs_mlp_sizes = msgs_mlp_sizes
#     self.use_ln = use_ln
#     self.use_triplets = use_triplets
#     self.nb_triplet_fts = nb_triplet_fts
#     self.gated = gated
#
#     # Lazy linear layers (depend on input dims)
#     self.m_1 = None
#     self.m_2 = None
#     self.m_e = None
#     self.m_g = None
#     self.o1 = None
#     self.o2 = None
#
#     # Triplet path
#     if self.use_triplets:
#       self.t_1 = None; self.t_2 = None; self.t_3 = None
#       self.t_e_1 = None; self.t_e_2 = None; self.t_e_3 = None
#       self.t_g = None
#       self.o3 = None
#
#     if self._msgs_mlp_sizes is not None:
#       self.msg_mlp = None  # lazy MLP
#
#     if use_ln:
#       self.ln = nn.LayerNorm(out_size)
#
#     if self.gated:
#       self.gate1 = None; self.gate2 = None; self.gate3 = None
#
#     self._lazy = True
#
#   def _maybe_lazy_init(self, z: _Array, edge_fts: _Array, graph_fts: _Array):
#     if self._lazy:
#       in_z = z.shape[-1]
#       in_e = edge_fts.shape[-1]
#       in_g = graph_fts.shape[-1]
#       self.m_1 = nn.Linear(in_z, self.mid_size)
#       self.m_2 = nn.Linear(in_z, self.mid_size)
#       self.m_e = nn.Linear(in_e, self.mid_size)
#       self.m_g = nn.Linear(in_g, self.mid_size)
#       self.o1 = nn.Linear(in_z, self.out_size)
#       self.o2 = nn.Linear(self.mid_size, self.out_size)
#       if self.use_triplets:
#         self.t_1 = nn.Linear(in_z, self.nb_triplet_fts)
#         self.t_2 = nn.Linear(in_z, self.nb_triplet_fts)
#         self.t_3 = nn.Linear(in_z, self.nb_triplet_fts)
#         self.t_e_1 = nn.Linear(in_e, self.nb_triplet_fts)
#         self.t_e_2 = nn.Linear(in_e, self.nb_triplet_fts)
#         self.t_e_3 = nn.Linear(in_e, self.nb_triplet_fts)
#         self.t_g = nn.Linear(in_g, self.nb_triplet_fts)
#         self.o3 = nn.Linear(self.nb_triplet_fts, self.out_size)
#       if self._msgs_mlp_sizes is not None:
#         layers = []
#         d = self.mid_size
#         for h in self._msgs_mlp_sizes:
#           layers.append(nn.Linear(d, h)); layers.append(nn.ReLU())
#           d = h
#         self.msg_mlp = nn.Sequential(*layers)
#       if self.gated:
#         self.gate1 = nn.Linear(in_z, self.out_size)
#         self.gate2 = nn.Linear(self.mid_size, self.out_size)
#         self.gate3 = nn.Linear(self.out_size, self.out_size)
#         with torch.no_grad():
#           self.gate3.bias.fill_(-3.0)
#       self._lazy = False
#       self.to(z.device)
#
#   def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused):
#     b, n, _ = node_fts.shape
#     assert edge_fts.shape[:-1] == (b, n, n)
#     assert graph_fts.shape[:-1] == (b,)
#     assert adj_mat.shape == (b, n, n)
#
#     z = torch.cat([node_fts, hidden], dim=-1)
#     self._maybe_lazy_init(z, edge_fts, graph_fts)
#
#     msg_1 = self.m_1(z)
#     msg_2 = self.m_2(z)
#     msg_e = self.m_e(edge_fts)
#     msg_g = self.m_g(graph_fts)
#
#     tri_msgs = None
#     if self.use_triplets:
#       # Triplet messages (Dudzik & Veličković, 2022)
#       tri_1 = self.t_1(z)
#       tri_2 = self.t_2(z)
#       tri_3 = self.t_3(z)
#       tri_e_1 = self.t_e_1(edge_fts)
#       tri_e_2 = self.t_e_2(edge_fts)
#       tri_e_3 = self.t_e_3(edge_fts)
#       tri_g = self.t_g(graph_fts)
#       triplets = (
#         tri_1.unsqueeze(2).unsqueeze(3) +      # (B,N,1,1,H)
#         tri_2.unsqueeze(1).unsqueeze(3) +      # (B,1,N,1,H)
#         tri_3.unsqueeze(1).unsqueeze(2) +      # (B,1,1,N,H)
#         tri_e_1.unsqueeze(3) +                 # (B,N,N,1,H)
#         tri_e_2.unsqueeze(2) +                 # (B,N,1,N,H)
#         tri_e_3.unsqueeze(1) +                 # (B,1,N,N,H)
#         tri_g.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B,1,1,1,H)
#       )
#       tri_msgs = self.o3(torch.amax(triplets, dim=1))  # (B,N,N,H)
#       if self.activation is not None:
#         tri_msgs = self.activation(tri_msgs)
#
#     msgs = (
#         msg_1.unsqueeze(1) + msg_2.unsqueeze(2) +
#         msg_e + msg_g.unsqueeze(1).unsqueeze(2)
#     )  # (B,N,N,H)
#
#     if self._msgs_mlp_sizes is not None:
#       msgs = self.msg_mlp(F.relu(msgs))
#
#     if self.mid_act is not None:
#       msgs = self.mid_act(msgs)
#
#     if self.reduction == 'mean':
#       msgs = torch.sum(msgs * adj_mat.unsqueeze(-1), dim=1)
#       msgs = msgs / (adj_mat.sum(dim=-1, keepdim=True) + 1e-9)
#     elif self.reduction == 'max':
#       maxarg = torch.where(adj_mat.unsqueeze(-1).bool(), msgs, torch.full_like(msgs, -BIG_NUMBER))
#       msgs = torch.amax(maxarg, dim=1)
#     else:  # 'sum' or custom handled upstream
#       msgs = torch.sum(msgs * adj_mat.unsqueeze(-1), dim=1)
#
#     h_1 = self.o1(z)
#     h_2 = self.o2(msgs)
#
#     ret = h_1 + h_2
#
#     if self.activation is not None:
#       ret = self.activation(ret)
#     if self.use_ln:
#       ret = self.ln(ret)
#
#     if self.gated:
#       gate = torch.sigmoid(self.gate3(F.relu(self.gate1(z) + self.gate2(msgs))))
#       ret = ret * gate + hidden * (1.0 - gate)
#
#     return ret, tri_msgs


class DeepSets(PGN):
  def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused):
    assert adj_mat.ndim == 3
    adj_mat = torch.ones_like(adj_mat) * torch.eye(adj_mat.shape[-1], device=adj_mat.device)
    return super().forward(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class MPNN(PGN):
  def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused):
    adj_mat = torch.ones_like(adj_mat)
    return super().forward(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class PGNMask(PGN):
  @property
  def inf_bias(self):
    return True

  @property
  def inf_bias_edge(self):
    return True


# ---------------------------------------------------------------------------
# End-to-End Memory Networks (masked / full)
# ---------------------------------------------------------------------------
class MemNetMasked(Processor):
  """End-to-End Memory Networks (Sukhbaatar et al., 2015)."""
  def __init__(
      self,
      vocab_size: int,
      sentence_size: int,
      linear_output_size: int,
      embedding_size: int = 16,
      memory_size: Optional[int] = 128,
      num_hops: int = 1,
      nonlin: Callable[[Any], Any] = F.relu,
      apply_embeddings: bool = True,
      init_func: Optional[Callable[..., torch.Tensor]] = None,
      use_ln: bool = False,
      name: str = 'memnet',
  ) -> None:
    super().__init__(name=name)
    self._vocab_size = vocab_size
    self._embedding_size = embedding_size
    self._sentence_size = sentence_size
    self._memory_size = memory_size
    self._linear_output_size = linear_output_size
    self._num_hops = num_hops
    self._nonlin = nonlin
    self._apply_embeddings = apply_embeddings
    self._use_ln = use_ln

    # Positional encodings (E) of shape [sentence_size, embedding_size]
    enc = _position_encoding(sentence_size, embedding_size)
    self.register_buffer('_encodings', torch.tensor(enc, dtype=torch.float32), persistent=False)

    # Embedding/bias parameters analogous to HK get_parameter
    if apply_embeddings:
      self.query_biases = nn.Parameter(torch.zeros(vocab_size - 1, embedding_size))
      self.stories_biases = nn.Parameter(torch.zeros(vocab_size - 1, embedding_size))
      self.memory_contents = nn.Parameter(torch.zeros(memory_size, embedding_size))
      self.output_biases = nn.Parameter(torch.zeros(vocab_size - 1, embedding_size))

    # Linear maps (no bias where HK used with_bias=False)
    self.intermediate_linear = nn.Linear(embedding_size, embedding_size, bias=False)
    self.output_linear = nn.Linear(embedding_size, linear_output_size, bias=False)
    self.final_linear = nn.Linear(linear_output_size, vocab_size, bias=False)

    if use_ln:
      self.ln = nn.LayerNorm(vocab_size)

  def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused):
    del hidden
    # node_fts: [B,N,Hn], graph_fts: [B,Hg] — here treated as word indices/sequences
    node_and_graph_fts = torch.cat([node_fts, graph_fts[:, None, :]], dim=1)  # [B,N+1,H]
    edge_fts_padded = (edge_fts * adj_mat.unsqueeze(-1))
    edge_fts_padded = F.pad(edge_fts_padded, (0,0,0,1,0,1))  # pad N->N+1 on both axes

    nxt_hidden = []
    B = node_and_graph_fts.shape[0]
    for t in range(node_and_graph_fts.shape[1]):
      q = node_and_graph_fts[:, t, :]                       # [B, sentence_size]
      S = edge_fts_padded[:, t, :, :]                       # [B, memory_size, sentence_size]
      nxt_hidden.append(self._apply(q, S))
    nxt_hidden = torch.stack(nxt_hidden, dim=1)             # [B, N+1, vocab]
    # Broadcast hidden at graph position across nodes and drop the last
    nxt_hidden = nxt_hidden[:, :-1, :] + nxt_hidden[:, -1:, :]
    return nxt_hidden, None

  def _apply(self, queries: _Array, stories: _Array) -> _Array:
    # queries: [B, sentence_size] (as ints or embeddings)
    # stories: [B, memory_size, sentence_size]
    B = queries.shape[0]
    if self._apply_embeddings:
      nil_word_slot = torch.zeros(1, self._embedding_size, device=queries.device)
      # A: story embeddings
      stories_biases = torch.cat([self.stories_biases, nil_word_slot], dim=0)
      idx = stories.long().clamp_min(0)  # assume inputs are >=0 indices
      mem_emb = F.embedding(idx, stories_biases)  # [B, M, S, E]
      pad_M = self._memory_size - mem_emb.shape[1]
      if pad_M > 0:
        mem_emb = F.pad(mem_emb, (0,0,0,0,0,pad_M))
      memory = (mem_emb * self._encodings[None, None, :, :]).sum(dim=2) + self.memory_contents  # [B,M,E]
      # B: query embedding
      query_biases = torch.cat([self.query_biases, nil_word_slot], dim=0)
      q_emb = F.embedding(queries.long().clamp_min(0), query_biases)  # [B,S,E]
      u = (q_emb * self._encodings[None, :, :]).sum(dim=1)            # [B,E]
    else:
      memory = stories
      u = queries

    # C: output embeddings
    if self._apply_embeddings:
      output_biases = torch.cat([self.output_biases, nil_word_slot], dim=0)
      out_emb = F.embedding(stories.long().clamp_min(0), output_biases)  # [B,M,S,E]
      pad_M = self._memory_size - out_emb.shape[1]
      if pad_M > 0:
        out_emb = F.pad(out_emb, (0,0,0,0,0,pad_M))
      Cj = (out_emb * self._encodings[None, None, :, :]).sum(dim=2)   # [B,M,E]
    else:
      Cj = stories

    # Hops
    x = u
    for hop in range(self._num_hops):
      probs = F.softmax((memory * x[:, None, :]).sum(dim=2), dim=-1)  # [B,M]
      o = (Cj * probs[:, :, None]).sum(dim=1)                         # [B,E]
      if hop == self._num_hops - 1:
        x = self.output_linear(x + o)
      else:
        x = self.intermediate_linear(x + o)
      if self._nonlin is not None:
        x = self._nonlin(x)

    ret = self.final_linear(x)  # [B, vocab]
    if self._use_ln:
      ret = self.ln(ret)
    return ret


class MemNetFull(MemNetMasked):
  def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused):
    adj_mat = torch.ones_like(adj_mat)
    return super().forward(node_fts, edge_fts, graph_fts, adj_mat, hidden)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
ProcessorFactory = Callable[[int], Processor]


def get_processor_factory(kind: str,
                          use_ln: bool,
                          nb_triplet_fts: int,
                          nb_heads: Optional[int] = None) -> ProcessorFactory:
  """Returns a processor factory (Torch version)."""
  def _factory(out_size: int):
    if kind == 'deepsets':
      return DeepSets(out_size=out_size, msgs_mlp_sizes=[out_size, out_size], use_ln=use_ln, use_triplets=False, nb_triplet_fts=0)
    elif kind == 'gat':
      return GAT(out_size=out_size, nb_heads=nb_heads, use_ln=use_ln)
    elif kind == 'gat_full':
      return GATFull(out_size=out_size, nb_heads=nb_heads, use_ln=use_ln)
    elif kind == 'gatv2':
      return GATv2(out_size=out_size, nb_heads=nb_heads, use_ln=use_ln)
    elif kind == 'gatv2_full':
      return GATv2Full(out_size=out_size, nb_heads=nb_heads, use_ln=use_ln)
    elif kind == 'memnet_full':
      return MemNetFull(vocab_size=out_size, sentence_size=out_size, linear_output_size=out_size)
    elif kind == 'memnet_masked':
      return MemNetMasked(vocab_size=out_size, sentence_size=out_size, linear_output_size=out_size)
    elif kind == 'mpnn':
      return MPNN(out_size=out_size, msgs_mlp_sizes=[out_size, out_size], use_ln=use_ln, use_triplets=False, nb_triplet_fts=0)
    elif kind == 'pgn':
      return PGN(out_size=out_size, msgs_mlp_sizes=[out_size, out_size], use_ln=use_ln, use_triplets=False, nb_triplet_fts=0)
    elif kind == 'pgn_mask':
      return PGNMask(out_size=out_size, msgs_mlp_sizes=[out_size, out_size], use_ln=use_ln, use_triplets=False, nb_triplet_fts=0)
    elif kind == 'triplet_mpnn':
      return MPNN(out_size=out_size, msgs_mlp_sizes=[out_size, out_size], use_ln=use_ln, use_triplets=True, nb_triplet_fts=nb_triplet_fts)
    elif kind == 'triplet_pgn':
      return PGN(out_size=out_size, msgs_mlp_sizes=[out_size, out_size], use_ln=use_ln, use_triplets=True, nb_triplet_fts=nb_triplet_fts)
    elif kind == 'triplet_pgn_mask':
      return PGNMask(out_size=out_size, msgs_mlp_sizes=[out_size, out_size], use_ln=use_ln, use_triplets=True, nb_triplet_fts=nb_triplet_fts)
    elif kind == 'gpgn':
      return PGN(out_size=out_size, msgs_mlp_sizes=[out_size, out_size], use_ln=use_ln, use_triplets=False, nb_triplet_fts=nb_triplet_fts, gated=True)
    elif kind == 'gpgn_mask':
      return PGNMask(out_size=out_size, msgs_mlp_sizes=[out_size, out_size], use_ln=use_ln, use_triplets=False, nb_triplet_fts=nb_triplet_fts, gated=True)
    elif kind == 'gmpnn':
      return MPNN(out_size=out_size, msgs_mlp_sizes=[out_size, out_size], use_ln=use_ln, use_triplets=False, nb_triplet_fts=nb_triplet_fts, gated=True)
    elif kind == 'triplet_gpgn':
      return PGN(out_size=out_size, msgs_mlp_sizes=[out_size, out_size], use_ln=use_ln, use_triplets=True, nb_triplet_fts=nb_triplet_fts, gated=True)
    elif kind == 'triplet_gpgn_mask':
      return PGNMask(out_size=out_size, msgs_mlp_sizes=[out_size, out_size], use_ln=use_ln, use_triplets=True, nb_triplet_fts=nb_triplet_fts, gated=True)
    elif kind == 'triplet_gmpnn':
      return MPNN(out_size=out_size, msgs_mlp_sizes=[out_size, out_size], use_ln=use_ln, use_triplets=True, nb_triplet_fts=nb_triplet_fts, gated=True)
    else:
      raise ValueError('Unexpected processor kind ' + str(kind))
  return _factory


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def _position_encoding(sentence_size: int, embedding_size: int) -> np.ndarray:
  encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
  ls = sentence_size + 1
  le = embedding_size + 1
  for i in range(1, le):
    for j in range(1, ls):
      encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
  encoding = 1 + 4 * encoding / embedding_size / sentence_size
  return encoding.T
