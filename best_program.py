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

class Processor(nn.Module):
  """Processor abstract base class."""

  def __init__(self, name: str):
    super().__init__()
    if not name.endswith(PROCESSOR_TAG):
      name = name + '_' + PROCESSOR_TAG
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
    raise NotImplementedError

  @property
  def inf_bias(self):
    return False

  @property
  def inf_bias_edge(self):
    return False


# ===== BEGIN EVOLVE PGN REGION =====
class PGN(Processor):
  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = F.relu,
      reduction: str = 'max',
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

    self.m_1 = None
    self.m_2 = None
    self.m_e = None
    self.m_g = None
    self.o1 = None
    self.o2 = None

    if self._msgs_mlp_sizes is not None:
      self.msg_mlp = None

    if use_ln:
      self.ln = nn.LayerNorm(out_size)

    if self.gated:
      self.gate1 = None
      self.gate2 = None
      self.gate3 = None

    self._lazy = True

  def _maybe_lazy_init(self, z: _Array, edge_fts: _Array, graph_fts: _Array):
    if self._lazy:
      in_z = z.shape[-1]
      in_e = edge_fts.shape[-1]
      in_g = graph_fts.shape[-1]

      self.m_1 = nn.Linear(in_z, self.mid_size)
      self.m_2 = nn.Linear(in_z, self.mid_size)
      self.m_e = nn.Linear(in_e, self.mid_size)
      self.m_g = nn.Linear(in_g, self.mid_size)
      self.o1 = nn.Linear(in_z, self.out_size)
      self.o2 = nn.Linear(self.mid_size, self.out_size)

      # ===== BEGIN EVOLVE TRIPLET PROJECTIONS REGION =====
      if self.use_triplets:
        # Factorized multi-head projections
        self.t_head = nn.Linear(in_z, self.nb_triplet_fts * 4)
        self.t_e_head = nn.Linear(in_e, self.nb_triplet_fts * 4)
        self.t_g = nn.Linear(in_g, self.nb_triplet_fts * 4)
        self.o3 = nn.Linear(self.nb_triplet_fts * 4, self.out_size)
        # Dynamic conditioning projections
        self.attn_proj_i = nn.Linear(in_z, self.nb_triplet_fts * 4)
        self.attn_proj_j = nn.Linear(in_z, self.nb_triplet_fts * 4)
        # Enhanced gating micro-MLP
        self.tri_gate_mlp = nn.Sequential(
            nn.Linear(self.nb_triplet_fts, 4),
            nn.ReLU(),
            nn.Linear(4, self.nb_triplet_fts),
            nn.Sigmoid()
        )
      # ===== END EVOLVE TRIPLET PROJECTIONS REGION =====

      if self._msgs_mlp_sizes is not None:
        layers: List[nn.Module] = []
        d = self.mid_size
        for h in self._msgs_mlp_sizes:
          layers.append(nn.Linear(d, h))
          layers.append(nn.ReLU())
          d = h
        self.msg_mlp = nn.Sequential(*layers)

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
    # Project to multi-head triplet space
    tri_head = self.t_head(z)  # (B, N, 4*H_t)
    tri_e_head = self.t_e_head(edge_fts)  # (B, N, N, 4*H_t)
    tri_g = self.t_g(graph_fts)  # (B, 4*H_t)

    # Reshape to head groups
    b, n, _ = z.shape
    h_t = self.nb_triplet_fts
    tri_head = tri_head.view(b, n, 4, h_t)
    tri_e_head = tri_e_head.view(b, n, n, 4, h_t)
    tri_g = tri_g.view(b, 1, 1, 4, h_t)

    # Dynamic conditioning vectors
    attn_i = self.attn_proj_i(z).view(b, n, 4, h_t)  # (B, N, 4, H_t)
    attn_j = self.attn_proj_j(z).view(b, n, 4, h_t)  # (B, N, 4, H_t)

    # Condition node features
    term_i = tri_head.unsqueeze(2) * attn_i.unsqueeze(2)  # (B, N, 1, 4, H_t)
    term_j = tri_head.unsqueeze(1) * attn_j.unsqueeze(1)  # (B, 1, N, 4, H_t)

    # Combine factors
    tri_msgs = (
        term_i + term_j + tri_e_head + tri_g
    )  # (B, N, N, 4, H_t)

    # Enhanced per-head gating
    tri_msgs_flat = tri_msgs.view(-1, h_t)  # (B*N*N*4, H_t)
    gate_flat = self.tri_gate_mlp(tri_msgs_flat)  # (B*N*N*4, H_t)
    gate = gate_flat.view(b, n, n, 4, h_t)  # (B, N, N, 4, H_t)
    tri_msgs = tri_msgs * gate

    # Flatten heads and project to output space
    tri_msgs_flat = tri_msgs.view(b, n, n, 4 * h_t)
    tri_msgs = self.o3(tri_msgs_flat)
    if self.activation is not None:
      tri_msgs = self.activation(tri_msgs)

    # Zero out non-edges
    tri_msgs = tri_msgs * adj_mat.unsqueeze(-1)
    # ===== END EVOLVE TRI_MSGS REGION =====

    return tri_msgs

  def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused):
    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = torch.cat([node_fts, hidden], dim=-1)
    self._maybe_lazy_init(z, edge_fts, graph_fts)

    msg_1 = self.m_1(z)
    msg_2 = self.m_2(z)
    msg_e = self.m_e(edge_fts)
    msg_g = self.m_g(graph_fts)

    msgs = (
        msg_1.unsqueeze(1) +
        msg_2.unsqueeze(2) +
        msg_e +
        msg_g.unsqueeze(1).unsqueeze(2)
    )

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
    else:
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

    # Pass adj_mat to triplet computation
    tri_msgs = self._compute_tri_msgs(z, edge_fts, graph_fts, adj_mat)

    return ret, tri_msgs
# ===== END EVOLVE PGN REGION =====

class MPNN(PGN):
  def forward(self, node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused):
    adj_mat = torch.ones_like(adj_mat)
    return super().forward(node_fts, edge_fts, graph_fts, adj_mat, hidden)

ProcessorFactory = Callable[[int], Processor]
def get_processor_factory(kind: str = "mpnn",
                          use_ln: bool = True,
                          nb_triplet_fts: int = 0,
                          nb_heads: Optional[int] = None
                          ) -> ProcessorFactory:
  def _factory(out_size: int) -> MPNN:
    return MPNN(
      out_size=out_size,
      msgs_mlp_sizes=(out_size, out_size),
      use_ln=use_ln,
      use_triplets=True,
      nb_triplet_fts=8,
      gated=True,
    )

  return _factory