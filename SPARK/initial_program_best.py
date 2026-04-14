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


# ===== BEGIN EVOLVE PGN REGION =====
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
        # Baseline design: simple Linear projections to a shared H_t space.
        self.t_1 = nn.Linear(in_z, self.nb_triplet_fts)
        self.t_2 = nn.Linear(in_z, self.nb_triplet_fts)
        self.t_3 = nn.Linear(in_z, self.nb_triplet_fts)
        self.t_e_1 = nn.Linear(in_e, self.nb_triplet_fts)
        self.t_e_2 = nn.Linear(in_e, self.nb_triplet_fts)
        self.t_e_3 = nn.Linear(in_e, self.nb_triplet_fts)
        self.t_g = nn.Linear(in_g, self.nb_triplet_fts)
        self.o3 = nn.Linear(self.nb_triplet_fts, self.out_size)

        # LLM1: You may create NEW operators here when LLM2 says prefer B.
        # e.g., new nn.Linear, nn.ModuleList, nn.MultiheadAttention, gates, etc.
        # All operators used in tri_msgs must be defined here.



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
  ) -> Optional[_Array]:



    if not self.use_triplets:
      return None

    # ===== BEGIN EVOLVE TRI_MSGS REGION =====
    tri_1 = self.t_1(z)          # (B, N, H_t)
    tri_2 = self.t_2(z)          # (B, N, H_t)
    tri_3 = self.t_3(z)          # (B, N, H_t)
    tri_e_1 = self.t_e_1(edge_fts)  # (B, N, N, H_t)
    tri_e_2 = self.t_e_2(edge_fts)  # (B, N, N, H_t)
    tri_e_3 = self.t_e_3(edge_fts)  # (B, N, N, H_t)
    tri_g = self.t_g(graph_fts)     # (B, H_t)

    triplets = (
        tri_1.unsqueeze(2).unsqueeze(3) +      # (B, N, 1, 1, H_t)
        tri_2.unsqueeze(1).unsqueeze(3) +      # (B, 1, N, 1, H_t)
        tri_3.unsqueeze(1).unsqueeze(2) +      # (B, 1, 1, N, H_t)
        tri_e_1.unsqueeze(3) +                 # (B, N, N, 1, H_t)
        tri_e_2.unsqueeze(2) +                 # (B, N, 1, N, H_t)
        tri_e_3.unsqueeze(1) +                 # (B, 1, N, N, H_t)
        tri_g.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B, 1, 1, 1, H_t)
    )
    tri_msgs = self.o3(torch.amax(triplets, dim=1))  # (B, N, N, out_size)
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
    tri_msgs = self._compute_tri_msgs(z, edge_fts, graph_fts)

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

