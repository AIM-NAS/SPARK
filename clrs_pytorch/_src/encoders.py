# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Encoder utilities (PyTorch reimplementation).

仅对原 encoders.py 中涉及 TensorFlow/JAX/Haiku 的位置做 PyTorch 等价改写，
其余逻辑、函数签名与返回值保持一致（尽量按行对应）。
"""

from __future__ import annotations

import functools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from clrs_pytorch._src import probing
from clrs_pytorch._src import specs

_Array = torch.Tensor
_DataPoint = probing.DataPoint
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Type = specs.Type


# ------------------------------ utils ------------------------------



# def _apply_linear(lin: nn.Linear, x: _Array) -> _Array:
#   in_features = x.shape[-1]
#   if lin.in_features != in_features:
#     # 在 lin 当前的 device/dtype 上重建
#     dev = lin.weight.device
#     dt = lin.weight.dtype
#     new_lin = nn.Linear(in_features=in_features, out_features=lin.out_features, bias=True).to(dev, dtype=dt)
#     with torch.no_grad():
#       # 用一个简单但稳定的方式初始化：沿输入维复制/均值，不改变尺度
#       w_mean = lin.weight.to(dt).to(dev)
#       if w_mean.dim() == 2 and w_mean.size(1) > 0:
#         w_fill = w_mean.mean(dim=1, keepdim=True).expand(-1, in_features)
#       else:
#         w_fill = torch.zeros(lin.out_features, in_features, device=dev, dtype=dt)
#       new_lin.weight.copy_(w_fill)
#       if lin.bias is not None and new_lin.bias is not None:
#         new_lin.bias.copy_(lin.bias.to(dt).to(dev))
#     # 真正替换参数到 lin（保持原模块注册关系）
#     lin.weight = nn.Parameter(new_lin.weight)
#     if lin.bias is not None:
#       lin.bias = nn.Parameter(new_lin.bias)
#     # 维护一下 in_features 元信息（便于后续判断）
#     lin.in_features = in_features  # type: ignore[attr-defined]
#   return lin(x)


# ------------------------------ public API ------------------------------

# encoders.py
import torch
import torch.nn as nn


def _make_linear(hidden_dim: int, init: str, stage: str, t: str, name: str) -> nn.Module:
    # 直接用 LazyLinear，首次前向会自动推断 in_features 并初始化
    return nn.LazyLinear(out_features=hidden_dim, bias=True)


def construct_encoders(stage: str, loc: str, t: str,
                       hidden_dim: int, init: str, name: str):
    if init not in ['default', 'xavier_on_scalars']:
        raise ValueError(f'Encoder initialiser {init} not supported.')
    encs = nn.ModuleList([_make_linear(hidden_dim, init, stage, t, f'{name}_enc0')])
    # EDGE + POINTER 需要第二条分支（senders/receivers 双向聚合）
    from .specs import Location as _Location, Type as _Type
    if loc == _Location.EDGE and t == _Type.POINTER:
        encs.append(_make_linear(hidden_dim, init, stage, t, f'{name}_enc1'))
    return encs

def preprocess(dp: _DataPoint, nb_nodes: int) -> _DataPoint:
  """Pre-process data point. (JAX→PyTorch)

  - POINTER: 压缩索引 → one-hot（末维大小为 nb_nodes）
  - SOFT_POINTER: 已经是分布，改 type_ 为 POINTER
  - 其他：转为 float32
  """
  new_type = dp.type_
  if dp.type_ == _Type.POINTER:
    # hk.one_hot → F.one_hot，保持 float32
    data = F.one_hot(dp.data.to(torch.long), num_classes=nb_nodes).to(torch.float32)
  else:
    data = dp.data.to(torch.float32)
    if dp.type_ == _Type.SOFT_POINTER:
      new_type = _Type.POINTER
  dp = probing.DataPoint(name=dp.name, location=dp.location, type_=new_type, data=data)
  return dp


# encoders.py

def accum_adj_mat(dp: _DataPoint, adj_mat: _Array) -> _Array:
  adj_mat = adj_mat.to(torch.float32)

  x = dp.data
  if not isinstance(x, torch.Tensor):
      return (adj_mat > 0.).to(torch.float32)

  # 允许 [B,N,N] 或更高维度：从最后两维取 [N,N]

  if x.dim() >= 2:
      N1, N2 = x.shape[-2], x.shape[-1]
      if N1 == adj_mat.shape[-2] and N2 == adj_mat.shape[-1]:
          x2 = x
          # 若多了时间维，把时间挤掉：对时间求 max/any（让任一步的连边都算入）
          while x2.dim() > 3:
              # 对非 (B,N,N) 的前置维度做 any
              x2 = (x2 > 0.5).any(dim=0).to(x2.dtype)
          if dp.location == _Location.NODE and dp.type_ in [_Type.POINTER, _Type.PERMUTATION_POINTER]:
              sym = (x2 + x2.transpose(-1, -2)) > 0.5
              adj_mat = adj_mat + sym.to(torch.float32)
          elif dp.location == _Location.EDGE and dp.type_ == _Type.MASK:
              sym = (x2 + x2.transpose(-1, -2)) > 0.0
              adj_mat = adj_mat + sym.to(torch.float32)

  return (adj_mat > 0.).to(torch.float32)



def accum_edge_fts(encoders, dp: _DataPoint, edge_fts: _Array) -> _Array:
  x = dp.data
  if not isinstance(x, torch.Tensor):
      return edge_fts

  # 统一把时间维挤掉：取当前步 or 对时间维做 mean/max（这里做 mean 稳定）
  while x.dim() > 3:
      x = x.mean(dim=0)

  if dp.location == _Location.NODE and dp.type_ in [_Type.POINTER, _Type.PERMUTATION_POINTER]:
      encoding = _encode_inputs(encoders, probing.DataPoint(dp.name, dp.location, dp.type_, x))
      # print("edge_fts:",edge_fts.shape,encoding.shape)
      edge_fts = edge_fts + encoding
      return edge_fts

  if dp.location == _Location.EDGE:

      encoding = _encode_inputs(encoders, probing.DataPoint(dp.name, dp.location, dp.type_, x))
      # print("first encoding1:", dp)
      if dp.type_ == _Type.POINTER:
          enc2 = encoders[1](x.unsqueeze(-1))  # [B,N,N,1] -> [B,N,N,H]
          edge_fts = edge_fts + encoding.mean(dim=1) + enc2.mean(dim=2)
      else:
          # print("edge_fts1:", edge_fts.shape, encoding.shape)
          edge_fts = edge_fts + encoding
  return edge_fts



def accum_node_fts(encoders, dp: _DataPoint, node_fts: _Array) -> _Array:
  """Encodes and accumulates node features. (JAX→PyTorch)"""
  is_pointer = (dp.type_ in [_Type.POINTER, _Type.PERMUTATION_POINTER])
  if ((dp.location == _Location.NODE and not is_pointer) or
      (dp.location == _Location.GRAPH and dp.type_ == _Type.POINTER)):
    encoding = _encode_inputs(encoders, dp)
    # print("node_fts 1:",node_fts.shape,encoding.shape)
    node_fts = node_fts + encoding
  return node_fts


def accum_graph_fts(encoders, dp: _DataPoint, graph_fts: _Array) -> _Array:
  """Encodes and accumulates graph features. (JAX→PyTorch)"""
  if dp.location == _Location.GRAPH and dp.type_ != _Type.POINTER:
    encoding = _encode_inputs(encoders, dp)
    graph_fts = graph_fts + encoding
  return graph_fts


def _encode_inputs(encoders, dp: _DataPoint) -> _Array:
    """
    PyTorch 版本：
    - CATEGORICAL：直接送入 encoders[0]
    - 其它类型：在末维扩一维后再送入 encoders[0]
    说明：encoders[0] 建议用 nn.LazyLinear 或 in_features 对齐的 nn.Linear。
    """
    # 保证张量/设备/精度与模块一致
    device = next(encoders[0].parameters()).device
    x = torch.as_tensor(dp.data, dtype=torch.float32, device=device)

    if dp.type_ == _Type.CATEGORICAL:
        encoding = encoders[0](x)                # [..., C] -> [..., H]
    else:
        encoding = encoders[0](x.unsqueeze(-1))  # [..., 1] -> [..., H]

    return encoding