# -*- coding: utf-8 -*-
# PyTorch reimplementation of probing.py (DeepMind CLRS) focusing on TF/JAX parts.
# Keep logic and shapes aligned with the original.

from __future__ import annotations

import functools
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np

from . import specs  # 注意：指向你工程里的 specs；路径按你的工程调整

_Location = specs.Location
_Stage = specs.Stage
_Type = specs.Type
_OutputClass = specs.OutputClass

_Tensor = torch.Tensor
_Data = Union[_Tensor, List[_Tensor]]
_DataOrType = Union[_Data, str]

ProbesDict = Dict[str, Dict[str, Dict[str, Dict[str, _DataOrType]]]]


def _convert_to_str(element):
    """兼容 bytes/np/torch 的字符串解码；去掉 tf 依赖。"""
    # torch.Tensor -> numpy 标量/数组
    if isinstance(element, torch.Tensor):
        # 期望是标量或 0D/1D 裸字节串，尽力处理
        try:
            element = element.detach().cpu().numpy()
        except Exception:
            element = element.item() if element.numel() == 1 else element

    # numpy 标量/数组
    if isinstance(element, np.ndarray):
        if element.shape == ():  # 标量
            element = element.item()
        else:
            # 若是存了字节数组，取第一个
            element = element.flatten()[0].item() if element.size > 0 else element

    # python bytes
    if isinstance(element, (bytes, np.bytes_)):
        try:
            return element.decode("utf-8")
        except Exception:
            return str(element)

    # python 标量/其他
    return element


class DataPoint:
    """Describes a data point. PyTorch version (keeps interface)."""

    def __init__(self, name: str, location: str, type_: str, data: _Tensor):
        self._name = name
        self._location = location
        self._type_ = type_
        self.data = data

    @property
    def name(self):
        return _convert_to_str(self._name)

    @property
    def location(self):
        return _convert_to_str(self._location)

    @property
    def type_(self):
        return _convert_to_str(self._type_)

    def __repr__(self):
        shp = tuple(self.data.shape) if isinstance(self.data, torch.Tensor) else "?"
        s = f'DataPoint(name="{self.name}",\tlocation={self.location},\t'
        return s + f"type={self.type_},\tdata=Tensor{shp})"

    # 兼容“树”接口（非必须，仅保留结构语义）
    def tree_flatten(self):
        data = (self.data,)
        meta = (self.name, self.location, self.type_)
        return data, meta

    @classmethod
    def tree_unflatten(cls, meta, data):
        name, location, type_ = meta
        subdata, = data
        return DataPoint(name, location, type_, subdata)


class ProbeError(Exception):
    pass


def _ensure_tensor(x, like: torch.device | None = None) -> torch.Tensor:
    """把各种输入转成 torch.Tensor；保留 dtype；可选搬到 device。"""
    if isinstance(x, torch.Tensor):
        return x.to(like) if isinstance(like, torch.device) else x
    if isinstance(x, (np.ndarray, np.number)):
        t = torch.from_numpy(np.array(x))
    else:
        t = torch.as_tensor(x)
    return t.to(like) if isinstance(like, torch.device) else t


def initialize(spec: specs.Spec) -> ProbesDict:
    """Initializes an empty `ProbesDict` corresponding with the provided spec."""
    probes: ProbesDict = dict()
    for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
        probes[stage] = {}
        for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
            probes[stage][loc] = {}

    for name in spec:
        stage, loc, t = spec[name]
        probes[stage][loc][name] = {}
        probes[stage][loc][name]["data"] = []
        probes[stage][loc][name]["type_"] = t

    return probes


def push(probes: ProbesDict, stage: str, next_probe):
    """Pushes a probe into an existing `ProbesDict`."""
    for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
        for name in probes[stage][loc]:
            if name not in next_probe:
                raise ProbeError(f"Missing probe for {name}.")
            if isinstance(probes[stage][loc][name]["data"], torch.Tensor):
                raise ProbeError("Attemping to push to finalized `ProbesDict`.")
            probes[stage][loc][name]["data"].append(_ensure_tensor(next_probe[name]))


def finalize(probes: ProbesDict):
    """Finalizes a `ProbesDict` by stacking/squeezing `data` field."""
    for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
        for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
            for name in probes[stage][loc]:
                data_field = probes[stage][loc][name]["data"]
                if isinstance(data_field, torch.Tensor):
                    raise ProbeError("Attemping to re-finalize a finalized `ProbesDict`.")

                # data_field 是 List[Tensor]
                if stage == _Stage.HINT:
                    # Hints 跨时间步堆叠： [T, ...]
                    probes[stage][loc][name]["data"] = torch.stack(data_field, dim=0)
                else:
                    # Input/Output 只有一个实例：去掉前导维
                    stacked = torch.stack(data_field, dim=0)  # [1, ...] 常见
                    probes[stage][loc][name]["data"] = torch.squeeze(stacked, dim=0)


def split_stages(
    probes: ProbesDict,
    spec: specs.Spec,
) -> Tuple[List[DataPoint], List[DataPoint], List[DataPoint]]:
    """Splits contents of `ProbesDict` into `DataPoint`s by stage."""
    inputs: List[DataPoint] = []
    outputs: List[DataPoint] = []
    hints: List[DataPoint] = []

    for name in spec:
        stage, loc, t = spec[name]

        if stage not in probes:
            raise ProbeError(f"Missing stage {stage}.")
        if loc not in probes[stage]:
            raise ProbeError(f"Missing location {loc}.")
        if name not in probes[stage][loc]:
            raise ProbeError(f"Missing probe {name}.")
        if "type_" not in probes[stage][loc][name]:
            raise ProbeError(f"Probe {name} missing attribute `type_`.")
        if "data" not in probes[stage][loc][name]:
            raise ProbeError(f"Probe {name} missing attribute `data`.")
        if t != probes[stage][loc][name]["type_"]:
            raise ProbeError(f"Probe {name} of incorrect type {t}.")

        data = probes[stage][loc][name]["data"]
        if not isinstance(data, torch.Tensor):
            raise ProbeError(
                (f'Invalid `data` for probe "{name}". Did you forget to call `probing.finalize`?')
            )

        if t in [_Type.MASK, _Type.MASK_ONE, _Type.CATEGORICAL]:
            # 检查取值为 {0,1,-1}
            vals_ok = ((data == 0) | (data == 1) | (data == -1)).all()
            if not bool(vals_ok):
                raise ProbeError(f"0|1|-1 `data` for probe \"{name}\"")
            # one-hot 检查（末轴和为 1）
            if t in [_Type.MASK_ONE, _Type.CATEGORICAL]:
                abs_sum = torch.sum(torch.abs(data), dim=-1)
                if not bool(torch.all(abs_sum == 1)):
                    raise ProbeError(f"Expected one-hot `data` for probe \"{name}\"")

        dim_to_expand = 1 if stage == _Stage.HINT else 0
        if dim_to_expand == 1:
            data_point = DataPoint(name=name, location=loc, type_=t, data=data.unsqueeze(1))
        else:
            data_point = DataPoint(name=name, location=loc, type_=t, data=data.unsqueeze(0))

        if stage == _Stage.INPUT:
            inputs.append(data_point)
        elif stage == _Stage.OUTPUT:
            outputs.append(data_point)
        else:
            hints.append(data_point)

    return inputs, outputs, hints


# ===== Helper probes (np -> torch) =====================================================

def array(A_pos):
  A_pos = _ensure_tensor(A_pos)
  N = A_pos.shape[0]
  probe = torch.arange(N, device=A_pos.device)
  for i in range(1, N):
    probe[A_pos[i]] = A_pos[i - 1]
  return probe

def array_cat(A, n: int) -> torch.Tensor:
  """Constructs an `array_cat` probe. One-hot with nb_classes=n."""
  assert n > 0
  A = _ensure_tensor(A)  # 统一成 torch.Tensor
  A = A.to(torch.long)  # 明确类别索引类型
  N = A.shape[0]
  probe = torch.zeros(N, n, device=A.device, dtype=torch.float32)
  probe[torch.arange(N, device=A.device), A] = 1.0
  return probe


def heap(A_pos, heap_size: int):
  A_pos = _ensure_tensor(A_pos)
  assert heap_size > 0
  N = A_pos.shape[0]
  probe = torch.arange(N, device=A_pos.device)
  for i in range(1, heap_size):
    probe[A_pos[i]] = A_pos[(i - 1) // 2]
  return probe


def _ensure_tensor(x, like: torch.device | None = None) -> torch.Tensor:
  if isinstance(x, torch.Tensor):
    return x.to(like) if isinstance(like, torch.device) else x
  if isinstance(x, (np.ndarray, np.number)):
    t = torch.from_numpy(np.array(x))
  else:
    t = torch.as_tensor(x)
  return t.to(like) if isinstance(like, torch.device) else t


def graph(A):
  """Constructs a `graph` probe. Accepts numpy or torch; adds self-loops."""
  A_t = _ensure_tensor(A)  # 关键：把 numpy 统一成 torch
  device, dtype = A_t.device, A_t.dtype
  I = torch.eye(A_t.shape[0], device=device, dtype=dtype)
  # (A + I) 非零处置 1；返回 float
  probe = ((A_t + I) != 0).to(torch.float32)
  return probe


def mask_one(i: int, n: int, device: torch.device | None = None) -> torch.Tensor:
    """Constructs a `mask_one` probe."""
    assert n > i
    probe = torch.zeros(n, device=device)
    probe[i] = 1.0
    return probe


def strings_id(T_pos: torch.Tensor, P_pos: torch.Tensor) -> torch.Tensor:
    """Constructs a `strings_id` probe."""
    probe_T = torch.zeros(T_pos.shape[0], device=T_pos.device)
    probe_P = torch.ones(P_pos.shape[0], device=P_pos.device)
    return torch.cat([probe_T, probe_P], dim=0)


def strings_pair(pair_probe: torch.Tensor) -> torch.Tensor:
    """Constructs a `strings_pair` probe."""
    n, m = pair_probe.shape
    device = pair_probe.device
    probe_ret = torch.zeros(n + m, n + m, device=device)
    for i in range(n):
        for j in range(m):
            probe_ret[i, j + n] = pair_probe[i, j]
    return probe_ret


def strings_pair_cat(pair_probe: torch.Tensor, nb_classes: int) -> torch.Tensor:
    """Constructs a `strings_pair_cat` probe."""
    assert nb_classes > 0
    n, m = pair_probe.shape
    device = pair_probe.device

    # 额外的分类用于 “blank”
    probe_ret = torch.zeros(n + m, n + m, nb_classes + 1, device=device)

    # 正类填入
    for i in range(n):
        for j in range(m):
            cls = int(pair_probe[i, j].item())
            probe_ret[i, j + n, cls] = float(_OutputClass.POSITIVE)

    # 填充 blank 区域为 MASKED
    blank = nb_classes
    for i1 in range(n):
        for i2 in range(n):
            probe_ret[i1, i2, blank] = float(_OutputClass.MASKED)
    for j1 in range(n):
        for x in range(n + m):
            probe_ret[j1 + n, x, blank] = float(_OutputClass.MASKED)
    return probe_ret


def strings_pi(T_pos, P_pos, pi):
  T_pos = _ensure_tensor(T_pos)
  P_pos = _ensure_tensor(P_pos)
  pi = _ensure_tensor(pi)
  device = T_pos.device
  N = T_pos.shape[0] + P_pos.shape[0]
  probe = torch.arange(N, device=device)
  for j in range(P_pos.shape[0]):
    probe[T_pos.shape[0] + P_pos[j]] = T_pos.shape[0] + pi[P_pos[j]]
  return probe


def strings_pos(T_pos, P_pos):
  T_pos = _ensure_tensor(T_pos)
  P_pos = _ensure_tensor(P_pos)
  probe_T = T_pos.clone().float() * 1.0 / T_pos.shape[0]
  probe_P = P_pos.clone().float() * 1.0 / P_pos.shape[0]
  return torch.cat([probe_T, probe_P], dim=0)


def strings_pred(T_pos, P_pos):
  T_pos = _ensure_tensor(T_pos)
  P_pos = _ensure_tensor(P_pos)
  device = T_pos.device
  N = T_pos.shape[0] + P_pos.shape[0]
  probe = torch.arange(N, device=device)
  for i in range(1, T_pos.shape[0]):
    probe[T_pos[i]] = T_pos[i - 1]
  for j in range(1, P_pos.shape[0]):
    probe[T_pos.shape[0] + P_pos[j]] = T_pos.shape[0] + P_pos[j - 1]
  return probe

# ===== JAX vectorized functions -> PyTorch (batched) ===================================

def _one_hot(idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(idx.long(), num_classes=num_classes).to(idx.dtype)


def predecessor_pointers_to_permutation_matrix(
    pointers: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch version of jnp.vectorize(signature='(n)->(n,n)') version.
    pointers: [..., N] (int)  -> returns [..., N, N] permutation matrices.
    """
    orig_shape = pointers.shape
    N = orig_shape[-1]
    device = pointers.device
    flat = pointers.reshape(-1, N)

    outs = []
    for row in flat:  # row: [N]
        oh = _one_hot(row, N)                # [N, N]  每行是 one-hot of predecessor
        last = oh.sum(dim=-2).argmin(dim=-1) # “没有被指向”的那个为尾节点（标量）

        perm = torch.zeros(N, N, device=device, dtype=row.dtype)
        # 从后往前填充
        curr_last = last
        for i in range(N - 1, -1, -1):
            perm[i] = _one_hot(curr_last, N)  # 第 i 行在 last 列放 1
            curr_last = row[curr_last]
        outs.append(perm)

    return torch.stack(outs, dim=0).reshape(*orig_shape[:-1], N, N)


def permutation_matrix_to_predecessor_pointers(
    perm: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch version of jnp.vectorize(signature='(n,n)->(n)').
    perm: [..., N, N]  -> returns [..., N] predecessor pointers.
    """
    orig_shape = perm.shape
    N = orig_shape[-1]
    device = perm.device
    flat = perm.reshape(-1, N, N)

    outs = []
    for mat in flat:  # [N,N]
        idx = mat.argmax(dim=-1)  # 每一行的位置索引
        pointers = torch.zeros(N, dtype=torch.long, device=device)
        pointers[idx[0]] = idx[0]
        for i in range(1, N):
            pointers[idx[i]] = idx[i - 1]
        pointers = torch.minimum(pointers, torch.tensor(N - 1, device=device))
        outs.append(pointers)

    return torch.stack(outs, dim=0).reshape(*orig_shape[:-2], N)


def predecessor_to_cyclic_predecessor_and_first(
    pointers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch version of jnp.vectorize(signature='(n)->(n,n),(n)').

    Input:  pointers [..., N]
    Return: (pointers_one_hot_cyclic [..., N, N], first_mask [..., N])
    """
    orig_shape = pointers.shape
    N = orig_shape[-1]
    device = pointers.device
    flat = pointers.reshape(-1, N)

    Ps = []
    Ms = []
    for row in flat:  # [N]
        oh = _one_hot(row, N)                 # [N,N]
        last = oh.sum(dim=-2).argmin(dim=-1)  # 尾节点
        first = torch.diagonal(oh).argmax(dim=-1)  # 指向自己的那个为首节点
        mask = _one_hot(first, N).to(row.dtype)

        # pointers_one_hot += mask[..., None] * one_hot(last)
        oh = oh + mask.unsqueeze(-1) * _one_hot(last, N).to(row.dtype)
        # pointers_one_hot -= mask[..., None] * mask
        oh = oh - mask.unsqueeze(-1) * mask.unsqueeze(0)
        Ps.append(oh.to(row.dtype))
        Ms.append(mask.to(row.dtype))

    P = torch.stack(Ps, dim=0).reshape(*orig_shape[:-1], N, N)
    M = torch.stack(Ms, dim=0).reshape(*orig_shape[:-1], N)
    return P, M
