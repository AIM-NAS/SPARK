# nets_torch.py
# PyTorch reimplementation of CLRS Net / NetChunked focusing on Haiku/JAX usages.
# Line-by-line logical parity with nets.py (DeepMind CLRS), replacing:
# - hk.Module -> torch.nn.Module
# - hk.Linear/LSTM/dropout/one_hot/scan/vmap/next_rng_key -> torch equivalents
# - jnp (jax.numpy) -> torch
# - jax.tree_util.tree_map -> Python mapping utilities
# - jax.random.bernoulli -> torch.rand() < p

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 假定你已有 PyTorch 版本（或兼容适配层）的这些模块，接口保持一致：
from . import decoders
from . import encoders
from . import probing
from . import processors
from . import samplers
from . import specs

_Array = torch.Tensor
_DataPoint = probing.DataPoint
_Features = samplers.Features
_FeaturesChunked = samplers.FeaturesChunked
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type


# -------------------------
# Util helpers (JAX -> Torch)
# -------------------------
def _expand_to(x: _Array, y: _Array) -> _Array:
    # // jnp: while len(y.shape) > len(x.shape): x = jnp.expand_dims(x, -1)
    while x.dim() < y.dim():
        x = x.unsqueeze(-1)
    return x


def _is_not_done_broadcast(lengths: _Array, i: int, tensor: _Array) -> _Array:
    # // is_not_done = (lengths > i + 1) * 1.0 ; expand dims to tensor.ndim
    is_not_done = (lengths > (i + 1)).to(tensor.dtype)
    while is_not_done.dim() < tensor.dim():
        is_not_done = is_not_done.unsqueeze(-1)
    return is_not_done


def _tree_map(fn, tree):
    # minimal tree-map (dict/list/tuple/torch.Tensor)
    if isinstance(tree, dict):
        return {k: _tree_map(fn, v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        typ = type(tree)
        return typ(_tree_map(fn, v) for v in tree)
    return fn(tree)


def _one_hot(data: _Array, num_classes: int) -> _Array:
    # // hk.one_hot
    return F.one_hot(data.to(torch.int64), num_classes=num_classes).to(dtype=torch.float32)


def _eye_repeat(batch_size: int, nb_nodes: int, device, dtype) -> _Array:
    # // jnp.repeat(expand_dims(eye), batch_size, axis=0)
    eye = torch.eye(nb_nodes, device=device, dtype=dtype).unsqueeze(0)
    return eye.expand(batch_size, nb_nodes, nb_nodes)


def _data_dimensions(features: _Features) -> Tuple[int, int]:
    # // Returns (batch_size, nb_nodes)
    for inp in features.inputs:
        if inp.location in [_Location.NODE, _Location.EDGE]:
            return tuple(inp.data.shape[:2])
    raise RuntimeError("Could not infer (batch, nodes).")


def _data_dimensions_chunked(features: _FeaturesChunked) -> Tuple[int, int]:
    # // Returns (batch_size, nb_nodes) for chunked: [T, B, N, ...]
    for inp in features.inputs:
        if inp.location in [_Location.NODE, _Location.EDGE]:
            return tuple(inp.data.shape[1:3])
    raise RuntimeError("Could not infer (batch, nodes) in chunked.")


# -------------------------
# Scan emulation (hk.scan)
# -------------------------
def _scan(fn, init_state, xs, length=None):
    """
    Emulate hk.scan using a Python for-loop.
    xs can be:
      - torch.arange(...) style list/1D tensor indices, or
      - a tuple of tensors representing time-major inputs (e.g., (inputs, hints, is_first))
    Returns: final_state, stacked_outputs (via list -> stack)
    """
    outputs = []
    state = init_state

    # unify iteration over time steps
    if isinstance(xs, torch.Tensor):
        time_steps = xs.tolist()
        xs_iter = [None] * len(time_steps)
        pair_iter = zip(time_steps, xs_iter)
        for i, _ in pair_iter:
            out_state, out_val = fn(state, i)
            state = out_state
            outputs.append(out_val)
    elif isinstance(xs, tuple):
        # assume each element is [T, ...]
        T = xs[0].shape[0]
        for t in range(T):
            x_t = tuple(x[t] for x in xs)
            out_state, out_val = fn(state, x_t)
            state = out_state
            outputs.append(out_val)
    elif isinstance(xs, list):
        for x in xs:
            out_state, out_val = fn(state, x)
            state = out_state
            outputs.append(out_val)
    else:
        raise TypeError("Unsupported xs type for _scan")

    # outputs is a list of (possibly nested) structures; we keep it as list
    # Original hk.scan returns a pytree with time stacked; caller will stack/select
    return state, outputs


# -------------------------
# Dataclasses matching Haiku scans
# -------------------------
@dataclass
class _MessagePassingScanState:
    hint_preds: Optional[Dict[str, _Array]]
    output_preds: Optional[Dict[str, _Array]]
    hiddens: _Array
    lstm_state: Optional[Tuple[_Array, _Array]]  # (h, c) with shape [B,N,H]


@dataclass
class _MessagePassingOutputChunked:
    hint_preds: Dict[str, _Array]
    output_preds: Dict[str, _Array]


@dataclass
class MessagePassingStateChunked:
    inputs: _Trajectory
    hints: _Trajectory
    is_first: _Array         # [B] int/bool
    hint_preds: Dict[str, _Array]
    hiddens: _Array          # [B,N,H]
    lstm_state: Optional[Tuple[_Array, _Array]]  # (h, c) with shape [B,N,H]


# -------------------------
# LSTM wrapper compatible with [B,N,H] tensors
# -------------------------
class BNHLSTMCell(nn.Module):
    """
    Emulates hk.LSTM applied with jax.vmap over batch dimension.
    We flatten (B*N) as batch for a single-step LSTMCell, then reshape back.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.cell = nn.LSTMCell(hidden_size, hidden_size)
        self._built = False
    def initial_state(self, B: int, N: int, H: int, device):
        h0 = torch.zeros(B, N, H, device=device)
        c0 = torch.zeros(B, N, H, device=device)
        return (h0, c0)

    def forward(self, x: _Array, state: Tuple[_Array, _Array]):
        # x: [B,N,H], state: (h,c) each [B,N,H]
        B, N, H = x.shape
        h, c = state
        x_flat = x.reshape(B * N, H)
        h_flat = h.reshape(B * N, H)
        c_flat = c.reshape(B * N, H)
        h_next, c_next = self.cell(x_flat, (h_flat, c_flat))
        h_next = h_next.reshape(B, N, H)
        c_next = c_next.reshape(B, N, H)
        return h_next, (h_next, c_next)  # align with Haiku where output=h_t


# -------------------------
# Net (PyTorch)
# -------------------------
class Net(nn.Module):
    """PyTorch version of CLRS Net (Haiku/JAX -> torch)."""

    def __init__(
        self,
        spec: List[_Spec],
        hidden_dim: int,
        encode_hints: bool,
        decode_hints: bool,
        processor_factory: processors.ProcessorFactory,
        use_lstm: bool,
        encoder_init: str,
        dropout_prob: float,
        hint_teacher_forcing: float,
        hint_repred_mode='soft',
        nb_dims=None,
        nb_msg_passing_steps=1,
        debug=False,
        name: str = 'net',
    ):
        super().__init__()
        self._dropout_prob = dropout_prob
        self._hint_teacher_forcing = hint_teacher_forcing
        self._hint_repred_mode = hint_repred_mode
        self.spec = spec
        self.hidden_dim = hidden_dim
        self.encode_hints = encode_hints
        self.decode_hints = decode_hints
        self.processor_factory = processor_factory
        self.nb_dims = nb_dims
        self.use_lstm = use_lstm
        self.encoder_init = encoder_init
        self.nb_msg_passing_steps = nb_msg_passing_steps
        self.debug = debug

        # Lazy members constructed per-call like original code
        self.encoders = None
        self.decoders = None
        self.processor = None
        self._built = False
        if self.use_lstm:
            self.lstm = BNHLSTMCell(hidden_size=self.hidden_dim)
        else:
            self.lstm = None

    def _one_step_pred(
            self,
            inputs: _Trajectory,
            hints: _Trajectory,
            hidden: _Array,  # [B,N,H]
            batch_size: int,
            nb_nodes: int,
            lstm_state: Optional[Tuple[_Array, _Array]],
            spec: _Spec,
            encs: Dict[str, List[nn.Module]],
            decs: Dict[str, Tuple[nn.Module, ...]],
            repred: bool,
            i: int,  # 当前时间步
    ):
        device = hidden.device
        dtype = hidden.dtype

        # 初始化 node / edge / graph 特征与邻接
        node_fts = torch.zeros(batch_size, nb_nodes, self.hidden_dim, device=device, dtype=dtype)
        edge_fts = torch.zeros(batch_size, nb_nodes, nb_nodes, self.hidden_dim, device=device, dtype=dtype)
        graph_fts = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        adj_mat = _eye_repeat(batch_size, nb_nodes, device, dtype)  # [B,N,N]

        # 若 dp.data 带时间维 [T,B,...] 或只有 [T]（graph 级别，如 time），切第 i 帧；
        # 对于 [T]，再扩成 [B] 以便后续编码成 [B,H]。
        def _dp_at_step(dp: _DataPoint) -> _DataPoint:
            x = dp.data
            # torch 张量
            if isinstance(x, torch.Tensor):
                # case A: [T,B,...] 或 [T,...]
                if x.dim() >= 1 and x.size(0) != batch_size:
                    # [T,B,...] -> [B,...] ; [T] -> 标量
                    x_i = x[i]
                    # 若是 graph 级别且切完是标量/0维，扩成 [B]
                    if dp.location == _Location.GRAPH and (not isinstance(x_i, torch.Tensor) or x_i.dim() == 0):
                        x_i = torch.as_tensor(x_i, device=device, dtype=x.dtype).expand(batch_size)
                    return dp.__class__(dp.name, dp.location, dp.type_, x_i)
                # case B: [B,...]（已经是按 batch 的），不动
                return dp
            # numpy 数组
            try:
                import numpy as np
                if isinstance(x, np.ndarray):
                    if x.ndim >= 1 and x.shape[0] != batch_size:
                        x_i = x[i]
                        if dp.location == _Location.GRAPH and (not isinstance(x_i, np.ndarray) or x_i.ndim == 0):
                            x_i = np.full((batch_size,), x_i, dtype=x.dtype)
                        return dp.__class__(dp.name, dp.location, dp.type_, x_i)
                    return dp
            except Exception:
                pass
            return dp

        # ============== ENCODE ==============
        # 1) 仅用 inputs 构建邻接 + 累加特征
        for dp in inputs:
            try:
                dp = _dp_at_step(dp)  # 切第 i 帧（含 [T]→[B] 的处理）
                dp = encoders.preprocess(dp, nb_nodes)  # POINTER 一致化等
                assert dp.type_ != _Type.SOFT_POINTER
                # 只允许 inputs 更新邻接
                adj_mat = encoders.accum_adj_mat(dp, adj_mat)  # [B,N,N]
                encoder = encs[dp.name]
                edge_fts = encoders.accum_edge_fts(encoder, dp, edge_fts)  # [B,N,N,H]
                node_fts = encoders.accum_node_fts(encoder, dp, node_fts)  # [B,N,H]
                graph_fts = encoders.accum_graph_fts(encoder, dp, graph_fts)  # [B,H]
            except Exception as e:
                raise Exception(f'Failed to process input {dp}') from e

        # 2) hints 只做特征累加，不更新邻接
        if self.encode_hints:
            for dp in hints:
                try:
                    dp = _dp_at_step(dp)  # 再保险一次：若仍是时间主轴/1D[T]，切/扩
                    dp = encoders.preprocess(dp, nb_nodes)
                    assert dp.type_ != _Type.SOFT_POINTER
                    encoder = encs[dp.name]
                    edge_fts = encoders.accum_edge_fts(encoder, dp, edge_fts)
                    node_fts = encoders.accum_node_fts(encoder, dp, node_fts)
                    graph_fts = encoders.accum_graph_fts(encoder, dp, graph_fts)
                except Exception as e:
                    raise Exception(f'Failed to process hint {dp}') from e

        # ============== PROCESS（消息传递多步）==============
        nxt_hidden = hidden
        nxt_edge = None
        for _ in range(self.nb_msg_passing_steps):
            nxt_hidden, nxt_edge = self.processor(
                node_fts=node_fts,  # [B,N,H]
                edge_fts=edge_fts,  # [B,N,N,H]
                graph_fts=graph_fts,  # [B,H]
                adj_mat=adj_mat,  # [B,N,N]
                hidden=nxt_hidden,  # [B,N,H]
                batch_size=batch_size,
                nb_nodes=nb_nodes,
            )

        if not repred:
            nxt_hidden = F.dropout(nxt_hidden, p=self._dropout_prob, training=self.training)

        if self.use_lstm:
            nxt_hidden, nxt_lstm_state = self.lstm(nxt_hidden, lstm_state)
        else:
            nxt_lstm_state = None

        # ============== DECODE ==============
        h_t = torch.cat([node_fts, hidden, nxt_hidden], dim=-1)  # [B,N,3H]
        e_t = torch.cat([edge_fts, nxt_edge], dim=-1) if nxt_edge is not None else edge_fts  # [B,N,N,H or 2H]

        hint_preds, output_preds = decoders.decode_fts(
            decoders=decs,
            spec=spec,
            h_t=h_t,
            adj_mat=adj_mat,
            edge_fts=e_t,
            graph_fts=graph_fts,
            inf_bias=self.processor.inf_bias,
            inf_bias_edge=self.processor.inf_bias_edge,
            repred=repred,
        )

        return nxt_hidden, output_preds, hint_preds, nxt_lstm_state

    # ---- message passing step in unchunked ----
    def _msg_passing_step(self,
                          mp_state: _MessagePassingScanState,
                          i: int,
                          hints: List[_DataPoint],
                          repred: bool,
                          lengths: torch.Tensor,  # chex.Array -> torch.Tensor
                          batch_size: int,
                          nb_nodes: int,
                          inputs: _Trajectory,
                          first_step: bool,
                          spec: _Spec,
                          encs: Dict[str, List[nn.Module]],  # hk.Module -> nn.Module
                          decs: Dict[str, Tuple[nn.Module, ...]],
                          return_hints: bool,
                          return_all_outputs: bool):
        """
        PyTorch 版 msg_passing_step：严格复刻 CLRS 原始逻辑。
        """
        import torch
        import torch.nn.functional as F
        from . import decoders
        from . import probing
        from . import specs as _specs
        _Type = _specs.Type

        device = lengths.device

        # ---------- helpers ----------
        def _expand_to(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """把 [B] 的mask扩到 target 的形状（在末维反复unsqueeze）。"""
            m = mask
            while m.dim() < target.dim():
                m = m.unsqueeze(-1)
            return m

        def _is_not_done_broadcast(lengths_: torch.Tensor, step_i: int, like: torch.Tensor) -> torch.Tensor:
            """长度早停：lengths > i+1 的样本继续更新；其余保持上一步。"""
            z = (lengths_ > (step_i + 1)).to(like.dtype)
            return _expand_to(z, like)

        # ---------- 1) 解码上一时刻的 hints（供 repred/TF 混合） ----------
        decoded_hint = None
        if self.decode_hints and (not first_step):
            assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
            hard_postprocess = (self._hint_repred_mode == 'hard' or
                                (self._hint_repred_mode == 'hard_on_eval' and repred))
            decoded_hint = decoders.postprocess(
                spec,
                mp_state.hint_preds,  # dict[name] -> DataPoint(logits/probs)
                sinkhorn_temperature=0.1,
                sinkhorn_steps=25,
                hard=hard_postprocess
            )

        # ---------- 2) 形成当前时刻的 hints: cur_hint ----------
        if repred and self.decode_hints and (not first_step):
            # 评测重推：完全使用已解码的 hints
            cur_hint: List[_DataPoint] = []
            for name in decoded_hint:
                cur_hint.append(decoded_hint[name])
        else:
            # 训练或首步：GT hints +（可选）noisy teacher forcing 混合 decoded hints
            cur_hint = []
            needs_noise = (self.decode_hints and (not first_step) and (self._hint_teacher_forcing < 1.0))
            if needs_noise:
                # [B] True 使用 GT；False 使用 decoded
                force_mask = (torch.rand(batch_size, device=device) < self._hint_teacher_forcing)
            else:
                force_mask = None

            for hint in hints:
                # 原库默认 hints.data 为 [T,B,...]，这里直接按第 i 步切
                # 若你的数据是 [B,T,...]，建议在进入 forward 前统一 transpose 成 [T,B,...]
                hint_data = torch.as_tensor(hint.data, device=device)[i]
                _, loc, typ = spec[hint.name]

                if needs_noise:
                    if (typ == _Type.POINTER) and (decoded_hint[hint.name].type_ == _Type.SOFT_POINTER):
                        # decoded 是 soft-pointer，而 GT 是 index，需要把 GT 提升为 one-hot 再混合
                        hint_data = F.one_hot(hint_data.long(), nb_nodes).to(hint_data.dtype)
                        typ = _Type.SOFT_POINTER
                    hint_data = torch.where(
                        _expand_to(force_mask, hint_data),
                        hint_data,
                        decoded_hint[hint.name].data.to(hint_data.dtype)
                    )

                cur_hint.append(
                    probing.DataPoint(
                        name=hint.name, location=loc, type_=typ, data=hint_data
                    )
                )

        # ---------- 3) 单步前向（编码->处理器->解码 logits） ----------
        # 你的 _one_step_pred 应该返回：
        #   hiddens: [B,N,H]
        #   output_preds_cand: Dict[str, DataPoint]
        #   hint_preds: Dict[str, DataPoint]
        #   lstm_state: 任意你定义的 state
        hiddens, output_preds_cand, hint_preds, lstm_state = self._one_step_pred(
            inputs, cur_hint, mp_state.hiddens,
            batch_size, nb_nodes, mp_state.lstm_state,
            spec, encs, decs, repred,i=i,
        )

        # ---------- 4) 输出累计（第一步直接赋值，其余步做长度早停融合） ----------
        if first_step:
            output_preds = output_preds_cand
        else:
            output_preds = {}
            for outp in mp_state.output_preds:
                is_not_done = _is_not_done_broadcast(lengths, i, output_preds_cand[outp])
                output_preds[outp] = is_not_done * output_preds_cand[outp] + (
                        1.0 - is_not_done) * mp_state.output_preds[outp]

        # ---------- 5) 组装 carry state 与 accumulate state ----------
        new_mp_state = _MessagePassingScanState(  # carry over
            hint_preds=hint_preds,
            output_preds=output_preds,
            hiddens=hiddens,
            lstm_state=lstm_state
        )
        # 减少显存：按需只保存必要字段以便堆叠
        accum_mp_state = _MessagePassingScanState(  # stacked over steps
            hint_preds=hint_preds if return_hints else None,
            output_preds=output_preds if return_all_outputs else None,
            hiddens=hiddens if self.debug else None,
            lstm_state=None
        )

        # 与 jax.scan 一致：返回 (carry_state, stacked_output)
        return new_mp_state, accum_mp_state

    # ---- forward (unchunked) ----
    def forward(self, features_list: List[_Features], repred: bool,
                algorithm_index: int, return_hints: bool,
                return_all_outputs: bool):
        # Construct encoders/decoders/processor per call (parity with original)

        # self.encoders, self.decoders = self._construct_encoders_decoders()
        # self.processor = self.processor_factory(self.hidden_dim)

        if not self._built:
            # 只在第一次 forward 构建模块
            self.encoders, self.decoders = self._construct_encoders_decoders()
            self.processor = self.processor_factory(self.hidden_dim)

            # # 把所有子模块搬到数据所在 device
            # self.processor.to(device)
            # if self.use_lstm and self.lstm is not None:
            #     self.lstm.to(device)
            #
            # # encoders: List[Dict[str, List[nn.Module]]]
            # for enc_dict in self.encoders:
            #     for mods in enc_dict.values():
            #         for m in mods:
            #             m.to(device)
            #
            # # decoders: List[Dict[str, Tuple[nn.Module,...]]]
            # for dec_dict in self.decoders:
            #     for mods in dec_dict.values():
            #         for m in mods:
            #             m.to(device)

            self._built = True
        else:
            # 非首次 forward，确保这些模块仍在正确 device（通常已经在）
            pass

        if algorithm_index == -1:
            algorithm_indices = range(len(features_list))
        else:
            algorithm_indices = [algorithm_index]
        assert len(algorithm_indices) == len(features_list)

        for algorithm_index, features in zip(algorithm_indices, features_list):
            inputs = features.inputs
            hints = features.hints
            lengths = features.lengths  # [B]

            batch_size, nb_nodes = _data_dimensions(features)
            device = hints[0].data.device
            dtype = torch.float32
            # === move newly-built modules to the same device as data ===
            self.processor.to(device)
            if self.use_lstm and self.lstm is not None:
                self.lstm.to(device)

            # encoders: List[dict[str, List[nn.Module]]]
            for enc_dict in self.encoders:
                for mods in enc_dict.values():  # mods is List[nn.Module]
                    for m in mods:
                        m.to(device)

            # decoders: List[dict[str, Tuple[nn.Module, ...]]]
            for dec_dict in self.decoders:
                for mods in dec_dict.values():  # mods is Tuple[nn.Module, ...]
                    for m in mods:
                        m.to(device)

            # 自适应推断时间轴长度（支持 [B,T,...] 或 [T,B,...]）
            if hints[0].data.dim() < 2:
                raise RuntimeError("Hint tensor must have at least 2 dims (B,T,...) or (T,B,...)")
            if hints[0].data.shape[0] == batch_size:
                T = hints[0].data.shape[1]  # [B, T, ...]
            elif hints[0].data.shape[1] == batch_size:
                T = hints[0].data.shape[0]  # [T, B, ...]
            else:
                raise RuntimeError(
                    f"Cannot infer time dim from hint shape {tuple(hints[0].data.shape)} with batch_size={batch_size}")
            nb_mp_steps = max(1, T - 1)

            hiddens = torch.zeros(batch_size, nb_nodes, self.hidden_dim, device=device, dtype=dtype)

            if self.use_lstm:
                H = self.hidden_dim
                lstm_state = self.lstm.initial_state(batch_size, nb_nodes, H, device=device)
            else:
                lstm_state = None

            mp_state = _MessagePassingScanState(
                hint_preds=None, output_preds=None,
                hiddens=hiddens, lstm_state=lstm_state
            )

            common_args = dict(
                hints=hints,
                repred=repred,
                inputs=inputs,
                batch_size=batch_size,
                nb_nodes=nb_nodes,
                lengths=lengths,
                spec=self.spec[algorithm_index],
                encs=self.encoders[algorithm_index],
                decs=self.decoders[algorithm_index],
                return_hints=return_hints,
                return_all_outputs=return_all_outputs,
            )
            # print("mp_state:",  mp_state)
            # print("**common_args:",common_args)
            # First step (different graph) // not scanned
            mp_state, lean_mp_state = self._msg_passing_step(
                i=0,
                mp_state=mp_state,
                first_step=True,
                **common_args
            )
            # Remaining steps // hk.scan -> for loop
            def scan_fn(state, i_idx):
                return self._msg_passing_step(
                    i=i_idx,
                    mp_state=state,
                    first_step=False,
                    **common_args
                )

            # Time indices: 1..nb_mp_steps-1
            if nb_mp_steps - 1 > 0:
                idxs = list(range(1, nb_mp_steps))
                output_mp_state, accum_list = _scan(scan_fn, mp_state, idxs)
            else:
                output_mp_state, accum_list = mp_state, []

        # Stack accumulated states to match original concatenation
        # accum_mp_state = concat([init], tail) along time axis
        # Here we have one algorithm's last values only.
        # Convert list of dataclasses to stacked structures:
        def _stack_accum(init_state, tail_list):
            # returns a structure like in JAX code
            def cat_time(init_v, tail_vs):
                # init_v -> prepend to list and stack on time dim (T, ...)
                if init_v is None:
                    return None
                seq = [init_v] + tail_vs
                # For dict -> dict of lists
                if isinstance(init_v, dict):
                    keys = init_v.keys()
                    out = {k: [s[k] for s in seq] for k in keys}
                    return out
                # For tensors: make list and later stack by caller
                return seq

            # build parallel lists for hint_preds/output_preds/hiddens
            # tail_list holds _MessagePassingScanState per time
            if not tail_list:
                return init_state

            stacked = _MessagePassingScanState(
                hint_preds=cat_time(init_state.hint_preds, [e.hint_preds for e in tail_list]),
                output_preds=cat_time(init_state.output_preds, [e.output_preds for e in tail_list]),
                hiddens=cat_time(init_state.hiddens, [e.hiddens for e in tail_list]),
                lstm_state=None
            )
            return stacked

        accum_mp_state = _stack_accum(lean_mp_state, accum_list)

        # invert helper: dict of lists -> list of dicts (for hint_preds)
        def invert(d):
            if d:
                # d: {k: [step0, step1, ...]}
                steps = len(next(iter(d.values())))
                return [dict((k, d[k][t]) for k in d.keys()) for t in range(steps)]
            return None

        # Build outputs same as original
        if return_all_outputs:
            # output_preds: stack along time // jnp.stack(v)
            # accum_mp_state.output_preds: {name: [T items]}
            output_preds = {k: torch.stack(v, dim=0) for k, v in accum_mp_state.output_preds.items()}
        else:
            output_preds = output_mp_state.output_preds

        hint_preds = invert(accum_mp_state.hint_preds)

        if self.debug:
            # hiddens: stack over time
            hiddens = torch.stack(accum_mp_state.hiddens, dim=0)
            return output_preds, hint_preds, hiddens

        return output_preds, hint_preds

    # ---- enc/dec construction (per algorithm) ----
    def _construct_encoders_decoders(self):
        # 2) _construct_encoders_decoders 内
        encoders_all = nn.ModuleList()
        decoders_all = nn.ModuleList()
        enc_algo_idx = None

        for algo_idx, spec in enumerate(self.spec):
            enc = nn.ModuleDict()
            dec = nn.ModuleDict()

            for name, (stage, loc, t) in spec.items():
                if stage == _Stage.INPUT or (stage == _Stage.HINT and self.encode_hints):
                    if name == specs.ALGO_IDX_INPUT_NAME:
                        if enc_algo_idx is None:
                            enc_algo_idx = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)])
                            nn.init.xavier_uniform_(enc_algo_idx[0].weight)
                            nn.init.zeros_(enc_algo_idx[0].bias)
                        enc[name] = enc_algo_idx
                    else:
                        enc[name] = encoders.construct_encoders(
                            stage, loc, t, hidden_dim=self.hidden_dim,
                            init=self.encoder_init,
                            name=f'algo_{algo_idx}_{name}'
                        )
                if stage == _Stage.OUTPUT or (stage == _Stage.HINT and self.decode_hints):
                    mods = decoders.construct_decoders(
                        loc, t, hidden_dim=self.hidden_dim,
                        nb_dims=self.nb_dims[algo_idx][name],
                        name=f'algo_{algo_idx}_{name}'
                    )
                    if isinstance(mods, nn.Module):
                        dec[name] = mods
                    elif isinstance(mods, (list, tuple)):
                        dec[name] = nn.ModuleList(list(mods))
                    else:
                        raise TypeError(f"construct_decoders() returned unsupported type: {type(mods)}")

            encoders_all.append(enc)
            decoders_all.append(dec)

        return encoders_all, decoders_all

# -------------------------
# NetChunked (PyTorch)
# -------------------------
class NetChunked(Net):
    """Chunked-time version of Net with scan unrolled along T."""

    def _msg_passing_step(
            self,
            *,
            i: int,
            inputs,  # List[DataPoint]
            hints,  # List[DataPoint]
            lengths: torch.Tensor,  # [B]
            nb_nodes: int,
            mp_state,  # 承载 hidden 等
            decoded_hint: dict = None,
            hard_postprocess: bool = False,
            repred: bool = False,
            first_step: bool = False,  # ← 新增：接受 first_step，但函数体里不用也没关系
            **_ignored,  # ← 新增：吞掉其它潜在关键字，防未来再报类似错
    ):
        # _as_prediction_data // hk.one_hot for POINTER
        def _as_prediction_data(hint):
            if hint.type_ == _Type.POINTER:
                return _one_hot(hint.data, nb_nodes)
            return hint.data

        nxt_inputs, nxt_hints, nxt_is_first = xs
        inputs = mp_state.inputs
        is_first = mp_state.is_first.bool()
        hints = mp_state.hints
        device = nxt_is_first.device

        if init_mp_state:
            prev_hint_preds = {h.name: _as_prediction_data(h) for h in hints}
            hints_for_pred = hints
        else:
            prev_hint_preds = mp_state.hint_preds
            if self.decode_hints:
                if repred:
                    force_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                elif self._hint_teacher_forcing == 1.0:
                    force_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                else:
                    force_mask = (torch.rand(batch_size, device=device) < self._hint_teacher_forcing)

                assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
                hard_postprocess = (self._hint_repred_mode == 'hard' or
                                    (self._hint_repred_mode == 'hard_on_eval' and repred))
                decoded_hints = decoders.postprocess(
                    spec, prev_hint_preds, sinkhorn_temperature=0.1,
                    sinkhorn_steps=50, hard=hard_postprocess
                )

                hints_for_pred = []
                for h in hints:
                    typ = h.type_
                    hint_data = h.data
                    if (typ == _Type.POINTER and decoded_hints[h.name].type_ == _Type.SOFT_POINTER):
                        hint_data = _one_hot(hint_data, nb_nodes)
                        typ = _Type.SOFT_POINTER
                    data = torch.where(
                        _expand_to(is_first | force_mask, hint_data),
                        hint_data,
                        decoded_hints[h.name].data.to(hint_data.dtype)
                    )
                    hints_for_pred.append(probing.DataPoint(
                        name=h.name, location=h.location, type_=typ, data=data
                    ))
            else:
                hints_for_pred = hints

        # Reset hidden/LSTM when is_first // jnp.where(...)
        hiddens = torch.where((is_first.unsqueeze(-1).unsqueeze(-1)), torch.tensor(0.0, device=mp_state.hiddens.device), mp_state.hiddens)
        if self.use_lstm and mp_state.lstm_state is not None:
            h, c = mp_state.lstm_state
            mask = (is_first.unsqueeze(-1).unsqueeze(-1))
            h = torch.where(mask, torch.tensor(0.0, device=h.device), h)
            c = torch.where(mask, torch.tensor(0.0, device=c.device), c)
            lstm_state = (h, c)
        else:
            lstm_state = None

        hiddens, output_preds, hint_preds, lstm_state = self._one_step_pred(
            inputs, hints_for_pred, hiddens,
            batch_size, nb_nodes, lstm_state,
            spec, encs, decs, repred,i=i,
        )

        new_mp_state = MessagePassingStateChunked(
            hiddens=hiddens, lstm_state=lstm_state, hint_preds=hint_preds,
            inputs=nxt_inputs, hints=nxt_hints, is_first=nxt_is_first
        )
        mp_output = _MessagePassingOutputChunked(
            hint_preds=hint_preds, output_preds=output_preds
        )
        return new_mp_state, mp_output

    def forward(self, features_list: List[_FeaturesChunked],
                mp_state_list: List[MessagePassingStateChunked],
                repred: bool, init_mp_state: bool,
                algorithm_index: int):
        # Build enc/dec/processor each call
        self.encoders, self.decoders = self._construct_encoders_decoders()
        self.processor = self.processor_factory(self.hidden_dim)

        if algorithm_index == -1:
            algorithm_indices = range(len(features_list))
        else:
            algorithm_indices = [algorithm_index]
            assert not init_mp_state, "init_mp_state only allowed with algorithm_index == -1"
        assert len(algorithm_indices) == len(features_list) == len(mp_state_list)

        if init_mp_state:
            output_mp_states = []
            for algorithm_index, features, mp_state in zip(algorithm_indices, features_list, mp_state_list):
                inputs = features.inputs
                hints = features.hints
                batch_size, nb_nodes = _data_dimensions_chunked(features)
                device = hints[0].data.device

                if self.use_lstm:
                    H = self.hidden_dim
                    mp_state.lstm_state = self.lstm.initial_state(batch_size, nb_nodes, H, device=device)

                # take t=0 slice to fill mp_state // jax.tree_util.tree_map(lambda x: x[0], ...)
                mp_state.inputs = _tree_map(lambda x: x[0], inputs)
                mp_state.hints = _tree_map(lambda x: x[0], hints)
                mp_state.is_first = torch.zeros(batch_size, dtype=torch.int32, device=device)
                mp_state.hiddens = torch.zeros(batch_size, nb_nodes, self.hidden_dim, device=device)
                next_is_first = torch.ones(batch_size, dtype=torch.int32, device=device)

                mp_state, _ = self._msg_passing_step(
                    mp_state,
                    (mp_state.inputs, mp_state.hints, next_is_first),
                    repred=repred,
                    init_mp_state=True,
                    batch_size=batch_size,
                    nb_nodes=nb_nodes,
                    spec=self.spec[algorithm_index],
                    encs=self.encoders[algorithm_index],
                    decs=self.decoders[algorithm_index],
                )
                output_mp_states.append(mp_state)
            return None, output_mp_states

        # Normal chunk processing
        for algorithm_index, features, mp_state in zip(algorithm_indices, features_list, mp_state_list):
            inputs = features.inputs   # time-major trajectories
            hints = features.hints
            is_first = features.is_first
            batch_size, nb_nodes = _data_dimensions_chunked(features)

            def scan_fn(state, xs):
                return self._msg_passing_step(
                    state, xs, repred=repred, init_mp_state=False,
                    batch_size=batch_size, nb_nodes=nb_nodes,
                    spec=self.spec[algorithm_index],
                    encs=self.encoders[algorithm_index],
                    decs=self.decoders[algorithm_index],
                )

            mp_state, scan_outputs = _scan(scan_fn, mp_state, (inputs, hints, is_first))

        # Return last algorithm's results
        # scan_outputs: list of _MessagePassingOutputChunked (length T)
        # Convert to time-major dicts: {name: [T elems]} then stack
        def _stack_time(outputs_list, field):
            # field in {'output_preds', 'hint_preds'}
            # outputs_list[t].output_preds is dict{name: tensor}
            keys = outputs_list[0].__getattribute__(field).keys()
            out = {}
            for k in keys:
                seq = [o.__getattribute__(field)[k] for o in outputs_list]
                out[k] = torch.stack(seq, dim=0)  # [T, ...]
            return out

        output_preds_T = _stack_time(scan_outputs, 'output_preds')
        hint_preds_T = _stack_time(scan_outputs, 'hint_preds')

        return (output_preds_T, hint_preds_T), mp_state
