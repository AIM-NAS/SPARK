# baselines_torch.py
# PyTorch reimplementation of CLRS baselines focusing on Haiku/JAX replacements.
# Keep the public API aligned with DeepMind CLRS baselines.py for drop-in use.

import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from . import decoders
from . import losses
from . import model
from . import probing
from . import processors
from . import samplers
from . import specs
# 顶部：若文件里还没引入 numpy
import numpy as np
import torch

def _to_tensor(x, device):
    """把 numpy / torch 统一成 torch.Tensor 并放到 device。dtype 维持原样。"""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        # 注意：保持原 dtype；后续模块若需要 float32/int64 会自行转换
        return torch.from_numpy(x).to(device)
    # 允许 Python 标量或列表（极少见）
    return torch.as_tensor(x, device=device)



# === type aliases ===
_Array = torch.Tensor
_DataPoint = probing.DataPoint
_Features = samplers.Features
_FeaturesChunked = samplers.FeaturesChunked
_Feedback = samplers.Feedback
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type
_OutputClass = specs.OutputClass

# IMPORTANT: assumes you use the PyTorch ports from nets_torch.py
from .nets import Net, NetChunked, MessagePassingStateChunked


# ---------------------------
# Helpers replacing JAX utils
# ---------------------------
def _nb_nodes(feedback: _Feedback, is_chunked: bool) -> int:
    for inp in feedback.features.inputs:
        if inp.location in [_Location.NODE, _Location.EDGE]:
            return inp.data.shape[2] if is_chunked else inp.data.shape[1]
    raise RuntimeError("Cannot infer nb_nodes.")

def _filter_param_names_in_processor(named_params):
    for name, p in named_params:
        if processors.PROCESSOR_TAG in name:
            yield name, p

def _filter_param_names_out_processor(named_params):
    for name, p in named_params:
        if processors.PROCESSOR_TAG not in name:
            yield name, p

def _tensor_tree_detach_clone(x):
    if isinstance(x, dict):
        return {k: _tensor_tree_detach_clone(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_tensor_tree_detach_clone(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_tensor_tree_detach_clone(v) for v in x)
    if torch.is_tensor(x):
        return x.detach().clone()
    return x

def _to_device_features1(feat: Union[_Features, _FeaturesChunked], device):
    def _move_dp(dp):  # DataPoint
        return probing.DataPoint(dp.name, dp.location, dp.type_, dp.data.to(device))
    return feat._replace(
        inputs=[_move_dp(dp) for dp in feat.inputs],
        hints=[_move_dp(dp) for dp in feat.hints],
        lengths=feat.lengths.to(device) if hasattr(feat, 'lengths') else feat.lengths,
        is_first=getattr(feat, 'is_first', None).to(device) if hasattr(feat, 'is_first') and feat.is_first is not None else None,
        is_last=getattr(feat, 'is_last', None).to(device) if hasattr(feat, 'is_last') and feat.is_last is not None else None,
    )
def _to_device_features(feat: Union[_Features, _FeaturesChunked], device):
    def _move_dp(dp):  # DataPoint
        data_t = _to_tensor(dp.data, device)
        return probing.DataPoint(dp.name, dp.location, dp.type_, data_t)

    # 某些字段（lengths/is_first/is_last）可能不存在或为 None/ndarray
    kwargs = dict(
        inputs=[_move_dp(dp) for dp in feat.inputs],
        hints=[_move_dp(dp) for dp in feat.hints],
    )
    if hasattr(feat, 'lengths') and feat.lengths is not None:
        kwargs['lengths'] = _to_tensor(feat.lengths, device)
    if hasattr(feat, 'is_first'):
        kwargs['is_first'] = None if feat.is_first is None else _to_tensor(feat.is_first, device)
    if hasattr(feat, 'is_last'):
        kwargs['is_last'] = None if feat.is_last is None else _to_tensor(feat.is_last, device)

    # _replace 是 Features/FeaturesChunked 的命名元组方法；保持其他字段不变
    return feat._replace(**kwargs)
# ---------------------------
# BaselineModel (non-chunked)
# ---------------------------
class BaselineModel(model.Model):
    """Model implementation with selectable message passing algorithm (PyTorch)."""

    def __init__(
        self,
        spec: Union[_Spec, List[_Spec]],
        dummy_trajectory: Union[List[_Feedback], _Feedback],
        processor_factory: processors.ProcessorFactory,
        hidden_dim: int = 32,
        encode_hints: bool = False,
        decode_hints: bool = True,
        encoder_init: str = 'default',
        use_lstm: bool = False,
        learning_rate: float = 0.005,
        grad_clip_max_norm: float = 0.0,
        checkpoint_path: str = '/tmp/clrs3',
        freeze_processor: bool = False,
        dropout_prob: float = 0.0,
        hint_teacher_forcing: float = 0.0,
        hint_repred_mode: str = 'soft',
        name: str = 'base_model',
        nb_msg_passing_steps: int = 1,
        debug: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        对应原 baselines.BaselineModel.__init__，去掉了 hk/optax/jit/pmap 依赖，
        保留参数与行为一致。原实现使用 hk.transform 和 pmap/jit 包装网络与更新流
        （见 baselines._create_net_fns / init / predict / _loss / feedback 等）。:contentReference[oaicite:1]{index=1}
        """
        super().__init__(spec=spec)

        if encode_hints and not decode_hints:
            raise ValueError('`encode_hints=True`, `decode_hints=False` is invalid.')
        assert hint_repred_mode in ['soft', 'hard', 'hard_on_eval']

        self.decode_hints = decode_hints
        self.checkpoint_path = checkpoint_path
        self.name = name
        self._freeze_processor = freeze_processor
        self.learning_rate = learning_rate
        self.grad_clip_max_norm = grad_clip_max_norm
        self.nb_msg_passing_steps = nb_msg_passing_steps
        self.debug = debug

        # device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ---- derive nb_dims from dummy_trajectory (unchanged logic) ----
        self.nb_dims: List[Dict[str, int]] = []
        if isinstance(dummy_trajectory, _Feedback):
            assert len(self._spec) == 1
            dummy_trajectory = [dummy_trajectory]
        for traj in dummy_trajectory:
            nb_dims = {}
            for inp in traj.features.inputs:
                nb_dims[inp.name] = inp.data.shape[-1]
            for hint in traj.features.hints:
                nb_dims[hint.name] = hint.data.shape[-1]
            for outp in traj.outputs:
                nb_dims[outp.name] = outp.data.shape[-1]
            self.nb_dims.append(nb_dims)

        # ---- create Net (replaces hk.transform(_use_net)) ----
        # 原代码将网络定义函数 transform 成 init/apply（见 baselines._create_net_fns）。:contentReference[oaicite:2]{index=2}
        self._net_ctor = lambda: Net(
            spec=self._spec,
            hidden_dim=hidden_dim,
            encode_hints=encode_hints,
            decode_hints=decode_hints,
            processor_factory=processor_factory,
            use_lstm=use_lstm,
            encoder_init=encoder_init,
            dropout_prob=dropout_prob,
            hint_teacher_forcing=hint_teacher_forcing,
            hint_repred_mode=hint_repred_mode,
            nb_dims=self.nb_dims,
            nb_msg_passing_steps=nb_msg_passing_steps,
            debug=debug,
        )
        self.net: Optional[nn.Module] = None

        # optimizer skeleton（代替 optax；原来 optax 在 __init__ 即构建）:contentReference[oaicite:3]{index=3}
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._opt_state_skeleton = None  # 为接口一致保留
        self._param_groups_ready = False

    # ---------------- init (replaces hk.init/apply) ----------------
    def init(self, features: Union[_Features, List[_Features]], seed: int):
        """
        对应原: self.params = self.net_fn.init(PRNGKey, features, True, ...)
        + self.opt.init(params)（见 baselines.BaselineModel.init）。:contentReference[oaicite:4]{index=4}
        """
        torch.manual_seed(seed)
        if not isinstance(features, list):
            assert len(self._spec) == 1
            features = [features]

        # create net & move to device
        self.net = self._net_ctor().to(self.device)
        self.net.eval()  # 与 JAX init 时不启用 dropout 对齐
        with torch.no_grad():
            # 运行一次前向完成各子模块懒初始化（对齐 hk.init）
            _ = self.net(
                features_list=[_to_device_features(f, self.device) for f in features],
                repred=False,
                algorithm_index=-1,
                return_hints=False,
                return_all_outputs=False,
            )

        # ---- 构建优化器（代替 optax.adam/chain） ----
        # freeze_processor 通过参数组或 requires_grad 控制
        named_params = list(self.net.named_parameters())
        if self._freeze_processor:
            encdec = [p for n, p in _filter_param_names_out_processor(named_params)]
            lstm_and_others = encdec
            # 处理器参数冻结
            for _, p in _filter_param_names_in_processor(named_params):
                p.requires_grad = False
            params = lstm_and_others
        else:
            params = [p for _, p in named_params]

        self._optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        self._param_groups_ready = True
        # 为了接口一致，给出一个 skeleton（占位）
        self._opt_state_skeleton = {"_": torch.tensor(0.0, device=self.device)}

    # --------------- params / opt_state properties ----------------
    @property
    def params(self):
        if self.net is None:
            return None
        return _tensor_tree_detach_clone(self.net.state_dict())

    @params.setter
    def params(self, state_dict):
        if self.net is None:
            self.net = self._net_ctor().to(self.device)
        self.net.load_state_dict(state_dict)

    @property
    def opt_state(self):
        if self._optimizer is None:
            return None
        return _tensor_tree_detach_clone(self._optimizer.state_dict())

    @opt_state.setter
    def opt_state(self, state):
        if self._optimizer is None:
            # 需要在 init 后才能设置
            raise RuntimeError("Call init() before setting opt_state.")
        self._optimizer.load_state_dict(state)

    # ---------------- core loss (replaces _loss in JAX) ----------------
    def _loss(self, feedback: _Feedback, algorithm_index: int):
        """
        对应原 baselines._loss: 使用 net.apply(repred=False, return_hints=True) 前向，
        累加 outputs 与 hints 的损失（见原文件 _loss 实现）。:contentReference[oaicite:5]{index=5}
        """
        self.net.train()
        features = _to_device_features(feedback.features, self.device)
        outs = self.net(
            features_list=[features],
            repred=True,
            algorithm_index=algorithm_index,
            return_hints=True,
            return_all_outputs=False,
        )
        if self.debug:
            output_preds, hint_preds, _hidden = outs
        else:
            output_preds, hint_preds = outs

        nb_nodes = _nb_nodes(feedback, is_chunked=False)
        lengths = feedback.features.lengths.to(self.device)
        total_loss = torch.tensor(0.0, device=self.device)

        # 输出损失
        for truth in feedback.outputs:
            total_loss = total_loss + losses.output_loss(
                truth=truth, pred=output_preds[truth.name], nb_nodes=nb_nodes
            )

        # hint 损失
        if self.decode_hints:
            for truth in feedback.features.hints:
                total_loss = total_loss + losses.hint_loss(
                    truth=truth,
                    preds=[x[truth.name] for x in hint_preds],
                    lengths=lengths,
                    nb_nodes=nb_nodes,
                )

        return total_loss

    # ---------------- training step (replaces feedback) ----------------
    def feedback(self, rng_key, feedback: _Feedback, algorithm_index=None) -> float:
        """
        对应原 baselines.feedback：计算梯度并应用优化器更新（无 jit/pmap）。:contentReference[oaicite:6]{index=6}
        """
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = 0

        assert self.net is not None and self._optimizer is not None
        self.net.train()
        self._optimizer.zero_grad(set_to_none=True)
        loss = self._loss(feedback, algorithm_index)
        loss.backward()

        if self.grad_clip_max_norm and self.grad_clip_max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_max_norm)

        self._optimizer.step()
        return float(loss.detach().cpu().item())

    # ---------------- grad-only step (replaces compute_grad) ----------------
    def compute_grad(self, rng_key, feedback: _Feedback, algorithm_index: Optional[int] = None):
        """
        对应原 baselines.compute_grad：返回 loss 和 grads（这里以 state_dict 形式返回）。:contentReference[oaicite:7]{index=7}
        """
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = 0
        assert self.net is not None
        self.net.train()
        self.net.zero_grad(set_to_none=True)
        loss = self._loss(feedback, algorithm_index)
        loss.backward()

        grads = {}
        for name, p in self.net.named_parameters():
            grads[name] = None if p.grad is None else p.grad.detach().clone()
        return float(loss.detach().cpu().item()), grads

    # ---------------- inference (replaces predict/apply+postprocess) ----------------
    def predict(self, rng_key, features: _Features,
                algorithm_index: Optional[int] = None,
                return_hints: bool = False,
                return_all_outputs: bool = False):
        """
        对应原 baselines.predict：net.apply(repred=True) + decoders.postprocess(hard=True)。:contentReference[oaicite:8]{index=8}
        """
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = 0
        assert self.net is not None

        self.net.eval()
        with torch.no_grad():
            outs = self.net(
                features_list=[_to_device_features(features, self.device)],
                repred=True,
                algorithm_index=algorithm_index,
                return_hints=return_hints,
                return_all_outputs=return_all_outputs,
            )
            if self.debug:
                outs_raw, hint_preds, hidden_states = outs
            else:
                outs_raw, hint_preds = outs
            outs_pp = decoders.postprocess(
                self._spec[algorithm_index],
                outs_raw,
                sinkhorn_temperature=0.1,
                sinkhorn_steps=50,
                hard=True,
            )
        if self.debug:
            return outs_pp, hint_preds, hidden_states
        else:
            return outs_pp, hint_preds

    # ---------------- save / restore (parity with pickle logic) ----------------
    def restore_model(self, file_name: str, only_load_processor: bool = False):
        path = os.path.join(self.checkpoint_path, file_name)
        with open(path, 'rb') as f:
            restored_state = pickle.load(f)
        # params
        params = restored_state['params']
        if only_load_processor:
            # 仅加载 processor 参数：保留现有，替换 processor 子树
            cur = self.params
            new = {}
            for k, v in params.items():
                if processors.PROCESSOR_TAG in k:
                    new[k] = v
                else:
                    new[k] = cur.get(k, v)
            self.params = new
        else:
            self.params = params
        # opt_state
        if 'opt_state' in restored_state and self._optimizer is not None:
            self.opt_state = restored_state['opt_state']

    def save_model(self, file_name: str):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        to_save = {'params': self.params, 'opt_state': self.opt_state}
        path = os.path.join(self.checkpoint_path, file_name)
        with open(path, 'wb') as f:
            pickle.dump(to_save, f)


# ---------------------------
# BaselineModelChunked
# ---------------------------
class BaselineModelChunked(BaselineModel):
    """Processes time-chunked data for training (PyTorch)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 将构造器换成 NetChunked（对应原 baselines.BaselineModelChunked._create_net_fns）:contentReference[oaicite:9]{index=9}
        def _ctor():
            return NetChunked(
                spec=self._spec,
                hidden_dim=self.net.hidden_dim if self.net is not None else args[3] if len(args) > 3 else 32,
                encode_hints=self.net.encode_hints if self.net is not None else kwargs.get('encode_hints', False),
                decode_hints=self.decode_hints,
                processor_factory=None,  # 将在 init 时替换为父类保存的
                use_lstm=self.net.use_lstm if self.net is not None else kwargs.get('use_lstm', False),
                encoder_init=kwargs.get('encoder_init', 'default'),
                dropout_prob=kwargs.get('dropout_prob', 0.0),
                hint_teacher_forcing=kwargs.get('hint_teacher_forcing', 0.0),
                hint_repred_mode=kwargs.get('hint_repred_mode', 'soft'),
                nb_dims=self.nb_dims,
                nb_msg_passing_steps=self.nb_msg_passing_steps,
            )
        self._net_chunked_ctor = None  # 我们在 init() 中用与父类一致的构造器重建

        # 状态容器（对齐原始实现中的 mp_states/init_mp_states）:contentReference[oaicite:10]{index=10}
        self.mp_states: List[List[MessagePassingStateChunked]] = []
        self.init_mp_states: List[List[MessagePassingStateChunked]] = []

    def _make_chunked_net_ctor(self, processor_factory):
        # 用与父类 _net_ctor 相同的参数，但类换成 NetChunked
        def _ctor():
            return NetChunked(
                spec=self._spec,
                hidden_dim=self.net.hidden_dim if self.net is not None else 32,
                encode_hints=self.net.encode_hints if self.net is not None else False,
                decode_hints=self.decode_hints,
                processor_factory=processor_factory,
                use_lstm=self.net.use_lstm if self.net is not None else False,
                encoder_init='default',
                dropout_prob=0.0,
                hint_teacher_forcing=self.hint_teacher_forcing,
                hint_repred_mode='soft',
                nb_dims=self.nb_dims,
                nb_msg_passing_steps=self.nb_msg_passing_steps,
            )
        self._net_chunked_ctor = _ctor

    # 重写 init：初始化 chunked mp_state（对应 baselines.BaselineModelChunked.init/_init_mp_state）:contentReference[oaicite:11]{index=11}
    def init(self, features: List[List[_FeaturesChunked]], seed: int):
        torch.manual_seed(seed)
        # 构建 NetChunked
        # 复用与父类相同的 processor_factory 等参数
        # 直接用父类保存的 _net_ctor 配置重新创建 NetChunked:
        # 简化：以当前父类 _net_ctor 的参数为准创建 NetChunked
        self.net = NetChunked(
            spec=self._spec,
            hidden_dim=self.net.hidden_dim if isinstance(self.net, Net) else 32,
            encode_hints=self.net.encode_hints if isinstance(self.net, Net) else False,
            decode_hints=self.decode_hints,
            processor_factory=self.net.processor_factory if isinstance(self.net, Net) else processors.Processor,  # 占位
            use_lstm=self.net.use_lstm if isinstance(self.net, Net) else False,
            encoder_init=self.net.encoder_init if isinstance(self.net, Net) else 'default',
            dropout_prob=self.net._dropout_prob if isinstance(self.net, Net) else 0.0,
            hint_teacher_forcing=self.net._hint_teacher_forcing if isinstance(self.net, Net) else 0.0,
            hint_repred_mode=self.net._hint_repred_mode if isinstance(self.net, Net) else 'soft',
            nb_dims=self.nb_dims,
            nb_msg_passing_steps=self.nb_msg_passing_steps,
        ).to(self.device)

        # 初始化 mp_states：对齐原实现 _init_mp_state 的“先以 init_mp_state=True 跑一次”逻辑:contentReference[oaicite:12]{index=12}
        self.mp_states = []
        self.init_mp_states = []
        for length_group in features:
            group_states = []
            for _feat in length_group:
                # 空状态（与原 empty_mp_state 字段一致）
                empty = MessagePassingStateChunked(
                    inputs=None, hints=None,
                    is_first=None, hint_preds=None,
                    hiddens=None, lstm_state=None
                )
                # 先触发 init_mp_state=True：返回更新后的 mp_state
                self.net.eval()
                with torch.no_grad():
                    _ = self.net(
                        features_list=[_to_device_features(_feat, self.device)],
                        mp_state_list=[empty],
                        repred=False,
                        init_mp_state=True,
                        algorithm_index=-1
                    )
                group_states.append(empty)
            self.mp_states.append(group_states)
            self.init_mp_states.append([s for s in group_states])

        # 准备优化器
        named_params = list(self.net.named_parameters())
        if self._freeze_processor:
            for n, p in named_params:
                if processors.PROCESSOR_TAG in n:
                    p.requires_grad = False
        params = [p for _, p in named_params if p.requires_grad]
        self._optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        self._param_groups_ready = True
        self._opt_state_skeleton = {"_": torch.tensor(0.0, device=self.device)}

    # 不提供 predict（与原版一致）
    def predict(self, rng_key, features: _FeaturesChunked, algorithm_index: Optional[int] = None):
        raise NotImplementedError

    # 对齐原 _loss(has_aux=True)：返回 (loss, (mp_state,))
    def _loss_chunked(self, feedback: _Feedback, mp_state: MessagePassingStateChunked, algorithm_index: int):
        self.net.train()
        (output_preds, hint_preds), mp_state = self.net(
            features_list=[_to_device_features(feedback.features, self.device)],
            mp_state_list=[mp_state],
            repred=True,
            init_mp_state=False,
            algorithm_index=algorithm_index
        )
        nb_nodes = _nb_nodes(feedback, is_chunked=True)
        total_loss = torch.tensor(0.0, device=self.device)
        is_first = feedback.features.is_first.to(self.device)
        is_last = feedback.features.is_last.to(self.device)

        # 输出损失（chunked）
        for truth in feedback.outputs:
            total_loss = total_loss + losses.output_loss_chunked(
                truth=truth, pred=output_preds[truth.name],
                is_last=is_last, nb_nodes=nb_nodes
            )

        # hint 损失（chunked）
        if self.decode_hints:
            for truth in feedback.features.hints:
                total_loss = total_loss + losses.hint_loss_chunked(
                    truth=truth, pred=hint_preds[truth.name],
                    is_first=is_first, nb_nodes=nb_nodes
                )

        return total_loss, mp_state

    # compute_grad：返回 loss 和 grads，并维护 mp_state（对齐原 compute_grad 返回值形态）:contentReference[oaicite:13]{index=13}
    def compute_grad(self, rng_key, feedback: _Feedback, algorithm_index: Optional[Tuple[int, int]] = None):
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = (0, 0)
        length_index, algo_index = algorithm_index
        mp_state = self.init_mp_states[length_index][algo_index]

        assert self.net is not None
        self.net.train()
        self.net.zero_grad(set_to_none=True)
        loss, mp_state = self._loss_chunked(feedback, mp_state, algo_index)
        loss.backward()
        grads = {n: (None if p.grad is None else p.grad.detach().clone())
                 for n, p in self.net.named_parameters()}
        # 更新缓存的 mp_state（与原逻辑一致）
        self.mp_states[length_index][algo_index] = mp_state
        return float(loss.detach().cpu().item()), grads

    # feedback：一步训练并更新 mp_state（对应原 jitted_feedback）:contentReference[oaicite:14]{index=14}
    def feedback(self, rng_key, feedback: _Feedback, algorithm_index=None) -> float:
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = (0, 0)
        length_index, algo_index = algorithm_index
        mp_state = self.init_mp_states[length_index][algo_index]

        assert self.net is not None and self._optimizer is not None
        self.net.train()
        self._optimizer.zero_grad(set_to_none=True)
        loss, mp_state = self._loss_chunked(feedback, mp_state, algo_index)
        loss.backward()
        if self.grad_clip_max_norm and self.grad_clip_max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_max_norm)
        self._optimizer.step()

        self.mp_states[length_index][algo_index] = mp_state
        return float(loss.detach().cpu().item())

    def verbose_loss(self, *args, **kwargs):
        raise NotImplementedError
