# evaluator.py

from __future__ import annotations
import os

# ---- 线程与后端：避免 fork + autograd 冲突（强制单线程 & CPU）----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import math
import inspect
import importlib
import importlib.util
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import os
import time
import torch
import torch.nn as nn

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# -------------------- 运行时猴补丁：拦截违规 API --------------------
# def _monkey_patch_forbidden():
# def _raise(*args, **kwargs):
#     raise RuntimeError("Use of forbidden API detected (nn.Linear/matmul/einsum/@)")
# nn.Linear = _raise  # type: ignore
# torch.matmul = _raise  # type: ignore
# torch.Tensor.__matmul__ = _raise  # type: ignore
# torch.einsum = _raise  # type: ignore
# # 新增：
# torch.dot = _raise
# torch.mm = _raise
# torch.bmm = _raise
# torch.mv = _raise
# torch.addmm = _raise
# torch.addmv = _raise

# -------------------- 静态扫描：忽略注释与字符串，只扫真实代码 --------------------
import io, tokenize, re


# -------------------- Cascade Evaluation API --------------------

def evaluate_stage1(candidate_program_path: str) -> dict:
    """
    Fast screening evaluation (no training).
    - Build candidate model
    - Run I/O smoke test
    - Measure median forward time on random data
    - Estimate MACs from hyperparams
    - Return a lightweight score (no CIFAR training)
    """
    import types
    print("instage1")
    # 读取并矫正 __future__ 位置
    with open(candidate_program_path, "r", encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines()
    future = [ln for ln in lines if ln.strip().startswith("from __future__ import")]
    others = [ln for ln in lines if not ln.strip().startswith("from __future__ import")]
    fixed_src = ""
    if future:
        fixed_src += "\n".join(future) + "\n"
    fixed_src += "\n".join(others) + "\n"

    module_name = "candidate_program_stage1"
    program_module = types.ModuleType(module_name)
    sys.modules[module_name] = program_module
    try:
        code_obj = compile(fixed_src, candidate_program_path, "exec")
        exec(code_obj, program_module.__dict__)

        build_model = getattr(program_module, "build_model")
        model, meta = build_model()
        device = "cpu"
        model.to(device)

        # I/O smoke check
        with torch.inference_mode():
            _probe = torch.randn(4, 3, 32, 32, device=device)
            _out = model(_probe)
            if not (isinstance(_out, torch.Tensor) and _out.dim() == 2
                    and _out.size(0) == 4 and _out.size(1) == 10):
                raise RuntimeError(f"Model I/O contract violated: got {_out.shape}")

        # Measure forward latency (median over random input)
        infer_time = _measure_forward_median(model, device=device)

        # Estimate MACs
        hp = _validate_hparams(meta)
        macs = _estimate_macs(hp)

        # Quick combined score: no accuracy term
        alpha_t = float(os.environ.get("OE_ALPHA", "0.005"))
        alpha_m = float(os.environ.get("OE_ALPHA_MACS", "0.000"))
        quick_score = - alpha_t * float(infer_time) - alpha_m * math.log1p(float(macs) / 3e4)

        return {
            "top1": 0.0,
            "infer_times_s": float(infer_time),
            "macs": float(macs),
            "score": float(quick_score),
            "combined_score": float(quick_score),
            "meta": meta,
            "timeout": False,
            "stage": "stage1",
            # === 新增：统一锚点/方向层特征 ===
            # "metrics": {"acc": 0.0},
            # "resources": {"infer_times_s": float(infer_time)},
            # "arch_feature_vec": meta.get("arch_feature_vec", []),
            # "arch_signature": meta.get("arch_signature", {}),
            # "anchor_features": {
            #     "metrics": {"acc": 0.0},
            #     "resources": {"infer_times_s": float(infer_time)},
            #     "arch_feature_vec": meta.get("arch_feature_vec", []),
            #     "arch_signature": meta.get("arch_signature", {})
            # }
        }

    except Exception as e:
        return {
            "combined_score": -1e9,
            "metrics": {"error": 1.0},
            "message": f"{type(e).__name__}: {e}",
            "stage": "stage1",
            "macs":1e9,
            "infer_times_s":1e9,
            "top1":-1e9,
        }


import subprocess, json, shlex, textwrap, tempfile, pathlib


def _run_stage2_in_spawn(candidate_program_path: str, timeout_s: int = 0) -> dict:
    """
    用全新 Python 进程(subprocess)跑 evaluate(candidate_program_path)。
    - 没有函数 pickling 问题
    - 不继承父进程中的线程/状态，规避 autograd×fork 冲突
    - 支持超时
    """
    py = sys.executable
    this_file = os.path.abspath(__file__)
    cand = os.path.abspath(candidate_program_path)

    # 在子进程里执行的代码：重新导入当前 evaluator 模块文件，调用 _evaluate 流水线
    # 注意：device 走 "auto"，并读取 OE_DEVICE 环境变量（cuda/cpu）
    child_code = textwrap.dedent(f"""
    import os, sys, types, json, traceback
    sys.path.insert(0, {repr(os.path.dirname(this_file))})
    # 动态导入 evaluator 本身（以 "evaluation_module" 名称装载，避免别名不一致）
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluation_module", {repr(this_file)})
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)

    path = {repr(cand)}
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        lines = src.splitlines()
        future = [ln for ln in lines if ln.strip().startswith("from __future__ import")]
        others = [ln for ln in lines if not ln.strip().startswith("from __future__ import")]
        fixed_src = ("\\n".join(future) + "\\n" if future else "") + "\\n".join(others) + "\\n"

        mod = types.ModuleType("candidate_program_spawn")
        sys.modules["candidate_program_spawn"] = mod
        code_obj = compile(fixed_src, path, "exec")
        exec(code_obj, mod.__dict__)

        # 调 em._evaluate；设备让 em._evaluate 内部根据 "auto"/OE_DEVICE 决定
        device = os.environ.get("OE_DEVICE_MODE", "auto")
        result = em._evaluate(mod, device=device)
        print(json.dumps({{"ok": True, "payload": result}}))
    except Exception as e:
        print(json.dumps({{"ok": False, "err": f"{{type(e).__name__}}: {{e}}\\n{{traceback.format_exc()}}"}}))
        sys.exit(2)
    """)

    # 调用子进程
    try:
        cp = subprocess.run(
            [py, "-u", "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s if timeout_s and timeout_s > 0 else None,
            env=os.environ.copy(),  # 继承 OE_DEVICE, OE_USE_AMP 等
        )
    except subprocess.TimeoutExpired:
        return {"combined_score": -1e9, "metrics": {"error": 1.0}, "message": "Stage2 timeout in subprocess"}

    # 解析子进程输出（stdout 的最后一行是我们的 JSON）
    out = (cp.stdout or "").strip().splitlines()
    if not out:
        # 把 stderr 也带上，便于排错
        return {"combined_score": -1e9, "metrics": {"error": 1.0},
                "message": f"Stage2 no output. stderr:\\n{cp.stderr}"}

    try:
        data = json.loads(out[-1])
    except Exception:
        return {"combined_score": -1e9, "metrics": {"error": 1.0},
                "message": f"Stage2 bad JSON: {out[-1]}\\nstderr:\\n{cp.stderr}"}

    if not data.get("ok"):
        return {"combined_score": -1e9, "metrics": {"error": 1.0}, "message": data.get("err", "Stage2 failed")}

    payload = data["payload"]
    return {"combined_score": payload.get("score", -1e9), **payload}


def evaluate_stage2(candidate_program_path: str) -> dict:
    # 子进程完整评测（可用 OE_STAGE2_TIMEOUT 控制秒数）
    t = int(os.environ.get("OE_STAGE2_TIMEOUT", "0"))
    return _run_stage2_in_spawn(candidate_program_path, timeout_s=t)


def merge_cascade_results(stage1_result: dict, stage2_result: dict) -> dict:
    """
    Merge cascade results.
    - Prefer stage2 (full evaluation) if it succeeded
    - Fall back to stage1 (fast screening) if stage2 failed or timed out
    - Attach both results for debugging/analysis
    """
    final = None
    print("inmerge_cascade_results")
    # 判定 stage2 是否有效
    if stage2_result and stage2_result.get("combined_score", -1e9) > -1e8 \
            and not stage2_result.get("timeout", False):
        final = stage2_result.copy()
        final["stage"] = "merged(stage2)"
    else:
        print("stage1")
        # 回退到 stage1
        final = stage1_result.copy() if stage1_result else {"combined_score": -1e9}
        final["stage"] = "merged(stage1_fallback)"

    # 为了方便后续分析，保留原始两个结果
    final["cascade_debug"] = {
        "stage1": stage1_result,
        "stage2": stage2_result,
    }

    return final


def _scan_source_forbidden(program_module):
    """
    静态源码扫描（仅在 OE_SCAN_FORBID=1 时启用）：
    检测候选程序中是否出现被禁止的算子调用。
    """
    if os.environ.get("OE_SCAN_FORBID", "0") != "1":
        return

    # 尝试获取源码
    src_chunks = []
    try:
        src_chunks.append(inspect.getsource(program_module))
    except Exception:
        # 遍历模块属性收集尽可能多的源码片段
        for name in dir(program_module):
            obj = getattr(program_module, name)
            try:
                src_chunks.append(inspect.getsource(obj))
            except Exception:
                pass
    src = "\n".join(src_chunks)

    # 需要禁止的 token（可按需增删）
    tokens = [
        # r"\bnn\.Linear\b",
        # r"\btorch\.matmul\b",
        # r"\beinsum\b",
        # r"@",
        # r"\btorch\.mm\b",
        # r"\btorch\.bmm\b",
        # r"\btorch\.mv\b",
        # r"\btorch\.dot\b",
        # r"\btorch\.addmm\b",
        # r"\btorch\.addmv\b",
    ]
    if not src:
        return
    pattern = re.compile("|".join(tokens))
    m = pattern.search(src)
    if m:
        hit = m.group(0)
        raise RuntimeError(f"Forbidden token detected in source: {hit}. "
                           f"Unset OE_SCAN_FORBID or remove forbidden calls.")


# -------------------- 训练/数据工具 --------------------
def _seed_all(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_cifar10_dataloaders(train_n: int = 1000, test_n: int = 100, seed: int = 42, bs: int = 50):
    _seed_all(seed)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ])
    # 注意：root 指向“包含 cifar-10-batches-py 的父目录”
    root = os.environ.get("CIFAR10_ROOT", "/data/lz/openevolve/dataset")
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)

    # 固定子集索引
    g = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train_set), generator=g)[:train_n].tolist()
    test_idx = torch.randperm(len(test_set), generator=g)[:test_n].tolist()

    # 避免 DataLoader 多进程：num_workers=0
    train_loader = DataLoader(Subset(train_set, train_idx), batch_size=bs, shuffle=True, num_workers=0)
    test_loader = DataLoader(Subset(test_set, test_idx), batch_size=bs, shuffle=False, num_workers=0)
    return train_loader, test_loader


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class _LocalTimeout(Exception):
    pass


import signal


class _LocalTimeoutCtx:
    def __init__(self, seconds: int):
        self.seconds = int(seconds) if seconds else 0
        self._old_handler = None

    def __enter__(self):
        if self.seconds <= 0:
            return

        def _handler(signum, frame):
            raise _LocalTimeout("local hard timeout")

        self._old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0:
            signal.alarm(0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
        # 不吞异常
        return False


REQUIRED_KEYS = ("in_dim", "num_classes", "hidden_dim", "lowrank_rank", "groups", "sparsity")


def _validate_hparams(meta: dict):
    hp = (meta or {}).get("hyperparams", {}) or {}
    defaults = {"in_dim": 3 * 32 * 32,
                "num_classes": 10,
                "hidden_dim": 0,
                "lowrank_rank": 0,
                "groups": 1,
                "sparsity": 1.0,
                }
    for k, v in defaults.items():
        hp.setdefault(k, v)
    return hp


# def _validate_hparams(meta: dict):
#     hp = (meta or {}).get("hyperparams", {}) or {}
#     miss = [k for k in REQUIRED_KEYS if k not in hp]
#     if miss:
#         raise ValueError(f"hyperparams missing keys: {miss}. "
#                          "Candidates MUST return these in meta['hyperparams'].")
#     return hp

def _estimate_macs(meta: dict) -> int:
    """
    估算循环 Linear 的乘法次数（MACs），优先读取结构超参；向后兼容旧逻辑。
    约定：
      - 低秩分解（W≈U@V）：macs = in_dim * r + r * C, 其中 r=lowrank_rank
      - 分组线性（g 组）：macs ≈ in_dim * C / g
      - 稀疏（非零比例 ρ）：在上述基础上乘以 ρ（或 1-sparsity）
      - 两层 MLP：in_dim * H + H * C
      - 否则：in_dim * C
    """
    hp = _validate_hparams(meta)

    # hp = meta.get("hyperparams", {}) or {}
    in_dim = int(hp.get("in_dim", 3 * 32 * 32))
    C = int(hp.get("num_classes", 10))
    H = int(hp.get("hidden_dim", 0) or 0)

    # 优先识别“能显著改变算术量”的结构
    r = int(hp.get("lowrank_rank", 0) or 0)
    g = int(hp.get("groups", 1) or 1)
    sparsity = float(hp.get("sparsity", hp.get("nonzero_ratio", 1.0)))
    if sparsity <= 0.0: sparsity = 1.0
    sparsity = max(0.0, min(1.0, sparsity))

    if r > 0:
        macs = in_dim * r + r * C
    elif g > 1:
        macs = (in_dim * C) // max(1, g)
    elif H > 0:
        macs = in_dim * H + H * C
    else:
        macs = in_dim * C

    macs = int(max(1, macs * sparsity))
    return macs


def _measure_forward_median(model: nn.Module, device: str = "cpu",
                            batch: int = None, warmup: int = 5, runs: int = 10) -> float:
    if batch is None:
        batch = int(os.environ.get("OE_BENCH_BATCH", "64"))
    model.eval()
    x = torch.randn(batch, 3, 32, 32, device=device)
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    return float(times[len(times) // 2])


# --- NEW: 顶层可pickle的子进程入口 ---
def _stage2_spawn_child(path: str, q):
    """
    在spawn子进程中执行：编译 candidate_program 并调用本模块的 _evaluate()。
    结果/错误以JSON字符串写入队列 q。
    """
    try:
        import sys, types, json, traceback
        # 读取并修正 __future__ 位置
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        lines = src.splitlines()
        future = [ln for ln in lines if ln.strip().startswith("from __future__ import")]
        others = [ln for ln in lines if not ln.strip().startswith("from __future__ import")]
        fixed_src = ("\n".join(future) + "\n" if future else "") + "\n".join(others) + "\n"

        mod = types.ModuleType("candidate_program_spawn")
        sys.modules["candidate_program_spawn"] = mod
        code_obj = compile(fixed_src, path, "exec")
        exec(code_obj, mod.__dict__)

        # 直接使用本模块中的 _evaluate（spawn 会重新导入本模块，顶层函数可见）
        result = _evaluate(mod, device=os.environ.get("OE_DEVICE_MODE", "auto"))
        q.put(json.dumps({"ok": True, "payload": result}))
    except Exception as e:
        q.put(json.dumps({"ok": False, "err": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}))


def _train_one_epoch(model: nn.Module,
                     train_loader,
                     device: str = "cpu",
                     lr: float = 1e-2,
                     weight_decay: float = 0.0,
                     max_batches: int = None) -> dict:
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    use_amp = device.startswith("cuda") and os.environ.get("OE_USE_AMP", "0") == "1"
    try:
        scaler = torch.amp.GradScaler(device if use_amp else "cpu", enabled=use_amp)
    except Exception:
        # 向后兼容老版本 PyTorch
        from torch.cuda.amp import GradScaler as _OldScaler
        scaler = _OldScaler(enabled=use_amp)

    total_loss = 0.0
    total_items = 0
    batches = 0

    for i, (xb, yb) in enumerate(train_loader):
        if max_batches is not None and i >= max_batches:
            break
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

        total_loss += loss.detach().item() * yb.size(0)
        total_items += yb.size(0)
        batches += 1

    avg_loss = total_loss / max(1, total_items)
    return {"train_loss": float(avg_loss), "batches": batches}


# def _monkey_patch_forbidden():
#     """
#     运行期“禁用”高阶算子，强制在 for-loop 搜索空间内演化。
#     建议仅在设置了 OE_FORBID_MM=1 时调用。
#     被禁用：nn.Linear, torch.matmul, @, einsum, dot, mm, bmm, mv, addmm, addmv
#     """
#
#     def _raise(*args, **kwargs):
#         raise RuntimeError("Use of forbidden API detected: nn.Linear/matmul/einsum/@/mm/dot/bmm/mv/addmm/addmv")
#
#     nn.Linear = _raise  # type: ignore
#     torch.matmul = _raise  # type: ignore
#     torch.Tensor.__matmul__ = _raise  # type: ignore
#     torch.einsum = _raise  # type: ignore
#     torch.dot = _raise
#     torch.mm = _raise
#     torch.bmm = _raise
#     torch.mv = _raise
#     torch.addmm = _raise
#     torch.addmv = _raise


# -------------------- 核心评测：固定 CPU；失败时自动降级为“无训练评测” --------------------
def _evaluate(program_module, device: str = "auto") -> dict:
    # 设备选择（已有就保留）
    if device == "auto":
        device = os.environ.get("OE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # 读取本地硬超时秒数（比如 60），未设置则为 0 表示不开
    local_hard_timeout = int(os.environ.get("OE_HARD_TIMEOUT", "0"))

    # ……你已有的猴补丁/源码扫描/构建模型等逻辑 ……
    build_model = getattr(program_module, "build_model")
    model, meta = build_model()
    model.to(device)
    # —— I/O 形状 smoke check（训练前） ——
    with torch.inference_mode():
        _probe = torch.randn(4, 3, 32, 32, device=device)
        _out = model(_probe)
        if not (isinstance(_out, torch.Tensor) and _out.dim() == 2 and _out.size(0) == 4 and _out.size(1) == 10):
            raise RuntimeError(
                f"Model I/O contract violated: expected [B,10], got {_out.shape if isinstance(_out, torch.Tensor) else type(_out)}")

    # 数据加载（建议 num_workers=0），以及训练/测试子集规模（可通过环境变量配置）
    train_n = int(os.environ.get("OE_TRAIN_N", "1000"))
    test_n = int(os.environ.get("OE_TEST_N", "100"))
    bs = int(os.environ.get("OE_BATCH", "50"))
    train_loader, test_loader = _get_cifar10_dataloaders(train_n=train_n, test_n=test_n, seed=42, bs=bs)

    try:
        with _LocalTimeoutCtx(local_hard_timeout):
            # ===== 训练（限批） =====
            max_batches = int(os.environ.get("OE_MAX_TRAIN_BATCHES", "5"))
            _train_one_epoch(model, train_loader, device=device, max_batches=100)

            # ===== 准确率 =====
            model.eval()
            with torch.inference_mode():
                acc_sum = 0
                n = 0
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    pred = logits.argmax(dim=1)
                    acc_sum += (pred == yb).sum().item()
                    n += yb.numel()
            top1 = acc_sum / max(1, n)

            # ===== 纯前向时延（中位数） =====
            infer_time = _measure_forward_median(model, device=device)

            # ===== 资源与综合分数 =====
            hp = _validate_hparams(meta)
            macs = _estimate_macs(hp)
            alpha = float(os.environ.get("OE_ALPHA", "0.005"))
            score = float(top1) - alpha * float(max(1e-9, infer_time))

            alpha_t = float(os.environ.get("OE_ALPHA", "0.005"))
            alpha_m = float(os.environ.get("OE_ALPHA_MACS", "0.005"))  # 默认为 0，不影响现有行为
            combined = float(top1) - alpha_t * float(infer_time) - alpha_m * math.log1p(float(macs) / 3e4)
            print("top1:", top1)
            return {
                "top1": float(top1),
                "infer_times_s": float(infer_time),
                "macs": float(macs),
                "score": float(score),
                "combined_score": float(combined),
                "meta": meta,
                "timeout": False,
                "stage": "stage2",
            }


    except (_LocalTimeout, TimeoutError):
        # 子进程内自己认怂，给一个“安全可解析”的结果，让 Future 正常返回
        print("chaoshi1")
        hp = _validate_hparams(meta)
        macs = _estimate_macs(hp)
        return {
            "top1": 0.0,
            "infer_times_s": float(local_hard_timeout),
            "macs": float(macs),
            # === 新增：统一锚点/方向层特征（超时兜底） ===
            # "metrics": {"acc": 0.0},
            # "resources": {"infer_times_s": float(local_hard_timeout)},
            # "arch_feature_vec": meta.get("arch_feature_vec", []),
            # "arch_signature": meta.get("arch_signature", {}),
            # "anchor_features": {
            #     "metrics": {"acc": 0.0},
            #     "resources": {"infer_times_s": float(local_hard_timeout)},
            #     "arch_feature_vec": meta.get("arch_feature_vec", []),
            #     "arch_signature": meta.get("arch_signature", {})
            # },

            "score": -1e9,  # 明确劣化，避免被当成好样本
            "meta": meta,
            "timeout": True,
        }


# -------------------- CLI 单测入口（不影响 OpenEvolve） --------------------
def main():
    module_path = os.environ.get("CANDIDATE_PATH", "initial_program.py")
    spec = importlib.util.spec_from_file_location("candidate_program", module_path)
    assert spec and spec.loader, f"cannot load program at {module_path}"
    program_module = importlib.util.module_from_spec(spec)
    print("porgram_module:", program_module)
    sys.modules["candidate_program"] = program_module
    spec.loader.exec_module(program_module)  # type: ignore
    result = _evaluate(program_module, device="auto")
    print({"combined_score": result["score"], **result})


# -------------------- OpenEvolve 标准入口：自动矫正 __future__ 位置 --------------------
def evaluate(candidate_program_path: str) -> dict:
    print("instage2")
    import types

    # 读取并矫正 __future__ 位置
    with open(candidate_program_path, "r", encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines()
    future = [ln for ln in lines if ln.strip().startswith("from __future__ import")]
    others = [ln for ln in lines if not ln.strip().startswith("from __future__ import")]
    fixed_src = ""
    if future:
        fixed_src += "\n".join(future) + "\n"
    fixed_src += "\n".join(others) + "\n"

    # 动态创建模块并执行
    module_name = "candidate_program"
    program_module = types.ModuleType(module_name)
    sys.modules[module_name] = program_module
    try:
        code_obj = compile(fixed_src, candidate_program_path, "exec")
        exec(code_obj, program_module.__dict__)
        result = _evaluate(program_module, device="auto")
        return {"combined_score": result["score"], **result}
    except RuntimeError as e:
        # 对 “Autograd & Fork” 等已知问题兜底：跳过训练再评估一次
        if "Autograd" in str(e) and "Fork" in str(e):
            os.environ["OE_SKIP_TRAIN"] = "1"
            try:
                result = _evaluate(program_module, device="auto")
                return {"combined_score": result["score"], **result}
            finally:
                os.environ.pop("OE_SKIP_TRAIN", None)
        # 其它异常：返回极低分，带上错误信息，保证演化不中断
        return {"combined_score": -1e9, "metrics": {"error": 1.0}, "message": f"{type(e).__name__}: {e}"}
    except Exception as e:
        return {"combined_score": -1e9, "metrics": {"error": 1.0}, "message": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    print("len(sys.argv):", len(sys.argv))
    if len(sys.argv) == 2:
        print(evaluate(sys.argv[1]))
    else:
        main()