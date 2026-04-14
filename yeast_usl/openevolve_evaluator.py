import os
import sys
import time
import json
import random
import argparse
import traceback
import importlib.util
import subprocess
import contextlib
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Reduce thread oversubscription, mirroring the working reference evaluator.
try:
    torch.set_num_threads(int(os.getenv("YEAST_TORCH_NUM_THREADS", "1")))
except Exception:
    pass
try:
    torch.set_num_interop_threads(int(os.getenv("YEAST_TORCH_NUM_INTEROP_THREADS", "1")))
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_NPZ = "/data1/lz/clrs/openevolve/yeast_usl/yeast_prepared/yeast_train.npz"
VAL_NPZ = "/data1/lz/clrs/openevolve/yeast_usl/yeast_prepared/yeast_val.npz"
TEST_NPZ = "/data1/lz/clrs/openevolve/yeast_usl/yeast_prepared/yeast_test.npz"
LABEL_NAMES_TXT = "/data1/lz/clrs/openevolve/yeast_usl/yeast_prepared/label_names.txt"

try:
    from openevolve.evaluation_result import EvaluationResult  # type: ignore
except Exception:
    @dataclass
    class EvaluationResult:
        metrics: Dict[str, Any]
        artifacts: Dict[str, Any]


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(int(default))).strip().lower()
    return value in {"1", "true", "t", "yes", "y", "on"}


def resolve_device(requested: str) -> str:
    requested = str(requested).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda"):
        return requested if torch.cuda.is_available() else "cpu"
    if requested == "mps":
        return "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    return "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_args(as_dict: bool = False):
    arg_dict = {
        "input_dim": int(os.getenv("YEAST_INPUT_DIM", "8")),
        "num_classes": int(os.getenv("YEAST_NUM_CLASSES", "10")),
        "learning_rate": float(os.getenv("YEAST_LR", "1e-3")),
        "weight_decay": float(os.getenv("YEAST_WEIGHT_DECAY", "1e-4")),
        "batch_size": int(os.getenv("YEAST_BATCH_SIZE", "64")),
        "epochs": int(os.getenv("YEAST_EPOCHS", "30")),
        "device": "cpu",
        "seed": int(os.getenv("YEAST_SEED", "42")),
        "max_params": int(os.getenv("YEAST_MAX_PARAMS", "1000000")),
        "stage2_timeout_sec": int(os.getenv("YEAST_STAGE2_TIMEOUT_SEC", "1200")),
        "stage2_train_limit": int(os.getenv("YEAST_STAGE2_TRAIN_LIMIT", "0")),
        "stage2_val_limit": int(os.getenv("YEAST_STAGE2_VAL_LIMIT", "0")),
        "stage2_test_limit": int(os.getenv("YEAST_STAGE2_TEST_LIMIT", "0")),
        "stage1_train_limit": int(os.getenv("YEAST_STAGE1_TRAIN_LIMIT", "0")),
        "stage1_val_limit": int(os.getenv("YEAST_STAGE1_VAL_LIMIT", "0")),
        "stage1_test_limit": int(os.getenv("YEAST_STAGE1_TEST_LIMIT", "0")),
        "seed_list": [int(s) for s in os.getenv("YEAST_SEEDS", "0,1").split(",") if s.strip()],
    }
    return arg_dict if as_dict else type("ObjectView", (object,), arg_dict)()


def prepare_runtime_args():
    args = get_model_args(as_dict=False)
    requested_device = os.getenv("YEAST_DEVICE", "cpu")
    resolved_device = resolve_device(requested_device)
    args.device = resolved_device
    return args, requested_device, resolved_device


def load_program_module(program_path: str):
    module_name = f"candidate_program_{int(time.time() * 1e6)}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def static_check_candidate(program_path: str):
    """
    Stage1 static/interface check only.
    No training. No backward. Minimal runtime validation only through inspect.
    """
    module = load_program_module(program_path)

    if not hasattr(module, "SimpleMLP"):
        raise AttributeError("Candidate program must define class SimpleMLP")

    cls = getattr(module, "SimpleMLP")
    if not inspect.isclass(cls):
        raise TypeError("SimpleMLP must be a class")

    init_fn = getattr(cls, "__init__", None)
    if init_fn is None:
        raise AttributeError("SimpleMLP must define __init__")

    init_sig = inspect.signature(init_fn)
    init_params = list(init_sig.parameters.keys())
    required_init = ["self", "input_dim", "num_classes"]
    for p in required_init:
        if p not in init_params:
            raise ValueError(
                f"SimpleMLP.__init__ must contain parameter `{p}`, got {init_params}"
            )

    forward_fn = getattr(cls, "forward", None)
    if forward_fn is None:
        raise AttributeError("SimpleMLP must define forward")
    if not callable(forward_fn):
        raise TypeError("SimpleMLP.forward must be callable")

    forward_sig = inspect.signature(forward_fn)
    forward_params = list(forward_sig.parameters.keys())
    if len(forward_params) < 2:
        raise ValueError(
            f"SimpleMLP.forward must accept at least (self, x), got {forward_params}"
        )

    return {
        "class_name": cls.__name__,
        "init_params": init_params,
        "forward_params": forward_params,
    }


def build_candidate_model(module, args):
    if not hasattr(module, "SimpleMLP"):
        raise AttributeError("Candidate program must define class SimpleMLP")
    model = module.SimpleMLP(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
    )
    if not isinstance(model, torch.nn.Module):
        raise TypeError("SimpleMLP(...) must return a torch.nn.Module instance")
    return model


def validate_candidate_model_runtime(module, args):
    model = build_candidate_model(module, args)
    model.eval()
    with torch.no_grad():
        dummy_x = torch.randn(4, args.input_dim)
        dummy_y = model(dummy_x)
    if not isinstance(dummy_y, torch.Tensor):
        raise TypeError("Model forward must return a torch.Tensor")
    if tuple(dummy_y.shape) != (4, args.num_classes):
        raise ValueError(
            f"Model output shape must be (4, {args.num_classes}), got {tuple(dummy_y.shape)}"
        )
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_macs(model: nn.Module, input_dim: int) -> float:
    model = model.cpu().eval()
    macs = 0

    def linear_hook(module, inputs, outputs):
        nonlocal macs
        x = inputs[0]
        batch_size = x.shape[0]
        macs += batch_size * module.in_features * module.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    with torch.no_grad():
        dummy_x = torch.randn(1, input_dim)
        _ = model(dummy_x)

    for h in hooks:
        h.remove()

    return float(macs)


def load_dense_split(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing file: {npz_path}")
    arr = np.load(npz_path)
    if "X" not in arr or "y" not in arr:
        raise KeyError(f"{npz_path} must contain keys `X` and `y`")
    x = np.asarray(arr["X"], dtype=np.float32)
    y = np.asarray(arr["y"], dtype=np.int64)
    return x, y


def maybe_limit_split(x: np.ndarray, y: np.ndarray, limit: int):
    if limit is None or int(limit) <= 0 or len(y) <= int(limit):
        return x, y
    limit = int(limit)
    return x[:limit], y[:limit]


def build_dataloader(x, y, batch_size: int, shuffle: bool):
    x_t = torch.from_numpy(x.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.int64))
    dataset = TensorDataset(x_t, y_t)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def accuracy(model, dataloader, device):
    model.eval()
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = yb.cpu().numpy()
            preds_all.append(preds)
            labels_all.append(labels)

    preds_all = np.concatenate(preds_all, axis=0) if preds_all else np.array([], dtype=np.int64)
    labels_all = np.concatenate(labels_all, axis=0) if labels_all else np.array([], dtype=np.int64)
    return float(np.mean(preds_all == labels_all)) if len(labels_all) > 0 else 0.0


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_num = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            total_num += bs

    return float(total_loss / max(total_num, 1))


def train_model(dataset, model, args, epochs: int, batch_size: int):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    x_train, y_train = dataset["x_train"], dataset["y_train"]
    x_val, y_val = dataset["x_val"], dataset["y_val"]
    x_test, y_test = dataset["x_test"], dataset["y_test"]

    train_loader = build_dataloader(x_train, y_train, batch_size, shuffle=True)
    val_loader = build_dataloader(x_val, y_val, max(batch_size, 512), shuffle=False)
    test_loader = build_dataloader(x_test, y_test, max(batch_size, 512), shuffle=False)
    full_train_eval_loader = build_dataloader(x_train, y_train, max(batch_size, 512), shuffle=False)

    model = model.to(args.device)

    results = {
        "train_losses": [],
        "val_losses": [],
        "test_losses": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
    }

    best_val_acc = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_num = 0

        for xb, yb in train_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            epoch_loss_sum += float(loss.item()) * bs
            epoch_num += bs

        epoch_train_loss = epoch_loss_sum / max(epoch_num, 1)
        results["train_losses"].append(float(epoch_train_loss))

        train_loss_eval = evaluate_loss(model, full_train_eval_loader, criterion, args.device)
        val_loss_eval = evaluate_loss(model, val_loader, criterion, args.device)
        test_loss_eval = evaluate_loss(model, test_loader, criterion, args.device)
        train_acc_eval = accuracy(model, full_train_eval_loader, args.device)
        val_acc_eval = accuracy(model, val_loader, args.device)
        test_acc_eval = accuracy(model, test_loader, args.device)

        results["val_losses"].append(float(val_loss_eval))
        results["test_losses"].append(float(test_loss_eval))
        results["train_acc"].append(float(train_acc_eval))
        results["val_acc"].append(float(val_acc_eval))
        results["test_acc"].append(float(test_acc_eval))

        if val_acc_eval > best_val_acc:
            best_val_acc = val_acc_eval
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        final_test_loss = evaluate_loss(model, test_loader, criterion, args.device)
        final_test_acc = accuracy(model, test_loader, args.device)
        final_val_acc = accuracy(model, val_loader, args.device)
        final_train_acc = accuracy(model, full_train_eval_loader, args.device)
    else:
        final_test_loss = results["test_losses"][-1] if results["test_losses"] else float("inf")
        final_test_acc = results["test_acc"][-1] if results["test_acc"] else 0.0
        final_val_acc = results["val_acc"][-1] if results["val_acc"] else 0.0
        final_train_acc = results["train_acc"][-1] if results["train_acc"] else 0.0

    results["best_val_acc"] = float(best_val_acc if best_val_acc >= 0 else 0.0)
    results["final_test_loss_bestval"] = float(final_test_loss)
    results["final_test_acc_bestval"] = float(final_test_acc)
    results["final_val_acc_bestval"] = float(final_val_acc)
    results["final_train_acc_bestval"] = float(final_train_acc)
    return results


def build_dataset(args, train_limit: int = 0, val_limit: int = 0, test_limit: int = 0):
    x_train, y_train = load_dense_split(TRAIN_NPZ)
    x_val, y_val = load_dense_split(VAL_NPZ)
    x_test, y_test = load_dense_split(TEST_NPZ)

    x_train, y_train = maybe_limit_split(x_train, y_train, train_limit)
    x_val, y_val = maybe_limit_split(x_val, y_val, val_limit)
    x_test, y_test = maybe_limit_split(x_test, y_test, test_limit)

    dataset = {
        "x_train": x_train.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "x_val": x_val.astype(np.float32),
        "y_val": y_val.astype(np.int64),
        "x_test": x_test.astype(np.float32),
        "y_test": y_test.astype(np.int64),
    }
    return dataset


def evaluate_stage1(program_path: str):
    """
    Static-only stage1.
    No training.
    """
    try:
        args, requested_device, resolved_device = prepare_runtime_args()
        info = static_check_candidate(program_path)

        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "combine_score": 0.0,
                "ood_acc": 0.0,
                "final_test_acc": 0.0,
                "final_test_loss": 0.0,
                "num_params": 0.0,
                "macs": 0.0,
                "time": 0.0,
                "stage1_ok": 1.0,
            },
            artifacts={
                "stage": "stage1",
                "status": "static_check_ok",
                "requested_device": str(requested_device),
                "resolved_device": str(resolved_device),
                "input_dim": int(args.input_dim),
                "num_classes": int(args.num_classes),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "class_name": info["class_name"],
                "init_params": info["init_params"],
                "forward_params": info["forward_params"],
                "train_npz": TRAIN_NPZ,
                "val_npz": VAL_NPZ,
                "test_npz": TEST_NPZ,
                "label_names_txt": LABEL_NAMES_TXT,
            },
        )
    except Exception as e:
        return EvaluationResult(
            metrics={
                "combined_score": -1e18,
                "combine_score": -1e18,
                "ood_acc": -1e18,
                "final_test_acc": 0.0,
                "final_test_loss": float("inf"),
                "num_params": 0.0,
                "macs": -1.0,
                "time": 0.0,
                "stage1_ok": 0.0,
                "error": str(e),
            },
            artifacts={
                "stage": "stage1",
                "status": "static_check_failed",
                "traceback": traceback.format_exc(),
                "train_npz": TRAIN_NPZ,
                "val_npz": VAL_NPZ,
                "test_npz": TEST_NPZ,
                "label_names_txt": LABEL_NAMES_TXT,
            },
        )


def _stage2_core(program_path: str) -> EvaluationResult:
    args, requested_device, resolved_device = prepare_runtime_args()
    set_seed(args.seed)

    dataset = build_dataset(
        args,
        train_limit=args.stage2_train_limit,
        val_limit=args.stage2_val_limit,
        test_limit=args.stage2_test_limit,
    )

    if not os.path.exists(LABEL_NAMES_TXT):
        raise FileNotFoundError(f"Missing file: {LABEL_NAMES_TXT}")
    label_names = [ln.strip() for ln in open(LABEL_NAMES_TXT, "r", encoding="utf-8").read().splitlines() if ln.strip()]
    if len(label_names) > 0:
        args.num_classes = len(label_names)

    module = load_program_module(program_path)
    validate_candidate_model_runtime(module, args)
    model = build_candidate_model(module, args)

    num_params = count_params(model)
    if num_params <= 0 or num_params > args.max_params:
        raise ValueError(f"Invalid num_params={num_params}; allowed range is (0, {args.max_params}]")
    macs = estimate_macs(model, args.input_dim)

    t0 = time.time()
    seed_results = []
    train_curves = []
    val_curves = []
    test_curves = []
    train_loss_curves = []
    val_loss_curves = []
    test_loss_curves = []

    for seed in args.seed_list:
        set_seed(seed)
        module = load_program_module(program_path)
        validate_candidate_model_runtime(module, args)
        model = build_candidate_model(module, args)
        results = train_model(
            dataset=dataset,
            model=model,
            args=args,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        seed_results.append(results)
        train_curves.append([float(v) for v in results["train_acc"]])
        val_curves.append([float(v) for v in results["val_acc"]])
        test_curves.append([float(v) for v in results["test_acc"]])
        train_loss_curves.append([float(v) for v in results["train_losses"]])
        val_loss_curves.append([float(v) for v in results["val_losses"]])
        test_loss_curves.append([float(v) for v in results["test_losses"]])

    train_time_sec = time.time() - t0

    final_test_acc = float(np.mean([r["final_test_acc_bestval"] for r in seed_results]))
    final_test_loss = float(np.mean([r["final_test_loss_bestval"] for r in seed_results]))
    final_val_acc = float(np.mean([r["final_val_acc_bestval"] for r in seed_results]))
    final_train_acc = float(np.mean([r["final_train_acc_bestval"] for r in seed_results]))
    best_val_acc = float(np.mean([r["best_val_acc"] for r in seed_results]))
    ood_acc = float(final_test_acc)

    combined_score = (
        100.0 * float(ood_acc)
        - 0.1 * float(final_test_loss)
        - 1e-6 * float(macs)
        - 1e-6 * float(num_params)
        - 1e-3 * float(train_time_sec)
    )

    return EvaluationResult(
        metrics={
            "combined_score": float(combined_score),
            "combine_score": float(combined_score),
            "ood_acc": float(ood_acc),
            "final_test_acc": float(final_test_acc),
            "final_test_loss": float(final_test_loss),
            "final_val_acc": float(final_val_acc),
            "final_train_acc": float(final_train_acc),
            "best_val_acc": float(best_val_acc),
            "num_params": float(num_params),
            "macs": float(macs),
            "time": float(train_time_sec),
            "stage2_ok": 1.0,
        },
        artifacts={
            "stage": "stage2",
            "dataset_num_train": int(dataset["x_train"].shape[0]),
            "dataset_num_val": int(dataset["x_val"].shape[0]),
            "dataset_num_test": int(dataset["x_test"].shape[0]),
            "requested_device": str(requested_device),
            "resolved_device": str(resolved_device),
            "seed": int(args.seed),
            "seed_list": [int(s) for s in args.seed_list],
            "train_time_sec": float(train_time_sec),
            "input_dim": int(args.input_dim),
            "num_classes": int(args.num_classes),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "train_npz": TRAIN_NPZ,
            "val_npz": VAL_NPZ,
            "test_npz": TEST_NPZ,
            "label_names_txt": LABEL_NAMES_TXT,
            "train_acc_curve": train_curves,
            "val_acc_curve": val_curves,
            "test_acc_curve": test_curves,
            "train_loss_curve": train_loss_curves,
            "val_loss_curve": val_loss_curves,
            "test_loss_curve": test_loss_curves,
        },
    )


def _result_to_jsonable(result: EvaluationResult) -> dict:
    return {"metrics": result.metrics, "artifacts": result.artifacts}


def _result_from_jsonable(obj: dict) -> EvaluationResult:
    return EvaluationResult(
        metrics=obj.get("metrics", {}),
        artifacts=obj.get("artifacts", {}),
    )


def evaluate_stage2(program_path: str):
    try:
        cmd = [sys.executable, os.path.abspath(__file__), "--stage2-worker", program_path]
        env = os.environ.copy()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=CURRENT_DIR,
            check=False,
        )

        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="", flush=True)

        if proc.returncode != 0:
            return EvaluationResult(
                metrics={
                    "combined_score": -1e18,
                    "combine_score": -1e18,
                    "ood_acc": -1e18,
                    "final_test_acc": 0.0,
                    "final_test_loss": float("inf"),
                    "num_params": 0.0,
                    "macs": -1.0,
                    "time": 0.0,
                    "stage2_ok": 0.0,
                    "error": f"stage2 worker failed with return code {proc.returncode}",
                },
                artifacts={
                    "stage": "stage2",
                    "worker_stdout": proc.stdout,
                    "worker_stderr": proc.stderr,
                    "train_npz": TRAIN_NPZ,
                    "val_npz": VAL_NPZ,
                    "test_npz": TEST_NPZ,
                },
            )

        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError("stage2 worker produced no JSON output")
        payload = json.loads(lines[-1])
        return _result_from_jsonable(payload)

    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return EvaluationResult(
            metrics={
                "combined_score": -1e18,
                "combine_score": -1e18,
                "ood_acc": -1e18,
                "final_test_acc": 0.0,
                "final_test_loss": float("inf"),
                "num_params": 0.0,
                "macs": -1.0,
                "time": 0.0,
                "stage2_ok": 0.0,
                "error": str(e),
            },
            artifacts={
                "stage": "stage2",
                "traceback": tb,
                "train_npz": TRAIN_NPZ,
                "val_npz": VAL_NPZ,
                "test_npz": TEST_NPZ,
            },
        )


def evaluate(program_path: str):
    s1 = evaluate_stage1(program_path)
    ok = float(s1.metrics.get("stage1_ok", 0.0)) > 0.0
    if not ok:
        return s1

    s2 = evaluate_stage2(program_path)
    try:
        s2.artifacts["cascade_debug"] = {"stage1_metrics": dict(s1.metrics)}
    except Exception:
        pass
    return s2


def _main_stage2_worker(program_path: str):
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = _stage2_core(program_path)
        print(json.dumps(_result_to_jsonable(result)), flush=True)
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        err_result = EvaluationResult(
            metrics={
                "combined_score": -1e18,
                "combine_score": -1e18,
                "ood_acc": -1e18,
                "final_test_acc": 0.0,
                "final_test_loss": float("inf"),
                "num_params": 0.0,
                "macs": -1.0,
                "time": 0.0,
                "stage2_ok": 0.0,
                "error": str(e),
            },
            artifacts={
                "stage": "stage2_worker",
                "traceback": tb,
                "train_npz": TRAIN_NPZ,
                "val_npz": VAL_NPZ,
                "test_npz": TEST_NPZ,
            },
        )
        print(tb, file=sys.stderr, flush=True)
        print(json.dumps(_result_to_jsonable(err_result)), flush=True)
        return 0


def _public_view(result: EvaluationResult) -> Dict[str, Any]:
    metrics = dict(result.metrics)
    artifacts = dict(result.artifacts)
    stage1_ok = float(metrics.get("stage1_ok", 0.0)) > 0.0
    stage2_ok = float(metrics.get("stage2_ok", 0.0)) > 0.0
    return {
        "stage1_pass": stage1_ok,
        "stage1_reason": artifacts.get("status", "ok") if stage1_ok else metrics.get("error", artifacts.get("status", "failed")),
        "stage2_pass": stage2_ok,
        "combine_score": float(metrics.get("combine_score", metrics.get("combined_score", -1e18))),
        "combined_score": float(metrics.get("combined_score", metrics.get("combine_score", -1e18))),
        "macs": float(metrics.get("macs", -1.0)),
        "time": float(metrics.get("time", 0.0)),
        "num_params": float(metrics.get("num_params", 0.0)),
        "final_test_acc": float(metrics.get("final_test_acc", 0.0)),
        "final_test_loss": float(metrics.get("final_test_loss", float("inf"))),
        "ood_acc": float(metrics.get("ood_acc", 0.0)),
        "error": metrics.get("error", ""),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("program_path", nargs="?", default=os.path.join(CURRENT_DIR, "openevolve_initial_program.py"))
    parser.add_argument("--stage", type=str, default="all", choices=["stage1", "stage2", "all"])
    parser.add_argument("--stage2-worker", type=str, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.stage2_worker:
        raise SystemExit(_main_stage2_worker(args.stage2_worker))

    target = args.program_path
    if args.self_test:
        s1 = evaluate_stage1(target)
        print("=== STAGE1 ===")
        print(json.dumps(_public_view(s1), ensure_ascii=False, indent=2))
        s2 = evaluate_stage2(target)
        print("=== STAGE2 ===")
        print(json.dumps(_public_view(s2), ensure_ascii=False, indent=2))
    elif args.stage == "stage1":
        print(json.dumps(_public_view(evaluate_stage1(target)), ensure_ascii=False, indent=2))
    elif args.stage == "stage2":
        print(json.dumps(_public_view(evaluate_stage2(target)), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(_public_view(evaluate(target)), ensure_ascii=False, indent=2))
