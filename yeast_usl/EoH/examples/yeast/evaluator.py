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
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


try:
    torch.set_num_threads(int(os.getenv("YEAST_TORCH_NUM_THREADS", "1")))
except Exception:
    pass
try:
    torch.set_num_interop_threads(int(os.getenv("YEAST_TORCH_NUM_INTEROP_THREADS", "1")))
except Exception:
    pass


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_NPZ = os.getenv("YEAST_TRAIN_NPZ", "/data1/lz/clrs/openevolve/yeast_usl/yeast_prepared/yeast_train.npz")
VAL_NPZ = os.getenv("YEAST_VAL_NPZ", "/data1/lz/clrs/openevolve/yeast_usl/yeast_prepared/yeast_val.npz")
TEST_NPZ = os.getenv("YEAST_TEST_NPZ", "/data1/lz/clrs/openevolve/yeast_usl/yeast_prepared/yeast_test.npz")
LABELS_TXT = os.getenv("YEAST_LABELS_TXT", "/data1/lz/clrs/openevolve/yeast_usl/yeast_prepared/label_names.txt")


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
    if requested in {"gpu", "cuda"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        return requested if torch.cuda.is_available() else "cpu"
    if requested == "mps":
        return "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _infer_num_classes() -> int:
    if os.path.exists(LABELS_TXT):
        with open(LABELS_TXT, "r", encoding="utf-8") as f:
            labels = [ln.strip() for ln in f if ln.strip()]
        if labels:
            return len(labels)
    ys = []
    for path in (TRAIN_NPZ, VAL_NPZ, TEST_NPZ):
        if os.path.exists(path):
            data = np.load(path)
            ys.append(np.asarray(data["y"], dtype=np.int64))
    if not ys:
        return 10
    return int(np.max(np.concatenate(ys, axis=0))) + 1


def get_model_args(as_dict=False):
    arg_dict = {
        "input_size": int(os.getenv("YEAST_INPUT_SIZE", "8")),
        "output_size": int(os.getenv("YEAST_NUM_CLASSES", str(_infer_num_classes()))),
        "hidden_size": int(os.getenv("YEAST_HIDDEN_SIZE", "32")),
        "dropout": float(os.getenv("YEAST_DROPOUT", "0.1")),
        "learning_rate": float(os.getenv("YEAST_LR", "1e-3")),
        "weight_decay": float(os.getenv("YEAST_WEIGHT_DECAY", "1e-4")),
        "batch_size": int(os.getenv("YEAST_BATCH_SIZE", "64")),
        "epochs": int(os.getenv("YEAST_EPOCHS", "30")),
        "device": "cpu",
        "seed": int(os.getenv("YEAST_SEED", "42")),
        "stage2_train_limit": int(os.getenv("YEAST_STAGE2_TRAIN_LIMIT", "0")),
        "stage2_val_limit": int(os.getenv("YEAST_STAGE2_VAL_LIMIT", "0")),
        "stage2_test_limit": int(os.getenv("YEAST_STAGE2_TEST_LIMIT", "0")),
    }
    return arg_dict if as_dict else type("ObjectView", (object,), arg_dict)()


def prepare_runtime_args():
    args = get_model_args(as_dict=False)
    requested_device = os.getenv("YEAST_DEVICE", "cpu")
    resolved_device = resolve_device(requested_device)
    args.device = resolved_device
    return args, requested_device, resolved_device


def load_program_module(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def static_check_candidate(program_path: str):
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
    required_params = ["self", "input_dim", "num_classes", "hidden_dim", "dropout"]
    for p in required_params:
        if p not in init_params:
            raise ValueError(f"SimpleMLP.__init__ must contain parameter `{p}`, got {init_params}")

    forward_fn = getattr(cls, "forward", None)
    if forward_fn is None:
        raise AttributeError("SimpleMLP must define forward")
    if not callable(forward_fn):
        raise TypeError("SimpleMLP.forward must be callable")

    forward_sig = inspect.signature(forward_fn)
    forward_params = list(forward_sig.parameters.keys())
    if len(forward_params) < 2:
        raise ValueError(f"SimpleMLP.forward must accept at least (self, x), got {forward_params}")

    return {
        "class_name": cls.__name__,
        "init_params": init_params,
        "forward_params": forward_params,
    }


def build_candidate_model(module, args):
    if not hasattr(module, "SimpleMLP"):
        raise AttributeError("Candidate program must define class SimpleMLP")
    model = module.SimpleMLP(
        input_dim=args.input_size,
        num_classes=args.output_size,
        hidden_dim=args.hidden_size,
        dropout=args.dropout,
    )
    if not isinstance(model, torch.nn.Module):
        raise TypeError("SimpleMLP(...) must return a torch.nn.Module instance")
    return model


def validate_candidate_model_runtime(module, args):
    model = build_candidate_model(module, args)
    model.eval()
    with torch.no_grad():
        dummy_x = torch.randn(4, args.input_size)
        dummy_y = model(dummy_x)
    if not isinstance(dummy_y, torch.Tensor):
        raise TypeError("Model forward must return a torch.Tensor")
    if tuple(dummy_y.shape) != (4, args.output_size):
        raise ValueError(f"Model output shape must be (4, {args.output_size}), got {tuple(dummy_y.shape)}")
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_macs(model: nn.Module, input_size: int) -> float:
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
        dummy_x = torch.randn(1, input_size)
        _ = model(dummy_x)

    for h in hooks:
        h.remove()

    return float(macs)


def _load_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    data = np.load(path)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    return X, y


def maybe_limit_split(X, y, limit: int):
    if limit is None or int(limit) <= 0 or len(y) <= int(limit):
        return X, y
    limit = int(limit)
    return X[:limit], y[:limit]


def build_dataloader(x, y, batch_size: int, shuffle: bool):
    x_t = torch.from_numpy(x.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.int64))
    dataset = torch.utils.data.TensorDataset(x_t, y_t)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


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
    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return 100.0 * float(np.mean(preds_all == labels_all))


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


def train_model(dataset, model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader = build_dataloader(dataset["x_train"], dataset["y_train"], args.batch_size, shuffle=True)
    val_loader = build_dataloader(dataset["x_val"], dataset["y_val"], max(args.batch_size, 256), shuffle=False)
    test_loader = build_dataloader(dataset["x_test"], dataset["y_test"], max(args.batch_size, 256), shuffle=False)
    train_eval_loader = build_dataloader(dataset["x_train"], dataset["y_train"], max(args.batch_size, 256), shuffle=False)

    model = model.to(args.device)
    results = {"train_losses": [], "val_losses": [], "test_losses": [], "train_acc": [], "val_acc": [], "test_acc": []}

    for _ in range(args.epochs):
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

        results["train_losses"].append(float(epoch_loss_sum / max(epoch_num, 1)))
        results["val_losses"].append(float(evaluate_loss(model, val_loader, criterion, args.device)))
        results["test_losses"].append(float(evaluate_loss(model, test_loader, criterion, args.device)))
        results["train_acc"].append(float(accuracy(model, train_eval_loader, args.device)))
        results["val_acc"].append(float(accuracy(model, val_loader, args.device)))
        results["test_acc"].append(float(accuracy(model, test_loader, args.device)))

    return results


def build_dataset(args):
    X_train, y_train = _load_npz(TRAIN_NPZ)
    X_val, y_val = _load_npz(VAL_NPZ)
    X_test, y_test = _load_npz(TEST_NPZ)

    X_train, y_train = maybe_limit_split(X_train, y_train, args.stage2_train_limit)
    X_val, y_val = maybe_limit_split(X_val, y_val, args.stage2_val_limit)
    X_test, y_test = maybe_limit_split(X_test, y_test, args.stage2_test_limit)

    return {
        "x_train": X_train,
        "y_train": y_train,
        "x_val": X_val,
        "y_val": y_val,
        "x_test": X_test,
        "y_test": y_test,
    }


def evaluate_stage1(program_path: str):
    try:
        args, requested_device, resolved_device = prepare_runtime_args()
        info = static_check_candidate(program_path)
        return EvaluationResult(
            metrics={
                "combine_score": 0.0,
                "combined_score": 0.0,
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
                "input_size": int(args.input_size),
                "output_size": int(args.output_size),
                "hidden_size": int(args.hidden_size),
                "dropout": float(args.dropout),
                "train_npz": TRAIN_NPZ,
                "val_npz": VAL_NPZ,
                "test_npz": TEST_NPZ,
                "class_name": info["class_name"],
                "init_params": info["init_params"],
                "forward_params": info["forward_params"],
            },
        )
    except Exception as e:
        return EvaluationResult(
            metrics={
                "combine_score": -1e18,
                "combined_score": -1e18,
                "ood_acc": -1e18,
                "final_test_acc": 0.0,
                "final_test_loss": float("inf"),
                "num_params": 0.0,
                "macs": -1.0,
                "time": 0.0,
                "stage1_ok": 0.0,
                "error": str(e),
            },
            artifacts={"stage": "stage1", "status": "static_check_failed", "traceback": traceback.format_exc()},
        )


def _stage2_core(program_path: str) -> EvaluationResult:
    args, requested_device, resolved_device = prepare_runtime_args()
    set_seed(args.seed)

    dataset = build_dataset(args)
    module = load_program_module(program_path)
    validate_candidate_model_runtime(module, args)
    model = build_candidate_model(module, args)

    num_params = count_params(model)
    macs = estimate_macs(model, args.input_size)

    t0 = time.time()
    results = train_model(dataset=dataset, model=model, args=args)
    train_time_sec = time.time() - t0

    max_val_acc = max(results["val_acc"]) if results["val_acc"] else 0.0
    final_val_acc = results["val_acc"][-1] if results["val_acc"] else 0.0
    final_test_acc = results["test_acc"][-1] if results["test_acc"] else 0.0
    final_test_loss = results["test_losses"][-1] if results["test_losses"] else float("inf")
    combined_score = float(max_val_acc) - 0.0001 * float(num_params)

    return EvaluationResult(
        metrics={
            "combine_score": combined_score,
            "combined_score": combined_score,
            "ood_acc": float(max_val_acc),
            "final_test_acc": float(final_test_acc),
            "final_test_loss": float(final_test_loss),
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
            "train_acc_curve": [float(v) for v in results["train_acc"]],
            "val_acc_curve": [float(v) for v in results["val_acc"]],
            "test_acc_curve": [float(v) for v in results["test_acc"]],
            "val_loss_curve": [float(v) for v in results["val_losses"]],
            "test_loss_curve": [float(v) for v in results["test_losses"]],
            "train_loss_curve": [float(v) for v in results["train_losses"]],
            "final_val_acc": float(final_val_acc),
        },
    )


def _result_to_jsonable(result: EvaluationResult) -> dict:
    return {"metrics": result.metrics, "artifacts": result.artifacts}


def _result_from_jsonable(obj: dict) -> EvaluationResult:
    return EvaluationResult(metrics=obj.get("metrics", {}), artifacts=obj.get("artifacts", {}))


def evaluate_stage2(program_path: str):
    try:
        cmd = [sys.executable, os.path.abspath(__file__), "--stage2-worker", program_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy(), cwd=CURRENT_DIR, check=False)

        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="", flush=True)

        if proc.returncode != 0:
            return EvaluationResult(
                metrics={
                    "combine_score": -1e18,
                    "combined_score": -1e18,
                    "ood_acc": -1e18,
                    "final_test_acc": 0.0,
                    "final_test_loss": float("inf"),
                    "num_params": 0.0,
                    "macs": -1.0,
                    "time": 0.0,
                    "stage2_ok": 0.0,
                    "error": f"stage2 worker failed with return code {proc.returncode}",
                },
                artifacts={"stage": "stage2", "worker_stdout": proc.stdout, "worker_stderr": proc.stderr},
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
                "combine_score": -1e18,
                "combined_score": -1e18,
                "ood_acc": -1e18,
                "final_test_acc": 0.0,
                "final_test_loss": float("inf"),
                "num_params": 0.0,
                "macs": -1.0,
                "time": 0.0,
                "stage2_ok": 0.0,
                "error": str(e),
            },
            artifacts={"stage": "stage2", "traceback": tb},
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
                "combine_score": -1e18,
                "combined_score": -1e18,
                "ood_acc": -1e18,
                "final_test_acc": 0.0,
                "final_test_loss": float("inf"),
                "num_params": 0.0,
                "macs": -1.0,
                "time": 0.0,
                "stage2_ok": 0.0,
                "error": str(e),
            },
            artifacts={"stage": "stage2_worker", "traceback": tb},
        )
        print(tb, file=sys.stderr, flush=True)
        print(json.dumps(_result_to_jsonable(err_result)), flush=True)
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-worker", type=str, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.stage2_worker:
        raise SystemExit(_main_stage2_worker(args.stage2_worker))

    target = os.path.join(CURRENT_DIR, "initial_program.py")
    if args.self_test:
        s1 = evaluate_stage1(target)
        print("=== STAGE1 ===")
        print(s1.metrics)
        s2 = evaluate_stage2(target)
        print("=== STAGE2 ===")
        print(s2.metrics)
    else:
        out = evaluate(target)
        print("=== METRICS ===")
        for k, v in out.metrics.items():
            print(f"{k}: {v}")
