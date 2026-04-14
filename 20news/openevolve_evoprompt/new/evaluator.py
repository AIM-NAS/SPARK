
import os
import sys
import time
import copy
import json
import random
import argparse
import traceback
import importlib.util
import subprocess
import contextlib
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Reduce thread-related issues and oversubscription, especially under process pools.
try:
    torch.set_num_threads(int(os.getenv("MNIST1D_TORCH_NUM_THREADS", "1")))
except Exception:
    pass
try:
    torch.set_num_interop_threads(int(os.getenv("MNIST1D_TORCH_NUM_INTEROP_THREADS", "1")))
except Exception:
    pass


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST1D_REPO_ROOT = os.path.join(CURRENT_DIR, "mnist1d-master")
if MNIST1D_REPO_ROOT not in sys.path:
    sys.path.insert(0, MNIST1D_REPO_ROOT)

from mnist1d.data import get_dataset_args, make_dataset
from mnist1d.utils import ObjectView

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


def get_model_args(as_dict=False):
    arg_dict = {
        "input_size": 40,
        "output_size": 10,
        "hidden_size": 256,
        "learning_rate": float(os.getenv("MNIST1D_LR", "1e-2")),
        "weight_decay": float(os.getenv("MNIST1D_WEIGHT_DECAY", "0")),
        "batch_size": int(os.getenv("MNIST1D_BATCH_SIZE", "100")),
        "total_steps": int(os.getenv("MNIST1D_TOTAL_STEPS", "8000")),
        "print_every": int(os.getenv("MNIST1D_PRINT_EVERY", "1000")),
        "eval_every": int(os.getenv("MNIST1D_EVAL_EVERY", "250")),
        "checkpoint_every": int(os.getenv("MNIST1D_CHECKPOINT_EVERY", "1000")),
        "device": "cpu",
        "seed": int(os.getenv("MNIST1D_SEED", "42")),
    }
    return arg_dict if as_dict else ObjectView(arg_dict)


def accuracy(model, inputs, targets):
    preds = model(inputs).argmax(-1).cpu().numpy()
    targets = targets.cpu().numpy().astype(np.float32)
    return 100 * float(np.mean(preds == targets))


def train_model(dataset, model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    x_train, x_test = torch.Tensor(dataset["x"]), torch.Tensor(dataset["x_test"])
    y_train, y_test = torch.LongTensor(dataset["y"]), torch.LongTensor(dataset["y_test"])

    model = model.to(args.device)
    x_train, x_test, y_train, y_test = [v.to(args.device) for v in [x_train, x_test, y_train, y_test]]

    results = {
        "checkpoints": [],
        "train_losses": [],
        "test_losses": [],
        "train_acc": [],
        "test_acc": [],
    }

    t0 = time.time()
    for step in range(args.total_steps + 1):
        bix = (step * args.batch_size) % len(x_train)
        x, y = x_train[bix:bix + args.batch_size], y_train[bix:bix + args.batch_size]

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        results["train_losses"].append(float(loss.item()))
        loss.backward()
        optimizer.step()

        if args.eval_every > 0 and step % args.eval_every == 0:
            with torch.no_grad():
                test_logits = model(x_test)
                test_loss = criterion(test_logits, y_test)
            results["test_losses"].append(float(test_loss.item()))
            results["train_acc"].append(float(accuracy(model, x_train, y_train)))
            results["test_acc"].append(float(accuracy(model, x_test, y_test)))

        if step > 0 and args.print_every > 0 and step % args.print_every == 0:
            t1 = time.time()
            print(
                "step {}, dt {:.2f}s, train_loss {:.3e}, test_loss {:.3e}, train_acc {:.1f}, test_acc {:.1f}".format(
                    step, t1 - t0, float(loss.item()), results["test_losses"][-1], results["train_acc"][-1], results["test_acc"][-1]
                ),
                flush=True,
            )
            t0 = t1

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            model.step = step
            results["checkpoints"].append(copy.deepcopy(model.state_dict()))

    return results


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_macs(model: nn.Module, input_size: int) -> float:
    model = copy.deepcopy(model).cpu().eval()
    macs = 0

    def conv1d_hook(module, inputs, outputs):
        nonlocal macs
        out = outputs
        if out.dim() != 3:
            return
        batch_size, out_channels, out_length = out.shape
        kernel_size = module.kernel_size[0]
        in_channels = module.in_channels
        groups = module.groups
        macs += batch_size * out_channels * out_length * kernel_size * (in_channels // groups)

    def linear_hook(module, inputs, outputs):
        nonlocal macs
        x = inputs[0]
        batch_size = x.shape[0]
        macs += batch_size * module.in_features * module.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv1d_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    with torch.no_grad():
        dummy_x = torch.randn(1, input_size)
        _ = model(dummy_x)

    for h in hooks:
        h.remove()

    return float(macs)


def load_program_module(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_candidate_model(module, args):
    if not hasattr(module, "build_model"):
        raise AttributeError("Candidate program must define build_model(input_size=40, output_size=10)")
    model = module.build_model(input_size=args.input_size, output_size=args.output_size)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("build_model(...) must return a torch.nn.Module")
    return model


def validate_candidate_model(module, args):
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


def build_dataset():
    dataset_args = get_dataset_args(as_dict=False)
    dataset_args.shuffle_seq = env_flag("MNIST1D_SHUFFLE", default=False)
    dataset = make_dataset(dataset_args)
    return dataset, dataset_args


def prepare_runtime_args():
    args = get_model_args(as_dict=False)
    requested_device = os.getenv("MNIST1D_DEVICE", "cpu")
    resolved_device = resolve_device(requested_device)
    args.device = resolved_device
    return args, requested_device, resolved_device


def evaluate_stage1(program_path: str):
    try:
        args, requested_device, resolved_device = prepare_runtime_args()
        module = load_program_module(program_path)
        model = validate_candidate_model(module, args)
        num_params = count_params(model)
        macs = estimate_macs(model, args.input_size)

        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "ood_acc": 0.0,
                "stage1_ok": 1.0,
                "num_params": float(num_params),
                "macs": float(macs),
            },
            artifacts={
                "stage": "stage1",
                "status": "interface_ok",
                "requested_device": requested_device,
                "resolved_device": resolved_device,
                "shuffle_seq": env_flag("MNIST1D_SHUFFLE", default=False),
            },
        )
    except Exception as e:
        return EvaluationResult(
            metrics={
                "combined_score": -1e18,
                "ood_acc": -1e18,
                "stage1_ok": 0.0,
                "num_params": 0.0,
                "macs": -1.0,
                "error": str(e),
            },
            artifacts={"stage": "stage1", "traceback": traceback.format_exc()},
        )


def _stage2_core(program_path: str) -> EvaluationResult:
    args, requested_device, resolved_device = prepare_runtime_args()
    set_seed(args.seed)

    dataset, dataset_args = build_dataset()
    module = load_program_module(program_path)
    validate_candidate_model(module, args)
    model = build_candidate_model(module, args)

    num_params = count_params(model)
    macs = estimate_macs(model, args.input_size)

    t0 = time.time()
    results = train_model(dataset, model, args)
    train_time_sec = time.time() - t0

    max_test_acc = max(results["test_acc"]) if results["test_acc"] else 0.0
    final_test_acc = results["test_acc"][-1] if results["test_acc"] else 0.0
    final_train_acc = results["train_acc"][-1] if results["train_acc"] else 0.0
    final_test_loss = results["test_losses"][-1] if results["test_losses"] else float("inf")
    last_train_loss = results["train_losses"][-1] if results["train_losses"] else float("inf")

    return EvaluationResult(
        metrics={
            "combined_score": float(max_test_acc) - 0.0001*float(num_params),
            "ood_acc": float(max_test_acc)- 0.0001*float(num_params),
            "max_test_acc": float(max_test_acc),
            "final_test_acc": float(final_test_acc),
            "final_train_acc": float(final_train_acc),
            "final_test_loss": float(final_test_loss),
            "last_train_loss": float(last_train_loss),
            "num_params": float(num_params),
            "macs": float(macs),
            "stage2_ok": 1.0,
        },
        artifacts={
            "stage": "stage2",
            "dataset_num_train": int(dataset["x"].shape[0]),
            "dataset_num_test": int(dataset["x_test"].shape[0]),
            "shuffle_seq": bool(dataset_args.shuffle_seq),
            "requested_device": str(requested_device),
            "resolved_device": str(resolved_device),
            "seed": int(args.seed),
            "train_time_sec": float(train_time_sec),
            "train_acc_curve": [float(v) for v in results["train_acc"]],
            "test_acc_curve": [float(v) for v in results["test_acc"]],
            "test_loss_curve": [float(v) for v in results["test_losses"]],
        },
    )


def _result_to_jsonable(result: EvaluationResult) -> dict:
    return {"metrics": result.metrics, "artifacts": result.artifacts}


def _result_from_jsonable(obj: dict) -> EvaluationResult:
    return EvaluationResult(metrics=obj.get("metrics", {}), artifacts=obj.get("artifacts", {}))


def evaluate_stage2(program_path: str):
    """
    Full evaluation is executed in a fresh Python subprocess.
    This avoids PyTorch autograd failures under fork-based multiprocessing used by OpenEvolve.
    """
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
                    "ood_acc": -1e18,
                    "max_test_acc": 0.0,
                    "final_test_acc": 0.0,
                    "final_train_acc": 0.0,
                    "final_test_loss": float("inf"),
                    "last_train_loss": float("inf"),
                    "num_params": 0.0,
                    "macs": -1.0,
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
                "combined_score": -1e18,
                "ood_acc": -1e18,
                "max_test_acc": 0.0,
                "final_test_acc": 0.0,
                "final_train_acc": 0.0,
                "final_test_loss": float("inf"),
                "last_train_loss": float("inf"),
                "num_params": 0.0,
                "macs": -1.0,
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
                "combined_score": -1e18,
                "ood_acc": -1e18,
                "max_test_acc": 0.0,
                "final_test_acc": 0.0,
                "final_train_acc": 0.0,
                "final_test_loss": float("inf"),
                "last_train_loss": float("inf"),
                "num_params": 0.0,
                "macs": -1.0,
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
