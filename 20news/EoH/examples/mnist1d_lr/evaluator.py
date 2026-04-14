import os
import sys
import time
import copy
import random
import argparse
import importlib.util
import traceback
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==============================
# Runtime / thread setup
# ==============================
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


@dataclass
class EvaluationResult:
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str, gpu_id: int = 0) -> str:
    device = str(device).strip().lower()
    if device == "auto":
        if torch.cuda.is_available():
            return f"cuda:{gpu_id}"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "gpu":
        return f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda:"):
        return device if torch.cuda.is_available() else "cpu"
    if device == "mps":
        return "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    return "cpu"


def get_eval_args(as_dict=False):
    arg_dict = {
        "input_size": 40,
        "output_size": 10,
        "lr": float(os.getenv("MNIST1D_LR", "1e-2")),
        "weight_decay": float(os.getenv("MNIST1D_WEIGHT_DECAY", "0.0")),
        "batch_size": int(os.getenv("MNIST1D_BATCH_SIZE", "100")),
        "total_steps": int(os.getenv("MNIST1D_TOTAL_STEPS", "8000")),
        "print_every": int(os.getenv("MNIST1D_PRINT_EVERY", "1000")),
        "eval_every": int(os.getenv("MNIST1D_EVAL_EVERY", "250")),
        "checkpoint_every": int(os.getenv("MNIST1D_CHECKPOINT_EVERY", "1000")),
        "requested_device": os.getenv("MNIST1D_DEVICE", "cpu"),
        "gpu_id": int(os.getenv("MNIST1D_GPU_ID", "0")),
        "seed": int(os.getenv("MNIST1D_SEED", "42")),
        "shuffle": bool(int(os.getenv("MNIST1D_SHUFFLE", "0"))),
    }
    arg_dict["device"] = resolve_device(arg_dict["requested_device"], arg_dict["gpu_id"])
    return arg_dict if as_dict else ObjectView(arg_dict)


def load_program_module(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate program from {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_macs(model: nn.Module, input_size: int) -> float:
    model = copy.deepcopy(model).cpu().eval()
    macs = 0

    def conv1d_hook(module, inputs, outputs):
        nonlocal macs
        out = outputs
        if not isinstance(out, torch.Tensor) or out.dim() != 3:
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


def accuracy(model, inputs, targets):
    preds = model(inputs).argmax(-1).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy().astype(np.int64)
    return 100.0 * float(np.mean(preds == targets))


def validate_candidate_model(module, input_size=40, output_size=10):
    if not hasattr(module, "build_model"):
        raise AttributeError("Candidate program does not define build_model(input_size, output_size).")

    model = module.build_model(input_size=input_size, output_size=output_size)

    if not isinstance(model, nn.Module):
        raise TypeError("build_model(...) must return a torch.nn.Module.")

    model.eval()
    with torch.no_grad():
        x = torch.randn(4, input_size)
        try:
            y = model(x)
        except TypeError:
            # Allow candidates whose forward signature is forward(self, x) only
            y = model.forward(x)

    if not isinstance(y, torch.Tensor):
        raise TypeError("Model forward must return a torch.Tensor.")
    if tuple(y.shape) != (4, output_size):
        raise ValueError(
            f"Model output shape must be (4, {output_size}), but got {tuple(y.shape)}"
        )

    return model


def build_dataset(shuffle: bool):
    dataset_args = get_dataset_args(as_dict=False)
    dataset_args.shuffle_seq = bool(shuffle)
    dataset = make_dataset(dataset_args)
    return dataset, dataset_args


def train_model(dataset, model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    x_train = torch.tensor(dataset["x"], dtype=torch.float32)
    x_test = torch.tensor(dataset["x_test"], dtype=torch.float32)
    y_train = torch.tensor(dataset["y"], dtype=torch.long)
    y_test = torch.tensor(dataset["y_test"], dtype=torch.long)

    model = model.to(args.device)
    x_train = x_train.to(args.device)
    x_test = x_test.to(args.device)
    y_train = y_train.to(args.device)
    y_test = y_test.to(args.device)

    results = {
        "train_losses": [],
        "test_losses": [],
        "train_acc": [],
        "test_acc": [],
    }

    train_start_time = time.time()
    last_print_time = train_start_time

    for step in range(args.total_steps + 1):
        step_start_time = time.time()

        bix = (step * args.batch_size) % len(x_train)
        x = x_train[bix:bix + args.batch_size]
        y = y_train[bix:bix + args.batch_size]

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        results["train_losses"].append(float(loss.item()))
        loss.backward()
        optimizer.step()

        step_time_sec = time.time() - step_start_time
        elapsed_time_sec = time.time() - train_start_time

        if args.eval_every > 0 and step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(x_test)
                test_loss = criterion(test_logits, y_test)
                train_acc = accuracy(model, x_train, y_train)
                test_acc = accuracy(model, x_test, y_test)
            model.train()
            results["test_losses"].append(float(test_loss.item()))
            results["train_acc"].append(float(train_acc))
            results["test_acc"].append(float(test_acc))

        if step > 0 and args.print_every > 0 and step % args.print_every == 0:
            dt = time.time() - last_print_time
            print(
                f"step {step}, elapsed {elapsed_time_sec:.2f}s, step_time {step_time_sec:.4f}s, "
                f"dt {dt:.2f}s, train_loss {loss.item():.4e}, "
                f"test_loss {results['test_losses'][-1]:.4e}, train_acc {results['train_acc'][-1]:.2f}, "
                f"test_acc {results['test_acc'][-1]:.2f}",
                flush=True,
            )
            last_print_time = time.time()

    return results


def evaluate_stage1(program_path: str) -> EvaluationResult:
    print("evaluate", flush=True)
    print("in stage1", flush=True)
    try:
        args = get_eval_args(as_dict=False)
        module = load_program_module(program_path)
        model = validate_candidate_model(
            module,
            input_size=args.input_size,
            output_size=args.output_size,
        )

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
                "program_path": str(program_path),
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
            artifacts={
                "stage": "stage1",
                "program_path": str(program_path),
                "traceback": traceback.format_exc(),
            },
        )


def evaluate_stage2(program_path: str) -> EvaluationResult:
    args = get_eval_args(as_dict=False)
    set_seed(args.seed)

    module = load_program_module(program_path)
    model = module.build_model(input_size=args.input_size, output_size=args.output_size)
    validate_candidate_model(module, input_size=args.input_size, output_size=args.output_size)

    dataset, dataset_args = build_dataset(args.shuffle)
    num_params = count_params(model)
    macs = estimate_macs(model, args.input_size)

    print("=" * 80)
    print("MNIST-1D candidate evaluation")
    print("=" * 80)
    print(f"shuffle_seq      : {dataset_args.shuffle_seq}")
    print(f"requested_device : {args.requested_device}")
    print(f"resolved_device  : {args.device}")
    print(f"seed             : {args.seed}")
    print(f"num_params       : {num_params}")
    print(f"macs             : {macs}")
    print(f"train/test       : {dataset['x'].shape[0]}/{dataset['x_test'].shape[0]}")
    print("=" * 80)

    t0 = time.time()
    results = train_model(dataset, model, args)
    train_time_sec = time.time() - t0

    max_test_acc = max(results["test_acc"]) if results["test_acc"] else 0.0
    final_test_acc = results["test_acc"][-1] if results["test_acc"] else 0.0
    final_train_acc = results["train_acc"][-1] if results["train_acc"] else 0.0
    final_test_loss = results["test_losses"][-1] if results["test_losses"] else float("inf")
    last_train_loss = results["train_losses"][-1] if results["train_losses"] else float("inf")
    combined_score = float(max_test_acc) - 0.0001 * float(num_params)

    result = EvaluationResult(
        metrics={
            "combined_score": combined_score,
            "max_test_acc": float(max_test_acc),
            "final_test_acc": float(final_test_acc),
            "final_train_acc": float(final_train_acc),
            "final_test_loss": float(final_test_loss),
            "last_train_loss": float(last_train_loss),
            "num_params": float(num_params),
            "macs": float(macs),
            "stage1_ok": 1.0,
        },
        artifacts={
            "stage": "stage2",
            "program_path": str(program_path),
            "shuffle_seq": bool(dataset_args.shuffle_seq),
            "requested_device": str(args.requested_device),
            "resolved_device": str(args.device),
            "seed": int(args.seed),
            "train_time_sec": float(train_time_sec),
            "train_acc_curve": [float(v) for v in results["train_acc"]],
            "test_acc_curve": [float(v) for v in results["test_acc"]],
            "test_loss_curve": [float(v) for v in results["test_losses"]],
        },
    )

    print("\nFinal metrics:")
    for k, v in result.metrics.items():
        print(f"{k}: {v}")

    return result


def evaluate(program_path: str) -> EvaluationResult:
    s1 = evaluate_stage1(program_path)
    ok = float(s1.metrics.get("stage1_ok", 0.0)) > 0.0
    if not ok:
        return s1
    return evaluate_stage2(program_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a candidate template_program.py on MNIST-1D")
    parser.add_argument("--program-path", type=str, required=True, help="Path to candidate program")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate(args.program_path)
    print(result)