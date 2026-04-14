import os
import sys
import time
import copy
import json
import random
import argparse
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==============================
# 你只需要改这里的 ConvBase
# ==============================
class ConvBase(nn.Module):
    def __init__(self):
        super(ConvBase, self).__init__()

        hidden_size = 6
        bidirectional = True
        input_size = 40
        output_size = 10
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        num_dirs = 2 if bidirectional else 1

        self.linear = nn.Linear(input_size * hidden_size * num_dirs, output_size)

    def forward(self, x, h0=None):
        x = x.unsqueeze(-1)

        num_dirs = 2 if self.bidirectional else 1
        if h0 is None:
            h0 = torch.zeros(num_dirs, x.shape[0], self.hidden_size, device=x.device)

        output, _ = self.gru(x, h0)
        output = output.reshape(output.shape[0], -1)
        return self.linear(output)

# ===== END EVOLVE PGN REGION =====

def build_model(input_size=40, output_size=10, hidden_size=128, dropout_rate=0.1):
    return ConvBase()


# ==============================
# 下面逻辑严格对齐 evaluator.py
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
        step_t0 = time.time()

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
            step_time = t1 - step_t0
            print(
                "step {}, dt {:.2f}s, step_time {:.4f}s, train_loss {:.3e}, test_loss {:.3e}, train_acc {:.1f}, test_acc {:.1f}".format(
                    step, t1 - t0, step_time, float(loss.item()), results["test_losses"][-1], results["train_acc"][-1], results["test_acc"][-1]
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


def validate_candidate_model(args):
    model = build_model(input_size=args.input_size, output_size=args.output_size)
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


def run_strict_eval():
    args, requested_device, resolved_device = prepare_runtime_args()
    set_seed(args.seed)

    dataset, dataset_args = build_dataset()
    validate_candidate_model(args)
    model = build_model(input_size=args.input_size, output_size=args.output_size)

    num_params = count_params(model)
    macs = estimate_macs(model, args.input_size)

    print("=" * 80)
    print("MNIST-1D strict evaluator-aligned test")
    print("=" * 80)
    print(f"shuffle_seq      : {bool(dataset_args.shuffle_seq)}")
    print(f"requested_device : {requested_device}")
    print(f"resolved_device  : {resolved_device}")
    print(f"seed             : {args.seed}")
    print(f"learning_rate    : {args.learning_rate}")
    print(f"weight_decay     : {args.weight_decay}")
    print(f"batch_size       : {args.batch_size}")
    print(f"total_steps      : {args.total_steps}")
    print(f"print_every      : {args.print_every}")
    print(f"eval_every       : {args.eval_every}")
    print(f"checkpoint_every : {args.checkpoint_every}")
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

    result = EvaluationResult(
        metrics={
            "combined_score": float(max_test_acc) - 0.0001 * float(num_params),
            "ood_acc": float(max_test_acc) - 0.0001 * float(num_params),
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

    print("\n最终结果:")
    for k, v in result.metrics.items():
        print(f"{k}: {v}")

    return result


def apply_cli_overrides(cli_args):
    # 为了尽量严格对齐 evaluator，这里只是把命令行参数写回同名环境变量，
    # 后续仍然走 evaluator 原本的 get_model_args / prepare_runtime_args / build_dataset 流程。
    if cli_args.shuffle:
        os.environ["MNIST1D_SHUFFLE"] = "1"
    elif cli_args.no_shuffle:
        os.environ["MNIST1D_SHUFFLE"] = "0"

    if cli_args.device is not None:
        device = cli_args.device.strip().lower()
        if device == "gpu":
            device = "cuda"
        if device == "cpu":
            os.environ["MNIST1D_DEVICE"] = "cpu"
        elif device == "auto":
            os.environ["MNIST1D_DEVICE"] = "auto"
        elif device == "mps":
            os.environ["MNIST1D_DEVICE"] = "mps"
        elif device == "cuda":
            os.environ["MNIST1D_DEVICE"] = f"cuda:{cli_args.gpu_id}"
        elif device.startswith("cuda:"):
            os.environ["MNIST1D_DEVICE"] = device
        else:
            raise ValueError(f"Unsupported --device: {cli_args.device}")

    if cli_args.seed is not None:
        os.environ["MNIST1D_SEED"] = str(cli_args.seed)
    if cli_args.lr is not None:
        os.environ["MNIST1D_LR"] = str(cli_args.lr)
    if cli_args.weight_decay is not None:
        os.environ["MNIST1D_WEIGHT_DECAY"] = str(cli_args.weight_decay)
    if cli_args.batch_size is not None:
        os.environ["MNIST1D_BATCH_SIZE"] = str(cli_args.batch_size)
    if cli_args.total_steps is not None:
        os.environ["MNIST1D_TOTAL_STEPS"] = str(cli_args.total_steps)
    if cli_args.print_every is not None:
        os.environ["MNIST1D_PRINT_EVERY"] = str(cli_args.print_every)
    if cli_args.eval_every is not None:
        os.environ["MNIST1D_EVAL_EVERY"] = str(cli_args.eval_every)
    if cli_args.checkpoint_every is not None:
        os.environ["MNIST1D_CHECKPOINT_EVERY"] = str(cli_args.checkpoint_every)


def parse_args():
    parser = argparse.ArgumentParser(description="严格对齐 evaluator.py 的 MNIST-1D 测试脚本")
    parser.add_argument("--shuffle", action="store_true", help="设置 MNIST1D_SHUFFLE=1")
    parser.add_argument("--no-shuffle", action="store_true", help="设置 MNIST1D_SHUFFLE=0")
    parser.add_argument("--device", type=str, default=None, help="cpu / gpu / cuda / cuda:0 / auto / mps")
    parser.add_argument("--gpu-id", type=int, default=0, help="当 --device 为 gpu/cuda 时使用的 GPU 编号")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--save-json", type=str, default=None, help="把最终结果保存到 json 文件")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    apply_cli_overrides(cli_args)
    result = run_strict_eval()

    if cli_args.save_json:
        payload = {"metrics": result.metrics, "artifacts": result.artifacts}
        with open(cli_args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {cli_args.save_json}")
