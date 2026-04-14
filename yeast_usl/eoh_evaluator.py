# This evaluator intentionally avoids sklearn and uses only numpy + torch.
import argparse
import ast
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ===== Fixed prepared Yeast paths =====
TRAIN_NPZ = Path(r"/data1/lz/clrs/openevolve/yeast_usl/yeast_prepared/yeast_train.npz")
VAL_NPZ = Path(r"/data1/lz/clrs/openevolve/yeast_usl/yeast_prepared/yeast_val.npz")
TEST_NPZ = Path(r"/data1/lz/clrs/openevolve/yeast_usl/yeast_prepared/yeast_test.npz")
LABEL_NAMES_TXT = Path(r"/data1/lz/clrs/openevolve/yeast_usl/yeast_prepared/label_names.txt")

DEVICE = torch.device("cpu")
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEEDS = [0, 1]
MAX_PARAMS = 1_000_000

FORBIDDEN_IMPORT_ROOTS = {
    "os", "subprocess", "socket", "requests", "urllib", "http", "ftplib",
    "shutil", "pathlib", "glob", "tempfile", "pickle", "joblib",
    "multiprocessing", "threading", "ctypes", "inspect"
}
FORBIDDEN_CALL_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__"
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_npz_split(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(path)
    return arr["X"].astype(np.float32), arr["y"].astype(np.int64)


def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def load_data() -> Dict[str, Any]:
    missing = [str(p) for p in [TRAIN_NPZ, VAL_NPZ, TEST_NPZ, LABEL_NAMES_TXT] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prepared Yeast files: {missing}")

    x_train, y_train = load_npz_split(TRAIN_NPZ)
    x_val, y_val = load_npz_split(VAL_NPZ)
    x_test, y_test = load_npz_split(TEST_NPZ)

    label_names = [line.strip() for line in LABEL_NAMES_TXT.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "train_loader": make_loader(x_train, y_train, shuffle=True),
        "val_loader": make_loader(x_val, y_val, shuffle=False),
        "test_loader": make_loader(x_test, y_test, shuffle=False),
        "input_dim": int(x_train.shape[1]),
        "num_classes": int(len(np.unique(np.concatenate([y_train, y_val, y_test])))),
        "label_names": label_names,
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
    }


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def stage1_rule_check(program_path: Path, required_symbol: str) -> Dict[str, Any]:
    if not program_path.exists():
        return {"stage1_pass": False, "stage1_reason": f"File not found: {program_path}"}

    try:
        source = program_path.read_text(encoding="utf-8")
    except Exception as e:
        return {"stage1_pass": False, "stage1_reason": f"Cannot read file: {e}"}

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"stage1_pass": False, "stage1_reason": f"SyntaxError: {e}"}

    has_required = False
    errors: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == required_symbol:
            has_required = True

        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split('.')[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    errors.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                root = node.module.split('.')[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    errors.append(f"forbidden import-from: {node.module}")
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            if name in FORBIDDEN_CALL_NAMES:
                errors.append(f"forbidden call: {name}")

    if not has_required:
        errors.append(f"required symbol not found: {required_symbol}")

    if errors:
        return {
            "stage1_pass": False,
            "stage1_reason": "; ".join(errors[:10]),
        }

    return {"stage1_pass": True, "stage1_reason": "ok"}


def import_module_from_path(program_path: Path):
    spec = importlib.util.spec_from_file_location("candidate_module", str(program_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_linear_macs(model: nn.Module, input_dim: int) -> int:
    macs = 0
    hooks = []

    def linear_hook(module, inputs, outputs):
        nonlocal macs
        x = inputs[0]
        batch = int(x.shape[0])
        macs += batch * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, input_dim)
        _ = model(dummy)

    for h in hooks:
        h.remove()
    return int(macs)


def evaluate_loss_and_acc(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * int(xb.size(0))
            total_count += int(xb.size(0))
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    acc = float((preds == targets).mean()) if len(targets) > 0 else 0.0
    avg_loss = float(total_loss / max(total_count, 1))
    return avg_loss, acc


def train_one_seed(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, seed: int) -> Dict[str, float]:
    set_seed(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for _epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        _val_loss, val_acc = evaluate_loss_and_acc(model, val_loader, criterion)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final_test_loss, final_test_acc = evaluate_loss_and_acc(model, test_loader, criterion)
    return {
        "final_test_loss": float(final_test_loss),
        "final_test_acc": float(final_test_acc),
        # In this Yeast setup, held-out test accuracy is used as the OOD/generalization proxy.
        "ood_acc": float(final_test_acc),
    }


def build_result(stage1_pass: bool, stage1_reason: str, **kwargs) -> Dict[str, Any]:
    out = {
        "stage1_pass": bool(stage1_pass),
        "stage1_reason": stage1_reason,
        "stage2_pass": False,
        "combine_score": -1e9,
        "macs": -1,
        "time": 0.0,
        "num_params": -1,
        "final_test_acc": 0.0,
        "final_test_loss": 1e9,
        "ood_acc": 0.0,
    }
    out.update(kwargs)
    return out



def construct_model(module, input_dim: int, num_classes: int) -> nn.Module:
    if not hasattr(module, "build_model"):
        raise AttributeError("Candidate must define function build_model")
    model = module.build_model(input_dim=input_dim, num_classes=num_classes)
    if not isinstance(model, nn.Module):
        raise TypeError("build_model(...) must return an nn.Module instance")
    return model


def stage2_evaluate(program_path: Path) -> Dict[str, Any]:
    t0 = time.time()
    data = load_data()
    module = import_module_from_path(program_path)
    model = construct_model(module, data["input_dim"], data["num_classes"])

    with torch.no_grad():
        dummy = torch.randn(4, data["input_dim"])
        out = model(dummy)
    if tuple(out.shape) != (4, data["num_classes"]):
        raise ValueError(f"Bad output shape: got {tuple(out.shape)}, expected {(4, data['num_classes'])}")

    num_params = count_parameters(model)
    if num_params <= 0 or num_params > MAX_PARAMS:
        raise ValueError(f"Invalid num_params={num_params}, allowed range: (0, {MAX_PARAMS}]")
    macs = estimate_linear_macs(model, data["input_dim"])

    seed_metrics = []
    for seed in SEEDS:
        set_seed(seed)
        model = construct_model(module, data["input_dim"], data["num_classes"])
        metrics = train_one_seed(model, data["train_loader"], data["val_loader"], data["test_loader"], seed)
        seed_metrics.append(metrics)

    final_test_acc = float(np.mean([m["final_test_acc"] for m in seed_metrics]))
    final_test_loss = float(np.mean([m["final_test_loss"] for m in seed_metrics]))
    ood_acc = float(np.mean([m["ood_acc"] for m in seed_metrics]))
    elapsed = float(time.time() - t0)

    combine_score = (
        100.0 * final_test_acc
        - 0.1 * final_test_loss
        - 1e-6 * float(macs)
        - 1e-6 * float(num_params)
        - 1e-3 * elapsed
    )

    return build_result(
        True,
        "ok",
        stage2_pass=True,
        combine_score=float(combine_score),
        macs=int(macs),
        time=elapsed,
        num_params=int(num_params),
        final_test_acc=final_test_acc,
        final_test_loss=final_test_loss,
        ood_acc=ood_acc,
    )


def evaluate(program_path: str, stage: str = "all") -> Dict[str, Any]:
    program_path = Path(program_path)
    s1 = stage1_rule_check(program_path, required_symbol="build_model")

    if stage == "stage1":
        return build_result(s1["stage1_pass"], s1["stage1_reason"])

    if not s1["stage1_pass"]:
        return build_result(False, s1["stage1_reason"])

    try:
        return stage2_evaluate(program_path)
    except Exception as e:
        return build_result(True, "ok", stage2_pass=False, stage2_error=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("program_path", type=str)
    parser.add_argument("--stage", type=str, default="all", choices=["stage1", "stage2", "all"])
    args = parser.parse_args()
    print(json.dumps(evaluate(args.program_path, stage=args.stage), ensure_ascii=False, indent=2))
