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
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# Reduce thread oversubscription.
try:
    torch.set_num_threads(int(os.getenv("RL_TORCH_NUM_THREADS", "1")))
except Exception:
    pass
try:
    torch.set_num_interop_threads(int(os.getenv("RL_TORCH_NUM_INTEROP_THREADS", "1")))
except Exception:
    pass


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


try:
    from openevolve.evaluation_result import EvaluationResult  # type: ignore
except Exception:
    @dataclass
    class EvaluationResult:
        metrics: Dict[str, Any]
        artifacts: Dict[str, Any]


# =========================
# Global config
# =========================
ENV_NAME = os.getenv("RL_ENV_NAME", "CartPole-v1")
INPUT_DIM = int(os.getenv("RL_INPUT_DIM", "4"))
OUTPUT_DIM = int(os.getenv("RL_OUTPUT_DIM", "2"))

MAX_PARAM_COUNT = int(os.getenv("RL_MAX_PARAM_COUNT", "10000"))
MAX_STEPS_PER_EPISODE = int(os.getenv("RL_MAX_STEPS_PER_EPISODE", "500"))

TRAIN_EPISODES = int(os.getenv("RL_TRAIN_EPISODES", "50"))
TEST_EPISODES = int(os.getenv("RL_TEST_EPISODES", "10"))

GAMMA = float(os.getenv("RL_GAMMA", "0.99"))
LR = float(os.getenv("RL_LR", "1e-2"))

# Only averaged results will be reported.
EVAL_SEEDS = [
    int(x) for x in os.getenv("RL_EVAL_SEEDS", "42,123").split(",") if x.strip()
]

# Penalize oversized models slightly in combined_score.
PARAM_PENALTY = float(os.getenv("RL_PARAM_PENALTY", "0.0001"))

# Device
REQUESTED_DEVICE = os.getenv("RL_DEVICE", "auto").strip().lower()


# =========================
# Utilities
# =========================
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


DEVICE = resolve_device(REQUESTED_DEVICE)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_program_module(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def discount_rewards(rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
    discounted = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        discounted.append(running)
    discounted.reverse()
    discounted = torch.tensor(discounted, dtype=torch.float32)
    if discounted.numel() > 1:
        discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
    return discounted


def _result_to_jsonable(result: EvaluationResult) -> dict:
    return {"metrics": result.metrics, "artifacts": result.artifacts}


def _result_from_jsonable(obj: dict) -> EvaluationResult:
    return EvaluationResult(
        metrics=obj.get("metrics", {}),
        artifacts=obj.get("artifacts", {}),
    )


# =========================
# Stage 1: static check only
# =========================
def static_check_candidate(program_path: str):
    """
    Pure static check:
    - import candidate
    - check build_model exists
    - check signature shape at inspect level only
    No tensor creation, no model instantiation, no forward.
    """
    module = load_program_module(program_path)

    if not hasattr(module, "build_model"):
        raise AttributeError(
            "Candidate program must define build_model(input_dim=4, output_dim=2)."
        )

    fn = getattr(module, "build_model")
    if not callable(fn):
        raise TypeError("build_model must be callable")

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    required_params = ["input_dim", "output_dim"]
    for p in required_params:
        if p not in params:
            raise ValueError(
                f"build_model must contain parameter `{p}`, got {params}"
            )

    return {
        "function_name": "build_model",
        "signature_params": params,
    }


# =========================
# Runtime validation / model build
# =========================
def build_candidate_model(module):
    model = module.build_model(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    if not isinstance(model, nn.Module):
        raise TypeError("build_model(...) must return a torch.nn.Module instance")
    return model


def validate_candidate_model_runtime(module):
    model = build_candidate_model(module)
    model.eval()

    param_count = count_params(model)
    if param_count <= 0:
        raise ValueError("Model has no trainable parameters.")
    if param_count > MAX_PARAM_COUNT:
        raise ValueError(
            f"Model too large: {param_count} parameters > {MAX_PARAM_COUNT}"
        )

    with torch.no_grad():
        dummy_x = torch.randn(4, INPUT_DIM)
        dummy_y = model(dummy_x)

    if not isinstance(dummy_y, torch.Tensor):
        raise TypeError("Model forward must return a torch.Tensor")

    if tuple(dummy_y.shape) != (4, OUTPUT_DIM):
        raise ValueError(
            f"Model output shape must be (4, {OUTPUT_DIM}), got {tuple(dummy_y.shape)}"
        )

    if not torch.isfinite(dummy_y).all():
        raise ValueError("Model output contains NaN or Inf.")

    return {
        "param_count": int(param_count),
    }


# =========================
# RL core
# =========================
def run_episode(env, model, train=True, optimizer=None, seed=None):
    if seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)

    log_probs = []
    rewards = []

    total_reward = 0.0
    steps = 0

    while True:
        x = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)

        if not torch.isfinite(probs).all():
            raise ValueError("Action probabilities contain NaN or Inf.")

        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action.item())

        if train:
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

        total_reward += float(reward)
        steps += 1
        obs = next_obs

        if terminated or truncated or steps >= MAX_STEPS_PER_EPISODE:
            break

    if train:
        returns = discount_rewards(rewards, gamma=GAMMA).to(DEVICE)
        loss = 0.0
        for log_prob, ret in zip(log_probs, returns):
            loss = loss - log_prob * ret

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    return float(total_reward)


def evaluate_greedy_episode(env, model, seed=None):
    if seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0

    while True:
        x = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        action = torch.argmax(logits, dim=-1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1

        if terminated or truncated or steps >= MAX_STEPS_PER_EPISODE:
            break

    return float(total_reward)


def train_and_evaluate_one_seed(module, seed: int):
    set_seed(seed)

    model = build_candidate_model(module).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)

    train_returns = []
    for ep in range(TRAIN_EPISODES):
        ep_seed = seed if ep == 0 else None
        reward = run_episode(
            train_env,
            model,
            train=True,
            optimizer=optimizer,
            seed=ep_seed,
        )
        train_returns.append(float(reward))

    model.eval()
    test_returns = []
    for ep in range(TEST_EPISODES):
        reward = evaluate_greedy_episode(
            test_env,
            model,
            seed=seed + 1000 + ep,
        )
        test_returns.append(float(reward))

    train_env.close()
    test_env.close()

    return {
        "mean_train_return": float(np.mean(train_returns)) if train_returns else 0.0,
        "mean_test_return": float(np.mean(test_returns)) if test_returns else 0.0,
        "best_test_return": float(np.max(test_returns)) if test_returns else 0.0,
    }


# =========================
# Stage 1
# =========================
def evaluate_stage1(program_path: str):
    """
    Static-only stage1.
    """
    try:
        info = static_check_candidate(program_path)

        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "ood_acc": 0.0,
                "time": 0.0,
                "stage1_ok": 1.0,
            },
            artifacts={
                "stage": "stage1",
                "status": "static_check_ok",
                "env_name": ENV_NAME,
                "input_dim": int(INPUT_DIM),
                "output_dim": int(OUTPUT_DIM),
                "train_episodes": int(TRAIN_EPISODES),
                "test_episodes": int(TEST_EPISODES),
                "eval_seeds": [int(s) for s in EVAL_SEEDS],
                "requested_device": str(REQUESTED_DEVICE),
                "resolved_device": str(DEVICE),
                "function_name": info["function_name"],
                "signature_params": info["signature_params"],
            },
        )
    except Exception as e:
        return EvaluationResult(
            metrics={
                "combined_score": -1e18,
                "ood_acc": -1e18,
                "time": 0.0,
                "stage1_ok": 0.0,
                "error": str(e),
            },
            artifacts={
                "stage": "stage1",
                "status": "static_check_failed",
                "traceback": traceback.format_exc(),
            },
        )


# =========================
# Stage 2 core
# =========================
def _stage2_core(program_path: str) -> EvaluationResult:
    t0 = time.time()

    # Lightweight validation before RL train/eval
    module = load_program_module(program_path)
    runtime_info = validate_candidate_model_runtime(module)
    num_params = int(runtime_info["param_count"])

    seed_results = []
    for seed in EVAL_SEEDS:
        result = train_and_evaluate_one_seed(module, seed)
        seed_results.append(result)

    mean_train_return = float(np.mean([x["mean_train_return"] for x in seed_results]))
    mean_test_return = float(np.mean([x["mean_test_return"] for x in seed_results]))
    best_test_return = float(np.mean([x["best_test_return"] for x in seed_results]))

    # Keep compatibility with your previous evaluator naming habit:
    # ood_acc is mapped to mean_test_return in this RL setup.
    ood_acc = mean_test_return
    combined_score = float(mean_test_return - PARAM_PENALTY * num_params)
    elapsed = float(time.time() - t0)

    return EvaluationResult(
        metrics={
            "combined_score": float(combined_score),
            "ood_acc": float(ood_acc),
            "time": float(elapsed),
            "mean_train_return": float(mean_train_return),
            "mean_test_return": float(mean_test_return),
            "best_test_return": float(best_test_return),
            "num_params": float(num_params),
            "macs": float(num_params),
            "stage2_ok": 1.0,
        },
        artifacts={
            "stage": "stage2",
            "env_name": ENV_NAME,
            "input_dim": int(INPUT_DIM),
            "output_dim": int(OUTPUT_DIM),
            "train_episodes": int(TRAIN_EPISODES),
            "test_episodes": int(TEST_EPISODES),
            "eval_seeds": [int(s) for s in EVAL_SEEDS],
            "requested_device": str(REQUESTED_DEVICE),
            "resolved_device": str(DEVICE),
            "gamma": float(GAMMA),
            "lr": float(LR),
            "max_steps_per_episode": int(MAX_STEPS_PER_EPISODE),
            "param_penalty": float(PARAM_PENALTY),
        },
    )


# =========================
# Stage 2 wrapper via subprocess
# =========================
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
                    "ood_acc": -1e18,
                    "time": 0.0,
                    "mean_train_return": 0.0,
                    "mean_test_return": 0.0,
                    "best_test_return": 0.0,
                    "num_params": 0.0,
                    "macs": 0.0,
                    "stage2_ok": 0.0,
                    "error": f"stage2 worker failed with return code {proc.returncode}",
                },
                artifacts={
                    "stage": "stage2",
                    "worker_stdout": proc.stdout,
                    "worker_stderr": proc.stderr,
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
                "ood_acc": -1e18,
                "time": 0.0,
                "mean_train_return": 0.0,
                "mean_test_return": 0.0,
                "best_test_return": 0.0,
                "num_params": 0.0,
                "macs": 0.0,
                "stage2_ok": 0.0,
                "error": str(e),
            },
            artifacts={
                "stage": "stage2",
                "traceback": tb,
            },
        )


# =========================
# Public evaluate
# =========================
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


# =========================
# Stage2 worker main
# =========================
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
                "time": 0.0,
                "mean_train_return": 0.0,
                "mean_test_return": 0.0,
                "best_test_return": 0.0,
                "num_params": 0.0,
                "macs": 0.0,
                "stage2_ok": 0.0,
                "error": str(e),
            },
            artifacts={
                "stage": "stage2_worker",
                "traceback": tb,
            },
        )
        print(tb, file=sys.stderr, flush=True)
        print(json.dumps(_result_to_jsonable(err_result)), flush=True)
        return 0


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-worker", type=str, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("program_path", nargs="?", default=None)
    args = parser.parse_args()

    if args.stage2_worker:
        raise SystemExit(_main_stage2_worker(args.stage2_worker))

    target = args.program_path
    if target is None:
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
        ordered_keys = [
            "combined_score",
            "ood_acc",
            "time",
            "mean_train_return",
            "mean_test_return",
            "best_test_return",
            "num_params",
            "macs",
        ]
        for k in ordered_keys:
            if k in out.metrics:
                print(f"{k}: {out.metrics[k]}")
        if "error" in out.metrics:
            print(f"error: {out.metrics['error']}")