import argparse
import importlib.util
import inspect
import json
import os
import random
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


try:
    torch.set_num_threads(int(os.getenv("RL_TORCH_NUM_THREADS", "1")))
except Exception:
    pass
try:
    torch.set_num_interop_threads(int(os.getenv("RL_TORCH_NUM_INTEROP_THREADS", "1")))
except Exception:
    pass


try:
    from openevolve.evaluation_result import EvaluationResult  # type: ignore
except Exception:
    @dataclass
    class EvaluationResult:
        metrics: Dict[str, Any]
        artifacts: Dict[str, Any]


ENV_NAME = os.getenv("RL_ENV_NAME", "CartPole-v1")
INPUT_DIM = int(os.getenv("RL_INPUT_DIM", "4"))
OUTPUT_DIM = int(os.getenv("RL_OUTPUT_DIM", "2"))
MAX_PARAM_COUNT = int(os.getenv("RL_MAX_PARAM_COUNT", "10000"))
MAX_STEPS_PER_EPISODE = int(os.getenv("RL_MAX_STEPS_PER_EPISODE", "500"))
TRAIN_EPISODES = int(os.getenv("RL_TRAIN_EPISODES", "100"))
TEST_EPISODES = int(os.getenv("RL_TEST_EPISODES", "20"))
GAMMA = float(os.getenv("RL_GAMMA", "0.99"))
LR = float(os.getenv("RL_LR", "1e-2"))
PARAM_PENALTY = float(os.getenv("RL_PARAM_PENALTY", "0.0001"))
REQUESTED_DEVICE = os.getenv("RL_DEVICE", "auto").strip().lower()
EVAL_SEEDS = [int(x) for x in os.getenv("RL_EVAL_SEEDS", "42,123").split(",") if x.strip()]


def _import_gym() -> Tuple[Any, str]:
    try:
        import gymnasium as gym  # type: ignore
        return gym, "gymnasium"
    except Exception:
        try:
            import gym  # type: ignore
            return gym, "gym"
        except Exception as e:
            raise RuntimeError(
                "Neither gymnasium nor gym is installed. Install one of them before running RL evaluation."
            ) from e


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_program_module(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate program from: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def discount_rewards(rewards: List[float], gamma: float) -> torch.Tensor:
    discounted: List[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        discounted.append(running)
    discounted.reverse()
    out = torch.tensor(discounted, dtype=torch.float32)
    if out.numel() > 1:
        out = (out - out.mean()) / (out.std() + 1e-8)
    return out


def static_check_candidate(program_path: str) -> Dict[str, Any]:
    module = load_program_module(program_path)

    if not hasattr(module, "build_model"):
        raise AttributeError("Candidate program must define build_model(input_dim=4, output_dim=2).")

    fn = getattr(module, "build_model")
    if not callable(fn):
        raise TypeError("build_model must be callable.")

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    for required in ["input_dim", "output_dim"]:
        if required not in params:
            raise ValueError(f"build_model must contain parameter `{required}`, but got {params}.")

    return {"function_name": "build_model", "signature_params": params}


def build_candidate_model(module) -> nn.Module:
    model = module.build_model(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    if not isinstance(model, nn.Module):
        raise TypeError("build_model(...) must return a torch.nn.Module instance.")
    return model


def validate_candidate_model_runtime(module) -> Dict[str, Any]:
    model = build_candidate_model(module)
    model.eval()

    param_count = count_params(model)
    if param_count <= 0:
        raise ValueError("Model has no trainable parameters.")
    if param_count > MAX_PARAM_COUNT:
        raise ValueError(f"Model too large: {param_count} parameters > {MAX_PARAM_COUNT}.")

    with torch.no_grad():
        dummy_x = torch.randn(4, INPUT_DIM)
        dummy_y = model(dummy_x)

    if not isinstance(dummy_y, torch.Tensor):
        raise TypeError("Model forward must return a torch.Tensor.")
    if tuple(dummy_y.shape) != (4, OUTPUT_DIM):
        raise ValueError(f"Model output shape must be (4, {OUTPUT_DIM}), but got {tuple(dummy_y.shape)}.")
    if not torch.isfinite(dummy_y).all():
        raise ValueError("Model output contains NaN or Inf.")

    return {"param_count": int(param_count)}


def _env_reset(env, seed=None):
    if seed is None:
        out = env.reset()
    else:
        try:
            out = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
            out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _env_step(env, action: int):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return next_obs, float(reward), done, info
    if isinstance(out, tuple) and len(out) == 4:
        next_obs, reward, done, info = out
        return next_obs, float(reward), bool(done), info
    raise ValueError(f"Unexpected env.step output: {out}")


def run_episode(env, model: nn.Module, train: bool, optimizer=None, seed=None) -> float:
    obs = _env_reset(env, seed=seed)
    total_reward = 0.0
    steps = 0
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []

    while True:
        x = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all():
            raise ValueError("Action probabilities contain NaN or Inf.")

        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        next_obs, reward, done, _ = _env_step(env, int(action.item()))

        if train:
            log_probs.append(dist.log_prob(action))
            rewards.append(float(reward))

        total_reward += float(reward)
        steps += 1
        obs = next_obs

        if done or steps >= MAX_STEPS_PER_EPISODE:
            break

    if train:
        returns = discount_rewards(rewards, gamma=GAMMA).to(DEVICE)
        loss = torch.tensor(0.0, device=DEVICE)
        for log_prob, ret in zip(log_probs, returns):
            loss = loss + (-log_prob * ret)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return float(total_reward)


def train_and_evaluate_one_seed(program_path: str, seed: int) -> Dict[str, float]:
    set_seed(seed)
    gym, gym_backend = _import_gym()
    env = gym.make(ENV_NAME)

    module = load_program_module(program_path)
    model = build_candidate_model(module).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_returns: List[float] = []
    test_returns: List[float] = []
    best_test_return = -float("inf")

    for episode_idx in range(TRAIN_EPISODES):
        train_ret = run_episode(
            env=env,
            model=model,
            train=True,
            optimizer=optimizer,
            seed=seed + episode_idx,
        )
        train_returns.append(train_ret)

    for episode_idx in range(TEST_EPISODES):
        with torch.no_grad():
            test_ret = run_episode(
                env=env,
                model=model,
                train=False,
                optimizer=None,
                seed=seed + 10000 + episode_idx,
            )
        test_returns.append(test_ret)
        best_test_return = max(best_test_return, test_ret)

    env.close()

    mean_train_return = float(np.mean(train_returns)) if train_returns else 0.0
    mean_test_return = float(np.mean(test_returns)) if test_returns else 0.0
    param_count = count_params(model)
    combined_score = mean_test_return - PARAM_PENALTY * float(param_count)

    return {
        "combined_score": float(combined_score),
        "mean_train_return": float(mean_train_return),
        "mean_test_return": float(mean_test_return),
        "best_test_return": float(best_test_return),
        "num_params": float(param_count),
        "gym_backend": gym_backend,
    }


def evaluate_stage1(program_path: str) -> EvaluationResult:
    try:
        static_info = static_check_candidate(program_path)
        module = load_program_module(program_path)
        runtime_info = validate_candidate_model_runtime(module)
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "ood_acc": 0.0,
                "stage1_ok": 1.0,
                "num_params": float(runtime_info["param_count"]),
            },
            artifacts={
                "stage": "stage1",
                "program_path": str(program_path),
                "static": static_info,
            },
        )
    except Exception as e:
        return EvaluationResult(
            metrics={
                "combined_score": -1e18,
                "ood_acc": -1e18,
                "stage1_ok": 0.0,
                "num_params": 0.0,
                "error": str(e),
            },
            artifacts={
                "stage": "stage1",
                "program_path": str(program_path),
                "traceback": traceback.format_exc(),
            },
        )


def evaluate_stage2(program_path: str) -> EvaluationResult:
    t0 = time.time()
    per_seed = [train_and_evaluate_one_seed(program_path, seed) for seed in EVAL_SEEDS]
    total_time = time.time() - t0

    mean_combined_score = float(np.mean([x["combined_score"] for x in per_seed]))
    mean_train_return = float(np.mean([x["mean_train_return"] for x in per_seed]))
    mean_test_return = float(np.mean([x["mean_test_return"] for x in per_seed]))
    best_test_return = float(np.max([x["best_test_return"] for x in per_seed]))
    mean_num_params = float(np.mean([x["num_params"] for x in per_seed]))
    gym_backend = per_seed[0]["gym_backend"] if per_seed else "unknown"

    result = EvaluationResult(
        metrics={
            "combined_score": mean_combined_score,
            "ood_acc": mean_test_return,
            "mean_train_return": mean_train_return,
            "mean_test_return": mean_test_return,
            "best_test_return": best_test_return,
            "num_params": mean_num_params,
            "time": float(total_time),
            "stage1_ok": 1.0,
        },
        artifacts={
            "stage": "stage2",
            "program_path": str(program_path),
            "env_name": ENV_NAME,
            "device": DEVICE,
            "gym_backend": gym_backend,
            "eval_seeds": list(EVAL_SEEDS),
            "train_episodes": TRAIN_EPISODES,
            "test_episodes": TEST_EPISODES,
            "per_seed": per_seed,
        },
    )

    print("=" * 80)
    print("CartPole candidate evaluation")
    print("=" * 80)
    print(f"env_name         : {ENV_NAME}")
    print(f"device           : {DEVICE}")
    print(f"eval_seeds       : {EVAL_SEEDS}")
    print(f"train_episodes   : {TRAIN_EPISODES}")
    print(f"test_episodes    : {TEST_EPISODES}")
    print(f"combined_score   : {mean_combined_score}")
    print(f"ood_acc          : {mean_test_return}")
    print(f"mean_train_return: {mean_train_return}")
    print(f"mean_test_return : {mean_test_return}")
    print(f"best_test_return : {best_test_return}")
    print(f"num_params       : {mean_num_params}")
    print(f"time             : {total_time}")
    print("=" * 80)

    return result


def evaluate(program_path: str) -> EvaluationResult:
    s1 = evaluate_stage1(program_path)
    if float(s1.metrics.get("stage1_ok", 0.0)) <= 0.0:
        return s1
    return evaluate_stage2(program_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CartPole candidate program.")
    parser.add_argument("--program-path", type=str, required=True, help="Path to candidate program")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate(args.program_path)
    print(json.dumps({"metrics": result.metrics, "artifacts": result.artifacts}, indent=2))
