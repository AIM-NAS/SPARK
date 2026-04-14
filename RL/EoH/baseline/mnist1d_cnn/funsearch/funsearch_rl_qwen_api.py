import os
import re
import time
import json
import copy
import traceback
import tempfile
import importlib.util
import multiprocessing
from typing import Collection

import requests

from implementation import funsearch
from implementation import config as config_lib
from implementation import sampler
from implementation import evaluator


INITIAL_PROGRAM_PATH = os.getenv(
    "FUNSEARCH_INITIAL_PROGRAM",
    os.path.join(os.getcwd(), "initial_program.py"),
)
EVALUATOR_PATH = os.getenv(
    "FUNSEARCH_EVALUATOR_PATH",
    os.path.join(os.getcwd(), "evaluator.py"),
)
FUNSEARCH_LOG_DIR = os.getenv(
    "FUNSEARCH_LOG_DIR",
    os.path.join(os.getcwd(), "logs", f"funsearch_rl_qwen_api_{int(time.time())}"),
)
os.makedirs(FUNSEARCH_LOG_DIR, exist_ok=True)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class _DecoratorShim:
    @staticmethod
    def run(fn):
        return fn

    @staticmethod
    def evolve(fn):
        return fn


def _build_specification_from_initial_program(initial_program_path: str) -> str:
    src = _read_text(initial_program_path)

    if "def build_model" not in src:
        raise ValueError("initial_program.py must define build_model(...)")

    # Ensure the specification has a real funsearch symbol for decorator parsing.
    if "from implementation import funsearch" not in src:
        src = "from implementation import funsearch\n\n" + src

    # Do NOT prepend any function before the user's imports/classes.
    # Append the dummy run function at the end instead.
    run_stub = '''

@funsearch.run
def evaluate_candidate(dummy_input=None):
    return 0.0
'''

    return src.rstrip() + "\n" + run_stub


SPECIFICATION = _build_specification_from_initial_program(INITIAL_PROGRAM_PATH)


def _trim_preface_of_body(sample: str) -> str:
    text = sample.replace("\r\n", "\n").strip("\n")
    lines = text.split("\n")

    for i, line in enumerate(lines):
        if line.startswith("def get_model_config") or line.startswith("def get_model_config_v"):
            body = lines[i + 1:]
            return "\n".join(body).rstrip() + "\n"
    return text.rstrip() + "\n"


class QwenAPI(sampler.LLM):
    def __init__(self, samples_per_prompt: int):
        self._samples_per_prompt = samples_per_prompt
        self._api_base = os.getenv(
            "FUNSEARCH_API_BASE",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).rstrip("/")
        self._api_key = os.getenv("FUNSEARCH_API_KEY", "").strip()
        self._model_name = os.getenv("FUNSEARCH_MODEL_NAME", "qwen-plus")
        self._timeout = int(os.getenv("FUNSEARCH_API_TIMEOUT", "180"))
        self._temperature = float(os.getenv("FUNSEARCH_TEMPERATURE", "0.9"))
        self._top_p = float(os.getenv("FUNSEARCH_TOP_P", "0.95"))
        self._max_new_tokens = int(os.getenv("FUNSEARCH_MAX_NEW_TOKENS", "512"))
        self._max_attempts = int(os.getenv("FUNSEARCH_MAX_ATTEMPTS", str(max(8, samples_per_prompt * 6))))

        if not self._api_key:
            raise ValueError("FUNSEARCH_API_KEY is empty")

        self._system_prompt = (
            "You are a Python code completion model for evolutionary search. "
            "Return code only. No markdown fences. No explanation."
        )
        self._extra_prompt = (
            "Complete only the body of get_model_config(input_dim=4, output_dim=2).\n"
            "Do not define any class.\n"
            "Do not redefine get_model_config.\n"
            "Return a valid Python function body.\n"
            "Return a dict with exactly these keys: 'hidden_dim', 'activation', 'num_layers'.\n"
            "hidden_dim must be a positive integer chosen from {16, 32, 64, 96, 128, 192, 256}.\n"
            "activation must be one of: 'relu', 'tanh', 'gelu', 'elu'.\n"
            "num_layers must be an integer between 1 and 4.\n"
            "Try diverse combinations across different samples.\n"
            "Avoid repeating the same configuration within a batch."
        )

    def _draw_sample(self, prompt: str) -> str:
        url = f"{self._api_base}/chat/completions"
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt + "\n\n" + self._extra_prompt},
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_new_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self._timeout)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _trim_preface_of_body(content)

    def draw_samples(self, prompt: str) -> Collection[str]:
        samples = []
        seen = set()
        last_error = None
        attempts = 0

        while len(samples) < self._samples_per_prompt and attempts < self._max_attempts:
            attempts += 1
            try:
                s = self._draw_sample(prompt).strip()
                if s in seen:
                    continue
                seen.add(s)
                samples.append(s)
            except Exception as exc:
                last_error = exc
                print(f"[WARN] sample attempt {attempts} failed: {exc}", flush=True)

        if not samples:
            raise RuntimeError(
                f"Failed to draw any sample after {attempts} attempts. last_error={last_error}"
            )
        return samples


class Sandbox(evaluator.Sandbox):
    def run(self, program: str, function_to_run: str, function_to_evolve: str, inputs, test_input, timeout_seconds: int):
        del function_to_run, function_to_evolve, inputs, test_input
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=self._compile_and_run_function, args=(program, q))
        p.start()
        p.join(timeout_seconds)

        if p.is_alive():
            p.terminate()
            p.join()
            return None, False

        if q.empty():
            return None, False
        return q.get()

    @staticmethod
    def _compile_and_run_function(program: str, q: multiprocessing.Queue):
        try:
            with tempfile.TemporaryDirectory(prefix="funsearch_rl_") as td:
                candidate_path = os.path.join(td, "candidate_program.py")
                with open(candidate_path, "w", encoding="utf-8") as f:
                    f.write(program)

                spec = importlib.util.spec_from_file_location("rl_eval_module", EVALUATOR_PATH)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load evaluator from: {EVALUATOR_PATH}")
                eval_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(eval_module)

                result = eval_module.evaluate(candidate_path)
                score = float(result.metrics["combined_score"])
                q.put((score, True))
        except Exception:
            traceback.print_exc()
            q.put((None, False))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def main() -> None:
    print(f"[INFO] INITIAL_PROGRAM_PATH = {INITIAL_PROGRAM_PATH}")
    print(f"[INFO] EVALUATOR_PATH       = {EVALUATOR_PATH}")
    print(f"[INFO] FUNSEARCH_LOG_DIR    = {FUNSEARCH_LOG_DIR}")

    cfg = config_lib.Config(
        programs_database=config_lib.ProgramsDatabaseConfig(
            functions_per_prompt=_env_int("FUNSEARCH_FUNCTIONS_PER_PROMPT", 2),
            num_islands=_env_int("FUNSEARCH_NUM_ISLANDS", 10),
            reset_period=_env_int("FUNSEARCH_RESET_PERIOD", 4 * 60 * 60),
            cluster_sampling_temperature_init=float(os.getenv("FUNSEARCH_CLUSTER_TEMP_INIT", "0.1")),
            cluster_sampling_temperature_period=_env_int("FUNSEARCH_CLUSTER_TEMP_PERIOD", 30000),
        ),
        num_samplers=_env_int("FUNSEARCH_NUM_SAMPLERS", 1),
        num_evaluators=_env_int("FUNSEARCH_NUM_EVALUATORS", 1),
        samples_per_prompt=_env_int("FUNSEARCH_SAMPLES_PER_PROMPT", 4),
    )
    class_cfg = config_lib.ClassConfig(llm_class=QwenAPI, sandbox_class=Sandbox)

    max_sample_nums = os.getenv("FUNSEARCH_MAX_SAMPLES", "100")
    max_sample_nums = None if str(max_sample_nums).lower() == "none" else int(max_sample_nums)

    funsearch.main(
        specification=SPECIFICATION,
        inputs=[None],
        config=cfg,
        max_sample_nums=max_sample_nums,
        class_config=class_cfg,
        log_dir=FUNSEARCH_LOG_DIR,
    )


if __name__ == "__main__":
    main()
