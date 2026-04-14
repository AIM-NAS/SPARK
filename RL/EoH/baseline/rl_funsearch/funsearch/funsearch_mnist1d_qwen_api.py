import json
import multiprocessing
import os
import tempfile
import importlib.util
from typing import Collection, Any

import requests
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FUNSEARCH_DIR = os.path.join(CURRENT_DIR, "funsearch")
if FUNSEARCH_DIR not in sys.path:
    sys.path.insert(0, FUNSEARCH_DIR)
from implementation import funsearch
from implementation import config
from implementation import sampler
from implementation import evaluator
from implementation import code_manipulation

# Use local evaluator by default, but allow override from env.
RL_EVALUATOR_PATH = "/data1/lz/clrs/openevolve/RL/evaluator.py"


def _trim_preface_of_body(sample: str) -> str:
    """Trim descriptions / def line before the generated function body."""
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    for lineno, line in enumerate(lines):
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    if find_def_declaration:
        code = ''
        for line in lines[func_body_lineno + 1:]:
            code += line + '\n'
        return code
    return sample


class QwenAPI(sampler.LLM):
    """Qwen API client for FunSearch."""

    def __init__(self, samples_per_prompt: int, trim: bool = True) -> None:
        super().__init__(samples_per_prompt)
        self._trim = trim
        self._api_base = os.getenv(
            "FUNSEARCH_API_BASE",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).rstrip("/")
        self._api_key = os.getenv("FUNSEARCH_API_KEY", "")
        self._model = os.getenv("FUNSEARCH_MODEL_NAME", "qwen-plus")
        self._timeout = int(os.getenv("FUNSEARCH_API_TIMEOUT", "180"))
        self._temperature = float(os.getenv("FUNSEARCH_TEMPERATURE", "0.8"))
        self._top_p = float(os.getenv("FUNSEARCH_TOP_P", "0.95"))
        self._max_tokens = int(os.getenv("FUNSEARCH_MAX_NEW_TOKENS", "512"))
        self._additional_prompt = (
            "Complete only the Python function body for the target function. "
            "Return code only. Do not output explanations, markdown, or surrounding text. "
            "The generated code must stay syntactically valid Python and preserve indentation."
        )
        if not self._api_key:
            raise ValueError(
                "FUNSEARCH_API_KEY is empty. Please export your DashScope-compatible API key."
            )

    def draw_samples(self, prompt: str) -> Collection[str]:
        samples = []
        while len(samples) < self._samples_per_prompt:
            try:
                sample = self._draw_sample(prompt)
                if self._trim:
                    sample = _trim_preface_of_body(sample)
                samples.append(sample)
            except Exception:
                continue
        return samples

    def _draw_sample(self, content: str) -> str:
        prompt = '\n'.join([content.strip('\n').strip(), self._additional_prompt])
        url = f"{self._api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Python code completion model for evolutionary search.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
        }
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class Sandbox(evaluator.Sandbox):
    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def run(
        self,
        program: str,
        function_to_run: str,
        function_to_evolve: str,
        inputs: Any,
        test_input: str,
        timeout_seconds: int,
        **kwargs,
    ) -> tuple[Any, bool]:
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, result_queue),
        )
        process.start()
        real_timeout_seconds = int(os.getenv("FUNSEARCH_EVAL_TIMEOUT", "3600"))
        process.join(timeout=real_timeout_seconds)

        if process.is_alive():
            print(
                f"\n[Sandbox Timeout] Candidate evaluation exceeded {real_timeout_seconds} seconds and was terminated."
            )
            process.terminate()
            process.join()
            results = (None, False)
        else:
            if not result_queue.empty():
                results = result_queue.get_nowait()
            else:
                results = (None, False)

        if self._verbose:
            print("================= Evaluated Program =================")
            try:
                program_obj = code_manipulation.text_to_program(text=program)
                function_obj = program_obj.get_function(function_to_evolve)
                print(str(function_obj).strip('\n'))
            except Exception as e:
                print(f"[WARN] pretty print failed: {e}")
            print("-----------------------------------------------------")
            print(f"Score: {str(results)}")
            print("=====================================================\n")

        return results

    @staticmethod
    def _compile_and_run_function(program: str, result_queue):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                candidate_path = os.path.join(tmpdir, "candidate_program.py")
                with open(candidate_path, "w", encoding="utf-8") as f:
                    f.write(program)

                spec = importlib.util.spec_from_file_location(
                    "rl_eval_module",
                    RL_EVALUATOR_PATH,
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load evaluator from {RL_EVALUATOR_PATH}")
                eval_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(eval_module)

                result = eval_module.evaluate(candidate_path)
                score = float(result.metrics["combined_score"])
                result_queue.put((score, True))
        except Exception:
            import traceback
            print("\n[Sandbox Error] Candidate evaluation failed:")
            traceback.print_exc()
            result_queue.put((None, False))


specification = r'''
import torch
import torch.nn as nn

class funsearch:
    @staticmethod
    def run(fn):
        return fn

    @staticmethod
    def evolve(fn):
        return fn


@funsearch.run
def evaluate_candidate(dummy_input=None) -> float:
    """Placeholder function only for passing decorator checks."""
    return 0.0


@funsearch.evolve
def build_model(input_dim=4, output_dim=2):
    """Return a torch.nn.Module policy network for CartPole-v1.

    Hard constraints:
    1. Must return an instance of torch.nn.Module.
    2. The model forward input is a float tensor of shape (B, input_dim).
    3. The model forward output must be a tensor of shape (B, output_dim).
    4. The output must be raw logits for discrete actions.
    5. Do not perform training here.
    6. Do not change the function signature.
    7. Keep the model lightweight and stable.
    """
    class SimplePolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            hidden_dim = 32
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    return SimplePolicyNet()


if __name__ == "__main__":
    model = build_model(input_dim=4, output_dim=2)
    x = torch.randn(8, 4)
    y = model(x)
    print("Output shape:", y.shape)
'''


if __name__ == '__main__':
    class_config = config.ClassConfig(llm_class=QwenAPI, sandbox_class=Sandbox)

    cfg = config.Config(
        samples_per_prompt=int(os.getenv("FUNSEARCH_SAMPLES_PER_PROMPT", "4"))
    )
    dummy_inputs = {"cartpole": None}

    global_max_sample_num = int(os.getenv("FUNSEARCH_MAX_SAMPLES", "20"))
    log_dir = os.getenv("FUNSEARCH_LOG_DIR", "logs/funsearch_cartpole_qwen_api")

    funsearch.main(
        specification=specification,
        inputs=dummy_inputs,
        config=cfg,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir=log_dir,
    )
