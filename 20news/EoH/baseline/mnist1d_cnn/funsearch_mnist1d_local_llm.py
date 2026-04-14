import json
import multiprocessing
import os
import sys
import tempfile
import importlib.util
from typing import Collection, Any

import requests

from implementation import funsearch
from implementation import config
from implementation import sampler
from implementation import evaluator
from implementation import code_manipulation

# ===== 固定为你的评估器路径，无需再改 =====
MNIST_EVALUATOR_PATH = "/data1/lz/clrs/openevolve/mnist1d/mnist1d_evolve/evaluator.py"


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


class LocalLLM(sampler.LLM):
    """Local LLM client. Reuses the same HTTP completion server style as your bin-packing runner."""

    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim: bool = True) -> None:
        super().__init__(samples_per_prompt)
        self._batch_inference = batch_inference
        self._trim = trim
        self._url = os.getenv("FUNSEARCH_LLM_URL", "http://127.0.0.1:11011/completions")

    def draw_samples(self, prompt: str) -> Collection[str]:
        while True:
            try:
                all_samples = []
                if self._batch_inference:
                    response = self._do_request(prompt)
                    for res in response:
                        all_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        response = self._do_request(prompt)
                        all_samples.append(response)

                if self._trim:
                    all_samples = [_trim_preface_of_body(sample) for sample in all_samples]
                return all_samples
            except Exception:
                continue

    def _do_request(self, content: str):
        content = content.strip('\n').strip()
        repeat_prompt = self._samples_per_prompt if self._batch_inference else 1
        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)
        response.raise_for_status()
        payload = response.json()["content"]
        return payload if self._batch_inference else payload[0]


class Sandbox(evaluator.Sandbox):
    """MNIST1D 专用 sandbox.

    与 bin-packing 不同，这里不是 exec(program) 后直接调用 @funsearch.run。
    而是：
    1) 把当前候选 program 写成临时 candidate_program.py
    2) 调用 /data1/.../mnist1d_evolve/evaluator.py 中的 evaluate(candidate_path)
    3) 取 result.metrics['combined_score'] 作为 FunSearch 分数
    """

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
        **kwargs
    ) -> tuple[Any, bool]:
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, result_queue)
        )
        process.start()
        process.join(timeout=timeout_seconds)

        if process.is_alive():
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

                spec = importlib.util.spec_from_file_location("mnist_eval_module", MNIST_EVALUATOR_PATH)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load evaluator from {MNIST_EVALUATOR_PATH}")
                eval_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(eval_module)

                result = eval_module.evaluate(candidate_path)
                score = float(result.metrics["combined_score"])
                result_queue.put((score, True))
        except Exception:
            result_queue.put((None, False))


specification = r'''
import torch
import torch.nn as nn
from implementation import funsearch


@funsearch.run
def evaluate_candidate(dummy_input=None) -> float:
    """占位函数，只用于通过 FunSearch 的 decorator 检查。"""
    return 0.0


@funsearch.evolve
def build_model(input_size=40, output_size=10):
    """Return a torch.nn.Module for MNIST1D classification.

    Hard constraints:
    1. Must return an instance of torch.nn.Module.
    2. The model forward input is a float tensor of shape (B, input_size).
    3. The model forward output must be a tensor of shape (B, output_size).
    4. Do not perform training here.
    5. Do not change function signature.
    6. Keep the model trainable and reasonably lightweight.
    """
    class CandidateNet(nn.Module):
        def __init__(self):
            super().__init__()
            hidden_size = 100
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linear3 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h = self.linear1(x).relu()
            h = h + self.linear2(h).relu()
            return self.linear3(h)

    return CandidateNet()


if __name__ == "__main__":
    model = build_model()
    x = torch.randn(8, 40)
    y = model(x)
    print("Output shape:", y.shape)
'''


if __name__ == '__main__':
    class_config = config.ClassConfig(llm_class=LocalLLM, sandbox_class=Sandbox)

    cfg = config.Config(samples_per_prompt=int(os.getenv("FUNSEARCH_SAMPLES_PER_PROMPT", "4")))

    # 这里只是让 evaluator.Evaluator 的 for current_input in self._inputs 能跑一轮。
    dummy_inputs = {"mnist1d": None}

    global_max_sample_num = int(os.getenv("FUNSEARCH_MAX_SAMPLES", "20"))
    log_dir = os.getenv("FUNSEARCH_LOG_DIR", "logs/funsearch_mnist1d_local_llm")

    funsearch.main(
        specification=specification,
        inputs=dummy_inputs,
        config=cfg,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir=log_dir,
    )
