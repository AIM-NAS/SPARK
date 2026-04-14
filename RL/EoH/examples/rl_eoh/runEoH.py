import os
import inspect

import eoh
import eoh.llm.api_general as ag
from eoh import eoh
from eoh.utils.getParas import Paras

from prob import CartPoleModelSearch

print("eoh imported from:", eoh.__file__)
print("api_general imported from:", ag.__file__)

# -------------------------
# Monkey-patch EoH parser
# -------------------------
from eoh.methods.eoh.eoh_evolution import Evolution as _EOHEvolution


def _strip_code_fences(text: str) -> str:
    text = str(text).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _extract_brace_description(text: str) -> str:
    text = str(text)
    l = text.find("{")
    if l != -1:
        r = text.find("}", l + 1)
        if r != -1:
            return text[l + 1:r].strip()

    def_pos = text.find("def ")
    if def_pos != -1:
        prefix = text[:def_pos].strip()
        if prefix:
            return prefix.splitlines()[-1].strip()

    return "No description"


def _extract_function_block(text: str, func_name: str, output_names) -> str:
    text = _strip_code_fences(text)
    header = f"def {func_name}("
    start = text.find(header)
    if start == -1:
        raise ValueError(f"Cannot find function header: {header}")

    text = text[start:]
    return_line = "return " + ", ".join(output_names)

    end = text.find(return_line)
    if end != -1:
        end += len(return_line)
        return text[:end].rstrip()

    lines = text.splitlines()
    if not lines:
        raise ValueError("Empty function block after locating function header.")

    kept = [lines[0]]
    for line in lines[1:]:
        if line.strip() == "":
            kept.append(line)
            continue
        if not line.startswith((" ", "\t")):
            break
        kept.append(line)

    return "\n".join(kept).rstrip()


def _patched_get_alg(self, prompt_content):
    response = self.interface_llm.get_response(prompt_content)

    n_retry = 1
    last_err = None
    while n_retry <= 4:
        try:
            if self.debug_mode:
                print("\n[PATCHED _get_alg] raw response:\n", response)

            algorithm = _extract_brace_description(response)
            code = _extract_function_block(
                response,
                self.prompt_func_name,
                self.prompt_func_outputs,
            )

            if self.debug_mode:
                print("\n[PATCHED _get_alg] parsed algorithm:\n", algorithm)
                print("\n[PATCHED _get_alg] parsed code:\n", code)

            return [code, algorithm]

        except Exception as e:
            last_err = e
            if self.debug_mode:
                print(f"[PATCHED _get_alg] parse failed on try {n_retry}: {repr(e)}")
            response = self.interface_llm.get_response(prompt_content)
            n_retry += 1

    raise RuntimeError(f"_patched_get_alg failed after retries: {repr(last_err)}")


_EOHEvolution._get_alg = _patched_get_alg

print("eoh_evolution imported from:", inspect.getsourcefile(_EOHEvolution))
print(">>> USING PATCHED Evolution._get_alg")

# -------------------------
# Normal EoH startup
# -------------------------
paras = Paras()
problem_local = CartPoleModelSearch()

paras.set_paras(
    method="eoh",
    problem=problem_local,
    llm_api_endpoint=os.getenv("EOH_LLM_API_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    llm_api_key=os.getenv("EOH_LLM_API_KEY", " "),
    llm_model=os.getenv("EOH_LLM_MODEL", "qwen-plus"),
    ec_pop_size=int(os.getenv("EOH_POP_SIZE", "4")),
    ec_n_pop=int(os.getenv("EOH_N_POP", "100")),
    exp_n_proc=int(os.getenv("EOH_N_PROC", "1")),
    exp_debug_mode=bool(int(os.getenv("EOH_DEBUG_MODE", "1"))),
    eva_numba_decorator=False,
)

print("llm_api_endpoint =", paras.llm_api_endpoint)
print("llm_model =", paras.llm_model)
print("debug_mode =", paras.exp_debug_mode)

evolution = eoh.EVOL(paras)
evolution.run()
