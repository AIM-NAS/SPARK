import importlib.util
import re
import tempfile
import textwrap
import traceback
from pathlib import Path

from prompts import GetPrompts


START_TAG = "# ===== BEGIN EVOLVE PGN REGION ====="
END_TAG = "# ===== END EVOLVE PGN REGION ====="
TARGET_FUNCTION = "build_candidate_net"


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class YeastSimpleMLPSearch:
    def __init__(self):
        self.root = Path(__file__).resolve().parent
        self.template_path = self.root / "initial_program.py"
        self.evaluator_path = self.root / "evaluator.py"
        if not self.template_path.exists():
            raise FileNotFoundError(f"Missing template program: {self.template_path}")
        if not self.evaluator_path.exists():
            raise FileNotFoundError(f"Missing evaluator: {self.evaluator_path}")
        self.prompts = GetPrompts(self.template_path)
        self.evaluator = load_module(self.evaluator_path, "yeast_evaluator")

    def _stitch_region_into_template(self, region_code: str) -> str:
        text = self.template_path.read_text(encoding="utf-8")
        if START_TAG not in text or END_TAG not in text:
            raise ValueError("Template program is missing evolve region tags.")
        s = text.index(START_TAG) + len(START_TAG)
        e = text.index(END_TAG)
        region_code = region_code.strip("\n")
        return text[:s] + "\n" + region_code + "\n" + text[e:]

    def _extract_code_block(self, text: str) -> str:
        text = str(text).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return text.strip()

    def _candidate_to_region_code(self, candidate) -> str:
        if isinstance(candidate, dict):
            for key in ("code", "algorithm", "text"):
                if key in candidate and candidate[key]:
                    candidate = candidate[key]
                    break
        text = self._extract_code_block(candidate)

        if f"def {TARGET_FUNCTION}(" in text:
            text = text[text.index(f"def {TARGET_FUNCTION}("):]
        else:
            m = re.search(rf"def\s+{TARGET_FUNCTION}\s*\(", text)
            if not m:
                raise ValueError(f"Candidate does not contain `def {TARGET_FUNCTION}(...):`")
            text = text[m.start():]

        lines = text.splitlines()
        if not lines:
            raise ValueError("Empty candidate code.")

        kept = [lines[0]]
        for line in lines[1:]:
            stripped = line.strip()
            if stripped == "":
                kept.append(line)
                continue
            indent = len(line) - len(line.lstrip(" "))
            if indent == 0 and not line.startswith((" ", "\t")):
                break
            kept.append(line)

        return textwrap.dedent("\n".join(kept)).strip() + "\n"

    def _validate_region_code(self, region_code: str) -> None:
        banned_substrings = [
            "```",
            "class ",
            "def build_model(",
            "if __name__ ==",
            "import ",
            "from torch",
            "print(",
        ]
        for token in banned_substrings:
            if token in region_code:
                raise ValueError(f"Forbidden content detected: {token}")

        required_patterns = [
            f"def {TARGET_FUNCTION}(input_dim, num_classes, hidden_dim=32, dropout=0.1):",
            "model =",
            "return model",
        ]
        for pat in required_patterns:
            if pat not in region_code:
                raise ValueError(f"Candidate missing required pattern: {pat}")

        return_count = len(re.findall(r"\breturn\b", region_code))
        if return_count != 1:
            raise ValueError(f"Candidate must contain exactly one return statement, got {return_count}.")

    def _normalize_candidate_code(self, candidate) -> str:
        region_code = self._candidate_to_region_code(candidate)
        self._validate_region_code(region_code)
        return self._stitch_region_into_template(region_code)

    def evaluate(self, candidate):
        try:
            print("[PROB] entering evaluate()", flush=True)
            candidate_code = self._normalize_candidate_code(candidate)
            print("[PROB] final candidate code:\n" + candidate_code, flush=True)

            with tempfile.TemporaryDirectory() as td:
                candidate_path = Path(td) / "candidate_program.py"
                candidate_path.write_text(candidate_code, encoding="utf-8")
                print(f"[PROB] written candidate to: {candidate_path}", flush=True)
                result = self.evaluator.evaluate(str(candidate_path))
                print(f"[PROB] evaluator returned: {result}", flush=True)
                metrics = getattr(result, "metrics", {}) or {}
                score = metrics.get("combined_score", metrics.get("combine_score", -1e18))
                print(f"[PROB] combined_score = {score}", flush=True)
                return float(score) if score is not None else -1e18
        except Exception as e:
            msg = (
                "\n" + "=" * 80 + "\n"
                + "[PROB] exception in evaluate()\n"
                + repr(e) + "\n"
                + traceback.format_exc() + "\n"
            )
            print(msg, flush=True)
            try:
                log_path = self.root / "eoh_eval_failures.log"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg)
            except Exception:
                pass
            return -1e18


# Backward compatibility alias
News20SimpleMLPSearch = YeastSimpleMLPSearch
