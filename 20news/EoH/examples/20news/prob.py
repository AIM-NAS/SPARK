import hashlib
import importlib.util
import traceback
from pathlib import Path

from prompts import GetPrompts


START = "# ===== BEGIN EVOLVE PGN REGION ====="
END = "# ===== END EVOLVE PGN REGION ====="


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class News20SimpleMLPSearch:
    def __init__(self):
        self.root = Path(__file__).resolve().parent
        self.template_path = self.root / "initial_program.py"
        self.evaluator_path = self.root / "evaluator.py"
        self.candidate_dir = self.root / "candidate_programs"

        if not self.template_path.exists():
            raise FileNotFoundError(f"Missing template program: {self.template_path}")
        if not self.evaluator_path.exists():
            raise FileNotFoundError(f"Missing evaluator: {self.evaluator_path}")

        self.candidate_dir.mkdir(parents=True, exist_ok=True)

        self.prompts = GetPrompts(self.template_path)
        self.evaluator = load_module(self.evaluator_path, "fixed_evaluator")

        if not hasattr(self.evaluator, "evaluate"):
            raise AttributeError(
                f"Evaluator at {self.evaluator_path} does not define evaluate(program_path)"
            )

        print("[PROB] template_path =", self.template_path, flush=True)
        print("[PROB] evaluator_path =", self.evaluator_path, flush=True)
        print("[PROB] candidate_dir =", self.candidate_dir, flush=True)
        print("[PROB] evaluator_file =", getattr(self.evaluator, "__file__", None), flush=True)

    def _strip_code_fence(self, text: str) -> str:
        text = str(text).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _validate_region_code(self, code: str) -> None:
        required_header = "def build_candidate_net(input_dim, num_classes, hidden_dim=128, dropout=0.2):"
        if required_header not in code:
            raise ValueError("Candidate missing required function header.")

        banned = [
            "import ",
            "class ",
            "if __name__ ==",
            "print(",
        ]
        for pat in banned:
            if pat in code:
                raise ValueError(f"Candidate contains forbidden content: {pat}")

        if "return" not in code:
            raise ValueError("Candidate function has no return statement.")

    def _stitch(self, region_code: str) -> str:
        text = self.template_path.read_text(encoding="utf-8")
        if START not in text or END not in text:
            raise ValueError("Template program is missing evolve region tags.")

        s = text.index(START) + len(START)
        e = text.index(END)
        return text[:s] + "\n" + region_code.strip() + "\n" + text[e:]

    def _normalize_candidate(self, candidate):
        if isinstance(candidate, dict):
            code = candidate.get("code")
            algorithm = candidate.get("algorithm")

            # 优先取 code，没有就取 algorithm
            if code is not None and str(code).strip():
                text = str(code).strip()
            elif algorithm is not None and str(algorithm).strip():
                text = str(algorithm).strip()
            else:
                raise ValueError(f"Empty candidate dict: {candidate!r}")
        else:
            text = str(candidate).strip()

        text = self._strip_code_fence(text)
        self._validate_region_code(text)
        return text

    def _make_candidate_path(self, full_code: str) -> Path:
        digest = hashlib.md5(full_code.encode("utf-8")).hexdigest()[:10]
        return self.candidate_dir / f"candidate_{digest}.py"

    def _extract_score(self, result):
        if result is None:
            return -1e18

        if hasattr(result, "metrics"):
            metrics = getattr(result, "metrics", {}) or {}
            score = metrics.get("combined_score", -1e18)
            return float(score) if score is not None else -1e18

        if isinstance(result, dict):
            if "metrics" in result and isinstance(result["metrics"], dict):
                score = result["metrics"].get("combined_score", -1e18)
                return float(score) if score is not None else -1e18
            score = result.get("combined_score", -1e18)
            return float(score) if score is not None else -1e18

        return float(result)

    def evaluate(self, candidate):
        try:
            print("[PROB] entering evaluate()", flush=True)

            region_code = self._normalize_candidate(candidate)
            full_code = self._stitch(region_code)

            candidate_path = self._make_candidate_path(full_code)
            candidate_path.write_text(full_code, encoding="utf-8")

            print(f"[PROB] written candidate to: {candidate_path}", flush=True)
            print("[PROB] candidate region preview:", flush=True)
            print(region_code[:2000], flush=True)

            result = self.evaluator.evaluate(str(candidate_path))
            print(f"[PROB] evaluator returned: {result}", flush=True)

            score = self._extract_score(result)
            print(f"[PROB] combined_score = {score}", flush=True)
            return score

        except Exception as e:
            print("[PROB] exception in evaluate()", flush=True)
            print(repr(e), flush=True)
            print(traceback.format_exc(), flush=True)
            return -1e18


# 兼容旧名字
MNIST1DModelSearch = News20SimpleMLPSearch