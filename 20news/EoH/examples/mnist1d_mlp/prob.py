import importlib.util
import tempfile
import traceback
import textwrap
from pathlib import Path

from prompts import GetPrompts


START_TAG = "# ===== BEGIN EVOLVE PGN REGION ====="
END_TAG = "# ===== END EVOLVE PGN REGION ====="


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MNIST1DModelSearch:
    def __init__(self):
        self.root = Path(__file__).resolve().parent
        self.template_path = self.root / "template_program.py"
        self.evaluator_path = self.root / "evaluator.py"

        if not self.template_path.exists():
            raise FileNotFoundError(f"Missing template program: {self.template_path}")
        if not self.evaluator_path.exists():
            raise FileNotFoundError(f"Missing evaluator: {self.evaluator_path}")

        self.prompts = GetPrompts(self.template_path)
        self.evaluator = load_module(self.evaluator_path, "mnist1d_evaluator")

    def _stitch_region_into_template(self, region_code: str) -> str:
        text = self.template_path.read_text(encoding="utf-8")
        if START_TAG not in text or END_TAG not in text:
            raise ValueError("Template program is missing evolve region tags.")

        s = text.index(START_TAG) + len(START_TAG)
        e = text.index(END_TAG)

        region_code = region_code.strip("\n")
        return text[:s] + "\n" + region_code + "\n" + text[e:]

    def _extract_code_block(self, code_string: str) -> str:
        code_string = str(code_string).strip()

        if code_string.startswith("```"):
            lines = code_string.splitlines()
            if len(lines) >= 2:
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
            code_string = "\n".join(lines).strip()

        return code_string

    def _sanitize_text(self, text: str) -> str:
        text = "" if text is None else str(text)
        text = self._extract_code_block(text)
        return text.strip("\n")

    def _dedent_code(self, text: str) -> str:
        text = self._sanitize_text(text)
        text = text.replace("\t", "    ")
        text = textwrap.dedent(text)
        lines = [ln.rstrip() for ln in text.splitlines()]
        return "\n".join(lines).strip("\n")

    def _rebuild_method_block(self, code: str) -> str:
        """
        Rebuild a method-only block into proper class-level indentation.

        Rules:
        - lines starting with 'def ' become class-level methods (4 spaces)
        - lines after a method def become method body (8 spaces)
        - blank lines are preserved
        """
        code = self._dedent_code(code)
        lines = code.splitlines()

        rebuilt = []
        inside_method = False

        for raw in lines:
            stripped = raw.lstrip()

            if stripped == "":
                rebuilt.append("")
                continue

            if stripped.startswith("def "):
                rebuilt.append("    " + stripped)
                inside_method = True
            else:
                if inside_method:
                    rebuilt.append("        " + stripped)
                else:
                    # fallback: if code starts with non-def content, still place inside class
                    rebuilt.append("    " + stripped)

        return "\n".join(rebuilt).rstrip()

    def _candidate_to_region_code(self, candidate) -> str:
        """
        Accept either:
        1) a plain code string
        2) a dict like {"algorithm": "...", "code": "...", ...}

        Prefer using a full class directly if available.
        """
        # Case 1: plain string candidate
        if isinstance(candidate, str):
            code_string = self._sanitize_text(candidate)

            if "class ConvBase(nn.Module):" in code_string:
                return self._dedent_code(code_string)

            if "def __init__(" in code_string and "def forward(" in code_string:
                return "class ConvBase(nn.Module):\n" + self._rebuild_method_block(code_string)

            raise ValueError(
                "Candidate string is neither a full ConvBase class nor a valid pair of method bodies."
            )

        # Case 2: dict candidate
        if isinstance(candidate, dict):
            algorithm = self._sanitize_text(candidate.get("algorithm", ""))
            code = self._sanitize_text(candidate.get("code", ""))

            # If code itself already contains the full class, use it directly
            if "class ConvBase(nn.Module):" in code:
                return self._dedent_code(code)

            # If algorithm accidentally contains a full class, use it directly
            if "class ConvBase(nn.Module):" in algorithm and "def __init__(" in algorithm:
                return self._dedent_code(algorithm)

            # Common EoH case:
            # algorithm = "class ConvBase(nn.Module):"
            # code = "def __init__(...)\n...\ndef forward(...)\n..."
            if "class ConvBase(nn.Module):" in algorithm:
                if not code:
                    raise ValueError("Candidate dict has class header but empty code body.")
                return "class ConvBase(nn.Module):\n" + self._rebuild_method_block(code)

            # If code only has methods, wrap it
            if "def __init__(" in code and "def forward(" in code:
                return "class ConvBase(nn.Module):\n" + self._rebuild_method_block(code)

            raise ValueError(
                "Could not reconstruct valid region code from candidate dict. "
                f"algorithm={repr(algorithm[:200])}, code={repr(code[:200])}"
            )

        raise TypeError(f"Unsupported candidate type: {type(candidate)}")

    def _validate_region_code(self, region_code: str) -> None:
        banned_patterns = [
            "def build_model(",
            "if __name__ ==",
            "import torch",
            "from torch",
        ]
        for pat in banned_patterns:
            if pat in region_code:
                raise ValueError(
                    f"Candidate contains forbidden full-program content: {pat}. "
                    "Only evolve-region code is allowed."
                )

        required_patterns = [
            "class ConvBase(nn.Module):",
            "def __init__(",
            "def forward(",
        ]
        for pat in required_patterns:
            if pat not in region_code:
                raise ValueError(f"Candidate missing required pattern: {pat}")

        if "return logits" in region_code and "logits =" not in region_code:
            raise ValueError("Candidate returns undefined variable 'logits'.")

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

                score = result.metrics.get("combined_score", -1e18)
                print(f"[PROB] combined_score = {score}", flush=True)

                if score is None:
                    return -1e18
                return float(score)

        except Exception as e:
            print("[PROB] exception in evaluate()", flush=True)
            print(repr(e), flush=True)
            print(traceback.format_exc(), flush=True)
            return -1e18