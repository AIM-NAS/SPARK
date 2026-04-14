
import importlib.util
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "template_program.py"
EVALUATOR_PATH = ROOT / "base_evaluator.py"

START_TAG = "# ===== BEGIN EVOLVE PGN REGION ====="
END_TAG = "# ===== END EVOLVE PGN REGION ====="


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_seed_region() -> str:
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    s = text.index(START_TAG) + len(START_TAG)
    e = text.index(END_TAG)
    return text[s:e].strip()


def stitch_program(region_code: str) -> str:
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    s = text.index(START_TAG) + len(START_TAG)
    e = text.index(END_TAG)
    return text[:s] + "\n" + region_code.strip() + "\n" + text[e:]


def evaluate_region(region_code: str) -> float:
    evaluator = load_module(EVALUATOR_PATH, "base_evaluator")

    with tempfile.TemporaryDirectory() as td:
        candidate_path = Path(td) / "candidate_program.py"
        candidate_path.write_text(stitch_program(region_code), encoding="utf-8")

        result = evaluator.evaluate(str(candidate_path))
        score = float(result.metrics["combined_score"])
        return score


if __name__ == "__main__":
    seed = extract_seed_region()
    score = evaluate_region(seed)
    print("Seed score:", score)