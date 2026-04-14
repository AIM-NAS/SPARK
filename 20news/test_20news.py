import os
import json
import argparse

from evaluator import evaluate_stage1, evaluate_stage2, evaluate


def pretty_print_result(name, result):
    print(f"\n===== {name} =====")

    print("\n[metrics]")
    for k, v in result.metrics.items():
        print(f"{k}: {v}")

    print("\n[artifacts]")
    for k, v in result.artifacts.items():
        if isinstance(v, list):
            if len(v) > 10:
                print(f"{k}: [len={len(v)}] first5={v[:5]} last5={v[-5:]}")
            else:
                print(f"{k}: {v}")
        else:
            print(f"{k}: {v}")

    train_time = result.artifacts.get("train_time_sec", None)
    if train_time is not None:
        print(f"\n[summary] training_time_sec: {train_time:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--program",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "initial_program.py"),
        help="Path to candidate program containing class SimpleMLP",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["stage1", "stage2", "full"],
        help="Which evaluation mode to run",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="Optional path to save result as JSON",
    )
    args = parser.parse_args()

    program_path = os.path.abspath(args.program)
    print(f"Using program: {program_path}")

    if args.mode == "stage1":
        result = evaluate_stage1(program_path)
        pretty_print_result("STAGE1", result)

    elif args.mode == "stage2":
        result = evaluate_stage2(program_path)
        pretty_print_result("STAGE2", result)

    else:
        print("\nRunning stage1...")
        s1 = evaluate_stage1(program_path)
        pretty_print_result("STAGE1", s1)

        print("\nRunning full cascade (stage1 + stage2)...")
        result = evaluate(program_path)
        pretty_print_result("FINAL", result)

    if args.save_json:
        payload = {
            "metrics": result.metrics,
            "artifacts": result.artifacts,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved result to: {args.save_json}")


if __name__ == "__main__":
    main()