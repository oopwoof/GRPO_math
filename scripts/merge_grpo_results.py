"""
Merge GRPO sweep result files from multiple pods into one unified JSON.

Usage:
    python scripts/merge_grpo_results.py \
        --inputs results/grpo_sweep_results_pod_a.json \
                 results/grpo_sweep_results_pod_b.json \
                 results/grpo_sweep_results_pod_c.json \
        --output results/grpo_sweep_results.json

De-duplicates by run_name (last file wins for conflicts).
Prints a summary table: experiment / run_name / val_reward / train_time_min.
"""

import argparse
import json
from pathlib import Path


def load_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        print(f"  WARNING: {path} is empty — skipping.")
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        print(f"  WARNING: {path} does not contain a JSON list — skipping.")
        return []
    return data


def merge(files: list[Path]) -> list[dict]:
    seen: dict[str, dict] = {}  # run_name → result dict
    for f in files:
        results = load_file(f)
        for r in results:
            key = r.get("run_name", "")
            if key in seen:
                print(f"  Duplicate run_name '{key}' in {f.name} — overwriting previous entry.")
            seen[key] = r
    # Return sorted by experiment then run_name for deterministic order
    return sorted(seen.values(), key=lambda r: (r.get("experiment", ""), r.get("run_name", "")))


def print_table(results: list[dict]) -> None:
    print()
    print("=" * 76)
    print("GRPO SWEEP SUMMARY")
    print("=" * 76)
    print(f"{'Experiment':<32} {'Run':<25} {'Val%':>6} {'Time':>8}")
    print("-" * 76)
    current_exp = None
    for r in results:
        exp = r.get("experiment", "?")
        if exp != current_exp:
            if current_exp is not None:
                print()
            current_exp = exp
        run = r.get("run_name", "?")
        val = r.get("val_reward", float("nan"))
        t   = r.get("train_time_min", float("nan"))
        val_str = f"{val:.1%}" if isinstance(val, (int, float)) else str(val)
        t_str   = f"{t:.1f}m"  if isinstance(t,   (int, float)) else str(t)
        print(f"{exp:<32} {run:<25} {val_str:>6} {t_str:>8}")
    print("=" * 76)
    print(f"Total runs: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Merge GRPO sweep result files")
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Input JSON files (one per pod)",
    )
    parser.add_argument(
        "--output", default="results/grpo_sweep_results.json",
        help="Output merged JSON path",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            print(f"  WARNING: {p} does not exist — skipping.")

    existing = [p for p in input_paths if p.exists()]
    if not existing:
        print("ERROR: No input files found.")
        raise SystemExit(1)

    print(f"Merging {len(existing)} file(s): {[str(p) for p in existing]}")
    merged = merge(existing)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMerged {len(merged)} runs → {out}")

    print_table(merged)


if __name__ == "__main__":
    main()
