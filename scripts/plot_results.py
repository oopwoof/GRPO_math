"""
Plotting scripts for all CS336 Section 4/8/9 figures.

Usage:
    # Plot SFT size sweep (Section 4.1)
    python scripts/plot_results.py --plot sft_size_sweep

    # Plot GRPO training curves from a metrics CSV (Section 8)
    python scripts/plot_results.py --plot grpo_curves --csv models/grpo/metrics.csv

    # Plot GRPO ablation comparison bar chart
    python scripts/plot_results.py --plot grpo_ablation --results results/grpo_sweep_results.json

    # Plot off-policy sweep (Section 8.6) — val reward vs steps + vs wall-clock
    python scripts/plot_results.py --plot off_policy_sweep

    # Plot all available figures
    python scripts/plot_results.py --plot all
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def save(fig, name: str):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)


def load_csv_metrics(csv_path: str) -> dict[str, list]:
    """Load training metrics CSV into dict of lists."""
    import csv
    data: dict[str, list] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, [])
                try:
                    data[k].append(float(v) if v else float("nan"))
                except ValueError:
                    data[k].append(v)
    return data


# ── Plot: SFT size sweep (Section 4.1) ────────────────────────────────────────

def plot_sft_size_sweep(results_path="results/sft_size_sweep.json"):
    if not Path(results_path).exists():
        print(f"[sft_size_sweep] {results_path} not found, skipping.")
        return

    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    sizes = [r["size"] for r in data]
    answer_acc = [r["answer_acc"] * 100 for r in data]
    format_acc = [r["format_acc"] * 100 for r in data]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, answer_acc, "o-", color="#2196F3", linewidth=2,
            markersize=7, label="Answer accuracy")
    ax.plot(sizes, format_acc, "s--", color="#FF9800", linewidth=1.5,
            markersize=6, label="Format accuracy")

    # Annotate points
    for s, a in zip(sizes, answer_acc):
        ax.annotate(f"{a:.1f}%", (s, a), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("SFT dataset size (# examples)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("SFT Dataset Size Sweep — GSM8K Test (r1_zero format)", fontsize=12)
    ax.set_ylim(50, 105)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(sizes)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "sft_size_sweep")


# ── Plot: SFT training loss curve ─────────────────────────────────────────────

def plot_sft_training_curve(csv_path: str, label: str = "SFT", out_name: str = "sft_training_curve"):
    if not Path(csv_path).exists():
        print(f"[sft_training_curve] {csv_path} not found, skipping.")
        return

    d = load_csv_metrics(csv_path)
    steps = d.get("step", [])
    loss = d.get("loss", [])
    entropy = d.get("token_entropy", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(steps, loss, color="#2196F3", linewidth=1.5)
    ax1.set_xlabel("Optimizer step")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{label} — Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, entropy, color="#9C27B0", linewidth=1.5)
    ax2.set_xlabel("Optimizer step")
    ax2.set_ylabel("Token entropy (nats)")
    ax2.set_title(f"{label} — Token Entropy")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, out_name)


# ── Plot: GRPO training curves ─────────────────────────────────────────────────

def plot_grpo_curves(csv_path: str, label: str = "GRPO", out_name: str = "grpo_training_curves"):
    if not Path(csv_path).exists():
        print(f"[grpo_curves] {csv_path} not found, skipping.")
        return

    d = load_csv_metrics(csv_path)
    steps = d.get("step", [])

    metrics = [
        ("loss",           "Loss",              "#2196F3"),
        ("mean_reward",    "Mean reward",        "#4CAF50"),
        ("grad_norm",      "Gradient norm",      "#FF5722"),
        ("token_entropy",  "Token entropy",      "#9C27B0"),
        ("clip_fraction",  "Clip fraction",      "#FF9800"),
        ("val_reward",     "Val reward",         "#F44336"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, title, color) in zip(axes.flat, metrics):
        vals = d.get(key, [float("nan")] * len(steps))
        # Filter out nan for val_reward (only logged every N steps)
        if key == "val_reward":
            valid = [(s, v) for s, v in zip(steps, vals) if not np.isnan(v)]
            if valid:
                vs, vv = zip(*valid)
                ax.plot(vs, vv, "o-", color=color, linewidth=2, markersize=5)
        else:
            ax.plot(steps, vals, color=color, linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Step", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label} — Training Diagnostics", fontsize=13)
    fig.tight_layout()
    save(fig, out_name)


# ── Plot: GRPO ablation comparison ────────────────────────────────────────────

def plot_grpo_ablation(results_path="results/grpo_sweep_results.json",
                       experiment_filter=None):
    if not Path(results_path).exists():
        print(f"[grpo_ablation] {results_path} not found, skipping.")
        return

    all_results = json.loads(Path(results_path).read_text(encoding="utf-8"))

    # Group by experiment
    groups: dict[str, list] = {}
    for r in all_results:
        exp = r["experiment"]
        if experiment_filter and exp not in experiment_filter:
            continue
        groups.setdefault(exp, []).append(r)

    if not groups:
        print("[grpo_ablation] No matching results, skipping.")
        return

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    for ax, (exp_name, runs) in zip(axes, groups.items()):
        run_names = [r["run_name"] for r in runs]
        val_rewards = [r["val_reward"] * 100 for r in runs]
        colors = plt.cm.tab10(np.linspace(0, 0.8, len(runs)))

        bars = ax.bar(range(len(runs)), val_rewards, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(run_names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Val accuracy (%)")
        ax.set_title(exp_name.replace("_", " "), fontsize=10)
        ax.set_ylim(0, max(val_rewards) * 1.15 if val_rewards else 100)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, val_rewards):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("GRPO Ablation Results — GSM8K Val Accuracy", fontsize=13)
    fig.tight_layout()
    save(fig, "grpo_ablation_summary")


# ── Plot: off-policy sweep (two figures as required by §8.6) ──────────────────

def plot_off_policy_sweep(sweep_dir="models/grpo-sweep/grpo_off_policy_sweep"):
    """Plot val reward vs steps AND vs wall-clock time for each off_policy_steps value."""
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.exists():
        print(f"[off_policy_sweep] {sweep_dir} not found, skipping.")
        return

    runs = sorted(sweep_dir.iterdir()) if sweep_dir.is_dir() else []
    run_data = []
    for run_dir in runs:
        csv_path = run_dir / "metrics.csv"
        if not csv_path.exists():
            continue
        d = load_csv_metrics(str(csv_path))
        run_data.append({"name": run_dir.name, "data": d})

    if not run_data:
        print("[off_policy_sweep] No metrics CSVs found, skipping.")
        return

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(run_data)))

    # Figure 1: val reward vs optimizer steps
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for rd, color in zip(run_data, colors):
        d = rd["data"]
        steps = d.get("step", [])
        val_r = d.get("val_reward", [float("nan")] * len(steps))
        valid = [(s, v) for s, v in zip(steps, val_r) if not np.isnan(v)]
        if valid:
            vs, vv = zip(*valid)
            ax1.plot(vs, [x * 100 for x in vv], "o-", color=color,
                     linewidth=2, markersize=5, label=rd["name"])

    ax1.set_xlabel("Optimizer steps", fontsize=11)
    ax1.set_ylabel("Val reward (%)", fontsize=11)
    ax1.set_title("Off-Policy Sweep — Val Reward vs Steps", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    save(fig1, "grpo_off_policy_vs_steps")

    # Figure 2: val reward vs wall-clock (approximate — use step as proxy if no timestamp)
    # If we had wall-clock time per step, we'd use it. For now, approximate from train_time.
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    results_path = Path("results/grpo_sweep_results.json")
    if results_path.exists():
        all_results = json.loads(results_path.read_text(encoding="utf-8"))
        op_results = [r for r in all_results if r["experiment"] == "grpo_off_policy_sweep"]
        if op_results:
            for rd, color in zip(run_data, colors):
                match = next((r for r in op_results if r["run_name"] == rd["name"]), None)
                if match:
                    # Approximate: assume uniform step time
                    d = rd["data"]
                    steps = d.get("step", [])
                    val_r = d.get("val_reward", [float("nan")] * len(steps))
                    total_steps = max(steps) if steps else 1
                    total_time = match["train_time_min"]
                    valid = [(s / total_steps * total_time * 60, v)
                             for s, v in zip(steps, val_r) if not np.isnan(v)]
                    if valid:
                        ts, vv = zip(*valid)
                        ax2.plot([t / 60 for t in ts], [x * 100 for x in vv], "o-",
                                 color=color, linewidth=2, markersize=5, label=rd["name"])

    ax2.set_xlabel("Wall-clock time (minutes)", fontsize=11)
    ax2.set_ylabel("Val reward (%)", fontsize=11)
    ax2.set_title("Off-Policy Sweep — Val Reward vs Wall-Clock", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    save(fig2, "grpo_off_policy_vs_wallclock")


# ── Main ───────────────────────────────────────────────────────────────────────

PLOT_FUNCTIONS = {
    "sft_size_sweep":       lambda args: plot_sft_size_sweep(),
    "sft_full_curve":       lambda args: plot_sft_training_curve(
        "models/sft-gsm8k-full/metrics.csv", "Full SFT (7473)", "sft_full_training_curve"),
    "sft_1k_curve":         lambda args: plot_sft_training_curve(
        "models/sft-gsm8k/metrics.csv", "SFT 1k", "sft_1k_training_curve"),
    "grpo_curves":          lambda args: plot_grpo_curves(
        args.csv or "models/grpo/metrics.csv", label=args.label or "GRPO",
        out_name=args.out or "grpo_training_curves"),
    "grpo_ablation":        lambda args: plot_grpo_ablation(
        results_path=args.results or "results/grpo_sweep_results.json"),
    "off_policy_sweep":     lambda args: plot_off_policy_sweep(),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", default="all",
                        help=f"Which plot(s) to generate. Options: all, "
                             f"{', '.join(PLOT_FUNCTIONS)}")
    parser.add_argument("--csv",     default=None, help="CSV path for grpo_curves")
    parser.add_argument("--results", default=None, help="JSON path for grpo_ablation")
    parser.add_argument("--label",   default=None, help="Label for grpo_curves title")
    parser.add_argument("--out",     default=None, help="Output filename stem for grpo_curves")
    args = parser.parse_args()

    to_run = list(PLOT_FUNCTIONS.keys()) if args.plot == "all" else [args.plot]

    for name in to_run:
        if name not in PLOT_FUNCTIONS:
            print(f"Unknown plot: {name}. Options: {list(PLOT_FUNCTIONS)}")
            continue
        print(f"\n[plot] {name}")
        PLOT_FUNCTIONS[name](args)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
