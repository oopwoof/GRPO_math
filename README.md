# Post-Training Pipeline: SFT → GRPO for Math Reasoning

A complete post-training pipeline implemented from scratch on the MATH dataset, taking Qwen2.5-Math-1.5B from a base model to a math reasoning model via SFT, Expert Iteration, and GRPO.

## Highlights

- **Full pipeline**: Zero-shot baseline → SFT → Expert Iteration → GRPO, each phase building on the last
- **Substantial accuracy gains**: Base zero-shot 19% → SFT 70.5% on GSM8K (r1_zero format); data-efficient scaling from 128 to 1024 examples shows log-linear improvement (61.5% → 72.0%)
- **GRPO with multiple loss variants**: `no_baseline` (REINFORCE), `reinforce_with_baseline` (group mean), `grpo_clip` (PPO-style IS-ratio clipping), and off-policy support
- **Full training diagnostics**: entropy (mode collapse detection), gradient norm, clip fraction, format/answer reward separation, logged to TensorBoard + CSV + W&B

## Pipeline

```
Base Model (Qwen2.5-Math-1.5B)
    │
    ▼
Zero-shot Eval          ← establish baseline, analyze failure modes
    │
    ▼
SFT                     ← supervised fine-tuning on MATH train split
    │
    ▼
Expert Iteration        ← filter self-generated correct rollouts, retrain
    │
    ▼
GRPO                    ← group relative policy optimization with reward shaping
```

## What's Implemented

| Module | Description |
|--------|-------------|
| `alignment/training.py` | SFT and GRPO training loops with gradient accumulation, mixed precision, full diagnostics |
| `tests/adapters.py` | Core tensor ops: `masked_mean`, `masked_normalize`, log-prob computation, entropy, all PG loss variants |
| `alignment/drgrpo_grader.py` | Reward functions: format checking + answer grading via string match → SymPy → LaTeX equivalence |
| `alignment/prompts/` | Prompt templates for each training phase (`r1_zero`, `alpaca_sft`, `question_only`) |
| `scripts/train_sft.py` | SFT training script with configurable dataset size, epochs, LR |
| `scripts/train_grpo.py` | GRPO training script with full ablation flags (loss type, std normalization, off-policy steps, LR) |
| `scripts/sft_size_sweep.py` | Automated data scaling sweep (128 → 1024 examples) |
| `scripts/eval_zero_shot.py` | Zero-shot evaluation on GSM8K/MATH with configurable prompt format |

## Experiment Results

### SFT Data Scaling (GSM8K, r1_zero format)

| Training Examples | Format Acc | Answer Acc | Train Time |
|:-----------------:|:----------:|:----------:|:----------:|
| 128               | 92.0%      | 61.5%      | 17.7 min   |
| 256               | 96.5%      | 63.5%      | 26.6 min   |
| 512               | 98.0%      | 67.0%      | 32.0 min   |
| 1024              | 99.0%      | 72.0%      | 57.9 min   |

**Finding**: Log-linear scaling — each 2× data adds ~3–5% answer accuracy. Format compliance saturates quickly (~99% by 1k examples). Even 128 examples yields 61.5% accuracy.

### Zero-shot Baseline

| Format | Acc | Format Compliance |
|--------|-----|-------------------|
| `question_only` (boxed) | 60.5% | 81.5% |
| `r1_zero` (think/answer tags) | 19.0% | 52.0% |

The base model handles `\boxed{}` format well but struggles with structured `<think>`/`<answer>` tags — motivating the SFT phase.

### GRPO Loss Variants (ablation)

| Loss Type | Description |
|-----------|-------------|
| `no_baseline` | REINFORCE: `loss = -log_π(a) * r` |
| `reinforce_with_baseline` | Group mean subtracted: `loss = -log_π(a) * (r - mean(r))` |
| `grpo_clip` | PPO-style: `loss = -min(ratio * adv, clip(ratio, 1±ε) * adv)` |

Off-policy support: reuse rollout buffer N steps before regenerating (configurable via `--off_policy_steps`).

## Setup

Uses `uv` for dependency management. `flash-attn` requires a two-step install:

```bash
uv sync --no-install-package flash-attn
uv sync
```

Run tests:

```bash
uv run pytest
```

### GRPO Training

```bash
# Default: grpo_clip, on-policy
python scripts/train_grpo.py

# Ablations
python scripts/train_grpo.py --loss_type no_baseline
python scripts/train_grpo.py --loss_type reinforce_with_baseline
python scripts/train_grpo.py --no_std_norm          # disable std normalization in group rewards
python scripts/train_grpo.py --off_policy_steps 4   # reuse rollouts 4× before regenerating
python scripts/train_grpo.py --lr 1e-5
```

### SFT Training

```bash
python scripts/train_sft.py --n_examples 1024 --n_epochs 2 --lr 1e-5

# Data scaling sweep
python scripts/sft_size_sweep.py
```

## Hardware Notes

Tested on NVIDIA RTX 3070 (8 GB VRAM). For full GRPO training (vLLM rollouts + optimizer), an A100 40 GB or equivalent is recommended.

| Task | Est. VRAM |
|------|-----------|
| Unit tests (no model) | CPU only |
| Zero-shot eval, fp16 | ~4 GB |
| SFT, bs=1, grad ckpt, fp16 | ~6–7 GB |
| Full GRPO (vLLM + train) | ~12–16 GB |
