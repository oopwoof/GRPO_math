import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

"""
GRPO training script on GSM8K.

Usage:
    # Quick test (default: grpo_clip, 200 steps)
    python scripts/train_grpo.py

    # Ablation: no baseline (REINFORCE)
    python scripts/train_grpo.py --loss_type no_baseline --output_dir models/grpo-no-baseline

    # Off-policy (reuse rollouts 4 times before regenerating)
    python scripts/train_grpo.py --off_policy_steps 4 --output_dir models/grpo-offpolicy-4

    # LR sweep
    python scripts/train_grpo.py --lr 1e-5 --output_dir models/grpo-lr-1e5
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.training import grpo_train, load_gsm8k_rl_data


def load_test_examples(test_jsonl: str, num: int = 200) -> list[dict]:
    raw = [json.loads(l) for l in open(test_jsonl, encoding="utf-8") if l.strip()]
    examples = []
    for ex in raw[:num]:
        m = re.search(r"####\s*(.+)", ex["answer"])
        final = m.group(1).strip().replace(",", "") if m else ex["answer"]
        examples.append({"problem": ex["question"], "answer": final})
    return examples


def main():
    parser = argparse.ArgumentParser(description="GRPO training on GSM8K")
    # Paths
    parser.add_argument("--model_path",    default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data",    default="data/gsm8k/train.jsonl")
    parser.add_argument("--val_data",      default="data/gsm8k/test.jsonl",
                        help="Validation split (we use test for GSM8K)")
    parser.add_argument("--output_dir",    default="models/grpo")
    # Core GRPO
    parser.add_argument("--loss_type",     default="grpo_clip",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--group_size",    type=int, default=8)
    parser.add_argument("--cliprange",     type=float, default=0.2)
    parser.add_argument("--no_std_norm",   action="store_true",
                        help="Disable std normalization in advantage computation")
    parser.add_argument("--off_policy_steps", type=int, default=1,
                        help="Reuse rollout buffer this many steps before regenerating. "
                             "1 = on-policy. >1 = off-policy.")
    # Optimizer
    parser.add_argument("--lr",            type=float, default=5e-6)
    parser.add_argument("--total_steps",   type=int, default=200)
    parser.add_argument("--warmup_ratio",  type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # Data / generation
    parser.add_argument("--prompts_per_step", type=int, default=4)
    parser.add_argument("--temperature",   type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_seq_len",   type=int, default=512)
    parser.add_argument("--grad_accum",    type=int, default=8)
    # Val / logging
    parser.add_argument("--val_every",     type=int, default=10)
    parser.add_argument("--num_val",       type=int, default=100)
    parser.add_argument("--log_every",     type=int, default=5)
    parser.add_argument("--save_every",    type=int, default=100)
    parser.add_argument("--seed",          type=int, default=42)
    # wandb
    parser.add_argument("--wandb_project", default="grpo-math-gsm8k")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--use_wandb",     action="store_true", default=False)
    args = parser.parse_args()

    train_examples = load_gsm8k_rl_data(args.train_data)
    val_examples   = load_test_examples(args.val_data, num=args.num_val * 2)

    print(f"Train pool: {len(train_examples)} problems")
    print(f"Val pool:   {len(val_examples)} problems")

    final_dir = grpo_train(
        model_path=args.model_path,
        train_examples=train_examples,
        val_examples=val_examples,
        reward_fn=r1_zero_reward_fn,
        output_dir=args.output_dir,
        loss_type=args.loss_type,
        group_size=args.group_size,
        cliprange=args.cliprange,
        normalize_by_std=not args.no_std_norm,
        off_policy_steps=args.off_policy_steps,
        lr=args.lr,
        total_steps=args.total_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        prompts_per_step=args.prompts_per_step,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
        gradient_accumulation_steps=args.grad_accum,
        val_every_n_steps=args.val_every,
        num_val=args.num_val,
        log_every_n_steps=args.log_every,
        save_every_n_steps=args.save_every,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_wandb=args.use_wandb,
    )
    print(f"\nModel saved to: {final_dir}")


if __name__ == "__main__":
    main()
