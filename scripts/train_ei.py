import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

"""
Expert Iteration training script on GSM8K (Section 5).

Algorithm:
  For n_ei_steps iterations:
    1. Sample prompts_per_step problems
    2. Generate group_size rollouts per problem
    3. Keep correct rollouts (reward=1) as SFT training data
    4. Fine-tune model on correct rollouts for sft_epochs
    5. Evaluate on val set and log

Usage:
    python scripts/train_ei.py
    python scripts/train_ei.py --model_path models/sft-gsm8k-full/final  # start from SFT
    python scripts/train_ei.py --n_ei_steps 3 --group_size 16 --prompts_per_step 32
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.training import expert_iteration_train, load_gsm8k_rl_data


def load_val_examples(test_jsonl: str, num: int = 200) -> list[dict]:
    raw = [json.loads(l) for l in open(test_jsonl, encoding="utf-8") if l.strip()]
    examples = []
    for ex in raw[:num]:
        m = re.search(r"####\s*(.+)", ex["answer"])
        final = m.group(1).strip().replace(",", "") if m else ex["answer"]
        examples.append({"problem": ex["question"], "answer": final})
    return examples


def main():
    parser = argparse.ArgumentParser(description="Expert Iteration on GSM8K")
    parser.add_argument("--model_path",       default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data",       default="data/gsm8k/train.jsonl")
    parser.add_argument("--val_data",         default="data/gsm8k/test.jsonl")
    parser.add_argument("--output_dir",       default="models/ei-gsm8k")
    # EI hyperparams
    parser.add_argument("--n_ei_steps",       type=int, default=5)
    parser.add_argument("--group_size",       type=int, default=8,
                        help="Rollouts per problem per step")
    parser.add_argument("--prompts_per_step", type=int, default=64,
                        help="Problems sampled per EI iteration")
    # SFT inner loop
    parser.add_argument("--sft_epochs",       type=int, default=1)
    parser.add_argument("--sft_lr",           type=float, default=1e-5)
    parser.add_argument("--grad_accum",       type=int, default=8)
    parser.add_argument("--max_seq_len",      type=int, default=512)
    # Generation
    parser.add_argument("--temperature",      type=float, default=0.7)
    parser.add_argument("--max_new_tokens",   type=int, default=512)
    # Val / logging
    parser.add_argument("--num_val",          type=int, default=200)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--wandb_project",    default="grpo-math-ei")
    parser.add_argument("--wandb_run_name",   default=None)
    parser.add_argument("--use_wandb",        action="store_true", default=False)
    args = parser.parse_args()

    train_examples = load_gsm8k_rl_data(args.train_data)
    val_examples   = load_val_examples(args.val_data, num=args.num_val)

    print(f"Train pool: {len(train_examples)} problems")
    print(f"Val pool:   {len(val_examples)} problems")
    print(f"EI config:  {args.n_ei_steps} steps × {args.prompts_per_step} prompts × {args.group_size} rollouts")
    print(f"            = {args.n_ei_steps * args.prompts_per_step * args.group_size} total rollouts")

    final_dir = expert_iteration_train(
        model_path=args.model_path,
        train_examples=train_examples,
        val_examples=val_examples,
        reward_fn=r1_zero_reward_fn,
        output_dir=args.output_dir,
        n_ei_steps=args.n_ei_steps,
        group_size=args.group_size,
        prompts_per_step=args.prompts_per_step,
        sft_epochs=args.sft_epochs,
        sft_lr=args.sft_lr,
        sft_grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_val=args.num_val,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_wandb=args.use_wandb,
    )
    print(f"\nFinal model: {final_dir}")


if __name__ == "__main__":
    main()
