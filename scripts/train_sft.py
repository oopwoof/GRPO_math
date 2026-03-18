import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

"""
SFT training script on GSM8K.

Usage:
    python scripts/train_sft.py --num_examples 1000 --epochs 2
    python scripts/train_sft.py --num_examples -1 --epochs 3 --output_dir models/sft-gsm8k-full
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from alignment.training import load_gsm8k_sft_data, sft_train


def main():
    parser = argparse.ArgumentParser(description="SFT fine-tuning on GSM8K")
    parser.add_argument("--model_path",     default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--data_path",      default="data/gsm8k/train.jsonl")
    parser.add_argument("--output_dir",     default="models/sft-gsm8k")
    parser.add_argument("--num_examples",   type=int, default=1000,
                        help="Number of training examples. -1 = all (~7473).")
    parser.add_argument("--epochs",         type=int, default=2)
    parser.add_argument("--lr",             type=float, default=1e-5)
    parser.add_argument("--batch_size",     type=int, default=1)
    parser.add_argument("--grad_accum",     type=int, default=8,
                        help="Effective batch = batch_size × grad_accum.")
    parser.add_argument("--normalize_constant", type=float, default=1.0,
                        help="Dr. GRPO length normalization constant. 1.0 = sum of log-probs.")
    parser.add_argument("--max_seq_len",    type=int, default=512)
    parser.add_argument("--save_every",     type=int, default=500)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--wandb_project",  default="grpo-math-sft")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--use_wandb",      action="store_true", default=False)
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path} ...")
    examples = load_gsm8k_sft_data(args.data_path)
    if args.num_examples > 0:
        examples = examples[:args.num_examples]
    print(f"Using {len(examples)} training examples.")

    # Print a sample so we can verify format
    print("\n[Sample] prompt (last 80 chars):", repr(examples[0]["prompt"][-80:]))
    print("[Sample] response (first 80 chars):", repr(examples[0]["response"][:80]))
    print()

    t0 = time.time()
    final_model_dir = sft_train(
        model_path=args.model_path,
        train_examples=examples,
        output_dir=args.output_dir,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        normalize_constant=args.normalize_constant,
        max_seq_len=args.max_seq_len,
        save_every_n_steps=args.save_every,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_wandb=args.use_wandb,
    )
    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed/60:.1f} min")
    print(f"Model saved to: {final_model_dir}")
    print("\nNext: run eval_zero_shot.py on the saved model:")
    print(f"  python scripts/eval_zero_shot.py --model_path {final_model_dir} "
          f"--data_dir data/gsm8k --split test --num_examples 200 "
          f"--prompt_format r1_zero --output results/sft_eval.json")


if __name__ == "__main__":
    main()
