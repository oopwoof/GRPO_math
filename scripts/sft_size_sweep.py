import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

"""
SFT dataset size sweep: trains models at multiple data sizes and evaluates each.

Usage:
    python scripts/sft_size_sweep.py
    python scripts/sft_size_sweep.py --sizes 128 256 512  # subset only
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from alignment.training import load_gsm8k_sft_data, sft_train

# Reuse eval logic from eval_zero_shot
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from alignment.drgrpo_grader import r1_zero_reward_fn

R1_ZERO_PROMPT = (
    "A conversation between User and Assistant. The User asks a question, and the "
    "Assistant solves it. The Assistant first thinks about the reasoning process in "
    "the mind and then provides the User with the answer. The reasoning process is "
    "enclosed within <think> </think> and answer is enclosed within <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>.\n"
    "User: {question}\n"
    "Assistant: <think>"
)


@torch.inference_mode()
def quick_eval(model_path: str, test_examples: list[dict], batch_size: int = 8,
               max_new_tokens: int = 512, num_eval: int = 200) -> dict:
    """Quick evaluation of a model on test examples."""
    import re
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    examples = test_examples[:num_eval]
    prompts = [R1_ZERO_PROMPT.format(question=ex["problem"]) for ex in examples]

    all_responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Eval", leave=False):
        batch = prompts[i: i + batch_size]
        tokenizer.padding_side = "left"
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id)
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        all_responses.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    responses = ["<think>" + r for r in all_responses]

    rewards, format_rewards, answer_rewards = [], [], []
    for ex, resp in zip(examples, responses):
        gt = ex["answer"]
        scores = r1_zero_reward_fn(resp, gt)
        rewards.append(scores["reward"])
        format_rewards.append(scores["format_reward"])
        answer_rewards.append(scores["answer_reward"])

    del model
    torch.cuda.empty_cache()

    return {
        "accuracy": sum(rewards) / len(rewards),
        "format_acc": sum(format_rewards) / len(format_rewards),
        "answer_acc": sum(answer_rewards) / len(answer_rewards),
        "n": len(rewards),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",   default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data",   default="data/gsm8k/train.jsonl")
    parser.add_argument("--test_data",    default="data/gsm8k/test.jsonl")
    parser.add_argument("--output_dir",   default="models/sft-sweep")
    parser.add_argument("--results_path", default="results/sft_size_sweep.json")
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[128, 256, 512, 1024],
                        help="Dataset sizes to sweep. Use -1 for full dataset.")
    parser.add_argument("--epochs",       type=int, default=2)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--grad_accum",   type=int, default=8)
    parser.add_argument("--num_eval",     type=int, default=200)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    # Load all data once
    print("Loading data...")
    import json as _json, re as _re
    all_train = load_gsm8k_sft_data(args.train_data)

    # Load test data in eval format
    test_raw = [_json.loads(l) for l in open(args.test_data, encoding="utf-8") if l.strip()]
    test_examples = []
    for ex in test_raw:
        m = _re.search(r"####\s*(.+)", ex["answer"])
        final = m.group(1).strip().replace(",", "") if m else ex["answer"]
        test_examples.append({"problem": ex["question"], "answer": final})

    results = []
    Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)

    for size in args.sizes:
        n = len(all_train) if size == -1 else min(size, len(all_train))
        train_subset = all_train[:n]

        print(f"\n{'='*60}")
        print(f"Training with {n} examples...")
        print(f"{'='*60}")

        output_subdir = str(Path(args.output_dir) / f"size-{n}")
        t0 = time.time()
        model_path = sft_train(
            model_path=args.base_model,
            train_examples=train_subset,
            output_dir=output_subdir,
            lr=args.lr,
            epochs=args.epochs,
            gradient_accumulation_steps=args.grad_accum,
            save_every_n_steps=99999,
            log_every_n_steps=max(1, n // (8 * 5)),  # ~5 log points per run
            seed=args.seed,
            wandb_project="grpo-math-sft-sweep",
            wandb_run_name=f"sft-size-{n}",
        )
        train_time = time.time() - t0

        print(f"\nEvaluating size={n} model...")
        metrics = quick_eval(model_path, test_examples, num_eval=args.num_eval)

        result = {
            "size": n,
            "model_path": model_path,
            "train_time_min": round(train_time / 60, 1),
            **metrics,
        }
        results.append(result)

        print(f"\n[size={n}] accuracy={metrics['accuracy']:.1%}, "
              f"format={metrics['format_acc']:.1%}, "
              f"answer={metrics['answer_acc']:.1%}")

        # Save intermediate results
        with open(args.results_path, "w", encoding="utf-8") as f:
            _json.dump(results, f, indent=2)
        print(f"Saved to {args.results_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(f"{'Size':>8} | {'Format%':>8} | {'Answer%':>8} | {'Time(m)':>8}")
    print("-" * 45)
    for r in results:
        print(f"{r['size']:>8} | {r['format_acc']:>7.1%} | {r['answer_acc']:>7.1%} | {r['train_time_min']:>7.1f}")


if __name__ == "__main__":
    main()
