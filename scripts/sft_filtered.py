import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

"""
SFT filtered-data experiment (Section 4.2):
  1. Run base model on GSM8K train (question_only, greedy) to collect `target_size`
     correct examples (scans up to `scan_size` examples)
  2. Train SFT on those filtered examples
  3. Compare to unfiltered-1024 baseline from existing sft_size_sweep results (72.0%)

Usage:
    python scripts/sft_filtered.py
    python scripts/sft_filtered.py --target_size 1024 --scan_size 2000
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from alignment.drgrpo_grader import question_only_reward_fn
from alignment.training import load_gsm8k_sft_data, sft_train

QUESTION_ONLY_PROMPT = "{question}"


@torch.inference_mode()
def filter_correct_examples(
    model_path: str,
    train_jsonl: str,
    target_size: int = 1024,
    scan_size: int = 2000,
    batch_size: int = 8,
    max_new_tokens: int = 512,
) -> list[dict]:
    """Run base model on train set; collect first `target_size` correct examples
    by scanning up to `scan_size` examples."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # Load raw train examples (up to scan_size)
    raw = [json.loads(l) for l in open(train_jsonl, encoding="utf-8") if l.strip()]
    raw = raw[:scan_size]

    problems = []
    for ex in raw:
        m = re.search(r"####\s*(.+)", ex["answer"])
        final_ans = m.group(1).strip().replace(",", "") if m else ex["answer"]
        problems.append({"question": ex["question"], "answer": final_ans})

    # Also load SFT-formatted versions for training
    all_sft = load_gsm8k_sft_data(train_jsonl)
    all_sft = all_sft[:scan_size]

    prompts = [QUESTION_ONLY_PROMPT.format(question=p["question"]) for p in problems]

    print(f"[Filter] Scanning {len(prompts)} train examples (target: {target_size} correct)...")
    all_responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Filter inference"):
        batch = prompts[i: i + batch_size]
        tokenizer.padding_side = "left"
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id)
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        all_responses.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    del model
    torch.cuda.empty_cache()

    # Score and collect correct examples (preserve original order)
    correct_sft_examples = []
    total_correct = 0
    for i, (prob, resp) in enumerate(zip(problems, all_responses)):
        scores = question_only_reward_fn(resp, prob["answer"])
        if scores["reward"] == 1.0:
            total_correct += 1
            if len(correct_sft_examples) < target_size:
                correct_sft_examples.append(all_sft[i])

    print(f"[Filter] Base model correct: {total_correct}/{len(problems)} = {total_correct/len(problems):.1%}")
    print(f"[Filter] Collected {len(correct_sft_examples)} filtered examples "
          f"(target={target_size}, scanned={len(problems)})")
    return correct_sft_examples


@torch.inference_mode()
def quick_eval(model_path: str, test_examples: list[dict],
               batch_size: int = 8, max_new_tokens: int = 512,
               num_eval: int = 200) -> dict:
    """Evaluate model on test set with r1_zero format."""
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
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
        scores = r1_zero_reward_fn(resp, ex["answer"])
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


def load_unfiltered_baseline(sweep_results_path: str, size: int) -> dict | None:
    """Load existing size-sweep result for a given training size."""
    path = Path(sweep_results_path)
    if not path.exists():
        return None
    sweep = json.loads(path.read_text(encoding="utf-8"))
    for entry in sweep:
        if entry["size"] == size:
            return entry
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",      default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data",      default="data/gsm8k/train.jsonl")
    parser.add_argument("--test_data",       default="data/gsm8k/test.jsonl")
    parser.add_argument("--output_dir",      default="models/sft-filtered-1k")
    parser.add_argument("--results_path",    default="results/sft_filtered_comparison.json")
    parser.add_argument("--sweep_results",   default="results/sft_size_sweep.json",
                        help="Existing size-sweep results to read unfiltered baseline from.")
    parser.add_argument("--target_size",     type=int, default=1024,
                        help="Number of correct examples to collect for filtered training.")
    parser.add_argument("--scan_size",       type=int, default=2000,
                        help="Max number of train examples to scan during filtering.")
    parser.add_argument("--epochs",          type=int, default=2)
    parser.add_argument("--lr",              type=float, default=1e-5)
    parser.add_argument("--grad_accum",      type=int, default=8)
    parser.add_argument("--num_eval",        type=int, default=200)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--skip_filter",     action="store_true",
                        help="Skip filtering step (reuse cached filtered examples).")
    parser.add_argument("--cached_filter",   default="results/sft_filtered_examples.json",
                        help="Cache file for filtered examples.")
    args = parser.parse_args()

    Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)
    results = {}

    # --- Step 1: Filter ---
    cached_path = Path(args.cached_filter)
    if args.skip_filter and cached_path.exists():
        print(f"[Filter] Loading cached filtered examples from {cached_path}")
        filtered = json.loads(cached_path.read_text(encoding="utf-8"))
    else:
        filtered = filter_correct_examples(
            args.base_model, args.train_data,
            target_size=args.target_size,
            scan_size=args.scan_size,
        )
        cached_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Filter] Cached {len(filtered)} filtered examples to {cached_path}")

    n_filtered = len(filtered)
    if n_filtered < args.target_size:
        print(f"[Warning] Only collected {n_filtered} correct examples "
              f"(target={args.target_size}). Proceeding with {n_filtered}.")

    # --- Load test data ---
    raw_test = [json.loads(l) for l in open(args.test_data, encoding="utf-8") if l.strip()]
    test_examples = []
    for ex in raw_test:
        m = re.search(r"####\s*(.+)", ex["answer"])
        final = m.group(1).strip().replace(",", "") if m else ex["answer"]
        test_examples.append({"problem": ex["question"], "answer": final})

    # --- Step 2: Train on filtered data ---
    print(f"\n{'='*60}")
    print(f"Training on FILTERED {n_filtered} examples...")
    print(f"{'='*60}")
    t0 = time.time()
    filtered_model_path = sft_train(
        model_path=args.base_model,
        train_examples=filtered,
        output_dir=args.output_dir,
        lr=args.lr,
        epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
        save_every_n_steps=99999,
        log_every_n_steps=max(1, n_filtered // (8 * 5)),
        seed=args.seed,
        wandb_project="grpo-math-sft-filtered",
        wandb_run_name=f"sft-filtered-{n_filtered}",
    )
    filtered_time = time.time() - t0

    print(f"\nEvaluating filtered model...")
    filtered_metrics = quick_eval(filtered_model_path, test_examples, num_eval=args.num_eval)
    results["filtered"] = {
        "n_train": n_filtered,
        "scan_size": args.scan_size,
        "model_path": filtered_model_path,
        "train_time_min": round(filtered_time / 60, 1),
        **filtered_metrics,
    }

    # --- Step 3: Load unfiltered baseline (no re-training needed) ---
    baseline = load_unfiltered_baseline(args.sweep_results, args.target_size)
    if baseline is not None:
        print(f"\n[Baseline] Loaded unfiltered-{args.target_size} from {args.sweep_results}")
        results["unfiltered"] = {
            "n_train": baseline["size"],
            "model_path": baseline["model_path"],
            "train_time_min": baseline["train_time_min"],
            "accuracy": baseline["accuracy"],
            "format_acc": baseline["format_acc"],
            "answer_acc": baseline["answer_acc"],
            "n": baseline["n"],
            "source": "existing sweep result",
        }
    else:
        print(f"\n[Warning] No unfiltered-{args.target_size} baseline found in {args.sweep_results}")

    # --- Save & print summary ---
    with open(args.results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.results_path}")

    print("\n" + "=" * 60)
    print("FILTERED vs UNFILTERED SFT COMPARISON")
    print("=" * 60)
    print(f"{'':15} | {'N train':>8} | {'Accuracy':>8} | {'Format%':>8} | {'Answer%':>8}")
    print("-" * 65)
    for key in ["filtered", "unfiltered"]:
        if key not in results:
            continue
        r = results[key]
        print(f"{key:15} | {r['n_train']:>8} | {r['accuracy']:>7.1%} | "
              f"{r['format_acc']:>7.1%} | {r['answer_acc']:>7.1%}")

    if "unfiltered" in results:
        delta = results["filtered"]["accuracy"] - results["unfiltered"]["accuracy"]
        sign = "+" if delta >= 0 else ""
        print(f"\nFiltered vs Unfiltered: {sign}{delta:.1%}")


if __name__ == "__main__":
    main()
