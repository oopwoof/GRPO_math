import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

"""
SFT filtered-data experiment (Section 4.2):
  1. Run base model on GSM8K train set (question_only, greedy) to find correct examples
  2. Train SFT on filtered (correct-only) examples
  3. Compare to unfiltered SFT of the same size

Usage:
    python scripts/sft_filtered.py
    python scripts/sft_filtered.py --num_filter_eval 500  # faster filter step
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
    num_eval: int = -1,
    batch_size: int = 8,
    max_new_tokens: int = 512,
) -> list[dict]:
    """Run base model on train set; return (prompt, response) pairs for correct examples."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # Load raw train examples
    raw = [json.loads(l) for l in open(train_jsonl, encoding="utf-8") if l.strip()]
    if num_eval > 0:
        raw = raw[:num_eval]

    # Build (question, answer) pairs
    problems = []
    for ex in raw:
        m = re.search(r"####\s*(.+)", ex["answer"])
        final_ans = m.group(1).strip().replace(",", "") if m else ex["answer"]
        problems.append({"question": ex["question"], "answer": final_ans})

    prompts = [QUESTION_ONLY_PROMPT.format(question=p["question"]) for p in problems]

    print(f"[Filter] Generating on {len(prompts)} train examples with base model...")
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

    # Score and filter
    correct, total = 0, len(problems)
    correct_sft_examples = []

    # Also load SFT-formatted versions for training
    all_sft = load_gsm8k_sft_data(train_jsonl)
    if num_eval > 0:
        all_sft = all_sft[:num_eval]

    for i, (prob, resp) in enumerate(zip(problems, all_responses)):
        scores = question_only_reward_fn(resp, prob["answer"])
        if scores["reward"] == 1.0:
            correct += 1
            correct_sft_examples.append(all_sft[i])

    print(f"[Filter] Base model correct: {correct}/{total} = {correct/total:.1%}")
    print(f"[Filter] Filtered training set size: {len(correct_sft_examples)}")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",      default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data",      default="data/gsm8k/train.jsonl")
    parser.add_argument("--test_data",       default="data/gsm8k/test.jsonl")
    parser.add_argument("--output_dir",      default="models/sft-filtered")
    parser.add_argument("--results_path",    default="results/sft_filtered.json")
    parser.add_argument("--num_filter_eval", type=int, default=-1,
                        help="How many train examples to run filtering on. -1 = all.")
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
            num_eval=args.num_filter_eval,
        )
        cached_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Filter] Cached {len(filtered)} filtered examples to {cached_path}")

    n_filtered = len(filtered)

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
        output_dir=str(Path(args.output_dir) / "filtered"),
        lr=args.lr,
        epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
        save_every_n_steps=99999,
        log_every_n_steps=max(1, n_filtered // (8 * 5)),
        seed=args.seed,
        wandb_project="grpo-math-sft-filtered",
        wandb_run_name="sft-filtered",
    )
    filtered_time = time.time() - t0

    print(f"\nEvaluating filtered model...")
    filtered_metrics = quick_eval(filtered_model_path, test_examples, num_eval=args.num_eval)
    results["filtered"] = {
        "n_train": n_filtered,
        "model_path": filtered_model_path,
        "train_time_min": round(filtered_time / 60, 1),
        **filtered_metrics,
    }

    # --- Step 3: Train on unfiltered data of same size (first n_filtered examples) ---
    all_sft = load_gsm8k_sft_data(args.train_data)
    unfiltered_subset = all_sft[:n_filtered]

    print(f"\n{'='*60}")
    print(f"Training on UNFILTERED {n_filtered} examples (baseline comparison)...")
    print(f"{'='*60}")
    t0 = time.time()
    unfiltered_model_path = sft_train(
        model_path=args.base_model,
        train_examples=unfiltered_subset,
        output_dir=str(Path(args.output_dir) / "unfiltered"),
        lr=args.lr,
        epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
        save_every_n_steps=99999,
        log_every_n_steps=max(1, n_filtered // (8 * 5)),
        seed=args.seed,
        wandb_project="grpo-math-sft-filtered",
        wandb_run_name="sft-unfiltered",
    )
    unfiltered_time = time.time() - t0

    print(f"\nEvaluating unfiltered model...")
    unfiltered_metrics = quick_eval(unfiltered_model_path, test_examples, num_eval=args.num_eval)
    results["unfiltered"] = {
        "n_train": n_filtered,
        "model_path": unfiltered_model_path,
        "train_time_min": round(unfiltered_time / 60, 1),
        **unfiltered_metrics,
    }

    # --- Summary ---
    with open(args.results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.results_path}")

    print("\n" + "=" * 60)
    print("FILTERED vs UNFILTERED SFT COMPARISON")
    print("=" * 60)
    print(f"{'':15} | {'Format%':>8} | {'Answer%':>8} | {'Time(m)':>8}")
    print("-" * 50)
    for key in ["filtered", "unfiltered"]:
        r = results[key]
        print(f"{key:15} | {r['format_acc']:>7.1%} | {r['answer_acc']:>7.1%} | {r['train_time_min']:>7.1f}")


if __name__ == "__main__":
    main()
