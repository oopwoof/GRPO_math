import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # triton stub breaks dynamo on Windows

"""
Zero-shot baseline evaluation on MATH dataset.

Usage:
    # Evaluate 100 examples with question_only format (fast baseline)
    python scripts/eval_zero_shot.py --model_path models/Qwen2.5-Math-1.5B --num_examples 100

    # Evaluate with r1_zero format (tests <think>/<answer> compliance)
    python scripts/eval_zero_shot.py --model_path models/Qwen2.5-Math-1.5B --prompt_format r1_zero

    # Full validation set
    python scripts/eval_zero_shot.py --model_path models/Qwen2.5-Math-1.5B --num_examples -1
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make sure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

# ── Prompt templates ──────────────────────────────────────────────────────────

QUESTION_ONLY_PROMPT = "{question}"

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

PROMPT_FORMATS = {
    "question_only": (QUESTION_ONLY_PROMPT, question_only_reward_fn),
    "r1_zero":       (R1_ZERO_PROMPT,       r1_zero_reward_fn),
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_math_dataset(split: str, num_examples: int, data_dir: str | None = None):
    """Load MATH/GSM8K dataset from local jsonl or HuggingFace."""
    # 1. Try local jsonl (assignment-provided format or GSM8K)
    if data_dir:
        path = Path(data_dir) / f"{split}.jsonl"
        if path.exists():
            examples = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            # Normalize GSM8K format: {question, answer} → {problem, answer}
            normalized = []
            for ex in examples:
                if "question" in ex and "problem" not in ex:
                    # GSM8K: extract final answer after ####
                    raw_ans = ex.get("answer", "")
                    import re
                    m = re.search(r"####\s*(.+)", raw_ans)
                    final_ans = m.group(1).strip().replace(",", "") if m else raw_ans
                    normalized.append({"problem": ex["question"], "answer": final_ans})
                else:
                    normalized.append(ex)
            print(f"Loaded {len(normalized)} examples from {path}")
            if num_examples > 0:
                normalized = normalized[:num_examples]
            return normalized

    # 2. Fall back to HuggingFace datasets
    try:
        from datasets import load_dataset
        print("Loading hendrycks/competition_math from HuggingFace...")
        hf_split = "test" if split == "validation" else split
        ds = load_dataset("hendrycks/competition_math", split=hf_split, trust_remote_code=True)
        examples = [{"problem": ex["problem"], "solution": ex["solution"]} for ex in ds]
        print(f"Loaded {len(examples)} examples from HuggingFace ({hf_split} split)")
        if num_examples > 0:
            examples = examples[:num_examples]
        return examples
    except Exception as e:
        raise RuntimeError(
            f"Could not load MATH dataset. Tried local ({data_dir}) and HuggingFace.\n"
            f"Install datasets: pip install datasets\nOriginal error: {e}"
        )


def get_ground_truth(example: dict) -> str:
    """Extract the ground truth answer from a MATH example."""
    # Assignment format: {"problem": ..., "answer": ...}
    if "answer" in example:
        return example["answer"]
    # HuggingFace format: solution contains \boxed{answer}
    import re
    solution = example.get("solution", "")
    m = re.search(r"\\boxed\{(.+?)\}", solution)
    return m.group(1) if m else solution


# ── Generation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> list[str]:
    """Batch-generate responses for a list of prompts."""
    model.eval()
    all_responses = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]

        # Tokenize with left-padding so all sequences end at the same position
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy for reproducibility
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens (exclude the prompt)
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        all_responses.extend(decoded)

    return all_responses


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    # Load model
    print(f"Loading model from {args.model_path} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    print(f"Model loaded on {device}.")

    # Load data
    examples = load_math_dataset(args.split, args.num_examples, args.data_dir)
    print(f"Evaluating {len(examples)} examples (split={args.split}).")

    # Build prompts
    prompt_template, reward_fn = PROMPT_FORMATS[args.prompt_format]
    prompts = [prompt_template.format(question=ex["problem"]) for ex in examples]

    # For r1_zero, the prompt already ends with "<think>" — we continue generation from there
    # The response we get back will be the continuation (reasoning + answer)
    responses_raw = generate_responses(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    # For r1_zero, reconstruct the full expected response format
    if args.prompt_format == "r1_zero":
        # Prepend "<think>" back so the grader sees the full "<think>...</think> <answer>..." string
        responses = ["<think>" + r for r in responses_raw]
    else:
        responses = responses_raw

    # Score
    results = []
    for ex, response in zip(examples, responses):
        gt = get_ground_truth(ex)
        scores = reward_fn(response, gt)
        results.append({
            "problem": ex["problem"][:80] + "...",
            "response_snippet": response[:120] + "...",
            "ground_truth": gt,
            **scores,
        })

    # Aggregate
    n = len(results)
    acc          = sum(r["reward"] for r in results) / n
    format_acc   = sum(r["format_reward"] for r in results) / n
    answer_acc   = sum(r["answer_reward"] for r in results) / n

    print("\n" + "=" * 60)
    print(f"  Prompt format : {args.prompt_format}")
    print(f"  Examples      : {n}")
    print(f"  Accuracy      : {acc:.1%}  (reward=1 means correct)")
    print(f"  Format acc    : {format_acc:.1%}")
    print(f"  Answer acc    : {answer_acc:.1%}")
    print("=" * 60)

    # Optionally save per-example results
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {n} results to {out_path}")

    return acc


def main():
    parser = argparse.ArgumentParser(description="Zero-shot MATH eval")
    parser.add_argument("--model_path",    default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--split",         default="test", choices=["train", "test", "validation"])
    parser.add_argument("--num_examples",  type=int, default=200,
                        help="Number of examples to evaluate. -1 = full split.")
    parser.add_argument("--prompt_format", default="question_only",
                        choices=["question_only", "r1_zero"],
                        help="question_only: just the math problem (lenient grader). "
                             "r1_zero: requires <think>/<answer> tags (strict grader).")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--data_dir",       default=None,
                        help="Path to local dir with train.jsonl / test.jsonl")
    parser.add_argument("--output",         default=None,
                        help="Path to save per-example JSON results")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
