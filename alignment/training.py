"""
SFT + GRPO training loops for Qwen2.5-Math-1.5B on GSM8K.

Memory budget on RTX 3070 (8 GB):
  - Model bf16:              ~3 GB
  - Gradients bf16:          ~3 GB   (gradient checkpointing reduces activation memory)
  - Adafactor states:        ~0.5 GB (factored 2nd-moment, vs 12 GB for AdamW fp32)
  ─────────────────────────────────
  Total:                     ~6.5 GB  (fits with headroom)

Why Adafactor:
  AdamW (fp32 states) = 2 × params × 4 bytes = 2 × 1.5B × 4 = 12 GB → OOM.
  Adafactor stores factored row/col statistics ≈ O(√n) per weight matrix → ~0.5 GB.
"""

import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Literal, Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Adafactor,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.adapters import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_sft_microbatch_train_step,
    run_compute_group_normalized_rewards,
    run_grpo_microbatch_train_step,
)

# ── Prompt template ────────────────────────────────────────────────────────────

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

# ── Data loading ───────────────────────────────────────────────────────────────

def load_gsm8k_sft_data(data_path: str) -> list[dict]:
    """Load GSM8K jsonl and return list of (prompt_str, response_str) dicts.

    Prompt ends with 'Assistant: <think>' (model continues from there).
    Response = '{gsm8k_reasoning}</think><answer>{final_number}</answer>'

    The response_mask will only cover response tokens, so the model learns
    to produce structured reasoning + answer in r1_zero format.
    """
    examples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            question = ex["question"]
            raw_answer = ex["answer"]

            # Split at '#### <number>'
            m = re.search(r"####\s*(.+)", raw_answer)
            if m:
                final_answer = m.group(1).strip().replace(",", "")
                reasoning = raw_answer[: m.start()].strip()
            else:
                final_answer = raw_answer.strip()
                reasoning = ""

            prompt = R1_ZERO_PROMPT.format(question=question)
            # Response continues after "<think>" which is already in the prompt
            response = f"{reasoning}</think> <answer>{final_answer}</answer>"
            examples.append({"prompt": prompt, "response": response})

    return examples


# ── SFT training loop ─────────────────────────────────────────────────────────

def sft_train(
    model_path: str,
    train_examples: list[dict],          # [{"prompt": str, "response": str}]
    output_dir: str,
    lr: float = 1e-5,
    epochs: int = 2,
    batch_size: int = 1,                 # increase only if >8 GB VRAM available
    gradient_accumulation_steps: int = 8,
    normalize_constant: float = 1.0,     # Dr. GRPO length-norm constant; 1.0 = sum
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.05,
    save_every_n_steps: int = 500,
    log_every_n_steps: int = 20,
    max_seq_len: int = 512,              # truncate to this many tokens per example
    seed: int = 42,
    wandb_project: str = "grpo-math-sft",
    wandb_run_name: str | None = None,
    use_wandb: bool = False,
) -> str:
    """Train model with SFT. Returns path to saved final model directory."""
    random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Loggers ---
    tb_dir = output_dir / "tensorboard"
    tb_writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[SFT] TensorBoard → {tb_dir}")
    print(f"[SFT]   Run:  tensorboard --logdir {tb_dir}")

    csv_path = output_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "epoch", "loss", "lr", "token_entropy"])

    _wandb_run = None
    if use_wandb:
        try:
            import wandb
            _wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=dict(
                    model_path=model_path, num_examples=len(train_examples),
                    lr=lr, epochs=epochs, batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    normalize_constant=normalize_constant, seed=seed,
                ),
            )
            print("[SFT] wandb run started:", _wandb_run.url)
        except Exception as e:
            print(f"[SFT] wandb init failed ({e}), continuing without it.")

    print(f"[SFT] Loading model from {model_path} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    # Gradient checkpointing: recompute activations during backward → saves ~2 GB VRAM
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    # Adafactor: memory-efficient optimizer (factored 2nd-moment)
    # scale_parameter=False + relative_step=False → use explicit lr (like Adam)
    optimizer = Adafactor(
        model.parameters(),
        lr=lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    # Steps & scheduler
    steps_per_epoch = max(1, len(train_examples) // (batch_size * gradient_accumulation_steps))
    total_optimizer_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_optimizer_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    print(f"[SFT] {len(train_examples)} examples × {epochs} epochs")
    print(f"[SFT] batch={batch_size}, grad_accum={gradient_accumulation_steps}, "
          f"effective_batch={batch_size * gradient_accumulation_steps}")
    print(f"[SFT] total optimizer steps={total_optimizer_steps}, warmup={warmup_steps}")
    print(f"[SFT] normalize_constant={normalize_constant}, lr={lr}")

    global_microstep = 0
    optimizer_step = 0
    running_loss = 0.0

    for epoch in range(epochs):
        indices = list(range(len(train_examples)))
        random.shuffle(indices)
        shuffled = [train_examples[i] for i in indices]

        pbar = tqdm(
            range(0, len(shuffled), batch_size),
            desc=f"Epoch {epoch + 1}/{epochs}",
        )
        for batch_start in pbar:
            batch = shuffled[batch_start : batch_start + batch_size]
            prompt_strs  = [ex["prompt"]   for ex in batch]
            response_strs = [ex["response"] for ex in batch]

            # --- Tokenize (prompt + response) ---
            tokens = run_tokenize_prompt_and_output(prompt_strs, response_strs, tokenizer)

            # Truncate to max_seq_len to guard against very long examples
            input_ids    = tokens["input_ids"][:, :max_seq_len].to(device)
            labels       = tokens["labels"][:, :max_seq_len].to(device)
            response_mask = tokens["response_mask"][:, :max_seq_len].to(device)

            # Skip if no response tokens survived truncation
            if response_mask.sum() == 0:
                continue

            # --- Forward pass (bf16 autocast) ---
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                result = run_get_response_log_probs(
                    model, input_ids, labels, return_token_entropy=True
                )
            log_probs = result["log_probs"]          # float32
            token_entropy = result["token_entropy"]  # float32

            # --- Backward pass (loss already /= gradient_accumulation_steps inside) ---
            loss, _ = run_sft_microbatch_train_step(
                log_probs, response_mask, gradient_accumulation_steps, normalize_constant
            )
            running_loss += loss.item() * gradient_accumulation_steps

            global_microstep += 1

            # --- Optimizer step every gradient_accumulation_steps microbatches ---
            if global_microstep % gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                if optimizer_step % log_every_n_steps == 0:
                    avg = running_loss / (log_every_n_steps * gradient_accumulation_steps)
                    lr_now = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}", opt_step=optimizer_step)
                    # Token entropy: mean over response tokens in this window
                    avg_entropy = (
                        (token_entropy * response_mask).sum() / response_mask.sum().clamp(min=1)
                    ).item()
                    # TensorBoard
                    tb_writer.add_scalar("train/loss", avg, optimizer_step)
                    tb_writer.add_scalar("train/lr", lr_now, optimizer_step)
                    tb_writer.add_scalar("train/token_entropy", avg_entropy, optimizer_step)
                    # CSV
                    csv_writer.writerow([optimizer_step, epoch + 1, f"{avg:.6f}", f"{lr_now:.2e}", f"{avg_entropy:.4f}"])
                    csv_file.flush()
                    # wandb (optional)
                    if _wandb_run:
                        _wandb_run.log({"train/loss": avg, "train/lr": lr_now, "train/epoch": epoch + 1},
                                       step=optimizer_step)
                    running_loss = 0.0

                if save_every_n_steps > 0 and optimizer_step % save_every_n_steps == 0:
                    ckpt = output_dir / f"checkpoint-{optimizer_step}"
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"\n[SFT] Checkpoint saved → {ckpt}")

    # --- Save final model ---
    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    tb_writer.close()
    csv_file.close()
    if _wandb_run:
        _wandb_run.finish()

    print(f"\n[SFT] Done. Final model → {final_dir}")
    print(f"[SFT] Metrics CSV  → {csv_path}")
    print(f"[SFT] TensorBoard  → {tb_dir}")
    return str(final_dir)


# ── GRPO rollout generation (transformers, no vLLM) ───────────────────────────

@torch.inference_mode()
def generate_rollouts(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    group_size: int,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    batch_size: int = 4,
    device: str = "cuda",
) -> tuple[list[str], list[str]]:
    """Generate `group_size` rollouts per prompt.

    Returns:
        rollout_responses: list[str] of length len(prompts) * group_size
        repeated_prompts:  same, each prompt repeated group_size times
    """
    # Each prompt needs group_size completions — expand list first
    expanded_prompts = [p for p in prompts for _ in range(group_size)]

    all_responses = []
    model.eval()
    tokenizer.padding_side = "left"

    for i in range(0, len(expanded_prompts), batch_size):
        batch = expanded_prompts[i: i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        all_responses.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    # Prepend "<think>" since prompt already ends with it
    rollout_responses = ["<think>" + r for r in all_responses]
    return rollout_responses, expanded_prompts


# ── GRPO val evaluation ────────────────────────────────────────────────────────

@torch.inference_mode()
def grpo_quick_val(
    model: torch.nn.Module,
    tokenizer,
    val_examples: list[dict],
    reward_fn,
    device: str = "cuda",
    num_val: int = 100,
    max_new_tokens: int = 512,
    batch_size: int = 8,
) -> dict:
    """Greedy-decode on val set and score with reward_fn."""
    examples = val_examples[:num_val]
    prompts = [R1_ZERO_PROMPT.format(question=ex["problem"]) for ex in examples]
    model.eval()
    tokenizer.padding_side = "left"

    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        all_responses.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    responses = ["<think>" + r for r in all_responses]
    rewards = [reward_fn(resp, ex["answer"])["reward"]
               for resp, ex in zip(responses, examples)]
    return {
        "val_reward": sum(rewards) / len(rewards),
        "n_val": len(rewards),
    }


# ── GRPO training loop ─────────────────────────────────────────────────────────

def grpo_train(
    model_path: str,
    train_examples: list[dict],          # [{"problem": str, "answer": str}]
    val_examples: list[dict],            # [{"problem": str, "answer": str}]
    reward_fn,                           # callable(response, ground_truth) -> {"reward": float, ...}
    output_dir: str,
    # Core GRPO hyperparams
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "grpo_clip",
    group_size: int = 8,
    cliprange: float = 0.2,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
    # Off-policy: number of reuse steps before re-generating rollouts
    # 1 = on-policy (re-generate every step)
    off_policy_steps: int = 1,
    # Optimizer / scheduler
    lr: float = 5e-6,
    total_steps: int = 200,
    warmup_ratio: float = 0.05,
    max_grad_norm: float = 1.0,
    # Data / generation
    prompts_per_step: int = 4,           # number of distinct problems per optimizer step
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    max_seq_len: int = 512,
    rollout_batch_size: int = 4,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    # Validation
    val_every_n_steps: int = 10,
    num_val: int = 100,
    # Logging / saving
    log_every_n_steps: int = 5,
    save_every_n_steps: int = 100,
    seed: int = 42,
    wandb_project: str = "grpo-math",
    wandb_run_name: str | None = None,
    use_wandb: bool = False,
) -> str:
    """GRPO training loop. Returns path to saved final model."""
    random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Loggers ---
    tb_dir = output_dir / "tensorboard"
    tb_writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[GRPO] TensorBoard → {tb_dir}")

    csv_path = output_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step", "loss", "lr", "grad_norm", "token_entropy",
        "mean_reward", "clip_fraction", "val_reward",
    ])

    _wandb_run = None
    if use_wandb:
        try:
            import wandb
            _wandb_run = wandb.init(
                project=wandb_project, name=wandb_run_name,
                config=dict(
                    model_path=model_path, loss_type=loss_type,
                    group_size=group_size, cliprange=cliprange,
                    normalize_by_std=normalize_by_std,
                    off_policy_steps=off_policy_steps, lr=lr,
                    total_steps=total_steps, prompts_per_step=prompts_per_step,
                    temperature=temperature,
                ),
            )
            print("[GRPO] wandb run started:", _wandb_run.url)
        except Exception as e:
            print(f"[GRPO] wandb init failed ({e}), continuing without it.")

    # --- Load model ---
    print(f"[GRPO] Loading model from {model_path} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    optimizer = Adafactor(
        model.parameters(), lr=lr,
        scale_parameter=False, relative_step=False, warmup_init=False,
    )
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"[GRPO] loss_type={loss_type}, group_size={group_size}, off_policy_steps={off_policy_steps}")
    print(f"[GRPO] total_steps={total_steps}, prompts_per_step={prompts_per_step}, "
          f"effective_rollouts_per_step={prompts_per_step * group_size}")
    print(f"[GRPO] lr={lr}, normalize_by_std={normalize_by_std}, cliprange={cliprange}")

    # Shuffle train examples
    train_pool = list(train_examples)
    random.shuffle(train_pool)
    pool_idx = 0

    # Off-policy buffer
    rollout_buffer: list[dict] | None = None
    buffer_uses = 0

    running = {
        "loss": 0.0, "entropy": 0.0, "reward": 0.0, "clip_frac": 0.0, "count": 0
    }

    pbar = tqdm(range(1, total_steps + 1), desc="GRPO")
    for step in pbar:
        # --- Fetch problem batch ---
        batch_problems = []
        for _ in range(prompts_per_step):
            if pool_idx >= len(train_pool):
                random.shuffle(train_pool)
                pool_idx = 0
            batch_problems.append(train_pool[pool_idx])
            pool_idx += 1

        prompts = [R1_ZERO_PROMPT.format(question=ex["problem"]) for ex in batch_problems]
        ground_truths = [ex["answer"] for ex in batch_problems]

        # --- Rollout generation (possibly off-policy) ---
        if rollout_buffer is None or buffer_uses >= off_policy_steps:
            # Generate new rollouts
            model.eval()
            with torch.inference_mode():
                rollout_responses, _ = generate_rollouts(
                    model, tokenizer, prompts,
                    group_size=group_size,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    batch_size=rollout_batch_size,
                    device=device,
                )
            model.train()

            # Compute group-normalized rewards
            repeated_gts = [gt for gt in ground_truths for _ in range(group_size)]
            advantages, raw_rewards, reward_meta = run_compute_group_normalized_rewards(
                reward_fn=reward_fn,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_gts,
                group_size=group_size,
                advantage_eps=advantage_eps,
                normalize_by_std=normalize_by_std,
            )

            # Tokenize rollouts to get old_log_probs
            # Build prompt+response strings for tokenization
            rollout_prompt_strs = [
                R1_ZERO_PROMPT.format(question=p)
                for p in [batch_problems[i // group_size]["problem"]
                          for i in range(len(rollout_responses))]
            ]
            # Response is everything after "<think>" prefix (the generation)
            rollout_response_strs = [
                r[len("<think>"):] for r in rollout_responses
            ]

            tokens = run_tokenize_prompt_and_output(
                rollout_prompt_strs, rollout_response_strs, tokenizer
            )
            inp_ids = tokens["input_ids"][:, :max_seq_len].to(device)
            lbl_ids = tokens["labels"][:, :max_seq_len].to(device)
            resp_mask = tokens["response_mask"][:, :max_seq_len].to(device)

            # Get old log probs with no grad
            model.eval()
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                    old_result = run_get_response_log_probs(
                        model, inp_ids, lbl_ids, return_token_entropy=False
                    )
            old_log_probs_buf = old_result["log_probs"].detach()  # (N, seq)
            model.train()

            rollout_buffer = {
                "rollout_responses": rollout_responses,
                "advantages": advantages,
                "raw_rewards": raw_rewards,
                "reward_meta": reward_meta,
                "inp_ids": inp_ids,
                "lbl_ids": lbl_ids,
                "resp_mask": resp_mask,
                "old_log_probs": old_log_probs_buf,
            }
            buffer_uses = 0

        buffer_uses += 1
        buf = rollout_buffer
        advantages = buf["advantages"]
        raw_rewards = buf["raw_rewards"]
        reward_meta = buf["reward_meta"]
        inp_ids = buf["inp_ids"]
        lbl_ids = buf["lbl_ids"]
        resp_mask = buf["resp_mask"]
        old_log_probs_buf = buf["old_log_probs"]

        # --- Policy gradient step ---
        optimizer.zero_grad()
        total_loss = 0.0
        total_clip_frac = 0.0
        total_entropy = 0.0
        n_microbatches = 0

        N = inp_ids.shape[0]
        for mb_start in range(0, N, train_batch_size):
            mb_end = mb_start + train_batch_size
            mb_inp = inp_ids[mb_start:mb_end]
            mb_lbl = lbl_ids[mb_start:mb_end]
            mb_mask = resp_mask[mb_start:mb_end]
            mb_adv = advantages[mb_start:mb_end].unsqueeze(1).to(device)
            mb_raw = raw_rewards[mb_start:mb_end].unsqueeze(1).to(device)
            mb_old = old_log_probs_buf[mb_start:mb_end]

            if mb_mask.sum() == 0:
                continue

            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                result = run_get_response_log_probs(
                    model, mb_inp, mb_lbl, return_token_entropy=True
                )
            log_probs = result["log_probs"]
            token_entropy = result["token_entropy"]

            loss, metadata = run_grpo_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=mb_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                loss_type=loss_type,
                raw_rewards=mb_raw,
                advantages=mb_adv,
                old_log_probs=mb_old,
                cliprange=cliprange,
            )

            total_loss += loss.item() * gradient_accumulation_steps
            if "clip_fraction" in metadata:
                cf = (metadata["clip_fraction"] * mb_mask).sum() / mb_mask.sum().clamp(min=1)
                total_clip_frac += cf.item()
            avg_ent = ((token_entropy * mb_mask).sum() / mb_mask.sum().clamp(min=1)).item()
            total_entropy += avg_ent
            n_microbatches += 1

        grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm).item()
        optimizer.step()
        scheduler.step()

        # --- Validation ---
        val_reward = float("nan")
        if step % val_every_n_steps == 0:
            val_metrics = grpo_quick_val(
                model, tokenizer, val_examples, reward_fn,
                device=device, num_val=num_val,
            )
            val_reward = val_metrics["val_reward"]
            model.train()

        # --- Logging ---
        avg_loss = total_loss / max(n_microbatches, 1)
        avg_clip = total_clip_frac / max(n_microbatches, 1)
        avg_entropy = total_entropy / max(n_microbatches, 1)
        mean_reward = reward_meta["mean_reward"]
        lr_now = optimizer.param_groups[0]["lr"]

        running["loss"] += avg_loss
        running["entropy"] += avg_entropy
        running["reward"] += mean_reward
        running["clip_frac"] += avg_clip
        running["count"] += 1

        pbar.set_postfix(
            loss=f"{avg_loss:.3f}",
            rew=f"{mean_reward:.2f}",
            grad=f"{grad_norm:.2f}",
            val=f"{val_reward:.2f}" if not torch.isnan(torch.tensor(val_reward)) else "—",
        )

        if step % log_every_n_steps == 0:
            c = running["count"]
            tb_writer.add_scalar("train/loss", running["loss"] / c, step)
            tb_writer.add_scalar("train/lr", lr_now, step)
            tb_writer.add_scalar("train/grad_norm", grad_norm, step)
            tb_writer.add_scalar("train/token_entropy", running["entropy"] / c, step)
            tb_writer.add_scalar("train/mean_reward", running["reward"] / c, step)
            tb_writer.add_scalar("train/clip_fraction", running["clip_frac"] / c, step)
            if not torch.isnan(torch.tensor(val_reward)):
                tb_writer.add_scalar("val/reward", val_reward, step)

            csv_writer.writerow([
                step,
                f"{running['loss'] / c:.6f}",
                f"{lr_now:.2e}",
                f"{grad_norm:.4f}",
                f"{running['entropy'] / c:.4f}",
                f"{running['reward'] / c:.4f}",
                f"{running['clip_frac'] / c:.4f}",
                f"{val_reward:.4f}" if not torch.isnan(torch.tensor(val_reward)) else "",
            ])
            csv_file.flush()

            if _wandb_run:
                log_dict = {
                    "train/loss": running["loss"] / c,
                    "train/lr": lr_now,
                    "train/grad_norm": grad_norm,
                    "train/token_entropy": running["entropy"] / c,
                    "train/mean_reward": running["reward"] / c,
                    "train/clip_fraction": running["clip_frac"] / c,
                }
                if not torch.isnan(torch.tensor(val_reward)):
                    log_dict["val/reward"] = val_reward
                _wandb_run.log(log_dict, step=step)

            running = {"loss": 0.0, "entropy": 0.0, "reward": 0.0, "clip_frac": 0.0, "count": 0}

        if save_every_n_steps > 0 and step % save_every_n_steps == 0:
            ckpt = output_dir / f"checkpoint-{step}"
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"\n[GRPO] Checkpoint → {ckpt}")

    # --- Save final ---
    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    tb_writer.close()
    csv_file.close()
    if _wandb_run:
        _wandb_run.finish()

    print(f"\n[GRPO] Done. Final model → {final_dir}")
    print(f"[GRPO] Metrics CSV → {csv_path}")
    print(f"[GRPO] TensorBoard → {tb_dir}")
    return str(final_dir)


# ── GSM8K data loading for GRPO ────────────────────────────────────────────────

def load_gsm8k_rl_data(data_path: str) -> list[dict]:
    """Load GSM8K as [{"problem": str, "answer": str}] for GRPO/EI training."""
    examples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            m = re.search(r"####\s*(.+)", ex["answer"])
            final_answer = m.group(1).strip().replace(",", "") if m else ex["answer"]
            examples.append({"problem": ex["question"], "answer": final_answer})
    return examples
