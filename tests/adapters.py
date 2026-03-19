from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    all_full_ids = []
    all_response_starts = []
    all_response_ends = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        full_ids = prompt_ids + output_ids

        # In labels (= full[1:]), output tokens start at len(prompt_ids)-1
        response_start = len(prompt_ids) - 1
        response_end = len(prompt_ids) + len(output_ids) - 1

        all_full_ids.append(full_ids)
        all_response_starts.append(response_start)
        all_response_ends.append(response_end)

    # Pad full_ids to max_full_len first, then derive input_ids and labels
    max_full_len = max(len(x) for x in all_full_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max_full_len - 1

    def pad_right(seq: list, pad_val: int, target: int) -> list:
        return seq + [pad_val] * (target - len(seq))

    padded_full = [pad_right(x, pad_id, max_full_len) for x in all_full_ids]

    input_ids_list = [x[:-1] for x in padded_full]
    labels_list = [x[1:] for x in padded_full]
    response_masks = [
        [1 if start <= j < end else 0 for j in range(max_len)]
        for start, end in zip(all_response_starts, all_response_ends)
    ]

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "response_mask": torch.tensor(response_masks, dtype=torch.long),
    }


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses,
    normalized by the group size.
    """
    n_total = len(rollout_responses)

    # Score all responses
    raw_rewards_list = []
    format_rewards_list = []
    answer_rewards_list = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, gt)
        raw_rewards_list.append(reward_dict["reward"])
        format_rewards_list.append(reward_dict.get("format_reward", float("nan")))
        answer_rewards_list.append(reward_dict.get("answer_reward", float("nan")))

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    advantages = torch.zeros_like(raw_rewards)

    # Group-normalize
    n_groups = n_total // group_size
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group_rewards = raw_rewards[start:end]
        mean = group_rewards.mean()
        if normalize_by_std:
            std = group_rewards.std()
            advantages[start:end] = (group_rewards - mean) / (std + advantage_eps)
        else:
            advantages[start:end] = group_rewards - mean

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": sum(format_rewards_list) / len(format_rewards_list),
        "mean_answer_reward": sum(answer_rewards_list) / len(answer_rewards_list),
    }

    return advantages, raw_rewards, metadata


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": token_log_probs.float()}
    if return_token_entropy:
        result["token_entropy"] = run_compute_entropy(logits).float()
    return result


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1):
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            the policy gradient per-token loss.
    """
    return -policy_log_probs * raw_rewards_or_advantages


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss."""
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    clip_fraction = ((ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)).float()
    metadata = {"clip_fraction": clip_fraction}

    return loss, metadata


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        return run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        return run_compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        return run_compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.
    """
    masked = tensor * mask
    if dim is None:
        return masked.sum() / mask.sum()
    else:
        return masked.sum(dim=dim) / mask.sum(dim=dim)


def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the SFT loss and backprop its gradients for a microbatch.

    Formula: sum token log-probs per sequence (divided by normalize_constant),
    average over batch, negate, then divide by gradient_accumulation_steps.
    """
    # Per-sequence: sum over response tokens / normalize_constant
    per_seq = run_masked_normalize(
        policy_log_probs, response_mask, dim=1, normalize_constant=normalize_constant
    )
    # Average over batch, negate (NLL), scale for gradient accumulation
    loss = -per_seq.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {}


def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch."""
    per_token_loss, metadata = run_compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    # Per-sequence: mean over response tokens, then average over batch
    per_seq = run_masked_mean(per_token_loss, response_mask, dim=1)
    loss = per_seq.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, metadata


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.
    """
    masked = tensor * mask
    if dim is None:
        return masked.sum() / normalize_constant
    else:
        return masked.sum(dim=dim) / normalize_constant


"""
The below adapters are used in the optional
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    raise NotImplementedError
