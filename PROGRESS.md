# GRPO Math — Training Log

## Experiment Index

| Date | Phase | Key Config | Val acc | Notes |
|------|-------|-----------|-------------|-------|
| 2026-03-17 | Setup | Implemented all core functions | N/A | All tensor/loss functions pass snapshot tests |
| 2026-03-17 | Zero-shot | GSM8K test, question_only, 200 ex | 60.5% | Base model strong on GSM8K; format_acc=81.5% |
| 2026-03-17 | Zero-shot | GSM8K test, r1_zero, 200 ex | 19.0% | format_acc=52% — base model doesn't reliably follow `<think>`/`<answer>` |
| 2026-03-17 | SFT v1 | 1k examples, 2 epochs, lr=1e-5 | 0.0% | Bug: `</think><answer>` missing space, all format checks fail |
| 2026-03-17 | SFT v2 | 1k examples, 2 epochs, lr=1e-5 | **70.5%** | Fixed format bug; format_acc=98.5%, answer_acc=70.5% (r1_zero) |
| 2026-03-17 | SFT size sweep | 128–1024, 2 epochs, lr=1e-5, r1_zero | 61.5%→72.0% | format_acc 92%→99%; log-linear data scaling |
| 2026-03-18 | SFT full dataset | 7473 examples, 2 epochs, lr=1e-5, r1_zero | **73.0%** | format_acc ~99.5%; full dataset scaling confirmed |

---

## 2026-03-17 — Phase: Implementation (Step 1 & 2)

### What Was Implemented

All core training functions in `tests/adapters.py`:

**Tensor utilities**
- `run_masked_mean`: `sum(tensor * mask, dim) / sum(mask, dim)` — mean over masked elements
- `run_masked_normalize`: `sum(tensor * mask, dim) / normalize_constant` — sum then divide by constant

**SFT path**
- `run_tokenize_prompt_and_output`: tokenizes prompt+output, builds response_mask (1 for output tokens in labels)
- `run_compute_entropy`: `H = -sum(softmax(logits) * log_softmax(logits), dim=-1)`
- `run_get_response_log_probs`: forward pass → log_softmax → gather at label positions
- `run_sft_microbatch_train_step`: `loss = -masked_normalize(log_probs, mask, dim=1).mean() / grad_accum`

**GRPO path**
- `run_compute_group_normalized_rewards`: per-group `(r - mean) / (std + eps)` [or without std]
- `run_compute_naive_policy_gradient_loss`: `-log_probs * rewards` (REINFORCE)
- `run_compute_grpo_clip_loss`: PPO-style `loss = -min(ratio * adv, clip(ratio) * adv)`
- `run_compute_policy_gradient_loss`: dispatcher for all 3 loss types
- `run_grpo_microbatch_train_step`: `loss = masked_mean(per_token_loss, mask, dim=1).mean() / grad_accum`

### Key Design Insights

**SFT vs GRPO normalization differ intentionally:**
- SFT uses `masked_normalize` (sum / constant): supports Dr. GRPO's length-invariant normalization
- GRPO uses `masked_mean` (sum / count): standard per-token averaging
- Both then do `.mean()` over batch dimension (divide by batch_size) and divide by `gradient_accumulation_steps`

**Gradient verification** (derived analytically):
- SFT grad at masked pos: `-1 / (normalize_constant × batch_size × grad_accum) = -0.25` ✓
- GRPO grad at unclipped pos: `per_token_grad / (count_per_seq × batch_size × grad_accum)` ✓

### Status
- [x] 模型相关测试可在本地跑（下载至 `models/Qwen2.5-Math-1.5B/`，conftest 已支持本地路径）
**Section 3 — Zero-shot Baseline**
- [x] eval 脚本（transformers 版）已跑，结果序列化
- [ ] evaluate_vllm 版本
- [ ] 三类统计分析文字（各举 ≥10 例）

**Section 3 — Zero-shot 分析**
- [x] eval 脚本 + 200 examples 已跑
- [x] 三类统计 + 12例×3类分析 → `analysis/section3_zero_shot_analysis.md`

**Section 4 — SFT**
- [x] SFT 训练循环（loss/lr/entropy → TensorBoard + CSV）
- [x] SFT 1k examples → **70.5% acc**
- [x] Size sweep 完成 → 61.5% / 63.5% / 67.0% / **72.0%**（图：`figures/sft_size_sweep.png`）
- [x] `scripts/sft_filtered.py` 实现
- [x] full dataset SFT 完成 → **73.0%** (7473 examples, 2 epochs)
- [ ] 过滤实验（4.2）⏳ 运行中（1000 base dataset，对比 filtered vs unfiltered）

**Section 5 — Expert Iteration**
- [x] `expert_iteration_train()` 实现（`alignment/training.py`）
- [x] `scripts/train_ei.py` CLI
- [ ] 实际跑 EI（需 GPU 空闲）

**Section 7 — GRPO 实现**
- [x] 所有 unit test 通过
- [x] `grpo_train()` + off-policy 支持（`alignment/training.py`）
- [x] `scripts/train_grpo.py` CLI（所有消融参数）
- [x] `scripts/grpo_sweep.py`（9 项消融自动化，支持 resume）

**Section 8 — GRPO 实验（9 项）**
- [ ] grpo_learning_rate（LR sweep）
- [ ] grpo_baselines（no_baseline vs reinforce）
- [x] think_about_length_normalization → `analysis/section8_length_normalization.md`
- [ ] grpo_length_normalization（需 GPU）
- [ ] grpo_group_standard_deviation（需 GPU）
- [x] grpo_off_policy 代码 ✓（`off_policy_steps` 参数）
- [ ] grpo_off_policy_sweep（需 GPU）
- [ ] grpo_off_policy_clip_ablation（需 GPU）
- [ ] grpo_prompt_ablation（需 GPU）

**绘图**
- [x] `scripts/plot_results.py`（支持所有 Section 4/8/9 图）
- [x] `figures/sft_size_sweep.png` ✓
- [x] `figures/sft_full_training_curve.png`（实时，训练中）

**Section 9 — Leaderboard**
- [ ] 租 GPU（2×H100） + 完整流程 + val acc vs wall-clock 截图

---

## 2026-03-17 — Phase: GRPO Train Loop Implementation (Section 7)

### What Was Implemented

**`alignment/training.py` — `grpo_train()`**
- Full GRPO training loop with all diagnostics
- Rollout generation via `generate_rollouts()` (transformers, greedy + temperature sampling)
- Group-normalized advantages via `run_compute_group_normalized_rewards`
- Policy gradient via `run_grpo_microbatch_train_step` (dispatches to no_baseline / reinforce / grpo_clip)
- **Logged per step**: loss, lr, grad_norm, token_entropy, mean_reward, clip_fraction, val_reward
- **Output**: TensorBoard + CSV + optional wandb (same pattern as SFT loop)

**Off-policy support**
- `off_policy_steps=1` → on-policy (regenerate every step)
- `off_policy_steps=N` → reuse rollout buffer N times, then regenerate
- old_log_probs computed once per buffer fill; IS ratio = exp(current - old) for clip loss

**`scripts/train_grpo.py`** — CLI runner with all ablation flags:
```bash
# Default: grpo_clip, 200 steps
python scripts/train_grpo.py

# Ablations
python scripts/train_grpo.py --loss_type no_baseline
python scripts/train_grpo.py --no_std_norm          # grpo_group_standard_deviation ablation
python scripts/train_grpo.py --off_policy_steps 4   # off-policy sweep
python scripts/train_grpo.py --lr 1e-5              # LR sweep
```

**`scripts/sft_filtered.py`** — Section 4.2 filtered SFT:
- Runs base model on train set (question_only), filters correct examples
- Trains SFT on filtered subset, compares to equal-size unfiltered SFT
- Caches filtered examples to `results/sft_filtered_examples.json`

### SFT Size Sweep Results (complete)

| Size | Format% | Answer% | Time(m) |
|------|---------|---------|---------|
| 128  | 92.0%   | 61.5%   | 17.7    |
| 256  | 96.5%   | 63.5%   | 26.6    |
| 512  | 98.0%   | 67.0%   | 32.0    |
| 1024 | 99.0%   | 72.0%   | 57.9    |

**Finding**: Log-linear scaling — each 2× data adds ~3–5% answer accuracy. Format accuracy saturates quickly (~99% by 1k). Data efficiency is excellent: even 128 examples gives 61.5% on GSM8K.

---

## 2026-03-17 — Debug: test_tokenize_prompt_and_output & test_get_response_log_probs

### 背景
在本地 WSL 环境首次跑模型相关测试（需要下载 Qwen2.5-Math-1.5B），遇到两个 snapshot 不匹配的错误。

---

### Bug 1：tokenize padding 顺序错误

**现象**
```
input_ids[0][7]: 151643 (ACTUAL), 0 (DESIRED)
```
ACTUAL 多了一个 EOS padding，但 DESIRED 在该位置是真实 token `!`（id=0）。

**根因**
原实现：`full_ids[:-1]` 先切掉最后一个 token，再 pad 到 `max_len`。
'Hello, world!' 只有 4+4=8 个 token，切后 7 个，pad 两个 → `[..., 151643, 151643]`。
但 'This is a test.' 有 5+5=10 个 token，切后 9 个（max_len=9），无需 padding。
正确做法：**先把 full_ids pad 到 `max_full_len`，再统一取 `[:-1]` 和 `[1:]`**，这样短序列的 padding 正好落在最末尾一位，而不会挤掉真实 token。

**修复**
```python
# 错误：先 slice 再 pad
input_ids = full_ids[:-1]  # 7 tokens
padded = pad(input_ids, max_len=9)  # → [..., PAD, PAD]

# 正确：先 pad 再 slice
padded_full = pad(full_ids, max_full_len=10)  # → [..., 0, PAD]
input_ids = padded_full[:-1]  # 9 tokens, 末尾只有 1 个 PAD
```

**关键发现**：Qwen2.5-Math 没有 BOS，且 `pad_token_id == eos_token_id == 151643`。`add_special_tokens=True/False` 对这个 tokenizer 没有差异。需直接 decode snapshot 内容逆推预期格式，比猜测更高效。

---

### Bug 2：模型 bfloat16 导致数值超差

**现象**
```
TypeError: Got unsupported ScalarType BFloat16  （第一轮）
AssertionError: max abs diff = 0.407  atol=0.01  （加了 .float() 之后）
```

**根因**
Qwen2.5-Math 模型 config 里 `torch_dtype: bfloat16`，`from_pretrained` 默认以 bfloat16 加载。snapshot 由 float32 生成，两者数值有系统性偏差（bfloat16 精度约 2 位有效十进制位），超出容差。

**修复**
在 `conftest.py` 的 `model` fixture 里加 `torch_dtype=torch.float32`，强制以 float32 加载，与 snapshot 精度对齐。

---

### 反思

1. **先看 snapshot，再猜实现**：遇到 snapshot 不匹配时，第一步应该直接 `np.load` + decode 看清楚期望值的结构，而不是在脑子里推导。花了很多时间在纯推理上，最后还是靠 decode token 内容才找到 bug。

2. **tokenizer 属性要实测**：假设 `pad_token_id=0` 或 `add_special_tokens` 有效果，都应该先用一行代码验证，不要依赖直觉。

3. **snapshot 精度问题要想到硬件差异**：模型测试出现"数值接近但超差"时，先排查 dtype（bfloat16 vs float32），这是跨机器/跨配置跑 snapshot 测试的常见坑。
