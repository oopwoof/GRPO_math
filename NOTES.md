# GRPO Math — 实现笔记

> 记录每个函数的设计思路、公式推导和关键 debug 经验。

---

## 目录

1. [项目结构](#1-项目结构)
2. [Tensor 工具函数](#2-tensor-工具函数)
3. [SFT 路径](#3-sft-路径)
4. [GRPO 路径](#4-grpo-路径)
5. [Zero-shot 评估脚本](#5-zero-shot-评估脚本)
6. [Debug 经验](#6-debug-经验)
7. [CS336 要求的指标与图表](#7-cs336-要求的指标与图表)

---

## 1. 项目结构

```
tests/adapters.py        ← 所有需要实现的函数（唯一需要改的文件）
alignment/
  drgrpo_grader.py       ← 已提供的 reward 函数（只读）
  prompts/               ← prompt 模板
scripts/
  eval_zero_shot.py      ← 零样本基线评估
```

### 三阶段训练流程

```
Base model (Qwen2.5-Math-1.5B, ~5% acc)
    │
    ▼ SFT（监督微调，用带推理链的正确样本）
    │
    ▼ Expert Iteration（生成 rollout → 过滤正确 → 继续 SFT）
    │
    ▼ GRPO（强化学习，目标 ≥25% acc）
```

---

## 2. Tensor 工具函数

### `run_masked_mean`

**用途**：对 tensor 取均值，但只计算 mask=1 的位置（忽略 prompt token 和 padding）。

```python
def run_masked_mean(tensor, mask, dim=None):
    masked = tensor * mask          # mask=0 的位置清零
    if dim is None:
        return masked.sum() / mask.sum()          # 全局：总和 / 有效元素数
    else:
        return masked.sum(dim=dim) / mask.sum(dim=dim)  # 沿某轴
```

**注意**：当某个切片内 mask 全为 0 时，结果为 `nan`（不添加 epsilon，与参考实现保持一致）。

**验证**：
- tensor 形状 (2,10,100)，mask ≈50% True，count ≈ 994
- `masked_mean_dimNone = -404.6 / 994 = -0.4070` ✓

---

### `run_masked_normalize`

**用途**：求和再除以**常数**（而非元素个数）。这是 Dr. GRPO 的归一化方式，避免长序列因 token 数多而梯度偏大。

```python
def run_masked_normalize(tensor, mask, dim=None, normalize_constant=1.0):
    masked = tensor * mask
    if dim is None:
        return masked.sum() / normalize_constant
    else:
        return masked.sum(dim=dim) / normalize_constant
```

**与 masked_mean 的区别**：

| | 除数 | 适用场景 |
|--|------|---------|
| `masked_mean` | 有效 token 数（变化） | GRPO loss |
| `masked_normalize` | 固定常数 | SFT loss（Dr. GRPO 风格） |

---

## 3. SFT 路径

### `run_tokenize_prompt_and_output`

**目标**：把 `(prompt, output)` 字符串对转为模型训练所需的三元组：

```
full_ids  = [p1, p2, p3, o1, o2, o3]      (prompt + output tokens)
input_ids = full_ids[:-1]                   (去掉最后一个)
labels    = full_ids[1:]                    (去掉第一个，即向左移一位)
response_mask[j] = 1  if j ≥ len(prompt)-1 else 0
```

**关键设计：先 pad 再 slice**

```python
# 错误做法：先切再 pad（短序列的 padding 位置错误）
input_ids = full_ids[:-1]          # 7 tokens
padded = pad(input_ids, 9)         # [t1...t7, PAD, PAD]  ← pad 占了真实位置

# 正确做法：先 pad 整个序列再切
padded_full = pad(full_ids, 10)    # [t1...t8, PAD, PAD]
input_ids = padded_full[:-1]       # [t1...t8, PAD]       ← pad 只在最末尾
labels    = padded_full[1:]        # [t2...t8, PAD, PAD]
```

**response_mask 的边界**：
- `response_start = len(prompt_ids) - 1`（在 labels 中，output 的第一个 token 对应位置）
- `response_end   = len(prompt_ids) + len(output_ids) - 1`
- `mask[j] = 1 if response_start <= j < response_end else 0`

**Qwen2.5-Math tokenizer 特性**（实测）：
- 没有 BOS token，`add_special_tokens=True/False` 无区别
- `pad_token_id == eos_token_id == 151643`

---

### `run_compute_entropy`

**公式**：$H = -\sum_v p_v \log p_v$

```python
def run_compute_entropy(logits):
    log_probs = F.log_softmax(logits, dim=-1)   # 数值稳定
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)      # 对 vocab 维度求和
```

**用途**：监控训练中的 entropy，entropy 过低说明模型坍缩（mode collapse），只会输出少数几种回答。

---

### `run_get_response_log_probs`

**作用**：给定 `input_ids` 和 `labels`，返回每个位置的条件 log 概率。

```python
def run_get_response_log_probs(model, input_ids, labels, return_token_entropy):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits                              # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)            # (B, T, V)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    result = {"log_probs": token_log_probs.float()}
    if return_token_entropy:
        result["token_entropy"] = run_compute_entropy(logits).float()
    return result
```

**注意**：
- `gather(-1, labels)` = 在 vocab 维度上取 `labels[i,j]` 处的值，即 $\log p(\text{label}_{i,j} | \text{context})$
- `.float()` 强制 float32，避免 bfloat16 精度不足导致 snapshot 测试失败

---

### `run_sft_microbatch_train_step`

**公式推导**（从 snapshot 反推）：

snapshot 中，所有 masked 位置的梯度恒为 `-0.25`。梯度 = $-1/K$，其中 $K=4$。

$$K = \text{normalize\_constant} \times \text{batch\_size} \times \text{grad\_accum} = 1.0 \times 2 \times 2 = 4$$

因此公式为：

$$\text{loss} = -\frac{1}{\text{grad\_accum}} \cdot \underbrace{\frac{1}{\text{batch\_size}} \sum_{i} \underbrace{\frac{\sum_j \log p_{ij} \cdot \text{mask}_{ij}}{\text{normalize\_constant}}}_{\text{masked\_normalize per seq}}}_{\text{batch mean}}$$

```python
def run_sft_microbatch_train_step(policy_log_probs, response_mask,
                                   gradient_accumulation_steps, normalize_constant=1.0):
    # 每条序列：sum(log_probs * mask) / normalize_constant
    per_seq = run_masked_normalize(policy_log_probs, response_mask,
                                   dim=1, normalize_constant=normalize_constant)
    loss = -per_seq.mean() / gradient_accumulation_steps   # batch 均值，再除以 grad_accum
    loss.backward()
    return loss, {}
```

**`normalize_constant` 的含义**：
- `= 1.0`（默认）：标准 SFT，loss = sum of log probs（不除 token 数）
- `= K`（如 42）：Dr. GRPO 变体，loss 除以常数 K，与序列长度无关

---

## 4. GRPO 路径

### `run_compute_group_normalized_rewards`

**原理**：GRPO 对同一问题采样 G 个答案，用组内相对好坏（advantage）代替绝对 reward。

$$a_i = \frac{r_i - \bar{r}_{\text{group}}}{\sigma_{\text{group}} + \varepsilon} \quad (\text{normalize\_by\_std=True})$$

$$a_i = r_i - \bar{r}_{\text{group}} \quad (\text{normalize\_by\_std=False})$$

```python
for g in range(n_groups):
    start, end = g * group_size, (g+1) * group_size
    group_rewards = raw_rewards[start:end]
    mean = group_rewards.mean()
    if normalize_by_std:
        advantages[start:end] = (group_rewards - mean) / (group_rewards.std() + eps)
    else:
        advantages[start:end] = group_rewards - mean
```

**直觉**：同一题目的 G 个 rollout 里，比组平均分好的得正 advantage（鼓励），差的得负 advantage（惩罚）。

---

### `run_compute_naive_policy_gradient_loss`

**REINFORCE 公式**：$\mathcal{L} = -\log\pi_\theta(a) \cdot r$（最大化期望 reward）

```python
def run_compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs):
    return -policy_log_probs * raw_rewards_or_advantages  # (B, T)
```

`raw_rewards_or_advantages` 形状为 `(B, 1)`，自动广播到 `(B, T)`，每个 token 都乘以同一个 reward。

**两种用法**：
- `no_baseline`：乘以原始 reward（高方差，训练不稳定）
- `reinforce_with_baseline`：乘以 advantage（减去组均值，方差更低）

---

### `run_compute_grpo_clip_loss`

**PPO-style 截断损失**，防止策略更新幅度过大：

$$\text{ratio} = \frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} = \exp(\log\pi_\theta - \log\pi_{\theta_\text{old}})$$

$$\mathcal{L}_\text{clip} = -\min\left(\text{ratio} \cdot A,\ \text{clip}(\text{ratio}, 1\pm\varepsilon) \cdot A\right)$$

```python
ratio = torch.exp(policy_log_probs - old_log_probs)
clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
loss = -torch.min(ratio * advantages, clipped * advantages)
```

**为什么取 min？**

| 情形 | ratio 大 (adv > 0) | ratio 小 (adv < 0) |
|------|--------------------|--------------------|
| 无 clip | 鼓励过猛，可能走偏 | 惩罚过猛，可能崩溃 |
| 有 clip | 超出 1+ε 时 clip，梯度清零 | 超出 1-ε 时 clip，梯度清零 |

**clip_fraction**：超出 `[1-ε, 1+ε]` 的比例，监控指标。若过高（>50%），说明策略偏离 reference 太远。

---

### `run_grpo_microbatch_train_step`

**与 SFT 步的区别**：用 `masked_mean`（除以 token 数）而非 `masked_normalize`（除以常数）。

**公式推导**（从 snapshot 反推）：

```
非 clip 位置梯度 = per_token_grad / (count_per_seq × batch_size × grad_accum)
              = 0.13439825 / (6 × 2 × 2) = 0.005599927 ✓
```

其中 `count_per_seq = 6`（该序列中 response_mask=1 的 token 数）。

```python
def run_grpo_microbatch_train_step(policy_log_probs, response_mask,
                                    gradient_accumulation_steps, loss_type,
                                    raw_rewards=None, advantages=None,
                                    old_log_probs=None, cliprange=None):
    per_token_loss, metadata = run_compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )
    per_seq = run_masked_mean(per_token_loss, response_mask, dim=1)  # 每条序列取均值
    loss = per_seq.mean() / gradient_accumulation_steps              # batch 均值
    loss.backward()
    return loss, metadata
```

---

### `run_compute_policy_gradient_loss` — 分发器

```python
if loss_type == "no_baseline":
    return naive_pg_loss(raw_rewards, policy_log_probs), {}
elif loss_type == "reinforce_with_baseline":
    return naive_pg_loss(advantages, policy_log_probs), {}
elif loss_type == "grpo_clip":
    return grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
```

---

## 5. Zero-shot 评估脚本

**脚本**：`scripts/eval_zero_shot.py`

### 两种 prompt 格式

| 格式 | Prompt | 打分器 | 适用阶段 |
|------|--------|--------|---------|
| `question_only` | 直接发题目 | `question_only_reward_fn`（找 `\boxed{}`） | 零样本基线 |
| `r1_zero` | 带 `<think>/<answer>` 指令 | `r1_zero_reward_fn`（严格格式） | GRPO 训练后评估 |

**为什么零样本用 question_only？** 模型未经 R1 训练，几乎不会自发输出 `<think>` 标签，严格格式会给所有样本 0 分，无法反映真实数学能力。

### 关键实现细节

```python
# 1. Left padding（decoder-only 模型必须左 pad）
tokenizer.padding_side = "left"

# 2. 只解码新生成的 token
new_tokens = outputs[:, inputs["input_ids"].shape[1]:]

# 3. r1_zero 格式重建（prompt 末尾是 "<think>"，生成内容不含它）
if prompt_format == "r1_zero":
    responses = ["<think>" + r for r in responses_raw]
```

### 运行

```bash
# 100 题快速测试
python scripts/eval_zero_shot.py \
    --model_path models/Qwen2.5-Math-1.5B \
    --num_examples 100 \
    --prompt_format question_only

# 预期结果：Accuracy ≈ 5-10%
```

---

## 6. Debug 经验

### Bug 1：tokenize 先 slice 再 pad 导致位置错位

**现象**：snapshot 不匹配，input_ids 中某位置出现 EOS padding，但 snapshot 期望真实 token。

**根因**：对长度不同的序列，先 `full[:-1]` 再右 pad 到 `max_len`，会导致短序列的 padding "挤掉"了最后一个真实 token 的预测位置。

**修复**：先把 `full_ids` pad 到 `max_full_len`，再统一做 `[:-1]` 和 `[1:]`，确保 padding 始终在序列末尾。

---

### Bug 2：bfloat16 精度导致 snapshot 超差

**现象**：
```
TypeError: Got unsupported ScalarType BFloat16      # 第一轮
AssertionError: max abs diff = 0.407  atol=0.01    # 加了 .float() 之后
```

**根因**：Qwen2.5-Math 的 config 设置 `torch_dtype: bfloat16`，`from_pretrained` 默认以 bfloat16 加载，但 snapshot 由 float32 生成，两者数值有系统性偏差（bfloat16 只有约 3 位有效十进制数字）。

**修复**：
```python
# conftest.py
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
# adapters.py
result["log_probs"] = token_log_probs.float()
```

**经验**：跨机器/跨配置运行 snapshot 测试时，dtype 不一致是常见坑，首先排查。

---

### 通用 Debug 原则

1. **先看 snapshot，再猜实现**：遇到不匹配，先 `np.load` 把期望值打印出来，直接对照数字推公式，比盲猜快 10 倍。

2. **tokenizer 属性要实测**：不要假设 `pad_token_id=0` 或 BOS/EOS 的存在，用一行代码验证：
   ```python
   print(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
   ```

3. **梯度反推公式**：snapshot 中保存了 `.grad`，通过梯度值可以精确反推 loss 公式（本项目用此方法确认了 SFT 和 GRPO 的归一化方式）。

---

## 7. CS336 作业完整要求（对齐 PDF）

> 本节是 PDF 要求的权威记录，按 Section 编号对齐。每项标注【完成状态】。

---

### Section 3 — Zero-Shot 基线（4 pts）

> ⚠️ **数据集说明**：PDF 要求在 **MATH validation set**（5000 题）上评估。
> 本项目因 HuggingFace 访问受限，改用 **GSM8K test set** 替代。
> 差异：GSM8K 是小学竞赛题（比 MATH 简单），基线准确率显著更高；
> 报告时需明确说明使用的是 GSM8K，zero-shot acc 约 19%（r1_zero）而非 ~5%。

**要求**：
1. **代码**：实现 `evaluate_vllm` 函数（加载 validation set、生成、评分、序列化到磁盘）
2. **分析**：统计三类结果数量，各类举 ≥10 个例子讨论
   - Cat 1：format ✓ + answer ✓
   - Cat 2：format ✓ + answer ✗
   - Cat 3：format ✗（两者都错）
3. **报告**：1-2 句话报告 zero-shot 准确率

**当前状态**：
- [x] eval 脚本 `scripts/eval_zero_shot.py`，支持序列化到 JSON
- [x] GSM8K test 200 examples 已跑，结果存 `results/zero_shot_baseline_gsm8k_*.json`
- [ ] ⚠️ `evaluate_vllm` 版本（用 vLLM 加速）尚未实现，当前用 transformers
- [ ] ⚠️ 三类统计 + ≥10 例分析尚未完成（需写分析文字）

**GSM8K 基线结果**（替代 MATH validation set）：

| 格式 | Format acc | Answer acc |
|------|-----------|-----------|
| question_only | 81.5% | 60.5% |
| r1_zero | 52.0% | 19.0% |

**三类统计脚本**：
```python
import json
results = json.load(open("results/zero_shot_baseline_gsm8k_r1_zero.json"))
cat1 = [r for r in results if r["format_reward"]==1 and r["answer_reward"]==1]
cat2 = [r for r in results if r["format_reward"]==1 and r["answer_reward"]==0]
cat3 = [r for r in results if r["format_reward"]==0]
print(f"Cat1 (format✓ answer✓): {len(cat1)}")
print(f"Cat2 (format✓ answer✗): {len(cat2)}")
print(f"Cat3 (format✗):         {len(cat3)}")
```

---

### Section 4 — SFT（5 pts 实现 + 2 pts 实验）

> ⚠️ **数据集说明**：PDF 要求使用 MATH train/validation set。
> 本项目用 **GSM8K train（7473 条）** 作为 SFT 数据，**GSM8K test（1319 条）** 作为 val。
> GSM8K 答案格式 `#### <number>` 已转换为 `</think> <answer><number></answer>`，与 r1_zero 格式兼容。
> 目标准确率对标调整：PDF 的 SFT 目标 ≥15% 是针对 MATH；GSM8K 上 SFT 目标参考已达到的 70.5%。

**实验 4.1 — Dataset size sweep**：
- 训练规模：`{128, 256, 512, 1024, full(~7473)}`（PDF 要求 5 条曲线）
- 绘图：x 轴 = dataset size，y 轴 = val accuracy（GSM8K test r1_zero acc）
- 脚本：`scripts/sft_size_sweep.py`

**实验 4.2 — 过滤 SFT 数据**：
- PDF 原意：从 MATH train set 里只保留模型能答对的样本
- GSM8K 调整方案：用零样本模型在 GSM8K train 上生成，过滤 `reward=1` 的样本，报告过滤后大小，与全量 SFT 对比
- 或等价：使用 SFT v2 模型在 train set 上生成，过滤正确答案后继续微调

**当前状态**：
- [x] SFT 训练循环 `alignment/training.py`，记录 loss / lr / token_entropy
- [x] SFT v2（1k examples, 2 epochs）→ **70.5% accuracy（r1_zero）**，format_acc=98.5%
- [x] Size sweep 脚本 `scripts/sft_size_sweep.py`（已启动，后台运行中）
- [ ] full 数据（7473 examples）的训练尚未跑
- [ ] 过滤实验（4.2）尚未实现

**SFT 已有结果**：

| Size | Format acc | Answer acc | 状态 |
|------|-----------|-----------|------|
| 1000（v2） | 98.5% | 70.5% | ✅ 完成 |
| 128 | - | - | 🔄 sweep 中 |
| 256 | - | - | 🔄 sweep 中 |
| 512 | - | - | 🔄 sweep 中 |
| 1024 | - | - | 🔄 sweep 中 |
| full | - | - | ⏳ 待跑 |

---

### Section 5 — Expert Iteration（2 pts，约 6 H100 hrs）

> ⚠️ **数据集说明**：PDF 用 MATH train 生成 rollout，MATH val 评估。
> 本项目用 **GSM8K train** 生成 rollout，**GSM8K test（前 200 条）** 评估。
> 目标准确率：PDF 要求 ≥15%（MATH），GSM8K 对应目标应 ≥75%（高于 SFT 的 70.5%）。

**要求**：
- 固定 `n_ei_steps=5`
- 变化：rollout 数 G（≥2 个值）× SFT epochs/step（≥2 个值）= ≥4 个配置
- 绘图 1：val accuracy vs. EI steps（多配置对比）
- 绘图 2：token entropy vs. training steps（监控模式崩溃）
- 文字：2 句对比 SFT 和各 EI step 的表现

**EI 算法流程**：
```
for ei_step in range(n_ei_steps):
    1. 用当前模型在 GSM8K train 上生成 G 个 rollout（vLLM 加速）
    2. 用 r1_zero_reward_fn 过滤：只保留 reward=1 的
    3. 在过滤后的数据上做 SFT（若干 epochs）
    4. 在 GSM8K test 上评估，记录 accuracy + entropy
```

**实现状态**：
- [ ] EI 训练循环尚未实现（需要 vLLM rollout 生成）
- [ ] ⚠️ 本地 RTX 3070：vLLM rollout (~4GB) 和 SFT 训练 (~6GB) 需分开跑（共 8GB）

---

### Section 7 — GRPO 实现（15 pts，含 grpo_train_loop）

**grpo_train_loop 需要记录（每步）**：

| 指标 | 说明 |
|------|------|
| `train/loss` | policy gradient loss |
| `train/grad_norm` | `clip_grad_norm_` 返回值（重要！监控训练稳定性）|
| `train/token_entropy` | 平均 token 熵（监控模式崩溃）|
| `train/reward` | 平均总奖励 |
| `train/format_reward` | 格式奖励均值 |
| `train/answer_reward` | 答案奖励均值 |
| `train/clip_fraction` | IS ratio 截断比例（仅 grpo_clip）|

**每 5~10 步记录**：
- `val/answer_reward`：在 val set 小批量上评估

**定性分析**：
- 记录不同训练阶段（early / mid / late）的 rollout 示例（问题 + 生成的推理链 + 答案）

**实现状态**：
- [ ] `grpo_train_loop` 尚未实现

---

### Section 8 — GRPO 实验（17 pts）

所有实验均绘制 **val answer reward vs. optimizer steps**（在同一坐标系内对比）。

| 问题 ID | 实验内容 | 输出 | 状态 |
|---------|---------|------|------|
| `grpo_learning_rate` | LR sweep（≥3 个值） | 曲线 + ≥25% + 2 句 | ⏳ |
| `grpo_baselines` | `no_baseline` vs `reinforce_with_baseline` | 曲线 + 2 句 | ⏳ |
| `think_about_length_normalization` | **纯文字**：masked_mean vs masked_normalize 优缺点 | 书面分析 | ⏳ |
| `grpo_length_normalization` | mean vs normalize 实验对比 | 曲线 + gradient norm 分析 | ⏳ |
| `grpo_group_standard_deviation` | `use_std=True` vs `False`（Dr. GRPO） | 曲线 + gradient norm 分析 | ⏳ |
| `grpo_off_policy` | 实现多 epoch off-policy 循环 | 代码 | ⏳ |
| `grpo_off_policy_sweep` | `epochs_per_batch × train_batch_size` 扫描 | **两张图**：val reward vs steps + vs wall-clock；entropy/length 分析 | ⏳ |
| `grpo_off_policy_clip_ablation` | GRPO-Clip vs GRPO-No-Clip | 曲线 + entropy/length/grad_norm 分析 | ⏳ |
| `grpo_prompt_ablation` | `r1_zero` vs `question_only` prompt | 曲线 + 各指标对比 | ⏳ |

**关键实现细节**：
- `think_about_length_normalization`（纯文字题）：
  - masked_mean：除以有效 token 数 → 长短序列 loss 同等权重，但长序列梯度更小
  - masked_normalize：除以固定常数 → 长序列获得更大梯度，可能不稳定；但 Dr. GRPO 用此防止对短序列过拟合
- off-policy 实验：需记录 `clip_fraction`，over-optimization 时 clip_fraction 会升高
- GRPO-No-Clip：去掉 `torch.min(...)` 中的 clipping，直接 `-ratio * adv`

---

### Section 9 — Leaderboard（16 pts）

**要求**：
- 4 小时内（2 H100）最大化 MATH validation accuracy（5000 examples）
- 报告最终准确率数值
- 截图：val accuracy vs. wall-clock time（x 轴 ≤ 4 小时）

**策略**（待定）：
- 从 SFT → EI → GRPO 最优 checkpoint 出发
- 在租借 GPU（2 × H100）上跑完整流程
- ⚠️ 需要先解决 MATH dataset 不可访问问题（HuggingFace 被禁）

---

### 需要绘制的图汇总（13 张）

| # | 图名 | 脚本/函数 | 状态 |
|---|------|---------|------|
| 1 | SFT dataset size sweep（5 条曲线） | `sft_size_sweep.py` | 🔄 running |
| 2 | SFT 过滤 vs 未过滤对比 | TBD | ⏳ |
| 3 | EI val accuracy（多配置） | TBD | ⏳ |
| 4 | EI token entropy vs steps | TBD | ⏳ |
| 5 | GRPO val reward vs steps（基础） | TBD | ⏳ |
| 6 | GRPO LR sweep | TBD | ⏳ |
| 7 | GRPO no_baseline vs reinforce | TBD | ⏳ |
| 8 | GRPO masked_mean vs masked_normalize | TBD | ⏳ |
| 9 | GRPO std=True vs std=False | TBD | ⏳ |
| 10a | Off-policy: val reward vs steps | TBD | ⏳ |
| 10b | Off-policy: val reward vs wall-clock | TBD | ⏳ |
| 11 | Clip vs No-Clip | TBD | ⏳ |
| 12 | r1_zero vs question_only prompt | TBD | ⏳ |
| 13 | Leaderboard: val acc vs wall-clock | TBD | ⏳ |

---

### 总体进度清单

**Section 3 (Zero-shot)**
- [x] eval 脚本（transformers 版）
- [ ] evaluate_vllm 版本
- [ ] 三类统计 + ≥10 例分析

**Section 4 (SFT)**
- [x] SFT 训练循环
- [x] size sweep 脚本启动
- [ ] size sweep 完整结果（含 full）
- [ ] SFT 过滤实验（4.2）

**Section 5 (Expert Iteration)**
- [ ] EI 训练循环
- [ ] EI 实验（G × epochs 扫描）

**Section 7 (GRPO 实现)**
- [x] 所有 unit test 函数（adapters.py）
- [ ] grpo_train_loop

**Section 8 (GRPO 实验 × 9)**
- [ ] 全部 9 个实验

**Section 9 (Leaderboard)**
- [ ] 租 GPU + 完整跑流程
