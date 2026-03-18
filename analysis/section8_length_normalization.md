# Section 8.3 — Think About Length Normalization

> **Assignment question**: In the SFT setting, we used `masked_normalize` (sum of log-probs divided by a constant) rather than `masked_mean` (mean over response tokens). Discuss: why does this choice matter? What are the trade-offs?

---

## Background: Two Normalization Schemes

In the SFT loss, we compute a per-sequence scalar from the token log-probs, then average over the batch:

**`masked_mean`** (standard token-average):
$$\mathcal{L}_{\text{mean}} = -\frac{1}{B} \sum_{i=1}^{B} \frac{\sum_{t \in \text{resp}_i} \log p_\theta(y_t | y_{<t}, x)}{\left|\text{resp}_i\right|}$$

**`masked_normalize`** (Dr. GRPO normalization, divide by constant $C$):
$$\mathcal{L}_{\text{norm}} = -\frac{1}{B} \sum_{i=1}^{B} \frac{\sum_{t \in \text{resp}_i} \log p_\theta(y_t | y_{<t}, x)}{C}$$

where $C$ is a fixed constant (e.g., $C = 1.0$ = sum; $C = L_{\max}$ = normalize to max length).

---

## Why Length Normalization Matters

### 1. Gradient Scale is Length-Dependent

With `masked_mean`, the gradient contributed by sequence $i$ is:

$$\nabla_\theta \mathcal{L}_{\text{mean}}^{(i)} = -\frac{1}{B \cdot |\text{resp}_i|} \sum_t \nabla_\theta \log p_\theta(y_t | \cdot)$$

The $1 / |\text{resp}_i|$ factor means **short responses get larger per-token gradient weight** than long responses. In a batch with variable-length responses, the optimizer step is dominated by short sequences even if the long ones contain equally important signal.

With `masked_normalize` ($C$ = constant), every sequence contributes equally regardless of length — only the total log-probability sum matters.

### 2. Implicit Bias Toward Short Responses

Under `masked_mean`, suppose two responses $A$ (10 tokens) and $B$ (100 tokens) have identical *total* log-probability. Response $A$ will produce 10× larger per-sample gradient because its normalization denominator is 10× smaller.

If the training signal correlates with length (e.g., harder problems require longer chains of thought), `masked_mean` will systematically underweight the gradient signal from long reasoning traces — exactly the cases where the model most needs to learn.

### 3. Connection to the GRPO Setting (Dr. GRPO)

In GRPO, the policy gradient loss is:

$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}_{o \sim \pi_\theta}\left[ -A(o) \cdot \frac{1}{|o|} \sum_t \log \frac{\pi_\theta(o_t)}{\pi_{\text{old}}(o_t)} \right]$$

The $1/|o|$ normalization (= `masked_mean`) means the *advantage signal per token* is what the optimizer sees. For long responses, the advantage is "diluted" across many tokens.

**Dr. GRPO** (Wen et al. 2025) identifies this as a problem: it argues for length-invariant normalization because:
- A correct long response should receive the same *total* reward signal as a correct short response, not a signal scaled by $1/\text{length}$.
- Otherwise, the model is incentivized to produce shorter correct responses even when the problem demands multi-step reasoning.

Setting $C = L_{\max}$ (maximum sequence length in the batch) approximates this: all sequences are normalized by the same constant, so the learning signal is proportional to *total* log-probability.

### 4. Practical Consequences

| Normalization | Tendency | Risk |
|---|---|---|
| `masked_mean` ($C = |o|$) | Learns to be concise | Underweights long reasoning; may produce truncated chains |
| `masked_normalize` ($C = 1$, i.e., sum) | Learns from all tokens equally | Long responses dominate gradient magnitude |
| `masked_normalize` ($C = L_{\max}$) | Length-invariant | Most principled; harder to tune $C$ |

In our GSM8K experiments:
- GSM8K answers are relatively short (1–5 tokens), but reasoning chains vary from 50–400 tokens.
- Using `masked_normalize` with $C = 1.0$ (sum normalization) means each example contributes proportional to its reasoning length — longer, more complex problems get more gradient weight.
- Empirically, SFT with $C = 1.0$ achieved 70.5% at 1k examples; we would need to run a controlled ablation to see the effect of switching to `masked_mean`.

---

## Trade-offs Summary

**Use `masked_mean` when**:
- You want uniform per-token importance across the dataset
- Responses are roughly equal length
- You're concerned about gradient instability from very long sequences

**Use `masked_normalize` (Dr. GRPO style) when**:
- Response lengths vary significantly (e.g., CoT reasoning vs. short-answer mix)
- You want the total-reward-per-sequence interpretation: correct long answers should be rewarded as much as correct short answers
- You're training a model that needs to learn multi-step reasoning (GRPO on hard math)

**In the GRPO context specifically**: using `masked_mean` (per-token average) creates a bias against long correct rollouts. The model may learn that the easiest way to maximize reward/token is to produce short, potentially low-quality answers. Dr. GRPO's normalization removes this bias by keeping the learning signal proportional to total correctness, not correctness-per-token.

---

## Experiment Plan (Section 8.4 in Assignment)

To empirically validate this, we run two GRPO variants:

```bash
# Per-token mean (standard)
python scripts/train_grpo.py \
    --loss_type grpo_clip \
    --output_dir models/grpo-length-mean \
    # (default: run_masked_mean in grpo_microbatch_train_step)

# Dr. GRPO normalize (length-invariant)
python scripts/train_grpo.py \
    --loss_type grpo_clip \
    --normalize_constant 512 \
    --output_dir models/grpo-length-normalize
```

Expected result: length-normalize should show higher val accuracy on problems requiring multi-step reasoning, at the cost of potentially noisier gradients from long-but-wrong rollouts.

*Results to be filled in after running experiments.*
