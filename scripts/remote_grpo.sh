#!/bin/bash
# ============================================================
# Remote GRPO ablation sweep (Section 8)
#
# Usage:
#   bash scripts/remote_grpo.sh <RUNPOD_API_KEY> [POD_ID] [EXPERIMENTS...]
#
# Examples:
#   # Run all experiments, auto-stop pod when done
#   bash scripts/remote_grpo.sh abc123key wdc6riqt8acr19
#
#   # Run only baselines ablation
#   bash scripts/remote_grpo.sh abc123key wdc6riqt8acr19 grpo_baselines
#
#   # Run without auto-stop (leave pod running)
#   bash scripts/remote_grpo.sh "" "" grpo_baselines
#
# Available experiments:
#   grpo_learning_rate           LR sweep (4 runs)
#   grpo_baselines               no_baseline / reinforce / grpo_clip (3 runs)
#   grpo_length_normalization    masked_mean vs normalize (2 runs)
#   grpo_group_standard_deviation  std norm on/off (2 runs)
#   grpo_off_policy_sweep        off_policy_steps 1/2/4/8 (4 runs)
#   grpo_off_policy_clip         off-policy + cliprange ablation (4 runs)
#   grpo_prompt_ablation         r1_zero vs question_only (2 runs)
# ============================================================
set -e

RUNPOD_API_KEY="${1:-}"
POD_ID="${2:-}"
shift 2 2>/dev/null || true
EXPERIMENTS=("$@")   # remaining args = experiment names; empty = all

WORKDIR="/workspace/GRPO_math"
MODEL_PATH="$WORKDIR/models/ei-gsm8k/final"   # start from EI model
FALLBACK_MODEL="$WORKDIR/models/sft-gsm8k-full/final"  # fallback if EI not done

echo "=============================="
echo "  GRPO Math - Remote GRPO Sweep"
echo "=============================="
cd "$WORKDIR"

# ---------- 1. Install deps ----------
echo ""
echo "[1/3] Installing dependencies..."
pip install -q \
    "transformers>=4.50.0" \
    torch tqdm wandb \
    "math-verify[antlr4-13-2]>=0.7.0" \
    pylatexenc==2.10 \
    accelerate safetensors datasets \
    huggingface_hub tensorboard \
    latex2sympy2_extended
pip install -q -e . --no-deps
echo "[1/3] Done."

# ---------- 2. Resolve model path ----------
echo ""
echo "[2/3] Resolving model path..."
if [ -f "$MODEL_PATH/config.json" ]; then
    echo "  Using EI model: $MODEL_PATH"
elif [ -f "$FALLBACK_MODEL/config.json" ]; then
    echo "  EI model not found, using full SFT: $FALLBACK_MODEL"
    MODEL_PATH="$FALLBACK_MODEL"
else
    echo "ERROR: Neither EI model nor SFT model found. Exiting."
    exit 1
fi

# ---------- 3. Run GRPO sweep ----------
echo ""
echo "[3/3] Running GRPO ablation sweep..."
mkdir -p "$WORKDIR/results"

# Build experiment args
if [ ${#EXPERIMENTS[@]} -gt 0 ]; then
    EXP_ARGS="--experiments ${EXPERIMENTS[*]}"
    echo "  Experiments: ${EXPERIMENTS[*]}"
else
    EXP_ARGS=""
    echo "  Running ALL experiments"
fi

WANDB_MODE=disabled python scripts/grpo_sweep.py \
    --model_path "$MODEL_PATH" \
    --output_dir models/grpo-sweep \
    --results_path results/grpo_sweep_results.json \
    --total_steps 200 \
    --num_val 200 \
    $EXP_ARGS

echo ""
echo "=============================="
echo "  GRPO sweep complete!"
echo "  Results: $WORKDIR/results/grpo_sweep_results.json"
echo "=============================="

# ---------- Auto-stop pod ----------
if [ -n "$RUNPOD_API_KEY" ] && [ -n "$POD_ID" ]; then
    echo ""
    echo "Stopping pod $POD_ID ..."
    curl -s --request POST \
        --header 'content-type: application/json' \
        --url "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
        --data "{\"query\": \"mutation { podStop(input: {podId: \\\"${POD_ID}\\\"}) { id } }\"}"
    echo "Pod stop signal sent."
else
    echo "(No API key/pod ID provided — pod left running)"
fi
