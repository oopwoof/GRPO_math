#!/bin/bash
# ============================================================
# Remote GRPO init + sweep for NEW pods (Pod B / Pod C)
#
# Usage:
#   bash scripts/remote_grpo_init.sh <API_KEY> <POD_ID> <POD_LABEL> [EXPERIMENTS...]
#
# POD_LABEL is used to write a pod-specific results file so that multiple pods
# sharing the same Network Volume do not overwrite each other's results.
# Results land in: results/grpo_sweep_results_<POD_LABEL>.json
#
# Examples:
#   # Pod B: off-policy experiments
#   bash scripts/remote_grpo_init.sh abc123key pod_b_id pod_b grpo_off_policy_sweep grpo_off_policy_clip
#
#   # Pod C: misc ablations
#   bash scripts/remote_grpo_init.sh abc123key pod_c_id pod_c \
#       grpo_length_normalization grpo_group_standard_deviation grpo_prompt_ablation
#
#   # No auto-stop (leave pod running)
#   bash scripts/remote_grpo_init.sh "" "" pod_b grpo_baselines
#
# This script assumes:
#   - The repo is already present at /workspace/GRPO_math  OR  will be cloned
#   - The EI model will be uploaded to /workspace/GRPO_math/models/ei-gsm8k/final
#     before (or shortly after) this script starts
#   - REPO_URL env var can override the git remote (default: auto-detect from origin)
# ============================================================
set -e

RUNPOD_API_KEY="${1:-}"
POD_ID="${2:-}"
POD_LABEL="${3:-pod_b}"
shift 3 2>/dev/null || true
EXPERIMENTS=("$@")   # remaining args = experiment names; empty = all

WORKDIR="/workspace/GRPO_math"
MODEL_PATH="$WORKDIR/models/ei-gsm8k/final"
FALLBACK_MODEL="$WORKDIR/models/sft-gsm8k-full/final"

# Default repo URL — override with REPO_URL env var if needed
REPO_URL="${REPO_URL:-https://github.com/oopwoof/GRPO_math.git}"

echo "=============================="
echo "  GRPO Math - Pod Init + Sweep"
echo "=============================="

# ---------- 1. Clone / update repo ----------
echo ""
echo "[1/5] Setting up repository..."
if [ -d "$WORKDIR/.git" ]; then
    echo "  Repo already present at $WORKDIR, pulling latest..."
    cd "$WORKDIR"
    git pull --ff-only || echo "  (git pull failed — continuing with existing code)"
else
    echo "  Cloning repo to $WORKDIR ..."
    mkdir -p "$(dirname "$WORKDIR")"
    git clone "$REPO_URL" "$WORKDIR"
    cd "$WORKDIR"
fi
cd "$WORKDIR"
echo "[1/5] Done."

# ---------- 2. Install dependencies ----------
echo ""
echo "[2/5] Installing dependencies..."
pip install -q \
    "transformers>=4.50.0" \
    torch tqdm wandb \
    "math-verify[antlr4-13-2]>=0.7.0" \
    pylatexenc==2.10 \
    accelerate safetensors datasets \
    huggingface_hub tensorboard \
    latex2sympy2_extended
pip install -q -e . --no-deps
echo "[2/5] Done."

# ---------- 3. Download GSM8K dataset ----------
echo ""
echo "[3/5] Downloading GSM8K dataset..."
mkdir -p "$WORKDIR/data/gsm8k"

if [ -f "$WORKDIR/data/gsm8k/train.jsonl" ]; then
    echo "[3/5] GSM8K already present, skipping."
else
    python - <<PYEOF
from datasets import load_dataset
import json, pathlib

ds = load_dataset("openai/gsm8k", "main")
out = pathlib.Path("$WORKDIR/data/gsm8k")
out.mkdir(parents=True, exist_ok=True)

for split, fname in [("train", "train.jsonl"), ("test", "test.jsonl")]:
    with open(out / fname, "w") as f:
        for ex in ds[split]:
            f.write(json.dumps(ex) + "\n")
    print(f"  {split}: {len(ds[split])} examples -> {out / fname}")
print("[3/5] Done.")
PYEOF
fi

# ---------- 4. Wait for EI model ----------
echo ""
echo "[4/5] Waiting for EI model at $MODEL_PATH ..."
MAX_WAIT=3600   # 60 minutes max
POLL_INTERVAL=30
waited=0

while [ ! -f "$MODEL_PATH/config.json" ]; do
    if [ $waited -ge $MAX_WAIT ]; then
        echo "  WARNING: Timed out waiting for EI model after ${MAX_WAIT}s."
        break
    fi
    echo "  EI model not ready yet. Waiting ${POLL_INTERVAL}s... (${waited}s elapsed)"
    sleep $POLL_INTERVAL
    waited=$((waited + POLL_INTERVAL))
done

# Resolve model path (EI → fallback SFT → error)
if [ -f "$MODEL_PATH/config.json" ]; then
    echo "  EI model found: $MODEL_PATH"
elif [ -f "$FALLBACK_MODEL/config.json" ]; then
    echo "  EI model not found, using full SFT fallback: $FALLBACK_MODEL"
    MODEL_PATH="$FALLBACK_MODEL"
else
    echo "ERROR: Neither EI model nor SFT fallback found. Exiting."
    exit 1
fi
echo "[4/5] Done."

# ---------- 5. Run GRPO sweep ----------
echo ""
echo "[5/5] Running GRPO ablation sweep..."
mkdir -p "$WORKDIR/results"

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
    --results_path "results/grpo_sweep_results_${POD_LABEL}.json" \
    --total_steps 200 \
    --num_val 200 \
    $EXP_ARGS

echo ""
echo "=============================="
echo "  GRPO sweep complete!"
echo "  Results: $WORKDIR/results/grpo_sweep_results_${POD_LABEL}.json"
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
