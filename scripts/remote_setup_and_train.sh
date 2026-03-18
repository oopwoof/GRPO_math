#!/bin/bash
# ============================================================
# Remote GPU setup + SFT filtered training (Section 4.2)
#
# Usage:
#   bash scripts/remote_setup_and_train.sh            # full run (filter + train)
#   bash scripts/remote_setup_and_train.sh --skip-filter  # train only (needs cached examples)
#
# Requirements: CUDA GPU with 16GB+ VRAM recommended (RTX 4090 / A100)
# Estimated time: ~20 min filter + ~60 min training on RTX 4090
# ============================================================
set -e

SKIP_FILTER=false
for arg in "$@"; do
    [[ "$arg" == "--skip-filter" ]] && SKIP_FILTER=true
done

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="Qwen/Qwen2.5-Math-1.5B"
MODEL_DIR="$WORKDIR/models/Qwen2.5-Math-1.5B"

echo "=============================="
echo "  GRPO Math - Remote Setup"
echo "  Workdir: $WORKDIR"
echo "=============================="
cd "$WORKDIR"

# ---------- 1. Install Python deps (SFT only — no vLLM needed) ----------
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -q \
    "transformers>=4.50.0" \
    torch \
    tqdm \
    wandb \
    "math-verify[antlr4-13-2]>=0.7.0" \
    pylatexenc==2.10 \
    accelerate \
    safetensors \
    datasets \
    huggingface_hub \
    tensorboard

pip install -q -e . --no-deps
echo "[1/4] Done."

# ---------- 2. Download base model ----------
echo ""
echo "[2/4] Downloading $MODEL_NAME ..."
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/config.json" ]; then
    echo "[2/4] Model already present, skipping download."
else
    python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$MODEL_NAME",
    local_dir="$MODEL_DIR",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)
print(f"[2/4] Model saved to $MODEL_DIR")
PYEOF
fi

# ---------- 3. Download GSM8K data ----------
echo ""
echo "[3/4] Downloading GSM8K dataset..."
mkdir -p "$WORKDIR/data/gsm8k"

if [ -f "$WORKDIR/data/gsm8k/train.jsonl" ]; then
    echo "[3/4] GSM8K already present, skipping."
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
print("[3/4] Done.")
PYEOF
fi

# ---------- 4. Run SFT filtered training ----------
echo ""
echo "[4/4] Running SFT filtered training..."
mkdir -p "$WORKDIR/results"

EXTRA_ARGS=""
if [ "$SKIP_FILTER" = true ]; then
    if [ -f "$WORKDIR/results/sft_filtered_examples.json" ]; then
        echo "  Using cached filtered examples (--skip-filter)"
        EXTRA_ARGS="--skip_filter"
    else
        echo "  WARNING: --skip-filter set but no cached examples found. Running full filter."
    fi
fi

python scripts/sft_filtered.py \
    --target_size 1024 \
    --scan_size 2000 \
    --base_model "$MODEL_DIR" \
    --output_dir models/sft-filtered-1k \
    --results_path results/sft_filtered_comparison.json \
    $EXTRA_ARGS

echo ""
echo "=============================="
echo "  Training complete!"
echo "  Results: $WORKDIR/results/sft_filtered_comparison.json"
echo "  Model:   $WORKDIR/models/sft-filtered-1k/"
echo "=============================="
