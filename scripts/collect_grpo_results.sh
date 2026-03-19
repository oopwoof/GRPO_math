#!/bin/bash
# ============================================================
# Collect GRPO sweep results from all pods and merge them.
#
# Since all 3 pods share the same Network Volume, all pod-specific
# result files are accessible via any single pod's SSH connection.
#
# Usage:
#   bash scripts/collect_grpo_results.sh <IP> <PORT>
#
# Example:
#   bash scripts/collect_grpo_results.sh 209.170.80.132 21752
#
# Outputs:
#   results/grpo_sweep_results_pod_a.json  (from shared NV via scp)
#   results/grpo_sweep_results_pod_b.json
#   results/grpo_sweep_results_pod_c.json
#   results/grpo_sweep_results.json        (merged, de-duplicated)
#
# Requirements:
#   - SSH key at ~/.ssh/id_ed25519
#   - python3 + scripts/merge_grpo_results.py in $PWD
# ============================================================
set -e

# ---- Parse args ----
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <IP> <PORT>"
    echo ""
    echo "Connect to any one of the 3 pods (they share a Network Volume)."
    echo "All pod result files will be fetched via that single connection."
    exit 1
fi

IP="$1"
PORT="$2"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="/workspace/GRPO_math/results"
LOCAL_DIR="results"
MERGED_PATH="$LOCAL_DIR/grpo_sweep_results.json"

mkdir -p "$LOCAL_DIR"

echo "=============================="
echo "  Collecting GRPO sweep results"
echo "=============================="
echo "  Source: root@${IP}:${PORT} (shared NV)"
echo ""

# ---- Fetch all pod result files from a single SSH connection ----
echo "[1/3] Downloading pod result files..."

fetch_file() {
    local label="$1"
    local remote_file="$2"
    local local_file="$3"

    scp -q \
        -P "$PORT" \
        -i "$SSH_KEY" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=30 \
        "root@${IP}:${remote_file}" \
        "$local_file" \
    && echo "  [${label}] -> $local_file" \
    || echo "  [${label}] WARNING: not found on NV — skipping."
}

fetch_file "pod_a" "$REMOTE_DIR/grpo_sweep_results_pod_a.json" \
    "$LOCAL_DIR/grpo_sweep_results_pod_a.json"
fetch_file "pod_b" "$REMOTE_DIR/grpo_sweep_results_pod_b.json" \
    "$LOCAL_DIR/grpo_sweep_results_pod_b.json"
fetch_file "pod_c" "$REMOTE_DIR/grpo_sweep_results_pod_c.json" \
    "$LOCAL_DIR/grpo_sweep_results_pod_c.json"

echo ""
echo "[2/3] Merging results..."

# Build list of files that actually exist
FILES=()
for f in \
    "$LOCAL_DIR/grpo_sweep_results_pod_a.json" \
    "$LOCAL_DIR/grpo_sweep_results_pod_b.json" \
    "$LOCAL_DIR/grpo_sweep_results_pod_c.json"
do
    [ -f "$f" ] && FILES+=("$f")
done

if [ ${#FILES[@]} -eq 0 ]; then
    echo "ERROR: No result files downloaded. Nothing to merge."
    exit 1
fi

python scripts/merge_grpo_results.py \
    --inputs "${FILES[@]}" \
    --output "$MERGED_PATH"

echo ""
echo "[3/3] Done."
echo "  Merged results: $MERGED_PATH"
echo ""

# Quick count
python - <<PYEOF
import json, pathlib
data = json.loads(pathlib.Path("$MERGED_PATH").read_text(encoding="utf-8"))
print(f"  Total runs in merged file: {len(data)}")
exps = {}
for r in data:
    exps.setdefault(r.get('experiment','?'), []).append(r['run_name'])
for exp, runs in sorted(exps.items()):
    print(f"    {exp}: {len(runs)} runs ({', '.join(runs)})")
PYEOF
