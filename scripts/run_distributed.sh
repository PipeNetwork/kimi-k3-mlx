#!/bin/bash
# Launch the rank-local Kimi-K3 pipeline. JACCL is mandatory and explicit.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE_HOST="${KIMI_REMOTE_HOST:-beast2.local}"
HOSTFILE="${KIMI_HOSTFILE:-hosts-jaccl.json}"
PYTHON="${KIMI_PYTHON:-/Users/agent/dev/kimi-k3-mlx-distributed/.venv/bin/python}"
REPO_DIR="${KIMI_REPO_DIR:-/Users/agent/dev/kimi-k3-mlx-distributed}"
RUN_ID="${KIMI_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

test -f "$HOSTFILE" || {
    echo "Missing $HOSTFILE; run scripts/configure_jaccl.sh after connecting Thunderbolt 5." >&2
    exit 1
}

scripts/install_model.sh "$PYTHON"
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd '$REPO_DIR' && scripts/install_model.sh '$PYTHON'"

KIMI_RUN_ID="$RUN_ID" "$REPO_DIR/.venv/bin/mlx.launch" \
    --verbose \
    --backend jaccl \
    --hostfile "$HOSTFILE" \
    --cwd "$REPO_DIR" \
    --python "$PYTHON" \
    --env MLX_METAL_FAST_SYNCH=1 \
    --env KIMI_RUN_ID="$RUN_ID" \
    -- "$PYTHON" scripts/distributed_generate.py "$@"
