#!/bin/bash
# Stream the verified pipeline downloads into two complete rank-local TP stages.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE_HOST="${KIMI_REMOTE_HOST:-beast2.local}"
LOCAL_RDMA_HOST="${KIMI_LOCAL_RDMA_HOST:-192.168.0.1}"
REMOTE_RDMA_HOST="${KIMI_REMOTE_RDMA_HOST:-192.168.0.2}"
REPO="${KIMI_REPO_DIR:-$PWD}"
PYTHON="${KIMI_PYTHON:-$REPO/.venv/bin/python}"
SOURCE_ROOT="$REPO/weights/Kimi-K3-2bit-UVMAX-pipeline-2"
OUTPUT_ROOT="$REPO/weights/Kimi-K3-2bit-UVMAX-tensor-2"
LOCK="work/.tensor-conversion.lock"

if ! mkdir "$LOCK" 2>/dev/null; then
    local_workers="$(pgrep -f '[s]tream_tensor_owner.py' || true)"
    remote_workers="$(ssh -o BatchMode=yes "$REMOTE_HOST" \
        "pgrep -f '[s]tream_tensor_owner.py' || true")"
    if [[ -n "$local_workers" || -n "$remote_workers" ]]; then
        echo "A tensor conversion is already active; monitor it instead." >&2
        exit 1
    fi
    rmdir "$LOCK" 2>/dev/null || {
        echo "Cannot recover stale conversion lock: $LOCK" >&2
        exit 1
    }
    mkdir "$LOCK"
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT HUP INT TERM

/usr/bin/ibv_devinfo | grep -q 'PORT_ACTIVE' || {
    echo "No active local RDMA port." >&2
    exit 1
}
ssh -o BatchMode=yes "$REMOTE_HOST" /usr/bin/ibv_devinfo | \
    grep -q 'PORT_ACTIVE' || {
        echo "No active peer RDMA port." >&2
        exit 1
    }
ssh -o BatchMode=yes "$REMOTE_RDMA_HOST" true
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "ssh -o BatchMode=yes '$LOCAL_RDMA_HOST' true"

local_common=(
    "$PYTHON" scripts/stream_tensor_owner.py
    --source-root "$SOURCE_ROOT/rank0"
    --output "$OUTPUT_ROOT/rank0"
    --local-rank 0
    --peer-host "$REMOTE_RDMA_HOST"
    --peer-output "$OUTPUT_ROOT/rank1"
    --peer-python "$PYTHON"
    --first 94
    --last 185
)
remote_command=(
    "$PYTHON" scripts/stream_tensor_owner.py
    --source-root "$SOURCE_ROOT/rank1"
    --output "$OUTPUT_ROOT/rank1"
    --local-rank 1
    --peer-host "$LOCAL_RDMA_HOST"
    --peer-output "$OUTPUT_ROOT/rank0"
    --peer-python "$PYTHON"
    --first 1
    --last 93
)

# Initialize both destinations before either producer transfers its first file.
"${local_common[@]}" --init-only
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd '$REPO' && ${remote_command[*]} --init-only"

/usr/bin/caffeinate -dimsu "${local_common[@]}" &
local_pid=$!
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd '$REPO' && /usr/bin/caffeinate -dimsu ${remote_command[*]}" &
remote_pid=$!
result=0
wait "$local_pid" || result=$?
wait "$remote_pid" || result=$?
((result == 0)) || exit "$result"

"$PYTHON" scripts/finalize_tensor_stage.py \
    --stage "$OUTPUT_ROOT/rank0" \
    --source-manifest "$SOURCE_ROOT/rank0/stage-manifest.json" \
    --rank 0 &
local_pid=$!
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd '$REPO' && '$PYTHON' scripts/finalize_tensor_stage.py \
      --stage '$OUTPUT_ROOT/rank1' \
      --source-manifest '$SOURCE_ROOT/rank1/stage-manifest.json' --rank 1" &
remote_pid=$!
result=0
wait "$local_pid" || result=$?
wait "$remote_pid" || result=$?
((result == 0)) || exit "$result"

echo "Both SHA-256-verified tensor stages are complete."
