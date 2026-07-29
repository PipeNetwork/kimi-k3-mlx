#!/bin/bash
# Launch the rank-local Kimi-K3 pipeline. JACCL is mandatory and explicit.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE_HOST="${KIMI_REMOTE_HOST:-beast2.local}"
HOSTFILE="${KIMI_HOSTFILE:-hosts-jaccl.json}"
PYTHON="${KIMI_PYTHON:-/Users/agent/dev/kimi-k3-mlx-distributed/.venv/bin/python}"
REPO_DIR="${KIMI_REPO_DIR:-/Users/agent/dev/kimi-k3-mlx-distributed}"
RUN_ID="${KIMI_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

args=("$@")
for ((index = 0; index < ${#args[@]}; index++)); do
    case "${args[$index]}" in
        --run-id)
            ((index + 1 < ${#args[@]})) || {
                echo "--run-id requires a value" >&2
                exit 2
            }
            RUN_ID="${args[$((index + 1))]}"
            ((index += 1))
            ;;
        --run-id=*)
            RUN_ID="${args[$index]#--run-id=}"
            ;;
    esac
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "Unsafe run id: $RUN_ID" >&2
    exit 2
}

LOCK_DIR="work/.distributed-run.lock"
LOCK_OWNER="$LOCK_DIR/owner"

release_lock() {
    if [[ -f "$LOCK_OWNER" ]] && [[ "$(awk '{print $1}' "$LOCK_OWNER")" == "$$" ]]; then
        rm -f -- "$LOCK_OWNER"
        rmdir "$LOCK_DIR" 2>/dev/null || true
    fi
}

acquire_lock() {
    mkdir -p work
    for ((attempt = 0; attempt < 3; attempt++)); do
        if mkdir "$LOCK_DIR" 2>/dev/null; then
            printf '%s %s\n' "$$" "$RUN_ID" >"$LOCK_OWNER"
            trap release_lock EXIT HUP INT TERM
            return 0
        fi

        # Allow the process that created the directory a moment to publish
        # its owner before deciding whether this is a stale crash/reboot lock.
        if [[ ! -f "$LOCK_OWNER" ]]; then
            sleep 1
            [[ -f "$LOCK_OWNER" ]] || rmdir "$LOCK_DIR" 2>/dev/null || true
            continue
        fi
        read -r owner_pid owner_run <"$LOCK_OWNER"
        if [[ "$owner_pid" =~ ^[0-9]+$ ]] && kill -0 "$owner_pid" 2>/dev/null; then
            echo "Distributed run $owner_run is already active as PID $owner_pid" >&2
            return 1
        fi
        rm -f -- "$LOCK_OWNER"
        rmdir "$LOCK_DIR" 2>/dev/null || true
    done
    echo "Could not acquire $LOCK_DIR" >&2
    return 1
}

acquire_lock

test -f "$HOSTFILE" || {
    echo "Missing $HOSTFILE; run scripts/configure_jaccl.sh after connecting Thunderbolt 5." >&2
    exit 1
}
HOSTFILE_SHA256="$(shasum -a 256 "$HOSTFILE" | awk '{print $1}')"
LOCAL_SUMMARY="work/benchmarks/${RUN_ID}-summary-rank0.json"
REMOTE_SUMMARY="work/benchmarks/${RUN_ID}-summary-rank1.json"

test ! -e "$LOCAL_SUMMARY" || {
    echo "Refusing to overwrite existing $LOCAL_SUMMARY" >&2
    exit 2
}
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "test ! -e '$REPO_DIR/$REMOTE_SUMMARY'" || {
    echo "Refusing to overwrite existing $REMOTE_HOST:$REPO_DIR/$REMOTE_SUMMARY" >&2
    exit 2
}

scripts/install_model.sh "$PYTHON"
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd '$REPO_DIR' && scripts/install_model.sh '$PYTHON'"

set +e
KIMI_RUN_ID="$RUN_ID" "$REPO_DIR/.venv/bin/mlx.launch" \
    --verbose \
    --backend jaccl \
    --hostfile "$HOSTFILE" \
    --cwd "$REPO_DIR" \
    --python "$PYTHON" \
    --env MLX_METAL_FAST_SYNCH=1 \
    --env KIMI_HOSTFILE_SHA256="$HOSTFILE_SHA256" \
    --env KIMI_RUN_ID="$RUN_ID" \
    -- /usr/bin/caffeinate -dimsu "$PYTHON" scripts/distributed_generate.py "$@"
launch_status=$?
set -e

((launch_status == 0)) || {
    echo "mlx.launch failed with status $launch_status" >&2
    exit "$launch_status"
}
test -s "$LOCAL_SUMMARY" || {
    echo "Rank 0 did not produce $LOCAL_SUMMARY; treating launch as failed" >&2
    exit 1
}
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "test -s '$REPO_DIR/$REMOTE_SUMMARY'" || {
    echo "Rank 1 did not produce $REMOTE_HOST:$REPO_DIR/$REMOTE_SUMMARY; treating launch as failed" >&2
    exit 1
}

echo "Both ranks completed run $RUN_ID successfully."
