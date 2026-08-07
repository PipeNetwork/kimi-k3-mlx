#!/bin/bash
# Discover the Thunderbolt fabric and write a JACCL-only MLX hostfile.
set -euo pipefail
cd "$(dirname "$0")/.."

HOSTFILE="${KIMI_HOSTFILE:-hosts-jaccl.json}"
PYTHON="${KIMI_PYTHON:-.venv/bin/python}"
CONFIG_TOOL="$(dirname "$PYTHON")/mlx.distributed_config"
REMOTE_HOST="${KIMI_REMOTE_HOST:-}"
if [[ -z "$REMOTE_HOST" && -f "$HOSTFILE" ]]; then
    REMOTE_HOST="$("$PYTHON" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["hosts"][1]["ssh"])' \
        "$HOSTFILE")"
fi
REMOTE_HOST="${REMOTE_HOST:-beast2.local}"

if ! /usr/bin/ibv_devinfo | grep -q 'PORT_ACTIVE'; then
    echo "No active local Thunderbolt RDMA port. Connect the two Studios with Thunderbolt 5." >&2
    exit 1
fi
if ! ssh -o BatchMode=yes "$REMOTE_HOST" /usr/bin/ibv_devinfo | grep -q 'PORT_ACTIVE'; then
    echo "No active Thunderbolt RDMA port on $REMOTE_HOST." >&2
    exit 1
fi

test -x "$CONFIG_TOOL" || {
    echo "Missing $CONFIG_TOOL; run uv sync --frozen first." >&2
    exit 1
}

"$CONFIG_TOOL" \
    --hosts "localhost,$REMOTE_HOST" \
    --over thunderbolt \
    --backend jaccl \
    --auto-setup \
    --verbose \
    --output-hostfile "$HOSTFILE"

"$PYTHON" - "$HOSTFILE" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
value = json.loads(path.read_text())
if value.get("backend") != "jaccl":
    raise SystemExit(f"hostfile is not JACCL-only: {value}")
print(f"[jaccl] validated {path}: {value}")
PY
