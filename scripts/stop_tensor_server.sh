#!/bin/bash
# Gracefully stop the resident Kimi-K3 server and release its distributed lock.
set -euo pipefail
cd "$(dirname "$0")/.."

LOCK_OWNER="work/.distributed-run.lock/owner"
[[ -f "$LOCK_OWNER" ]] || {
    echo "No resident distributed run is locked."
    exit 0
}
read -r _owner_pid run_id <"$LOCK_OWNER"
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "Unsafe run id in $LOCK_OWNER: $run_id" >&2
    exit 1
}
READY="work/server/$run_id/ready.json"
[[ -s "$READY" ]] || {
    echo "Run $run_id is not a ready tensor server; refusing to stop it." >&2
    exit 1
}
service_pid="$(.venv/bin/python -c \
    'import json,sys; value=json.load(open(sys.argv[1])); assert value["backend"] == "jaccl"; print(value["pid"])' \
    "$READY")"
[[ "$service_pid" =~ ^[0-9]+$ ]] && kill -0 "$service_pid" 2>/dev/null || {
    echo "Server PID $service_pid is not alive." >&2
    exit 1
}
kill -TERM "$service_pid"
for _attempt in {1..120}; do
    if [[ ! -e "$LOCK_OWNER" ]]; then
        echo "Stopped resident server $run_id cleanly."
        exit 0
    fi
    sleep 0.5
done
echo "Server $run_id did not release its lock within 60 seconds." >&2
exit 1
