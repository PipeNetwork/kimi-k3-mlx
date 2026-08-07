#!/bin/bash
# Keep one Kimi-K3 TP model resident and serve it over mandatory JACCL/RDMA.
set -euo pipefail
cd "$(dirname "$0")/.."

export KIMI_DISTRIBUTED_PROGRAM=scripts/tensor_server.py
export KIMI_DISTRIBUTED_COMPLETION=exit
exec scripts/run_distributed.sh \
    --model-root weights/Kimi-K3-2bit-UVMAX-tensor-2 \
    "$@"
