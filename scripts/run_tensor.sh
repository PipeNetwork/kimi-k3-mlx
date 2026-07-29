#!/bin/bash
# Launch the primary rank-local tensor-parallel deployment through JACCL/RDMA.
set -euo pipefail
cd "$(dirname "$0")/.."

export KIMI_DISTRIBUTED_PROGRAM=scripts/tensor_generate.py
exec scripts/run_distributed.sh \
    --model-root weights/Kimi-K3-2bit-UVMAX-tensor-2 \
    "$@"
