# Contributing

Thank you for helping make unpruned Kimi-K3 inference on Apple Silicon faster,
more reliable, and easier to reproduce.

## Before opening a change

Open an issue for architectural changes, checkpoint formats, new quantization
schemes, or any change that can invalidate benchmark comparability. Small bug
fixes, tests, and documentation corrections may go directly to a pull request.

Keep pull requests focused. Explain the problem, the design choice, hardware
used, and validation performed. Link upstream work and retain copyright/license
headers when code is adapted. Do not commit model weights, tokens, machine-local
hostfiles, private paths, or benchmark claims without raw records.

## Development setup

```bash
uv sync --frozen
scripts/install_model.sh .venv/bin/python
scripts/test_all.sh
```

Distributed changes must also pass the tiny two-rank parity test:

```bash
.venv/bin/python scripts/make_tiny_uvmax_fixture.py
.venv/bin/mlx.launch --backend ring --hosts 127.0.0.1,127.0.0.1 -- \
  "$PWD/.venv/bin/python" scripts/distributed_smoke.py

# Rank-local tensor conversion and 64-token/32-step TP parity.
.venv/bin/python -m unittest tests.test_tensor_stage
.venv/bin/mlx.launch -n 2 -- \
  "$PWD/.venv/bin/python" scripts/distributed_tensor_smoke.py
```

Ring is used only for local CI parity. Full-model performance or compatibility
claims require the JACCL/RDMA profile in `docs/BENCHMARKING.md`.

## Pull-request requirements

- Tests cover failure and success paths.
- `git diff --check` is clean and the full suite passes.
- Public interfaces and operational changes are documented.
- Weight/checkpoint provenance is immutable and auditable.
- Performance changes include before/after JSON records and correctness proof.
- New dependencies are justified, pinned when stability requires it, and
  regenerated in `uv.lock`.
- Contributions do not weaken the production JACCL requirement or introduce a
  silent fallback.

By submitting a contribution, you certify that you have the right to submit it
and agree that your new contribution may be distributed under the license you
state in the pull request. Inherited files keep their existing terms; see
`NOTICE.md`. Until the original project's top-level software license is
clarified, do not describe the entire combined work as MIT-licensed.

Be direct and respectful in technical discussion. Maintainers may close work
that cannot be reproduced, removes attribution, includes secrets/weights, or
repeatedly ignores review and safety requirements.
