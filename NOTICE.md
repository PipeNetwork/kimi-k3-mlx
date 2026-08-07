# Notices and provenance

This repository is a fork, not a clean-room reimplementation.

## Inherited project

The fork point is PipeNetwork's
[`kimi-k3-mlx`](https://github.com/PipeNetwork/kimi-k3-mlx), commit
`15ecbd0319da9e538b95658b021467b479c5871f` (short form `15ecbd0`). The original
port and conversion/pruning work is principally by David Rhodus / PipeNetwork.
Preserve its history and credit in derived publications and model cards.

The fork point did not contain a top-level `LICENSE` file, and GitHub does not
identify a repository license. This fork cannot retroactively grant rights in
inherited files. Contributors should help obtain an explicit license from the
original author; until then, reuse of inherited code is governed by permission
from its copyright holder.

## Moonshot source and model weights

Files under `reference/` are unmodified files from
[`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3). Kimi-K3 model
weights, converted weights, configuration, and applicable reference code retain
Moonshot's Kimi K3 license. LLaVA-derived portions identified upstream remain
under Apache 2.0. Consult the model repository's current license before use or
redistribution.

## MLX and UVMAX loader

`kimi_k3_uvmax.py` and `mlx_lm_compat/tool_parsers/kimi_k3.py` start from Tarjei
Mandt's Kimi-K3 contribution in
[`ml-explore/mlx-lm` PR #1626](https://github.com/ml-explore/mlx-lm/pull/1626),
commit `7d505c285b801108a52c23353c7fb6af07204717`. The original Apple copyright
headers are retained. mlx-lm is distributed under the MIT License.

The preferred checkpoint is
[`kernelpool/Kimi-K3-2bit-UVMAX`](https://huggingface.co/kernelpool/Kimi-K3-2bit-UVMAX),
pinned here to revision
`edb5113218df612f4a92f95145680f3f8eacd375`. Its weights retain the Kimi K3
license. Credit checkpoint author Tarjei Mandt (`kernelpool`) when publishing
results derived from it.

## Fork additions

Copyright (c) 2026 Michal Komar and contributors. Original, separable
distributed scripts, tests, benchmark tooling, and documentation first authored
in `michalkomar/kimi-k3-mlx-distributed` are offered under the MIT License in
`LICENSES/Fork-Additions-MIT.txt`. That grant covers only copyright held by
those contributors and does not relicense inherited or third-party material.

The root `LICENSE` file is a scope manifest, not a claim that the combined work
has one project-wide license. A conventional project-wide OSS license still
requires the original author to license the inherited body.
