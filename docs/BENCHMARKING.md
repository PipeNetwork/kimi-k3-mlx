# Benchmarking policy

Benchmarks in this project are evidence, not decoration. A result is accepted
only when another owner of equivalent hardware can reproduce it from a commit,
checkpoint revision, host topology, command, and machine-readable record.

## Official distributed profile

- Two M3 Ultra Mac Studios, 512 GiB unified memory each.
- Direct Thunderbolt 5 RDMA connection.
- JACCL backend; TCP ring and Ethernet results must be labeled development-only.
- MLX 0.32.0 and mlx-lm 0.31.3 from `uv.lock`.
- `kernelpool/Kimi-K3-2bit-UVMAX` revision
  `edb5113218df612f4a92f95145680f3f8eacd375`.
- Tensor-parallel rank-local checkpoint on each host; both execute all layers.
- `MLX_METAL_FAST_SYNCH=1` (set by `scripts/run_tensor.sh` through the shared launcher).
- Deterministic sampling (`temperature=0`, seed 0) unless the result explicitly
  studies another sampler.
- Default JACCL mesh collectives. Set `MLX_JACCL_RING=1` only for an explicitly
  labeled comparison; the runner records and validates the selected mode.

## Canonical reference result

Commit `4f8564c`, suite run
`4f8564c-tp-canonical-20260729T211125Z`, completed all nine paired records:

| Metric | Result |
|---|---:|
| Decode median | 3.632 tok/s |
| Decode range | 3.436–3.649 tok/s |
| Prefill median | 47.911 tok/s |
| Peak memory | 414.302 GB |
| Rank agreement | 9/9 exact generated texts |

The [raw rank records and validated summary](../benchmarks/results/4f8564c-tp-canonical-20260729T211125Z)
are committed. A matched 256-token factual comparison measured JACCL ring at
3.64854 tok/s and default mesh at 3.64891 tok/s; ring provided no speedup, so
mesh remains the primary topology. The earlier serial pipeline proof decoded
at 0.0891 tok/s and remains a correctness baseline only.

## Required procedure

1. Reboot or record significant concurrent workloads; stop unrelated model
   servers. Do not purge filesystem cache between normal interactive runs.
2. Confirm both RDMA ports are active and save the JACCL topology/hostfile.
3. Run one 32-token warm-up to compile kernels and fault/wire weights.
4. Run at least three prompts representing factual prose, code, and long-form
   reasoning. Generate 256 tokens or to EOS, whichever comes first.
5. Repeat the prompt set three times. Report the median decode tok/s and the
   complete range; retain every JSON record.
6. Record prompt tok/s, decode tok/s, prompt/generated token counts, peak memory,
   load time, checkpoint revision, code commit, OS build, and exact MLX versions.

One-off example:

```bash
scripts/run_tensor.sh \
  --run-id "$(git rev-parse --short HEAD)-factual-1" \
  --prompt 'Who is Albert Einstein?' \
  --max-tokens 256
```

Canonical suite (one model load, warm-up, three prompts, three repetitions):

```bash
scripts/run_tensor.sh \
  --run-id "$(git rev-parse --short HEAD)-canonical" \
  --suite benchmarks/prompts.json \
  --warmup-tokens 32 \
  --repetitions 3 \
  --max-tokens 256
```

The runner writes `work/benchmarks/<run-id>-rank<rank>.json` locally on each
host, plus a summary containing median/range. Copy rank 1's records back before
summarizing a run, then validate both ranks and produce the system result:

```bash
scripts/summarize_benchmark.py "$(git rev-parse --short HEAD)-canonical"
```

The validator rejects dirty/mismatched commits, divergent output, missing
active-RDMA evidence, a missing JACCL hostfile hash, or a disabled MLX fast-sync
setting. It conservatively uses the slower rank's observed throughput for each
paired generation. Never hand-copy just a tok/s number into a pull request.

## Correctness gates

A faster run is rejected if any of these fail:

- Stage provenance or byte-size validation.
- Tiny-model bit-exact prefill/decode parity.
- Deterministic output agreement between ranks.
- Coherent full-model smoke output.
- No JACCL/RDMA proof in the launch log.
- OOM, page-fault thrashing, truncated generation, or changed checkpoint.

Use `mlx_lm.stream_generate`, which enters mlx-lm's wired-memory context. The
upstream project previously demonstrated why a hand-written loop can understate
decode by roughly 27x when weights are repeatedly faulted from SSD.

## Performance work

Every optimization pull request must include:

- Before/after JSON records from the same commit parent, prompt set, and stage.
- Median and range, not the best run.
- A correctness test that would fail if the optimization changed outputs.
- Peak-memory and prompt-throughput changes, even when optimizing decode.
- An explanation of communication volume and synchronization count when the
  change touches distributed code.

Prefer optimizations that reduce bytes read or JACCL synchronization without
changing checkpoint quality. Pipeline results are a separate serialized
correctness baseline and must not be mixed with the primary tensor-parallel
series.
