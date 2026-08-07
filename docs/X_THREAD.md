# X launch thread

Prepared from the validated resident-service benchmark in
`benchmarks/results/b7c115b-resident-batch-20260807T063408Z`. Each numbered
paragraph is one post.

**1/10**

We now have full Kimi-K3 inference running across two M3 Ultra Mac Studios — 1 TB unified memory total — using MLX tensor parallelism over mandatory JACCL/RDMA.

The result: 39.95 decode tok/s at concurrency 32, reproducibly. 🧵

**2/10**

This is the real 2.78T-parameter Kimi-K3 (104B active), not a toy fixture or pruned substitute.

Checkpoint: kernelpool/Kimi-K3-2bit-UVMAX, 816.77 GB on disk. Each Mac holds a 383.34 GiB rank-local stage.

https://huggingface.co/kernelpool/Kimi-K3-2bit-UVMAX

**3/10**

Validated benchmark, 3 identical concurrency-32 runs:

• decode: 39.15 / 39.95 / 40.30 tok/s
• end-to-end: 22.27 / 24.27 / 25.17 tok/s
• median decode: 39.95 tok/s
• median E2E: 24.27 tok/s

64 generated tokens/request.

**4/10**

Correctness mattered as much as speed:

• exact cross-rank output digest match
• zero errors across 128 requests / 8,144 generated tokens
• strict JACCL mesh verification
• no silent fallback to TCP/ring

Raw records are committed with the code.

**5/10**

The big speed lesson: keep ~766.7 GiB of rank-local weights resident, then batch concurrent requests through one distributed decode loop.

That removes repeated model loading and amortizes synchronization across the two machines. Single-stream latency is still the next frontier.

**6/10**

The fork adds rank-local TP conversion, durable two-host launch, strict RDMA preflight, persistent OpenAI-compatible serving, continuous batching, parity checks, crash-safe run locking, and reproducible benchmark artifacts.

https://github.com/michalkomar/kimi-k3-mlx-distributed

**7/10**

Major kudos to David Rhodus / PipeNetwork. Their kimi-k3-mlx port, converter, REAP work, pipeline implementation, and CUDA/fla-backed KDA parity correction are the upstream foundation we built on.

https://github.com/PipeNetwork/kimi-k3-mlx

**8/10**

Also credit where it belongs:

• Moonshot AI — Kimi-K3
• Tarjei Mandt / kernelpool — UVMAX checkpoint + MLX-LM loader work
• Apple MLX + MLX-LM teams
• JACCL contributors — Apple-Silicon RDMA collectives

This result depends on all of them.

**9/10**

We merged the latest upstream direction while preserving auditable history. UVMAX already used the corrected per-head KDA gate, so the distributed result remains valid.

Next: lower single-stream latency and push sustained throughput higher.

**10/10**

Code, exact model revision, launch path, validation digest, and raw JSON are public. Dual-M3 Ultra reproductions and contributions are welcome.

Licensing is documented per source; inherited upstream code still needs an explicit license from its author.
