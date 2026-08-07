# Resident batch benchmark

This result uses the complete `kernelpool/Kimi-K3-2bit-UVMAX` checkpoint on two
M3 Ultra 512 GiB Mac Studios. Each rank loaded a verified 383.336 GiB tensor
stage. The service ran at commit `b7c115bb31b4f9e9f0949c0ba6e35c5527a8a075`
with MLX 0.32.0, mlx-lm 0.31.3, `MLX_METAL_FAST_SYNCH=1`, JACCL mesh, and one
active Thunderbolt RDMA device (`rdma_en3`) per host.

The concurrency sweep generated 64 tokens for each simultaneous request. All
requested concurrencies formed one exact server batch; no split batches were
accepted by the benchmark.

| concurrency | end-to-end aggregate tok/s | decode-only aggregate tok/s |
|---:|---:|---:|
| 1 | 1.650 | 2.130 |
| 2 | 3.845 | 4.263 |
| 4 | 8.052 | 9.183 |
| 8 | 12.527 | 15.441 |
| 16 | 19.895 | 28.175 |
| 32 | 25.173 | 40.303 |

Two further concurrency-32 repetitions measured 22.272 and 24.273 tok/s
end-to-end, and 39.951 and 39.146 tok/s decode-only. Across all three runs the
median was **24.273 tok/s end-to-end** and **39.951 tok/s decode-only**. The
range was 22.272–25.173 and 39.146–40.303 tok/s respectively.

All three concurrency-32 batches produced the same cross-rank digest
`9545a7ccc2b7ea89fbb44f491548cdd16b8f75a63a3ca9bee86aae004a751e54`.
The service remained ready after 128 requests and 8,144 generated tokens, with
no JACCL, RDMA, parity, or allocation errors. It was deliberately left resident
after the proof to avoid another JACCL initialization.

`sweep.json` and `c32-repeat.json` are the raw client records.
`launcher-metadata.json` records the exact two-rank JACCL topology and command.
