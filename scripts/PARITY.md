# Layer parity: MLX port vs Moonshot's torch reference

The vision tower has torch parity (1.5e-6, all 27 layers). The **text** tower
never did, because its KDA path needs `fla` — a CUDA/Triton package — so
`tests/test_vision_parity.py` stubs it out and the comparison never happens.
Every text-tower claim therefore rests on architecture tests, key coverage and
per-expert cosine, none of which can catch a wrong recurrence.

A DGX Spark runs the real kernels. This is that missing test.

## Layout

| | |
|---|---|
| `parity_extract.py` | pull one layer's non-expert tensors out of the source shards (Mac) |
| `parity_dump.py` | build the MLX module, run seeded input, bundle weights + I/O (Mac) |
| `parity_check.py` | build the reference module, run the same bytes, compare (CUDA box) |
| `parity_probe_gate.py` | what `fla` actually does with K3's length-128 `A_log` (CUDA box) |

The bundle carries the **source-format weights alongside the I/O**, so the CUDA
side needs no shard access and the two implementations cannot drift on weight
loading. What is being compared is the arithmetic.

Layer `L` lives entirely in shard `L+1`, so a layer costs one shard read.

## Run

Mac:

```bash
scripts/install_model.sh                       # register kimi_k3.py into mlx-lm
scripts/parity_extract.py --layer 13           # -> out/parity/layer13_src.safetensors
scripts/parity_dump.py    --layer 13 --seqlen 64
scp -c aes128-gcm@openssh.com out/parity/layer13_mlx.safetensors spark:~/k3-parity/
```

Spark (`~/k3-parity`, venv already built):

```bash
.venv/bin/python parity_check.py --bundle layer13_mlx.safetensors \
    --config config.json --dtype float32
```

`--dtype float32` removes bf16 rounding as a variable; results are unchanged
either way, which is itself worth knowing.

### Spark environment

`~/k3-parity/.venv` — torch 2.13.0+cu130 (aarch64), `fla-core` 0.5.2,
triton 3.7.1, **transformers pinned to 4.57.1**. v5 drops
`transformers.utils.generic.OutputRecorder`, which the reference imports.
`python3-dev` is required or Triton cannot JIT its `cuda_utils` shim.

Both KDA kernels (`chunk_kda`, `fused_recurrent_kda`) compile and run on
sm_121 at full K3 dims — see `kda_smoke.py`.

## Results, seqlen 64, seed 0

| layer | kind | before | after | |
|---|---|---|---|---|
| 3 | MLA | 0.99998 | 0.99998 | pass (control, untouched) |
| 12 | KDA | 0.94717 | **0.99999** | fixed |
| 13 | KDA | 0.92224 | **0.99999** | fixed |

The first run found a real bug in `_compute_g`; see [What this found](#what-this-found).

MLA passing is the control: it shares the bundle, the loader, the comparison and
the dtype handling with the KDA runs, so the harness demonstrably reaches
parity. It also clears the `[K3-4]` items — q-LoRA, NoPE, sigmoid output gate.

The KDA gap is dtype-independent (bf16 and fp32 agree to 5 decimals) and
systematic: MLX's output runs **7-18% hotter** than the reference in mean
magnitude. Ruled out so far:

- **output gate** — `FusedRMSNormGated(activation='sigmoid')` computes
  `rms(x)·σ(g)`, exactly the MLX form (verified against both candidate formulas).
- **short conv** — both `silu`, same causal alignment.
- **gate space** — MLX's kernel takes `g` in exp space and `mlx_lm`'s stock
  `compute_g` agrees; `fla`'s precomputed path is log space. Two kernels, two
  conventions, internally consistent. Feeding fla exp-space g explodes to 1e24.

## What this found

`_compute_g` was wrong in two independent ways, and they partly cancelled, which
is why the port produced fluent text instead of garbage.

### 1. `A_log` is per-HEAD, not per-channel

`fla/ops/kda/gate.py` — a file whose header reads *"modified and supported by the
Moonshot AI Team"* — loads it as

```python
i_t, i_h = tl.program_id(0), tl.program_id(1)   # grid = (cdiv(T, BT), H)
b_A = tl.load(A_log + i_h)                      # H = g.shape[-2] = num_heads
b_yg = ... exp(b_A) ...                         # scalar, broadcast over head_dim
```

so `A_log` is indexed by **head** and broadcast as a scalar across `head_dim`.
Its docstring says *"Parameter tensor with `H` elements"*, and `naive_kda_gate`
does `A_log.view(H, 1)`. Contrast `dt_bias`, which is `view(H, -1)` — genuinely
per-head-per-channel.

The checkpoint ships `A_log` as `[128]` because it is **zero-padded from 96**:

```
layer 12  [0:96]   min -0.4684  max +1.0066  mean +0.2912  std 0.3263
          [96:128] all exactly 0.0
```

That padding is what makes the per-channel reading look forced by the shape.
Passing the `[128]` tensor to fla is silently harmless — Triton reads entries
0..95 by pointer arithmetic and ignores the rest. The reference's
`torch.empty(num_heads)` init was never stale.

### 2. The safe-gate branch is a different function, not a clamp

```python
USE_LOWER_BOUND: b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)
else:            b_gate = -exp(b_A) * softplus(b_s)
```

K3 sets `gate_lower_bound = -5.0`, so the **sigmoid branch is the live one**.
There is no softplus in it and no clamp anywhere. `_compute_g` was computing
`max(-exp(A_log)·softplus(s), -5)`.

### Scored against the kernel

`parity_probe_gate2.py`, layer 13, real weights:

| candidate | cosine |
|---|---|
| **`lower_bound·sigmoid(exp(A_log_head)·s)`** (fla) | **0.99999505** |
| `lower_bound·sigmoid(exp(A_log_chan)·s)` | 0.94442797 |
| `clamp(-exp(A_log_head)·softplus(s))` | 0.90362370 |
| `clamp(-exp(A_log_chan)·softplus(s))` — the old `_compute_g` | 0.91321528 |

Both errors made the decay too weak, so MLX retained more state and ran 7-18%
hot in output magnitude — the systematic bias that showed up before any of this
was localised.

`tests/test_kimi_k3.py` asserted the per-channel reading (`[K3-5]`); it now pins
the per-head one, that the zero padding is ignored, and that the lower bound is
reached by saturation rather than clamping.
