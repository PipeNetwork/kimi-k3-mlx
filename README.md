# Kimi-K3 MLX Distributed

This is a real GitHub fork of
[PipeNetwork/kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx), focused
on the fastest reproducible **unpruned Kimi-K3 inference across two 512 GiB M3
Ultra Mac Studios**. The deployment path is deliberately strict: Thunderbolt
RDMA and MLX's JACCL backend are mandatory; the production runner refuses a TCP
ring fallback.

The preferred checkpoint is
[kernelpool/Kimi-K3-2bit-UVMAX](https://huggingface.co/kernelpool/Kimi-K3-2bit-UVMAX),
a 816.77 GB mixed-precision MLX build whose 2.72T routed-expert parameters are
2-bit. It keeps the quality-sensitive attention, shared experts, routers, and
embeddings at 4–8 bits while fitting as two rank-local tensor-parallel
checkpoints. The original
883 GB uniform 2-bit streaming conversion remains supported as the fully
reproducible build path.

Current distributed work, exact storage, cabling, and commands are documented
in [docs/DISTRIBUTED.md](docs/DISTRIBUTED.md). Benchmark submissions must follow
[docs/BENCHMARKING.md](docs/BENCHMARKING.md), and contributions follow
[CONTRIBUTING.md](CONTRIBUTING.md).

The canonical two-host proof at commit `4f8564c` completed all nine paired
JACCL/RDMA records with identical rank outputs: median decode **3.632 tok/s**
(range 3.436–3.649), median prefill **47.911 tok/s**, and **414.302 GB** peak
memory. The complete machine-readable records are in
[`benchmarks/results/4f8564c-tp-canonical-20260729T211125Z`](benchmarks/results/4f8564c-tp-canonical-20260729T211125Z).

MLX (Apple Silicon) port of [**moonshotai/Kimi-K3**](https://huggingface.co/moonshotai/Kimi-K3) —
Moonshot's 2.78T-total / 104B-active native-multimodal MoE, built on Kimi Delta
Attention (KDA) and Attention Residuals (AttnRes), with a 1M-token context.

`kimi_k3.py` is a single, stock-`mlx-lm`-compatible model definition for the text
tower; `kimi_k3_vision.py` is the vision tower and projector; `kimi_k3_vl/` is
the mlx-vlm wrapper that joins them.
`scripts/convert.py` is a streaming converter — `mlx_lm convert` cannot be used
here, because it materialises the whole model and K3 is 5.6 TB in bf16.

> **Read this before planning a deployment.** The unpruned model does not fit
> one Mac. This fork partitions it across two 512 GiB M3 Ultras; the upstream
> pruned checkpoints remain the right choice for single-machine inference. See
> [Reality check](#reality-check).

## Architecture

| Feature | Detail |
|---|---|
| MoE | **896 routed experts, top-16**, `moe_intermediate_size=3072`; sigmoid router + aux-loss-free `e_score_correction_bias`; **2 shared experts** on every token |
| **Stable LatentMoE** | routed experts run in a **3584-d latent**, not the 7168-d residual: `down_proj` → experts → `RMSNorm` → `up_proj` |
| Attention | 93 layers: **69 KDA** + **24 gated MLA**, 96 heads, `head_dim=128` |
| KDA | short conv (k=4) on q/k/v, low-rank forget gate (`f_a`/`f_b`), **full-rank output gate**, `gate_lower_bound=-5.0` |
| MLA | q-LoRA rank 1536, kv-LoRA rank 512, **NoPE** (no rotary at all), **sigmoid output gate** (`g_proj`) |
| **AttnRes** | every 12 layers pushes the residual onto a stack; each sub-layer input is a **softmax mixture** over that stack, scored by a rank-1 `[1, 7168]` direction |
| Activation | **SiTU-GLU**: `β·tanh(g/β)·σ(g) · β_lin·tanh(u/β_lin)`, β=4, β_lin=25 |
| Dense layer | layer 0 only (`first_k_dense_replace=1`, `intermediate_size=33792`) |
| Vocab | 163,840 · tiktoken · `tie_word_embeddings=False` |
| Source format | routed experts ship as **MXFP4** (`weight_packed` u8 + e8m0 `weight_scale`, group 32); everything else bf16 |

Parameter budget — **97.94% of the model is routed experts**:

| component | params | share |
|---|---|---|
| routed experts (MXFP4 in source) | 2722.7 B | 97.94% |
| KDA attention ×69 | 30.6 B | 1.10% |
| shared experts | 12.2 B | 0.44% |
| MLA attention ×24 | 5.6 B | 0.20% |
| latent-MoE up/down | 4.7 B | 0.17% |
| embed + lm_head | 2.4 B | 0.08% |
| everything else | 1.8 B | 0.07% |
| **total** | **2780.0 B** | |

This model reproduces the published repo size exactly (1.561 TB predicted vs
1.561 TB actual), which is what validates the accounting above.

### Porting notes

Five things in K3 exist nowhere else in MLX. `mlx-lm`'s `kimi_linear.py`
(Kimi-Linear-48B-A3B) covers KDA, MLA and DeepSeek-style sparse MoE; everything
below is new, and each is marked `[K3-n]` in `kimi_k3.py`:

1. **SiTU-GLU** activation, computed in fp32 (the tanh saturation points are far
   enough out that bf16 rounding is visible).
2. **AttnRes** — a softmax mixture over a growing stack of block residuals,
   threaded through every layer and applied once more at the model output.
3. **LatentMoE** — experts operate in 3584 dims; shared experts do not.
4. **q-LoRA + output-gated MLA** — Kimi-Linear had a plain `q_proj` and no gate.
5. **Per-channel KDA decay.** The reference `modeling_kimi_linear.py` allocates
   `A_log` as `[num_heads]` = `[96]`, but **every shard ships `[128]` = `head_dim`**,
   while `b_proj` is `[96, H]` and `dt_bias` is `[12288]`. fla's kernel broadcasts
   `A_log` against `g` of shape `(B,T,96,128)`, so a length-128 vector can only
   align with the trailing axis: the decay is per-channel and shared across
   heads. The reference init code is stale w.r.t. the released weights — the
   shapes are authoritative. Assuming the Kimi-Linear layout here produces
   silent garbage, so `tests/test_kimi_k3.py` asserts it.

### Vision tower

`kimi_k3_vision.py` — a 3D (video-capable) MoonViT, 27 layers / 447 M params,
plus the patchmergerv2 projector. mlx-vlm's `kimi_vl/vision.py` (MoonViT for
Kimi-VL) is the nearest existing MLX code; this shares its complex64 2D-rope
approach and differs in six ways, marked `[K3-V*]` in the file:

| | difference |
|---|---|
| `[K3-V1]` | 3D grids `(t,h,w)`: a 1D sin-cos **time** embedding on top of the learnable 2D grid, and rope frequencies repeated per frame |
| `[K3-V2]` | **bilinear** position-embedding interpolation (Kimi-VL used bicubic); mlx-vlm ships bicubic/nearest kernels only, so this is written here against torch's `align_corners=False` half-pixel convention |
| `[K3-V3]` | `qkv_hidden_size` 1536 ≠ hidden 1024 — attention runs wider than the residual, so `wo` is `[1024, 1536]` and **not square** |
| `[K3-V4]` | RMSNorm, not LayerNorm |
| `[K3-V5]` | no biases anywhere |
| `[K3-V6]` | `sd2_tpool` merge: mean-pool over frames, then 2×2 spatial merge, then a 2-layer MLP projector with **exact (erf)** GELU and a post-RMSNorm |

The ViT blocks use **tanh-approximate** GELU while the projector uses **exact**
GELU. The reference deliberately uses both; they are different functions.

Unlike the 2.78T text tower, the vision tower is small enough to validate for
real — `tests/test_vision_parity.py` runs Moonshot's actual torch code on CPU
against the MLX port **at full K3 dimensions, all 27 layers**, and matches to
**1.5e-6 relative error** end to end (float32 round-off).

One judgement call: the reference builds `nn.RMSNorm(dim)` with no `eps`, so
torch falls back to `finfo(dtype).eps` — 1.19e-7 in fp32 but **7.8e-3 in bf16**,
materially different normalizers. `VIT_NORM_EPS` uses the fp32 value, matching an
upcast implementation. `projector_ln_eps` (1e-5) is given explicitly by the
config and applies only to the projector's post-norm.

### Multimodal wrapper

`kimi_k3_vl/` is the mlx-vlm-shaped package: `Model` (glue), `LanguageModel`,
`VisionModel`, and config plumbing. The load-bearing piece is the **expanding
merge**.

K3's processor rewrites each `<|kimi_image_placeholder|>` in the raw text into

```
<|media_begin|>image {W}x{H}<|media_content|><|media_pad|><|media_end|>
```

so the tokenized prompt carries **exactly one** `<|media_pad|>` (id 163605) per
image, which must expand into that image's *entire* feature block. That is
**not** how most LLaVA-style models work, and not what mlx-vlm's existing
Kimi-VL glue does — there the processor has already emitted one placeholder per
image token and merging is a same-length scatter.

Applying the scatter here would keep one token per image and silently discard
the rest: no crash, no error, just a model that writes fluent prose while being
effectively blind to most of the picture. `tests/test_vl_wrapper.py` asserts the
expansion directly against hand-built expected sequences.

Rows are built by concatenating the spans between placeholders — O(images)
concatenations, and far easier to check than the reference's index arithmetic.
With batch > 1 rows are left-padded, matching the reference's `left_padding`
branch.

Two further traps:

- `full_attn_layers` / `kda_layers` in the config are **1-based**: config `[4,8,…]`
  means tensor layers `3,7,…`. MLA lands on `3,7,…,91,92`.
- The source declares `quant_method: "compressed-tensors"`, which stock mlx-lm
  maps to **affine 4-bit group 32** — wrong for `mxfp4-pack-quantized` data. The
  converter strips that block and writes an explicit `quantization` block.

## Tiers

**The source experts are already MXFP4 — 4 bits of real information.** That
governs the whole tier list.

MLX has a native `mxfp4` quantization mode using the *identical* encoding
(low-nibble-first codes, e8m0 scales). So the source bytes can be reinterpreted
into MLX with **no arithmetic at all** — the `mxfp4` tier is bit-exact, verified
to zero error end-to-end. Requantizing those same weights to affine 4-bit costs
**9.8% mean relative error** and is *larger* (4.5 vs 4.25 bpw), so a plain
`4bit` tier is strictly dominated and is not built.

| tier | expert bpw | non-experts | size | expert error vs source |
|---|---|---|---|---|
| **mxfp4** | 4.25 | bf16 | 1.56 TB (1.45 TB experts + 115 GB rest) | **0 — bit-exact** |
| `3bit` | 3.50 | 4-bit | 1.22 TB | lossy (2nd quantization pass) |
| `mixed2` | 2.50 | **bf16** | 0.97 TB | lossy |
| `2bit` | 2.50 | 4-bit | 0.88 TB | lossy |

The bpw column is **expert** bits/weight, not the model average — experts are
97.94% of parameters, so it is the number that moves the size. `mixed2` and
`2bit` quantize experts identically and differ *only* in what happens to the
other 2%: `mixed2` leaves them bf16, which costs 83 GB and isolates the question
"are 2-bit experts survivable when nothing else is degraded?". Both accept
`--nonexpert-bits` to trade that down.

(Those two profiles were byte-identical until they were separated; the size and
bpw figures here are recomputed from the parameter table above, which reproduces
the measured 1.561 TB for `mxfp4` exactly. `tests/test_convert_roundtrip.py`
now asserts no two profiles describe the same quantization.)

`6bit`, `8bit` and `bf16` are not built: they would store upcast 4-bit values at
2.26 / 2.95 / 5.56 TB, larger than the source, with zero quality gain.

## Multimodal: verified end to end

`scripts/vl_generate.py` runs the whole chain on a *published* artifact — PIL
image → `KimiK3VisionProcessor` → MLX vision tower → `<|media_pad|>` expansion →
quantized text tower → tokens:

```
image (448, 448) -> 1024 patches, grid (1, 32, 32), 256 image tokens
prompt 14 tokens, 1 media_pad -> expands to 269
prefill 269 merged positions in 9.8s (expected 269)
--> The image shows two geometric shapes rendered as
```

The test image contains exactly two shapes (a red square, a blue circle) and the
model reports "two geometric shapes" — correct count and category from an image
it has never seen.

The `269` is the load-bearing number. A same-length scatter — what mlx-vlm's
Kimi-VL glue does, and the natural thing to copy — would have merged to 14
positions, shown the model one token instead of 256, and still produced fluent
text. `scripts/vision_test.py` covers the tower separately: weights bit-identical
to source, token counts matching the processor, and cosine 0.21 between two
different images (a tower ignoring its input would pass every shape check).

## Published

| repo | size | experts | calibrated on | tok/s |
|---|---|---|---|---|
| [Kimi-K3-REAP73-MLX-mxfp4-q8](https://huggingface.co/pipenetwork/Kimi-K3-REAP73-MLX-mxfp4-q8) | 451 GB | 242/896 | mixed (11 sources) | **5.51** |
| [Kimi-K3-REAP80-MLX-mxfp4-q8](https://huggingface.co/pipenetwork/Kimi-K3-REAP80-MLX-mxfp4-q8) | 350 GB | 179/896 | mixed | **5.54** |
| [Kimi-K3-REAP73-zh-code-MLX-mxfp4-q8](https://huggingface.co/pipenetwork/Kimi-K3-REAP73-zh-code-MLX-mxfp4-q8) | 451 GB | 242/896 | **Chinese + code** | **5.51** |
| [Kimi-K3-REAPgraded-MLX-mxfp4-q8](https://huggingface.co/pipenetwork/Kimi-K3-REAPgraded-MLX-mxfp4-q8) | 451 GB | 326/896, two-bank | mixed | **2.68** |

Surviving experts are bit-exact copies of Moonshot's MXFP4 in every build; the
only information lost is the pruning itself. `--nonexpert-bits 8` is required,
not cosmetic: with bf16 non-experts the 242-expert build is 504 GB and is
OOM-killed during load on a 512 GiB machine.

**These are interactive.** ~5.5 tok/s on a 512 GiB M3 Ultra — 58% of the 9.5
tok/s that per-token traffic allows against ~819 GB/s, so bandwidth-bound as
expected rather than pathological. Each token reads ~87 GB of weights, 61 GB of
which is non-expert tensors touched every token despite being 2% of parameters.
Figures are the mean of three prompts × 96 tokens, cross-checked against a
separate harness; spread within a tier is under 1%.

Two things the numbers say that the earlier text got backwards:

- **Pruning buys memory, not speed.** Per-token traffic depends on `top_k` and
  non-expert precision, never on how many experts are *stored* — so the 350 GB
  179-expert build and the 451 GB 242-expert build decode identically. Prune to
  fit a smaller machine, not to go faster.
- **The graded two-bank build costs 2.06× decode** (2.68 vs 5.51 at the same
  footprint). See [REAP expert pruning](#reap-expert-pruning); the per-layer sync
  that a single-layer microbenchmark measured at 1.16× does not amortise across
  92 layers.

> **Correction, 2026-07-28.** This table previously read 0.14–0.20 tok/s, and
> this section described an irreducible bandwidth wall that "no prune ratio
> fixes". Both were wrong, by roughly 27x. `scripts/smoke.py` hand-rolled its
> decode loop and so never entered the `wired_limit` context manager that
> `mlx_lm.stream_generate` applies (`generate.py:714`); the weights were left
> unwired and every decoded token faulted them back from SSD instead of reading
> RAM. It survived scrutiny because **prefill is compute-bound and touches each
> page once** — prompt processing looked healthy at ~73 tok/s throughout while
> decode ran at 2% of the memory system's capability, and only decode re-reads
> every weight per token. The "gains track size almost exactly" observation that
> appeared to confirm the bandwidth theory was really tracking how much
> page-faulting each build incurred. **Users were never affected:** `generate`,
> `stream_generate`, the CLI, `BatchGenerator` and `server.py` all wire
> correctly, so this was only ever wrong in our own measurements. Reported by
> [@pudepiedj](https://huggingface.co/pudepiedj), who measured 5.608 tok/s on
> REAP80 against a card claiming 0.20. `scripts/mlxmem.py` now wires to the same
> limit mlx-lm picks, and any harness reporting tok/s must call it.

## Reality check

**No unpruned tier is runnable on one Apple Silicon machine.** Peak unified
memory tops out at 512 GB (M3 Ultra Mac Studio); the original smallest full tier
is 883 GB. Fitting 4-bit into 512 GB would need ≤1.38 bits/weight. That is what
[REAP pruning](#reap-expert-pruning) exists to solve for one Mac. This fork's
second route is a two-machine JACCL tensor-parallel deployment. Each host keeps
one half of every sharded matrix, so the 816.77 GB UVMAX checkpoint becomes
383.335683 GiB per rank. A serial 47/46-layer pipeline remains as a
correctness/debugging baseline, not the performance deployment.

Consequences, stated plainly:

- In the original single-node work, the **unpruned tiers** had never produced a
  token because they cannot fit one M3 Ultra. Do not reinterpret the upstream
  single-node benchmark table as a distributed result. Two-node results in this
  fork are recorded separately under `work/benchmarks/` and only promoted to
  this README after a reproducible JACCL run.
- What *is* verified for them: bit-exactness of the mxfp4 tier against the
  source, 100% checkpoint-key coverage over all 497,220 tensors, prefill/decode
  agreement of the architecture at small scale, and per-expert cosine similarity
  against the source for the lossy tiers. See [Verification](#verification).
- The 2-bit tiers are a **double quantization** on top of Moonshot's MXFP4, and
  it costs real quality. This is now measured rather than guessed: a REAP73
  build with 2-bit experts is 262 GB, small enough to run with room to spare,
  and scores **17.03** held-out perplexity while looping on Chinese and French
  prompts where the mxfp4-expert graded build does not. `mixed2` keeps
  non-experts at bf16 for exactly this reason.

## Verification

```bash
scripts/test_all.sh                     # registers both loaders, runs all 107 tests
scripts/verify.py --path out/Kimi-K3-MLX-mxfp4 --src Kimi-K3-src
scripts/perplexity.py --path out/<tier> --calib-text out/calib.txt \
    --skip-tokens <past the calibration prefix> --out out/ppl.npz
```

| suite | what it covers | |
|---|---|---|
| `test_prune_apply.py` | REAP plan application + router renumbering | 18 |
| `test_kimi_k3.py` | architecture + 497,220-key coverage | 19 |
| `test_vl_wrapper.py` | placeholder expansion + multimodal path | 11 |
| `test_reap.py` | saliency accumulation and planning modes | 10 |
| `test_vision_parity.py` | vision tower vs torch reference | 9 |
| `test_processor_integration.py` | real processor -> MLX tower | 7 |
| `test_convert_roundtrip.py` | converter + profile distinctness, on a mini-K3 | 5 |
| `test_uvmax.py` | loader, stage integrity, benchmark/RDMA proof, pipeline split | 8 |
| `test_tensor_stage.py` | offline TP conversion, index, strict rank-local load | 1 |
| `test_tensor_server.py` | persistent control protocol, parity digest, HTTP batching | 6 |

Use `scripts/test_all.sh` rather than running suites directly: the tests import
`mlx_lm.models.kimi_k3`, so editing `kimi_k3.py` without re-registering it
silently tests the previous version.

`tests/test_processor_integration.py` runs Kimi-K3's actual
`KimiK3VisionProcessor` on real PIL images and asserts the MLX tower emits
exactly the token count `media_tokens_calculator` predicts (352 / 64 / 2691 for
611x437, 224x224, 1920x1080). That equality is what keeps the prompt's reserved
image slots aligned with the features that fill them.

`tests/test_vision_parity.py` imports Moonshot's real `modeling_kimi_k3.py` and
runs it on CPU with matched weights (`fla` is stubbed — it is a CUDA/Triton
dependency of the *text* tower's KDA kernels, never touched here).

`tests/test_convert_roundtrip.py` builds a structurally faithful miniature in the
*source* on-disk format (MXFP4 expert pairs, `language_model.*` prefixes, sharded
with an index) and pushes it through `convert.py` → `mlx_lm.load` → forward pass
for all four profiles. It asserts the mxfp4 experts come out bit-identical.

`scripts/verify.py` checks a real converted tier without loading it: config
coherence, index integrity, exact module coverage derived from `ModelArgs`, and a
numeric spot-check that dequantizes sampled experts from both the output and the
source. It reports **cosine similarity** per expert — quantization noise barely
moves it, but a mis-stacked expert or a transposed projection drops it to ~0
while magnitude statistics still look plausible.

## REAP expert pruning

`scripts/reap_calibrate.py` + `scripts/reap_plan.py`. This is the only route to
a K3 that *runs* on a 512 GB machine — see [Reality check](#reality-check).

REAP ([Cerebras](https://github.com/CerebrasResearch/reap)) scores each expert by
`S_j = (1/|C|) Σ g_j(x)·‖e_j(x)‖₂` over a calibration set, keeps the most salient
per layer, and lets the router renormalize.

Calibration nominally means running a 1.56 TB model. It does not have to: the
saliency of layer L depends only on the hidden states entering L, its router
gates, and its expert outputs — **nothing downstream**. So it streams exactly
like the converter: build one layer, push the whole calibration set through it,
record, free, advance. Experts run as real mxfp4 quantized layers, so the
arithmetic scored is the arithmetic the published tier executes.

| calibration | peak RAM | expert compute |
|---|---|---|
| 32 × 2048 (64k tok) | ~32 GB | 6.4 PFLOP |
| 64 × 2048 (128k tok) | ~41 GB | 12.7 PFLOP |
| 128 × 2048 (256k tok) | ~58 GB | 25.5 PFLOP |

Peak is dominated by the AttnRes block stack (8 × tokens × 7168), not the
weights. Disk reads 1.56 TB once.

```bash
scripts/make_calib.py --out out/calib.txt --mb 12
scripts/reap_calibrate.py --src Kimi-K3-src --out out/reap_saliency.npz \
    --calib-text out/calib.txt --seqs 64 --seqlen 2048
scripts/reap_plan.py --saliency out/reap_saliency.npz --mode graded \
    --hi 0.15 --lo 0.20 --out out/reap_plan.json
```

Then apply the plan with any profile:

```bash
scripts/convert.py --src Kimi-K3-src --out out/Kimi-K3-REAP73-mxfp4 \
    --profile mxfp4 --prune-plan out/reap_plan.json
```

Pruning **renumbers** experts — new index i is old expert `keep[i]` — so the
router's `gate.weight` rows and `e_score_correction_bias` are reordered to
match. Get that wrong and the model loads, runs, and emits fluent text while
routing every token to the wrong expert; no shape check sees it. The converter
asserts rather than guards on those two tensors, and
`tests/test_prune_apply.py` pins the behaviour: an identity prune is bit-for-bit
a no-op, and a pruned model must equal the full model with the dropped experts'
correction bias set to −inf. A third test rotates the router rows to prove that
equivalence check can actually detect misrouting.

Per-layer expert counts are recorded in `config.json` as `expert_counts`, so
`global` mode's uneven layers load correctly (`kimi_k3.experts_in_layer`).

Planning modes: `uniform` (what published REAP models use), `global` (rank all
layers together, with a routable floor of 4·top_k), and `graded`.

`graded` uses REAP's saliency ranking to allocate **bit-width** rather than
discarding it: top experts at mxfp4 (free on K3), mid experts lower, tail
dropped — ~30% more experts in the same memory (313 vs 240 at 448 GB).

MLX pins one bit width per expert tensor (`QuantizedSwitchLinear` stores scalar
`bits`/`group_size`/`mode`; `gather_qmm` takes them as scalars), so the two
tiers live in two banks inside `kimi_k3.TwoBankSwitchGLU`.

The naive two-bank forward costs 2× expert compute — run both banks over every
routed pair and select. Instead it exploits the sort `SwitchGLU` already
performs: once (token, slot) pairs are sorted by expert index, a bank owning a
contiguous index range owns a contiguous *slice*, so each bank does exactly its
own share. Isolated at K3 REAP dims (3584→3072, 240 experts, top-16), one layer:

| | vs single-bank |
|---|---|
| prefill (256 tok) | 1.05× |
| decode (1 tok) | 1.16× |
| memory, same expert count | **82%** |

The split point is data-dependent, so it must reach the host to slice on — and
that sync is essentially the entire overhead. Decode routes only `top_k` pairs,
too few to amortise two kernel launches plus a device sync, so below 1024 pairs
the indices come down in one transfer and partition with numpy. That alone took
the single-layer decode figure from 1.44× to 1.16×.

**That microbenchmark does not survive contact with the real model.** Measured
end to end on the published builds, all at ~450 GB and all wired:

| build | experts | tok/s |
|---|---|---|
| REAP73 single-bank | 242 | **5.51** |
| REAPgraded two-bank | 326 | **2.68** |

A **2.06× decode penalty** — essentially the full cost of the run-both-and-select
scheme this design exists to avoid. The single-layer 1.16× is real but
misleading: the overhead is a per-layer host sync, and K3 has 92 MoE layers, so
what amortises within one layer's kernel launch does not amortise across 92 of
them serialised down the decode critical path. Graded should if anything be
*faster* — part of its routed experts are 2-bit rather than mxfp4, so it moves
less memory per token — which places the entire 2.06× on the two-bank machinery.

The honest summary: graded buys ~35% more experts and measurably better Chinese
for **half the decode speed**. Whether that trade is worth taking depends
entirely on the workload, and the microbenchmark above should not be read as
predicting the cost.

Correctness rests on one test: give both banks the *same* weights and precision
and the result must match a plain `SwitchGLU` **exactly** (it does, bit-for-bit,
wherever single-bank also sorts). That isolates the partition/unsort machinery
from quantization error, so a bug there cannot hide behind "the low tier is
lossy".

**Calibration data is a modelling choice, not a formality.** Whatever the
corpus under-represents gets pruned away silently — no error, no warning, just a
model that looks fine until someone writes in the missing language.
`scripts/make_calib.py` builds a deliberate mix: 40% code (multi-language +
real Python files), 30% English web, 15% Chinese, 15% split across ja/ru/ko/de/
fr/es/ar. Two things that had to be fixed by measurement rather than assumption:

- C4's pooled `multilingual` config measured **97% Latin script** over its first
  200 documents — effectively a second helping of English. Named per-language
  configs replaced it.
- With that pooled stream, CJK came to **0.03%** of the corpus. Chinese now has
  an explicit share and lands at ~11% (14% of the prefix actually consumed).

Sources are interleaved, not concatenated: `reap_calibrate.py` reads the first
`seqs × seqlen` tokens, so a concatenated corpus would calibrate entirely on
whichever source was written first. C4's `zh` split also carries some
double-encoded documents whose mojibake still matches a naive "has CJK" test, so
a CJK-fraction threshold filters them.

A caveat worth stating: K3's LatentMoE applies RMSNorm to the *combined* expert
output before `up_proj`, so absolute expert norms are partly renormalized away
downstream. That likely makes K3 more robust to pruning than a standard MoE, but
it also means `‖e_j(x)‖` measures something partly cancelled — the criterion may
want reformulating as relative contribution. Two shared experts fire on every
token regardless and survive any pruning, giving a floor nothing touches.

## Experts cluster by domain — measured

`scripts/reap_calibrate.py` tags every calibration token by source corpus and
accumulates saliency per language in one pass (a single fused `(source, expert)`
scatter). `scripts/reap_overlap.py` then compares the top-N expert sets.

Two random top-242 picks from 896 experts overlap **27%** by chance. Against
that baseline:

| pair | overlap | vs chance |
|---|---|---|
| code-python ↔ code-multi | 57.2% | 2.1× |
| lang-de ↔ lang-es | 59.3% | 2.2× |
| lang-de ↔ web-en | 56.5% | 2.1× |
| chinese ↔ lang-ja | 42.8% | 1.6× |
| **chinese ↔ code-python** | **17.8%** | **0.66× — below chance** |

Code, European languages and CJK each form a cluster; code and Chinese are
*anti*-correlated. `scripts/reap_subset.py` exploits this: a tagged run already
holds per-source saliency, so a domain-targeted build needs no second
calibration — just sum the buckets you want.

**It works, and it cuts both ways.** Five builds, all 451 GB (bar one), differing
only in calibration target. Same prompts, greedy decoding, 24 tokens:

| build | retained | Chinese | code |
|---|---|---|---|
| mixed | 59.1% | good, loops back to the prompt by ~18 tok | correct |
| REAP-80 (179 experts) | ~50% | hard repetition loop | correct |
| English+code | 68.4% | **total collapse** | correct |
| Chinese-only | 79.8% | no loop, but vague | **total collapse** |
| **Chinese+code** | **69.3%** | **best — correct and specific, no loop** | **correct** |

Prune a domain's experts and that domain dies; keep them and it improves; combine
two domains and both hold. The expert set determines the capability, and you
select it by selecting the corpus.

Verbatim on `机器学习的基本原理是` ("the basic principle of machine learning is"):

```
mixed         ，机器学习是人工智能的核心，是一切计算机视觉化、网络化的基础。
              机器学习的基本原理是，机器学习是人工智能      <- loops
English+code  ，机器学习的基本原理是，机器学习的基本原理是，机器学习的基本原理是  <- collapse
Chinese-only  ：通过计算机模拟人脑的思维，由计算机实现人类对自然的延伸和扩展
Chinese+code  ：通过训练数据，学习算法，然后对未知数据进行预测。
              机器学习的过程是：输入数据→学习算法→          <- correct and specific
```

and on the code prompt, where Chinese-only lost the capability entirely:

```
mixed / En+code / zh-code    if not intervals: return [] ; intervals.sort(...) ; merged = [...]
Chinese-only                 """Merge overlapping intervals."   x4          <- collapse
```

**A caveat on strength of evidence.** This is one prompt per domain, 24 tokens,
greedy. It is consistent with two independent measurements — expert overlap and
saliency retention — which is why it is believable, but it is not a rigorous
eval. A real claim needs many prompts and held-out perplexity per domain. An
earlier version of this file recorded targeted calibration as a *null* result on
the strength of a single code prompt so over-determined that every build emits
identical tokens; that conclusion was wrong, and the same skepticism applies to
the positive result here.

That caveat is now partly discharged: `scripts/perplexity.py` scores a build on
held-out text bucketed by source corpus, which is the rigorous version of the
eval this section asks for. It requires `--skip-tokens` rather than defaulting
it, because `reap_calibrate.py` consumes the *first* `seqs × seqlen` tokens of
the corpus and scoring from offset 0 grades every build on the data its own plan
was fitted to.

## AWQ on the 2-bit experts — measured

`scripts/awq.py`. K3 makes AWQ unusually cheap: LatentMoE routes every expert in
a layer through the **same** 3584-d latent, so one scale vector per layer covers
all 896 experts — 1.3 MB, against the 1.2 GB a per-expert table would need — and
the inverse folds exactly into `routed_expert_down_proj`:

```
w1' = w1 · s        w3' = w3 · s        down_proj' = down_proj / s
```

`alpha` is grid-searched per layer against activation-weighted reconstruction
error, which is what makes this calibrated rather than a heuristic. Held-out
perplexity over 65,504 tokens against an otherwise byte-identical build:

| | perplexity |
|---|---|
| 2-bit experts, no AWQ | 17.0288 |
| 2-bit experts + AWQ | **16.6573** — −2.18%, 95% CI [−4.56%, −0.34%] |

**The aggregate is a poor summary of what happened.** AWQ does not shift the
typical token: the median per-token NLL delta is −0.0001 nats and only 51% of
tokens improve at all. It *redistributes*, sorted by how hard the control found
each token:

| control NLL band | tokens | mean Δ (nats) |
|---|---|---|
| below median | 32,752 | **+0.0403 — AWQ worse** |
| median–p90 | 26,201 | −0.0359 |
| p90–p99 | 5,895 | −0.2442 |
| top 0.1% | 65 | −0.9426 |

That is the mechanism AWQ advertises — protect the channels carrying the
activation — showing up as **tail repair rather than a mean shift**. It buys hard
tokens by spending easy ones. The net is 13,795 nats gained against 12,350 lost,
a 1.12:1 residue of two much larger opposing effects, which is why it is
fragile: dropping the three most favourable sequences takes −2.18% to −0.76%.

It is also domain-skewed, and the skew tracks the calibration corpus. Chinese was
36.4% of the AWQ calibration set and improves 5.7%; Arabic was 1.3% and regresses
6.6%, Spanish 1.4%, Japanese 2.0%, all with CIs excluding zero. One scale vector
per layer cannot be optimal for every domain at once — the same lesson the
calibration section above records for pruning, arriving by another route.
`scripts/make_balanced_calib.py` exists to remove that confound: it re-samples an
existing corpus to equal token counts per source, round-robined by document so
sequences stay mixed, and takes `--skip-tokens` to stay clear of held-out
regions.

Two things worth knowing before reaching for this:

- The weight-space probe **underestimated the effect by an order of magnitude**.
  Activation-weighted reconstruction error at the chosen alpha beats alpha=0 by
  only 0.18%, because it averages uniformly and is blind to a change that wrecks
  easy tokens while rescuing hard ones.
- AWQ is **incompatible with a graded plan** and `convert.py` refuses the
  combination. The fold divides `down_proj` by `s` once per layer, but the graded
  hi bank is passed through unscaled to stay bit-exact, so those experts would
  read `latent/s`. At `s ≈ 1 ± 5%` that degrades quietly rather than failing.

## Build

```bash
scripts/download.sh                  # 1.56 TB, resumable
scripts/build_all.sh                 # mxfp4, 3bit, mixed2, 2bit
scripts/build_all.sh mxfp4           # or one tier
```

Peak converter memory is a few tens of GB: it walks one layer at a time, and for
the lossy tiers dequantizes and requantizes one expert at a time rather than
stacking 896 of them in bf16 (which would be 59 GB per layer).

## Usage

Once published, the quants run on stock **mlx-lm**; the only extra step is
registering `kimi_k3.py`, since the architecture isn't upstream yet.

```bash
pip install mlx-lm

python - <<'PY'
import os, shutil, mlx_lm
from huggingface_hub import hf_hub_download
dst = os.path.join(os.path.dirname(mlx_lm.__file__), "models", "kimi_k3.py")
shutil.copy(hf_hub_download("pipenetwork/Kimi-K3-MLX-mxfp4", "kimi_k3.py"), dst)
print("registered kimi_k3 ->", dst)
PY
```

For the unpruned two-node deployment, use the pinned environment and rank-local
checkpoint builder instead:

```bash
uv sync --frozen
scripts/install_model.sh .venv/bin/python

# Beast1 is rank 0; download rank 1 on Beast2 in parallel.
.venv/bin/python scripts/prepare_uvmax_stage.py --rank 0 --verify-sha256
ssh beast2.local 'cd ~/dev/kimi-k3-mlx-distributed && \
  .venv/bin/python scripts/prepare_uvmax_stage.py --rank 1 --verify-sha256'

# After connecting Thunderbolt 5 and confirming RDMA is active:
scripts/configure_jaccl.sh
scripts/run_tensor_conversion.sh
scripts/run_tensor.sh --prompt 'Who is Albert Einstein?' --max-tokens 256
```

`scripts/tensor_generate.py` initializes `backend="jaccl"` in strict mode and
requires exactly two ranks. All attention and FFN shards execute concurrently,
with their MLX tensor-parallel reductions carried by JACCL over Thunderbolt
RDMA. A missing RDMA/JACCL fabric is an error, not a performance-degrading
fallback. `scripts/distributed_generate.py` retains the separately proven
point-to-point pipeline for diagnosis and regression testing.

For regular use, keep one model/JACCL context resident. This avoids the costly
reload and the JACCL protection-domain exhaustion associated with repeated
initialisation, while continuous micro-batching recovers throughput from this
dispatch-bound model:

```bash
scripts/configure_jaccl.sh
scripts/run_tensor_server.sh --decode-concurrency 32 --prompt-concurrency 4

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Kimi-K3-2bit-UVMAX","messages":[{"role":"user","content":"Who is Albert Einstein?"}],"max_tokens":256}'

# Measure aggregate throughput without reloading the resident model.
.venv/bin/python scripts/benchmark_server.py --concurrencies 1,2,4,8,16,32
```

The HTTP endpoint runs on Beast1. Each micro-batch is forwarded to Beast2 over
a small control socket, but every model collective is still strict JACCL over
Thunderbolt RDMA. Beast1 checks a SHA-256 digest of rank-local token outputs
before returning the OpenAI-compatible, non-streaming response. `/health` shows
resident-service and queue state. Bind remains loopback-only by default; use an
SSH tunnel for remote access rather than exposing the unauthenticated endpoint.

## Status

- [x] text architecture port (`kimi_k3.py`)
- [x] **vision tower** (`kimi_k3_vision.py`) — parity vs torch at full K3 dims, 1.5e-6
- [x] streaming converter + verifier, 4 profiles, round-trip tested
- [x] MXFP4 → MLX mxfp4 bit-exact passthrough proven
- [x] 107 tests discovered: 100 passing, 7 hardware/reference-data skips
- [x] source download (1.56 TB)
- [x] real conversions
- [x] **mlx-vlm wrapper** (`kimi_k3_vl/`) — expanding `<|media_pad|>` merge,
      validated against the real processor end to end
- [x] publish to `pipenetwork/` (4 tiers)
- [x] **held-out perplexity** (`scripts/perplexity.py`), bucketed per source
- [x] AWQ implemented and measured — [small, fragile, domain-skewed](#awq-on-the-2-bit-experts--measured)
- [x] latest stable MLX 0.32.0 + mlx-lm 0.31.3 environment pinned by `uv.lock`
- [x] resumable 47/46-layer rank-local UVMAX downloader (about 384.5 GiB/rank)
- [x] exact two-rank prefill and cached-decode parity on a miniature K3
- [x] exact 64-token/32-step tensor parity over real two-host JACCL/RDMA
- [x] production runner hard-requires JACCL and two ranks
- [x] full tensor-parallel two-M3-Ultra JACCL generation and canonical benchmark
- [x] persistent OpenAI-compatible serving with deterministic micro-batching and cross-rank output proof
- [ ] AWQ re-run on a balanced calibration corpus (in progress)
- [ ] tok/s re-measured for all published tiers after the wiring fix

## Provenance and licensing

`reference/` contains **unmodified files from
[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)**, vendored so
the port can be diffed and numerically tested against the code it targets:
`modeling_kimi_k3.py`, `modeling_kimi_linear.py`, `configuration_kimi_k3.py`,
the processors, `config.json`, and a gzipped `model.safetensors.index.json`.
Those files are Moonshot's, under the Kimi K3 License (with llava-derived parts
under Apache 2.0) — see the LICENSE in the upstream repo. Converted weights
carry the upstream licence too; `scripts/convert.py` copies it into every tier.

Everything outside `reference/` — `kimi_k3.py`, `kimi_k3_vision.py`,
`kimi_k3_vl/`, `scripts/`, `tests/` — is this port.

Distributed additions in this fork retain and credit that work. The UVMAX
loader is based on Tarjei Mandt's
[mlx-lm PR #1626](https://github.com/ml-explore/mlx-lm/pull/1626), pinned to
commit `7d505c285b801108a52c23353c7fb6af07204717` and kept under mlx-lm's MIT
terms. The UVMAX weights remain under Moonshot's Kimi K3 license. See
[NOTICE.md](NOTICE.md) for the file-by-file provenance and licensing caveat:
the fork point did not include a top-level software license, so inherited code
cannot be retroactively relicensed by this fork.
