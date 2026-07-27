# kimi-k3-mlx

MLX (Apple Silicon) port of [**moonshotai/Kimi-K3**](https://huggingface.co/moonshotai/Kimi-K3) —
Moonshot's 2.78T-total / 104B-active native-multimodal MoE, built on Kimi Delta
Attention (KDA) and Attention Residuals (AttnRes), with a 1M-token context.

`kimi_k3.py` is a single, stock-`mlx-lm`-compatible model definition for the text
tower; `kimi_k3_vision.py` is the vision tower and projector; `kimi_k3_vl/` is
the mlx-vlm wrapper that joins them.
`scripts/convert.py` is a streaming converter — `mlx_lm convert` cannot be used
here, because it materialises the whole model and K3 is 5.6 TB in bf16.

> **Read this before planning a deployment.** Nothing in this repo runs on a
> single Mac. The smallest tier is ~870 GB against a 512 GB ceiling on the
> largest Apple Silicon machine that exists. See [Reality check](#reality-check).

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

| tier | bpw | size | expert error vs source |
|---|---|---|---|
| **mxfp4** | 4.25 | ~1.48 TB (+114 GB bf16 rest = 1.56 TB) | **0 — bit-exact** |
| `3bit` | 3.50 | ~1.22 TB | lossy (2nd quantization pass) |
| `mixed2` | ~2.9 | ~1.02 TB | experts 2-bit, rest 4-bit |
| `2bit` | 2.50 | ~0.87 TB | lossy |

`6bit`, `8bit` and `bf16` are not built: they would store upcast 4-bit values at
2.26 / 2.95 / 5.56 TB, larger than the source, with zero quality gain.

## Reality check

**No tier is runnable on any Apple Silicon machine.** Peak unified memory tops
out at 512 GB (M3 Ultra Mac Studio); the smallest tier here is 869 GB. Fitting
4-bit into 512 GB would need ≤1.38 bits/weight.

Consequences, stated plainly:

- These artifacts have **never produced a token.** No perplexity, no generation,
  no smoke test — that is not a shortcut taken, it is arithmetically impossible
  on this hardware.
- What *is* verified: bit-exactness of the mxfp4 tier against the source,
  100% checkpoint-key coverage over all 497,220 tensors, prefill/decode
  agreement of the architecture at small scale, and per-expert cosine similarity
  against the source for the lossy tiers. See [Verification](#verification).
- The 2-bit tiers are a **double quantization** on top of Moonshot's MXFP4.
  Quality there is genuinely unknown and could be poor; `mixed2` exists because
  that is how the Laguna 2-bit tier stayed coherent.

## Verification

```bash
scripts/test_all.sh                     # registers kimi_k3.py, runs all 39 tests
scripts/verify.py --path out/Kimi-K3-MLX-mxfp4 --src Kimi-K3-src
```

| suite | what it covers | |
|---|---|---|
| `test_kimi_k3.py` | architecture + 497,220-key coverage | 13 |
| `test_vision_parity.py` | vision tower vs torch reference | 9 |
| `test_vl_wrapper.py` | placeholder expansion + multimodal path | 11 |
| `test_convert_roundtrip.py` | converter, on a synthetic mini-K3 | 3 |
| `test_processor_integration.py` | real processor -> MLX tower | 3 |

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
scripts/reap_calibrate.py --src Kimi-K3-src --out out/reap_saliency.npz \
    --calib-text corpus.txt --seqs 64 --seqlen 2048
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
own share. Measured at K3 REAP dims (3584→3072, 240 experts, top-16):

| | vs single-bank |
|---|---|
| prefill (256 tok) | **1.05×** |
| decode (1 tok) | **1.16×** |
| memory, same expert count | **82%** |

The split point is data-dependent, so it must reach the host to slice on — and
that sync is essentially the entire overhead. Decode routes only `top_k` pairs,
too few to amortise two kernel launches plus a device sync, so below 1024 pairs
the indices come down in one transfer and partition with numpy. That alone took
decode from 1.44× to 1.16×.

Correctness rests on one test: give both banks the *same* weights and precision
and the result must match a plain `SwitchGLU` **exactly** (it does, bit-for-bit,
wherever single-bank also sorts). That isolates the partition/unsort machinery
from quantization error, so a bug there cannot hide behind "the low tier is
lossy".

A caveat worth stating: K3's LatentMoE applies RMSNorm to the *combined* expert
output before `up_proj`, so absolute expert norms are partly renormalized away
downstream. That likely makes K3 more robust to pruning than a standard MoE, but
it also means `‖e_j(x)‖` measures something partly cancelled — the criterion may
want reformulating as relative contribution. Two shared experts fire on every
token regardless and survive any pruning, giving a floor nothing touches.

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

## Status

- [x] text architecture port (`kimi_k3.py`)
- [x] **vision tower** (`kimi_k3_vision.py`) — parity vs torch at full K3 dims, 1.5e-6
- [x] streaming converter + verifier, 4 profiles, round-trip tested
- [x] MXFP4 → MLX mxfp4 bit-exact passthrough proven
- [x] 39/39 tests passing
- [ ] source download (1.56 TB, in progress)
- [ ] real conversions
- [x] **mlx-vlm wrapper** (`kimi_k3_vl/`) — expanding `<|media_pad|>` merge,
      validated against the real processor end to end
- [ ] publish to `pipenetwork/`

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
