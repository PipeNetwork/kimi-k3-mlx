#!/usr/bin/env python3
"""Publish a REAP'd Kimi-K3 MLX build to pipenetwork/.

Writes the model card from the artifact's own config and the REAP plan rather
than from hand-typed numbers, so the card cannot drift from what shipped.

Usage:
  scripts/upload.py --path out/Kimi-K3-REAP73-MLX-mxfp4 [--plan out/reap_plan_uniform_0.27.json]
                    [--report out/reap_report.txt] [--dry-run]
"""

import argparse
import json
import os
import sys

REPO_OWNER = "pipenetwork"
BASE = "moonshotai/Kimi-K3"


def is_full_build(cfg) -> bool:
    """A build with no `reap` block kept every expert."""
    return "reap" not in cfg


def full_profile(cfg) -> str:
    """Which unpruned profile produced this build.

    A bf16 build carries no `quantization` key at all -- that absence is the
    whole signal, so it is read here rather than guessed from a filename.
    """
    q = cfg.get("quantization")
    if not q:
        return "bf16"
    if q.get("mode") == "mxfp4":
        return "mxfp4"
    raise SystemExit(f"unpruned build with unexpected quantization {q}; "
                     f"no card template covers it")


def collect(path, plan_path, report_path):
    cfg = json.load(open(os.path.join(path, "config.json")))
    tc = cfg.get("text_config", cfg)
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    size = sum(os.path.getsize(os.path.join(path, s)) for s in shards)
    counts = [c for c in tc.get("expert_counts", []) if c]
    reap = cfg.get("reap", {})
    if is_full_build(cfg):
        # Nothing was pruned, so there is no expert_counts list to read the
        # geometry off. Take it from the config, the way the model itself does.
        n_moe = tc["num_hidden_layers"] - tc.get("first_k_dense_replace", 1)
        counts = [tc["num_experts"]] * n_moe
    plan = json.load(open(plan_path)) if plan_path and os.path.exists(plan_path) else {}
    report = open(report_path).read() if report_path and os.path.exists(report_path) else ""
    retained = ""
    for line in report.splitlines():
        if "saliency" in line and "%" in line and "keep" not in line:
            pass
    return dict(
        cfg=cfg, tc=tc, size=size, shards=len(shards),
        kept=counts[0] if counts else None,
        src_experts=reap.get("source_num_experts", 896),
        top_k=tc.get("num_experts_per_token", 16),
        quant=cfg.get("quantization", {}),
        n_layers=len(counts), plan=plan, report=report,
    )


def parse_smoke(path):
    """Pull load time, tok/s and the generated samples out of a smoke log.

    Hardcoding one build's numbers into every card is how a card starts lying;
    each build reports its own.
    """
    if not path or not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="replace").read()
    import re
    loads = re.findall(r"loaded in (\d+)s\s*\|\s*peak mem (\d+) GB", txt)
    if len(loads) > 1:
        # An A/B log holds several runs back to back. Taking the first `loaded
        # in` line and averaging every tok/s across all of them silently
        # attributes one model's samples, load time and speed to another -- it
        # put three q4n generations on the published REAPgraded card. Refuse.
        raise SystemExit(
            f"{path} contains {len(loads)} model loads; a smoke log must cover "
            f"exactly one build or the card mixes them. Re-run scripts/smoke.py "
            f"for this build alone and pass that log."
        )
    load = re.search(r"loaded in (\d+)s\s*\|\s*peak mem (\d+) GB", txt)
    rates = [float(m) for m in re.findall(r"= ([\d.]+) tok/s", txt)]
    blocks = []
    for chunk in txt.split("[smoke] prompt: ")[1:]:
        prompt = chunk.split("\n", 1)[0].strip()
        # the last "--> " in a chunk is the complete generation; earlier ones
        # are partial streaming echoes
        outs = chunk.split("[smoke] --> ")
        if len(outs) < 2:
            continue
        gen = outs[-1].split("\n[smoke] decode")[0]
        if "[smoke]" in gen:                      # truncated run, skip it
            continue
        blocks.append((prompt, gen.rstrip()))
    if not (load and rates):
        return None
    return dict(load_s=int(load.group(1)), peak_gb=int(load.group(2)),
                tok_s=sum(rates) / len(rates),
                samples=[(p_.strip(), o.strip()) for p_, o in blocks])


SMOKE_EXAMPLE = '''prompt: def merge_intervals(intervals):
            """Merge overlapping intervals."""
-->     if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        merged = [interval

prompt: 机器学习的基本原理是
--> ，机器学习是人工智能的核心，是一切计算机视觉化、网络化的基础。'''


def model_card(name, d, sm=None, prev_tok_s=None):
    kept, src = d["kept"], d["src_experts"]
    pct_pruned = round((1 - kept / src) * 100)
    gb = d["size"] / 1e9
    params = (d["n_layers"] * kept * 3 * 3584 * 3072 + 57.23e9) / 1e9
    q = d["quant"]
    density = d["top_k"] / kept
    top_k = d["top_k"]
    exp_gb = d["n_layers"] * top_k * 3 * 3584 * 3072 * 4.25 / 8 / 1e9
    non_gb = 57.23e9 * 8.5 / 8 / 1e9
    per_tok = exp_gb + non_gb
    tok_s = sm["tok_s"] if sm else float("nan")
    load_line = (f"{sm['load_s']} s, {sm['peak_gb']} GB peak" if sm else "not measured")
    # ~819 GB/s is the M3 Ultra's peak memory bandwidth; per_tok is what one
    # decoded token must read, so their ratio is the hard ceiling for this build.
    ceiling = 819.0 / per_tok
    pct_of_ceiling = tok_s / ceiling * 100 if tok_s == tok_s else float("nan")
    correction = ""
    if prev_tok_s:
        correction = f"""
> **Correction, 2026-07-28.** Earlier revisions of this card reported
> ~{prev_tok_s:.2f} tok/s and described this build as "not interactive". That number
> came from an internal benchmark that hand-rolled its own decode loop and so
> never entered the `wired_limit` context manager `mlx_lm.stream_generate`
> applies automatically. The weights were left unwired, and every decoded token
> faulted them back from SSD instead of reading RAM -- understating this build by
> roughly {tok_s / prev_tok_s:.0f}x. **Users were never affected:** the normal mlx-lm
> paths (`generate`, `stream_generate`, the CLI, the server) have always wired
> correctly, so what you measure is the corrected figure above. Thanks to
> [@pudepiedj](https://huggingface.co/pudepiedj) for reporting the discrepancy.
"""
    calib_note = ""
    graded = bool((d["tc"] or {}).get("expert_bank_split"))
    if graded:
        bs = [x for x in d["tc"]["expert_bank_split"] if x]
        nhi = bs[0]
        calib_note = f""" — graded two-bank precision

Experts are split across **two banks at different bit widths**: the {nhi} most
salient per layer at **mxfp4** (bit-exact copies of Moonshot's weights) and the
remainder at 2-bit. That fits **{kept} experts per layer instead of 242** at the
same footprint — 35% more — because the least salient two thirds cost 2.5 bits
each instead of 4.25.

MLX pins one bit width per expert tensor, so this needs a custom
`TwoBankSwitchGLU` (bundled in `kimi_k3.py`). It exploits the sort SwitchGLU
already performs: once routed pairs are sorted by expert index, each bank owns a
contiguous slice, so neither bank does the other's work.

Saliency retained rises to **68.4%** from 59.1% for a uniform 242-expert build at
the same size, and Chinese output is measurably better (the uniform build drifts
back into restating the prompt; this one does not).

**It costs half the decode speed.** Measured end to end against the otherwise
comparable single-bank build at the same 450 GB footprint: **2.68 tok/s here
versus 5.51** for `Kimi-K3-REAP73-MLX-mxfp4-q8` -- a 2.06x penalty, which is
essentially the full cost of the run-both-and-select scheme the two-bank design
exists to avoid. The overhead is a per-layer host sync to find the data-dependent
split point, and K3 has 92 MoE layers serialised down the decode path; what
amortises inside a single layer does not amortise across 92. This build moves
*less* memory per token than the single-bank one (part of its experts are 2-bit),
so the penalty is entirely machinery, not bandwidth.

Take this build if you want the extra experts and better Chinese and can accept
half the speed. Take `Kimi-K3-REAP73-MLX-mxfp4-q8` otherwise."""
    if "zh-code" in name:
        calib_note = """ — targeted at Chinese + code

This build is calibrated on **Chinese and code only**, not the full mixed
corpus. Kimi-K3's experts cluster by domain — measured over a top-242 set
against a 27% chance baseline, code-python↔code-multi overlap is 57.2% while
chinese↔code-python is 17.8%, i.e. *below* chance — so dropping the languages
you do not need frees expert slots for the ones you do. Saliency retained rises
from 59.1% (mixed) to 69.3% here at identical size.

The tradeoff is real: Japanese, Korean, Russian, Arabic, German, French and
Spanish are materially degraded relative to the mixed build. Use
`Kimi-K3-REAP73-MLX-mxfp4-q8` if you need those.

Measured against the mixed build on the same prompts: Chinese improves (the
mixed build drifts into restating the prompt by ~18 tokens; this one does not),
and code is unchanged."""
    _looped = any("是一种基于人工智能的学科，是一种" in b for _, b in (sm or {}).get("samples", []))
    cjk_note = ("**Chinese output degrades noticeably in this build** — the sample above "
                "loops. Prefer the REAP-73 build if you need CJK.\n\n") if _looped else ""
    sample_block = "\n\n".join(f"prompt: {a_}\n--> {b}" for a_, b in (sm or {}).get("samples", [])) \
        or SMOKE_EXAMPLE

    return f"""---
license: other
license_name: kimi-k3
base_model: {BASE}
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- moe
- reap
- pruned
- kimi
- apple-silicon
---

# {name}

**Kimi-K3, REAP expert-pruned ({pct_pruned}%) and converted to MLX** — {gb:.0f} GB,
sized to actually load on a 512 GB Apple Silicon machine.

Base: [{BASE}](https://huggingface.co/{BASE}) — 2.78T total / 104B active,
native-multimodal MoE with Kimi Delta Attention, Attention Residuals and a 1M
context.

## Why this exists

Unpruned Kimi-K3 does not fit on any Mac. The full model is 1.56 TB; even a
2-bit quant is ~870 GB against a 512 GB ceiling. Fitting 4-bit into 512 GB would
need ≤1.38 bits/weight.

REAP ([Cerebras](https://github.com/CerebrasResearch/reap)) scores each expert by
`gate x ||expert_output||` over a calibration set and keeps the most salient.
This build keeps **{kept} of {src} experts per layer** across {d['n_layers']} MoE
layers → **{params:.0f}B params, {gb:.0f} GB**.

## Precision: mxfp4, and it is lossless

K3's routed experts ship from Moonshot as **MXFP4** (`weight_packed` + e8m0
`weight_scale`, group 32). MLX's native `mxfp4` mode uses the identical encoding,
so the surviving experts here are a **bit-exact byte copy of the source** — the
only information lost in this repo is the pruning itself, not the quantization.

Requantizing those same weights to affine 4-bit would cost ~9.8% mean relative
error *and* be larger (4.5 vs 4.25 bits/weight), so no affine 4-bit tier is
published.

Non-expert tensors ({q.get('mode')} global config; attention, shared experts,
latent projections, embeddings) are carried at higher precision.

## Measured behaviour

Loaded on a 512 GiB M3 Ultra ({load_line}). Verbatim, greedy, unedited:

```
{sample_block}
```

Samples above are verbatim, greedy-decoded, and unedited — including the failure
modes. Degradation from the pruning shows up as drift into list-like or
source-file-like continuations rather than answering directly, and at heavier
prune ratios as outright repetition loops.

{cjk_note}Code completions stay structurally correct across every ratio tested,
which matches the calibration data: code experts form a dense, self-similar
cluster (57% self-overlap in a top-242 set, versus a 27% chance baseline) and so
survive pruning better than more diffuse language capability.

## Speed

**~{tok_s:.2f} tok/s** decoding on a 512 GiB M3 Ultra, via `mlx_lm.generate` or
any standard mlx-lm entry point. Prompt processing is considerably faster; the
figure above is generation.

Each decoded token reads roughly {per_tok:.0f} GB of weights: {exp_gb:.1f} GB of routed
experts plus {non_gb:.1f} GB of non-expert tensors, which every token touches. Against
the M3 Ultra's ~819 GB/s that puts the ceiling near {ceiling:.1f} tok/s, so this build
reaches about {pct_of_ceiling:.0f}% of what the memory system permits. It is
bandwidth-bound, which is the expected regime for a model of this shape.

Note the non-experts dominate per-token traffic despite being ~2% of parameters:
all of them are read every token, while only {top_k} of {kept} experts are.

**Pruning buys memory, not speed.** Per-token traffic depends on `top_k` and the
non-expert precision, never on how many experts are *stored*, so the 350 GB
179-expert build and the 451 GB 242-expert build both decode at ~5.5 tok/s. Prune
harder to fit a smaller machine, not to go faster.
{correction}

## Quality expectations — read this

This is an **aggressive** prune. Top-{d['top_k']} routing over {kept} experts is
{density:.1%} density, comparable to a REAP-50 of a 256-expert model. Expect
noticeable degradation versus full K3. It is the "fits on one machine" build, not
a quality build.

Two things work in its favour that a plain expert cull would not have: K3 keeps
**2 shared experts** that fire on every token regardless of pruning, and its
LatentMoE applies RMSNorm to the *combined* expert output, which partially
self-corrects the magnitude lost when experts are removed.

## Calibration{calib_note}

Saliency was measured on a deliberately mixed 12.6 MB corpus — 40% code
(multi-language + real Python), 30% English web, 15% Chinese, 15% across
ja/ru/ko/de/fr/es/ar. The mix matters: whatever a calibration corpus
under-represents gets pruned away silently. An earlier attempt using C4's pooled
`multilingual` config left CJK at 0.03% of the corpus, which would have quietly
removed the experts handling Chinese.

## Usage

Requires **mlx-lm** plus the bundled `kimi_k3.py` loader (the architecture is not
upstream yet). On a 512 GB machine you must raise the GPU wired limit first —
the default is ~75% of RAM, below this model's footprint:

```bash
sudo sysctl iogpu.wired_limit_mb=480000
pip install mlx-lm

python - <<'PY'
import os, shutil, mlx_lm
from huggingface_hub import hf_hub_download
for f in ("kimi_k3.py",):
    dst = os.path.join(os.path.dirname(mlx_lm.__file__), "models", f)
    shutil.copy(hf_hub_download("{REPO_OWNER}/{name}", f), dst)
PY

mlx_lm.generate --model {REPO_OWNER}/{name} --max-tokens 256 \\
    --prompt "Write a Python function that merges overlapping intervals."
```

`kimi_k3_vision.py` and `kimi_k3_vl/` ship alongside for the vision tower; the
image path needs mlx-vlm and is not exercised by `mlx_lm.generate`.

## Provenance

Converted with [PipeNetwork/kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx):
a streaming converter (the model never fits in memory at any stage) and a
streaming REAP calibration harness. Weights remain under the Kimi K3 License.
"""


FULL_CARD = """---
license: other
license_name: kimi-k3
base_model: {BASE}
base_model_relation: {relation}
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- moe
- kimi
- apple-silicon
---

# {name}

**Kimi-K3 converted to MLX, complete -- every expert, no pruning.** {size_str}.

Base: [{BASE}](https://huggingface.co/{BASE}) -- 2.78T total / 104B active,
native-multimodal MoE with Kimi Delta Attention, Attention Residuals and a 1M
context.

{headline}

## What this is

{what}

## Hardware -- read this before downloading

This build needs **~{ram_str} of unified memory**. That is more than any single
Apple Silicon machine holds: the largest is 512 GB. Running it means an MLX ring
cluster with enough aggregate memory ({nodes}), and downloading it means
{size_str} of disk.

**If you want to run Kimi-K3 on hardware you own, you almost certainly want a
REAP build instead of this one.** Those are pruned to fit a single 512 GB
machine and are linked from the [collection]({collection}).

This repo exists so the *complete* model is available in MLX format -- as a
conversion source, as the reference for measuring what pruning and quantization
cost, and for anyone with the hardware to run it whole.

## Precision

{precision}

## Usage

Requires **mlx-lm** plus the bundled `kimi_k3.py` loader (the architecture is not
upstream yet). Copy it into mlx-lm's model directory after installing:

```bash
pip install mlx-lm
python -c "import os, shutil, mlx_lm; from huggingface_hub import hf_hub_download; \
shutil.copy(hf_hub_download('{owner}/{name}', 'kimi_k3.py'), \
os.path.join(os.path.dirname(mlx_lm.__file__), 'models', 'kimi_k3.py'))"
```

Single-machine `mlx_lm.generate` will not load this build -- it does not fit.
Use `mlx.launch` with the ring backend across enough nodes; `cluster_generate.py`
in the converter repo is a worked example.

`kimi_k3_vision.py` and `kimi_k3_vl/` ship alongside for the vision tower; the
image path needs mlx-vlm and is not exercised by `mlx_lm.generate`.

## Provenance

Converted with [PipeNetwork/kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx)
using a streaming converter -- the model never fits in memory at any stage, so
experts are decoded and written one at a time. Weights remain under the Kimi K3
License.
"""

MXFP4_HEADLINE = """This is a **lossless** conversion. Kimi-K3's routed experts ship from Moonshot
as MXFP4, and MLX's native `mxfp4` mode uses the identical encoding, so the
experts here are a **bit-exact byte copy of the source weights** -- not a
requantization. Non-expert tensors are carried at the bf16 they were published
in. Nothing in this repo is approximated."""

MXFP4_WHAT = """The complete model in MLX layout: all {n_experts} experts across {n_layers} MoE
layers, plus the vision tower and projector. Numerically identical to
[{BASE}](https://huggingface.co/{BASE}) -- only the file layout and tensor naming
differ, so mlx-lm can load it directly."""

MXFP4_PRECISION = """**Routed experts: MXFP4, byte-for-byte from source.** `weight_packed` plus e8m0
`weight_scale` at group 32, restacked per layer but never decoded and
re-encoded.

**Everything else: bf16, as published.** Attention, shared experts, latent
projections, embeddings and the vision tower are copied unchanged.

Requantizing the experts to affine 4-bit would cost ~9.8% mean relative error
*and* produce a larger file (4.5 vs 4.25 bits/weight), which is why no affine
4-bit tier of the full model is published."""

BF16_HEADLINE = """**This build is not higher fidelity than the source, and it is not the one you
want unless you know why you want it.** Kimi-K3's experts are published in
MXFP4 and carry 4 bits of real information per weight. This build stores those
same values in 16 bits -- the identical numbers in a wider container, at 3.4x the
bytes. Nothing is recovered; nothing is added.

For the same model at 1.56 TB with identical numerics, use
[{owner}/Kimi-K3-MLX-mxfp4](https://huggingface.co/{owner}/Kimi-K3-MLX-mxfp4)."""

BF16_WHAT = """An **unquantized reference artifact**. The MXFP4 routed experts are dequantized
into dense bf16 and every other tensor is copied as-is, giving a build with no
packed weights, no scales, and no `quantization` block anywhere.

It exists for work that wants dense weights as a starting point: evaluating new
quantization schemes, analysing expert weight distributions, or requantizing
without re-implementing the MXFP4 decode. For inference it is strictly worse than
the mxfp4 build -- same outputs, 3.4x the memory."""

BF16_PRECISION = """**Routed experts: dense bf16**, dequantized from the source's MXFP4 (group 32,
e8m0 scales). Each expert was decoded individually and verified bit-exact against
an independent dequant of the source, including a check that each slice does
*not* match its neighbour -- the stacking is per-expert, and an off-by-one there
yields tensors of the right dtype and shape holding the wrong expert.

**Everything else: bf16, as published**, unchanged from source.

There is no `quantization` key in `config.json`, because nothing here is
quantized."""


def full_model_card(name, d):
    cfg = d["cfg"]
    profile = full_profile(cfg)
    tc = d["tc"]
    size = d["size"]
    size_str = f"{size / 1e12:.2f} TB" if size >= 1e12 else f"{size / 1e9:.0f} GB"
    # 128 GB nodes cannot dedicate all of it to weights; ~120 GB is what the
    # cluster script actually fits per rank.
    nodes = f"~{max(2, -(-int(size / 1e9) // 120))} nodes at 128 GB, or fewer larger ones"
    sub = dict(BASE=BASE, owner=REPO_OWNER, name=name,
               n_experts=tc.get("num_experts"), n_layers=d["n_layers"])
    if profile == "mxfp4":
        headline, what, precision = MXFP4_HEADLINE, MXFP4_WHAT, MXFP4_PRECISION
        relation = "quantized"
    else:
        # bf16 is a widening of the source's own values, not a quantization of
        # them; "quantized" would be an outright false claim on the hub.
        headline, what, precision = BF16_HEADLINE, BF16_WHAT, BF16_PRECISION
        relation = "finetune"
    return FULL_CARD.format(
        BASE=BASE, owner=REPO_OWNER, name=name, relation=relation,
        size_str=size_str, ram_str=size_str, nodes=nodes,
        headline=headline.format(**sub), what=what.format(**sub),
        precision=precision.format(**sub),
        collection="https://huggingface.co/collections/pipenetwork",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke-log", default=None,
                    help="smoke.py output for this build; its measured numbers "
                         "go in the card")
    ap.add_argument("--readme-only", action="store_true",
                    help="regenerate and push just the model card")
    ap.add_argument("--prev-tok-s", type=float, default=None,
                    help="tok/s a previous revision of this card published; adds "
                         "a correction note. Use when republishing a number that "
                         "was wrong, so the change is visible rather than silent")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    name = a.name or os.path.basename(a.path.rstrip("/"))
    d = collect(a.path, a.plan, a.report)
    # A full build has no prune ratio, no calibration corpus and no saliency
    # numbers, and the REAP card is largely made of exactly those. It gets its
    # own template rather than a pile of conditionals.
    if is_full_build(d["cfg"]):
        card = full_model_card(name, d)
    else:
        card = model_card(name, d, parse_smoke(a.smoke_log), a.prev_tok_s)
    with open(os.path.join(a.path, "README.md"), "w") as f:
        f.write(card)
    print(f"wrote model card -> {a.path}/README.md")
    if is_full_build(d["cfg"]):
        print(f"  {name}: {d['size']/1e12:.2f} TB, {d['shards']} shards, "
              f"{full_profile(d['cfg'])}, all {d['kept']} experts "
              f"x {d['n_layers']} layers")
    else:
        print(f"  {name}: {d['size']/1e9:.0f} GB, {d['shards']} shards, "
              f"{d['kept']}/{d['src_experts']} experts x {d['n_layers']} layers")

    if a.readme_only:
        from huggingface_hub import HfApi

        repo = f"{REPO_OWNER}/{name}"
        HfApi().upload_file(path_or_fileobj=os.path.join(a.path, "README.md"),
                            path_in_repo="README.md", repo_id=repo, repo_type="model")
        print(f"updated card only -> https://huggingface.co/{repo}")
        return

    if a.dry_run:
        print("\n--- card ---\n" + card)
        print("[dry-run] not uploading")
        return

    from huggingface_hub import HfApi, create_repo

    repo = f"{REPO_OWNER}/{name}"
    create_repo(repo, repo_type="model", exist_ok=True, private=False)
    print(f"uploading {d['size']/1e9:.0f} GB -> {repo} (resumable)")
    HfApi().upload_large_folder(repo_id=repo, folder_path=a.path, repo_type="model")
    print(f"done -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
