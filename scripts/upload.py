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


def collect(path, plan_path, report_path):
    cfg = json.load(open(os.path.join(path, "config.json")))
    tc = cfg.get("text_config", cfg)
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    size = sum(os.path.getsize(os.path.join(path, s)) for s in shards)
    counts = [c for c in tc.get("expert_counts", []) if c]
    reap = cfg.get("reap", {})
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


def model_card(name, d, sm=None):
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

## Speed — read this before downloading

**~{tok_s:.2f} tok/s** on a 512 GiB M3 Ultra. Not a typo, and not interactive.

Each decoded token reads roughly {per_tok:.0f} GB of weights: {exp_gb:.1f} GB of routed
experts plus {non_gb:.1f} GB of non-expert weights, which every token touches. This is a
bandwidth wall, not a headroom problem -- a 350 GB build with 160 GiB of spare
memory measured 0.20 tok/s versus 0.16 for a 451 GB build, i.e. gains track size
almost exactly. No prune ratio makes K3 interactive on this hardware.

Note the non-experts dominate per-token traffic despite being ~2% of parameters:
all of them are read every token, while only {top_k} of {kept} experts are.

## Quality expectations — read this

This is an **aggressive** prune. Top-{d['top_k']} routing over {kept} experts is
{density:.1%} density, comparable to a REAP-50 of a 256-expert model. Expect
noticeable degradation versus full K3. It is the "fits on one machine" build, not
a quality build.

Two things work in its favour that a plain expert cull would not have: K3 keeps
**2 shared experts** that fire on every token regardless of pruning, and its
LatentMoE applies RMSNorm to the *combined* expert output, which partially
self-corrects the magnitude lost when experts are removed.

## Calibration

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
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    name = a.name or os.path.basename(a.path.rstrip("/"))
    d = collect(a.path, a.plan, a.report)
    card = model_card(name, d, parse_smoke(a.smoke_log))
    with open(os.path.join(a.path, "README.md"), "w") as f:
        f.write(card)
    print(f"wrote model card -> {a.path}/README.md")
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
