#!/usr/bin/env python3
"""Run a real image through a published build's vision tower.

Everything else about the vision port was validated against the reference
implementation with random weights. This exercises the weights that actually
shipped: load vision_tower.* / mm_projector.* out of the converted repo, push a
real PIL image through Kimi-K3's own processor, and check the result.

Three things are checked, in increasing order of what they'd catch:
  1. the artifact's vision weights are bit-identical to the source (the
     converter claims to copy them verbatim in bf16 -- verify, don't assume)
  2. a real image produces the exact token count the processor predicts
  3. the embeddings are finite, non-degenerate, and DIFFER between images
     (a tower that returns a constant would pass 1 and 2)

Usage: scripts/vision_test.py --path out/Kimi-K3-REAP73-MLX-mxfp4-q8 --src Kimi-K3-src
"""

import argparse
import json
import os
import sys
import types

import mlx.core as mx
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import kimi_k3_vision as mlxv  # noqa: E402


def load_vision_weights(path):
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    keys = [k for k in idx if k.startswith(("vision_tower.", "mm_projector."))]
    shards = sorted({idx[k] for k in keys})
    W = {}
    for sh in shards:
        d = mx.load(os.path.join(path, sh))
        W.update({k: d[k] for k in keys if k in d})
    return W


def make_processor(src):
    if "k3src" not in sys.modules:
        pkg = types.ModuleType("k3src")
        pkg.__path__ = [os.path.abspath(src)]
        sys.modules["k3src"] = pkg
    from k3src import kimi_k3_vision_processing as k3vp

    cfg = json.load(open(os.path.join(src, "preprocessor_config.json")))
    kw = {k: v for k, v in cfg.items()
          if k not in ("processor_class", "image_processor_type", "auto_map")}
    return k3vp.KimiK3VisionProcessor(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--src", default="Kimi-K3-src")
    ap.add_argument("--image", default=None, help="PIL-readable image; synthesised if absent")
    a = ap.parse_args()

    from PIL import Image, ImageDraw

    print(f"[vision] artifact: {a.path}")
    W = load_vision_weights(a.path)
    print(f"[vision] loaded {len(W)} vision/projector tensors "
          f"({sum(v.nbytes for v in W.values())/1e9:.2f} GB)")

    # ---- 1. bit-identical to source?
    src_idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    checked = mismatched = 0
    cache = {}
    for k in sorted(W)[:: max(1, len(W) // 12)]:
        sh = src_idx.get(k)
        if not sh:
            continue
        if sh not in cache:
            cache.clear()
            cache[sh] = mx.load(os.path.join(a.src, sh))
        ref = cache[sh][k]
        same = bool(mx.all(W[k].astype(mx.float32) == ref.astype(mx.float32)))
        checked += 1
        mismatched += (not same)
    print(f"[vision] weights vs source: {checked - mismatched}/{checked} sampled tensors "
          f"bit-identical{'' if not mismatched else '  <-- MISMATCH'}")

    # ---- build the tower from the artifact's own config
    cfg = json.load(open(os.path.join(a.path, "config.json")))
    vcfg = mlxv.VisionConfig.from_dict(cfg["vision_config"])
    tower = mlxv.VisionModel(vcfg)
    tower.load_weights(list(tower.sanitize(W).items()))
    mx.eval(tower.parameters())
    print(f"[vision] tower built: {vcfg.vt_num_hidden_layers} layers, "
          f"hidden {vcfg.vt_hidden_size}, out {vcfg.text_hidden_size}")

    proc = make_processor(a.src)

    # ---- 2 & 3: real images through the real processor
    def run(img, label):
        out = proc.preprocess([{"type": "image", "image": img}])
        px, gt = out["pixel_values"], out["grid_thws"]
        grids = [tuple(int(v) for v in g) for g in gt.tolist()]
        predicted = proc.media_tokens_calculator({"type": "image", "image": img})
        feats = tower(mx.array(px.numpy()), grids)
        mx.eval(feats)
        f = feats[0]
        t, h, w = grids[0]
        ok = f.shape[0] == predicted
        print(f"[vision] {label:<14} {img.size[0]}x{img.size[1]} -> grid {t}x{h}x{w} -> "
              f"{f.shape[0]} tokens (processor says {predicted}) {'OK' if ok else 'MISMATCH'}"
              f" | dim {f.shape[1]} | std {float(f.std()):.4f} | "
              f"finite {not bool(mx.any(mx.isnan(f)))}")
        return np.array(f.astype(mx.float32))   # bf16 has no numpy view

    if a.image:
        imgs = [(Image.open(a.image).convert("RGB"), "supplied")]
    else:
        # two structurally different images: a tower returning a constant would
        # pass every shape check, so we need something that must differ
        d1 = Image.new("RGB", (448, 448), "white")
        dr = ImageDraw.Draw(d1)
        dr.rectangle([60, 60, 200, 200], fill="red")
        dr.ellipse([240, 240, 390, 390], fill="blue")
        dr.line([0, 0, 448, 448], fill="black", width=9)
        d2 = Image.fromarray(
            (np.random.default_rng(0).random((448, 448, 3)) * 255).astype(np.uint8))
        imgs = [(d1, "shapes"), (d2, "noise")]

    embs = [run(img, lab) for img, lab in imgs]

    if len(embs) == 2:
        a_, b_ = embs
        cos = float((a_ * b_).sum() / (np.linalg.norm(a_) * np.linalg.norm(b_)))
        print(f"[vision] shapes vs noise: cosine {cos:.4f}  "
              f"({'DEGENERATE - tower ignores input' if cos > 0.999 else 'distinct - tower responds to content'})")

    print("[vision] VISION-TEST-DONE")


if __name__ == "__main__":
    main()
