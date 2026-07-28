#!/usr/bin/env python3
"""End-to-end multimodal generation on a published Kimi-K3 build.

Closes the last gap: the vision tower is numerically exact and its weights ship
in every repo, but until this runs no image has ever reached the text tower of a
published artifact. Everything before it validated halves.

Path exercised: PIL image -> KimiK3VisionProcessor -> MLX vision tower ->
<|media_pad|> expansion -> quantized text tower -> tokens.

Usage: scripts/vl_generate.py --path out/Kimi-K3-REAP73-MLX-mxfp4-q8 --max-tokens 8
"""

import argparse
import json
import os
import sys
import time
import types

import mlx.core as mx
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

MEDIA_BEGIN, MEDIA_CONTENT, MEDIA_PAD, MEDIA_END = 163602, 163603, 163605, 163604


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--src", default="Kimi-K3-src")
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default="Describe this image.")
    ap.add_argument("--max-tokens", type=int, default=8)
    a = ap.parse_args()

    from PIL import Image, ImageDraw

    import kimi_k3_vl
    from reap_calibrate import build_tokenizer

    enc = build_tokenizer(a.src if os.path.exists(os.path.join(a.src, "tiktoken.model")) else a.path)
    print(f"[vl] tokenizer ok (n_vocab {enc.n_vocab})", flush=True)

    # ---- image + processor
    if a.image:
        img = Image.open(a.image).convert("RGB")
    else:
        img = Image.new("RGB", (448, 448), "white")
        d = ImageDraw.Draw(img)
        d.rectangle([80, 80, 220, 220], fill="red")
        d.ellipse([250, 250, 390, 390], fill="blue")
    if "k3src" not in sys.modules:
        pkg = types.ModuleType("k3src")
        pkg.__path__ = [os.path.abspath(a.src)]
        sys.modules["k3src"] = pkg
    from k3src import kimi_k3_vision_processing as k3vp

    pcfg = json.load(open(os.path.join(a.src, "preprocessor_config.json")))
    proc = k3vp.KimiK3VisionProcessor(**{k: v for k, v in pcfg.items()
                                         if k not in ("processor_class", "image_processor_type", "auto_map")})
    out = proc.preprocess([{"type": "image", "image": img}])
    px = mx.array(out["pixel_values"].numpy())
    grids = [tuple(int(v) for v in g) for g in out["grid_thws"].tolist()]
    n_img_tok = proc.media_tokens_calculator({"type": "image", "image": img})
    print(f"[vl] image {img.size} -> {px.shape[0]} patches, grid {grids[0]}, "
          f"{n_img_tok} image tokens", flush=True)

    # ---- prompt: exactly ONE <|media_pad|>, as K3's processor emits
    w, h = img.size
    head = enc.encode_ordinary(f"image {w}x{h}")
    tail = enc.encode_ordinary("\n" + a.prompt)
    ids = [MEDIA_BEGIN] + head + [MEDIA_CONTENT, MEDIA_PAD, MEDIA_END] + tail
    print(f"[vl] prompt {len(ids)} tokens, {ids.count(MEDIA_PAD)} media_pad "
          f"-> expands to {len(ids) - 1 + n_img_tok}", flush=True)

    # ---- load
    t0 = time.time()
    print(f"[vl] loading {a.path} ...", flush=True)
    model = kimi_k3_vl.Model.from_pretrained(a.path)
    print(f"[vl] loaded in {time.time()-t0:.0f}s | peak {mx.get_peak_memory()/1e9:.0f} GB",
          flush=True)

    # ---- prefill with the image, then decode
    x = mx.array([ids])
    cache = model.make_cache()
    t0 = time.time()
    logits = model(x, pixel_values=px, grid_thws=grids, cache=cache)
    mx.eval(logits)
    print(f"[vl] prefill {logits.shape[1]} merged positions in {time.time()-t0:.1f}s "
          f"(expected {len(ids) - 1 + n_img_tok})", flush=True)
    assert logits.shape[1] == len(ids) - 1 + n_img_tok, "merge produced wrong length"

    tok = int(mx.argmax(logits[0, -1]))
    gen, shown = [], ""
    t0 = time.time()
    print("[vl] --> ", end="", flush=True)
    for _ in range(a.max_tokens):
        gen.append(tok)
        text = enc.decode(gen)
        if len(text) > len(shown):
            print(text[len(shown):], end="", flush=True)
            shown = text
        logits = model(mx.array([[tok]]), cache=cache)
        mx.eval(logits)
        tok = int(mx.argmax(logits[0, -1]))
        if tok in (163584, 163586):
            break
    print(f"\n[vl] {len(gen)} tok in {time.time()-t0:.0f}s = "
          f"{len(gen)/max(time.time()-t0,1e-9):.2f} tok/s", flush=True)
    print("[vl] VL-GENERATE-DONE")


if __name__ == "__main__":
    main()
