#!/usr/bin/env python3
"""Streaming REAP calibration for Kimi-K3.

REAP (Cerebras, github.com/CerebrasResearch/reap) scores each expert by

    S_j = (1/|C|) * sum_{x in C_j} g_j(x) * ||e_j(x)||_2

-- router gate times expert output norm, over a calibration set -- then keeps
the most salient experts per layer and lets the router renormalize.

The obstacle for K3 is that calibration nominally means running a 1.56 TB model,
which does not fit on any Apple Silicon machine. It does not have to: the
saliency of layer L depends only on the hidden states entering L, the router
gates at L, and the expert outputs at L. Nothing downstream. So calibration
streams exactly like scripts/convert.py -- build one layer, push the whole
calibration set through it, record stats, free it, move on. Peak memory is one
layer (~16 GB with mxfp4 experts) plus the resident activations.

Experts are run as real mxfp4 quantized layers, so the arithmetic scored here is
the arithmetic the published tier actually executes.

Output: an .npz with `saliency` (n_moe_layers, n_experts) holding summed
gate*norm, `counts` (routing frequency), plus the token total and layer index,
consumed by scripts/reap_plan.py.

Usage:
  scripts/reap_calibrate.py --src Kimi-K3-src --out out/reap_saliency.npz \
      --calib-text corpus.txt --seqs 64 --seqlen 2048
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import convert  # ShardReader, expert_stack_passthrough  # noqa: E402
from mlx_lm.models import kimi_k3  # noqa: E402
from mlx_lm.models.base import create_attention_mask, create_ssm_mask  # noqa: E402


# ----------------------------------------------------------------- tokenizer


def build_tokenizer(src: str):
    """K3 ships a plain tiktoken BPE; build it directly rather than through
    transformers, whose API moved after the reference was written."""
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    vocab = os.path.join(src, "tiktoken.model")
    if not os.path.exists(vocab):
        raise FileNotFoundError(f"{vocab} not downloaded yet")

    tk_py = os.path.join(src, "tokenization_kimi.py")
    if not os.path.exists(tk_py):
        raise FileNotFoundError(tk_py)

    # Read pat_str by parsing the AST, not by regexing the source. The pattern
    # is a list of TRIPLE-quoted raw strings joined by "|", and a naive
    # `r?"(...)"` scrape matches the empty string between the `"""` delimiters
    # -- yielding empty alternatives that make tiktoken's Rust core panic with
    # "range end index 2 out of range for slice of length 0". literal_eval gives
    # the exact strings, escapes and all.
    import ast

    pat_str = None
    for node in ast.walk(ast.parse(open(tk_py).read())):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pat_str" for t in node.targets):
            continue
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "join"
        ):
            pat_str = ast.literal_eval(call.func.value).join(
                ast.literal_eval(call.args[0])
            )
        else:
            pat_str = ast.literal_eval(call)
        break
    if not pat_str:
        raise RuntimeError(f"could not recover pat_str from {tk_py}")

    ranks = load_tiktoken_bpe(vocab)
    # 256 reserved special-token ids sit above the merges; encode_ordinary never
    # emits them, but declaring them keeps n_vocab honest at 163,840.
    base = len(ranks)
    specials = {f"<|reserved_special_token_{i}|>": base + i for i in range(256)}
    return tiktoken.Encoding(
        name="kimi_k3", pat_str=pat_str, mergeable_ranks=ranks, special_tokens=specials
    )


def load_calibration(args, src: str):
    """-> (seqs, seqlen) int32 token ids, and matching (seqs, seqlen) source ids.

    Source ids let one calibration pass produce a saliency vector PER corpus
    (code / English / Chinese / ...), which is what answers "do these languages
    use the same experts?". Tokenizing document-by-document keeps the mapping
    exact rather than approximating it from character offsets.
    """
    if args.calib_tokens:
        toks = np.load(args.calib_tokens)
        print(f"[calib] pre-tokenized {toks.shape} from {args.calib_tokens}")
        t = toks[: args.seqs, : args.seqlen].astype(np.int32)
        return mx.array(t), None, []

    if not args.calib_text:
        raise SystemExit(
            "need --calib-text (a corpus) or --calib-tokens (pre-tokenized .npy).\n"
            "REAP saliency is only as representative as its calibration data; "
            "there is no sensible default."
        )

    enc = build_tokenizer(src)
    text = open(args.calib_text, errors="ignore").read()
    need = args.seqs * args.seqlen

    side = args.calib_text + ".sources.json"
    labels, ids, src_ids = [], [], []
    if os.path.exists(side):
        man = json.load(open(side))
        labels = sorted(set(man["order"]))
        lut = {l: i for i, l in enumerate(labels)}
        pos = 0
        for lab, n in zip(man["order"], man["chars"]):
            if len(ids) >= need:
                break
            doc_ids = enc.encode_ordinary(text[pos:pos + n])
            pos += n
            ids.extend(doc_ids)
            src_ids.extend([lut[lab]] * len(doc_ids))
        print(f"[calib] per-source tokenization over {len(labels)} sources: {labels}")
    else:
        ids = enc.encode_ordinary(text)
        print("[calib] no .sources.json manifest -> single-bucket saliency")

    if len(ids) < need:
        raise SystemExit(
            f"calibration corpus has {len(ids):,} tokens, need {need:,} "
            f"({args.seqs} x {args.seqlen}). Use a bigger file or fewer seqs."
        )
    arr = np.array(ids[:need], dtype=np.int32).reshape(args.seqs, args.seqlen)
    sarr = (np.array(src_ids[:need], dtype=np.int32).reshape(args.seqs, args.seqlen)
            if src_ids else None)
    print(f"[calib] tokenized {len(ids):,} tokens -> {arr.shape}")
    if sarr is not None:
        for i, l in enumerate(labels):
            print(f"[calib]   {l:<13} {(sarr==i).sum():>8,} tokens "
                  f"({(sarr==i).mean():.1%})")
    return mx.array(arr), (mx.array(sarr) if sarr is not None else None), labels


# ------------------------------------------------------------ layer building


class LayerStreamer:
    """Builds one KimiDecoderLayer at a time and loads its weights from source."""

    def __init__(self, src: str, args: kimi_k3.ModelArgs, index: Dict[str, str]):
        self.src, self.args, self.index = src, args, index
        self.reader = convert.ShardReader(src, index, keep=3)

    def embed(self, tokens: mx.array) -> mx.array:
        w = self.reader.get("language_model.model.embed_tokens.weight")
        out = w[tokens]
        mx.eval(out)
        return out

    def build(self, i: int) -> kimi_k3.KimiDecoderLayer:
        a = self.args
        layer = kimi_k3.KimiDecoderLayer(a, i)

        if layer.is_moe:
            # Score the same arithmetic the artifact runs: mxfp4 experts, which
            # for K3 are a bit-exact reinterpretation of the source bytes.
            nn.quantize(
                layer.block_sparse_moe.switch_mlp, group_size=32, bits=4, mode="mxfp4"
            )

        prefix = f"language_model.model.layers.{i}."
        raw = {
            k: self.reader.get(k)
            for k in self.index
            if k.startswith(prefix) and ".experts." not in k
        }
        mapped = kimi_k3.remap_checkpoint(raw, a, layer_indices=[i], stack_experts=False)
        strip = f"model.layers.{i}."
        w = {k[len(strip):]: v for k, v in mapped.items() if k.startswith(strip)}

        if layer.is_moe:
            for src_name, dst in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
                t = convert.expert_stack_passthrough(self.reader, i, src_name, a.num_experts)
                w[f"block_sparse_moe.switch_mlp.{dst}.weight"] = t["weight"]
                w[f"block_sparse_moe.switch_mlp.{dst}.scales"] = t["scales"]

        layer.load_weights(list(w.items()))
        mx.eval(layer.parameters())
        return layer


# -------------------------------------------------------------------- main


def calibrate(args):
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    margs = kimi_k3.ModelArgs.from_dict(cfg)
    index = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))["weight_map"]

    tokens, source_ids, labels = load_calibration(args, args.src)
    n_seq, seq_len = tokens.shape
    n_src = max(1, len(labels))
    n_tokens = n_seq * seq_len

    streamer = LayerStreamer(args.src, margs, index)
    n_layers = args.layers or margs.num_hidden_layers
    moe_layers = [i for i in range(n_layers) if kimi_k3.is_moe_layer(margs, i)]

    print(f"[reap] {n_layers} layers ({len(moe_layers)} MoE), {margs.num_experts} experts, "
          f"top-{margs.num_experts_per_token}")
    print(f"[reap] calibration {n_seq} x {seq_len} = {n_tokens:,} tokens, "
          f"chunk {args.batch} seq")

    # split into chunks; activations for the whole set stay resident while each
    # layer is loaded once, which is the entire point of streaming
    chunks = [tokens[i : i + args.batch] for i in range(0, n_seq, args.batch)]
    src_chunks = ([source_ids[i : i + args.batch] for i in range(0, n_seq, args.batch)]
                  if source_ids is not None else [None] * len(chunks))
    H = [streamer.embed(c) for c in chunks]
    hidden = margs.hidden_size
    use_res = margs.attn_res_block_size is not None
    BR = [
        mx.zeros((c.shape[0] * c.shape[1], 0, hidden), dtype=H[0].dtype) if use_res else None
        for c in chunks
    ]
    ssm_masks = [create_ssm_mask(h, None) for h in H]
    attn_masks = [create_attention_mask(h, None, return_array=True) for h in H]

    saliency = np.zeros((margs.num_hidden_layers, margs.num_experts), np.float64)
    counts = np.zeros((margs.num_hidden_layers, margs.num_experts), np.float64)
    # per-source saliency: (source, layer, expert). 12 x 93 x 896 float64 = 8 MB.
    per_src = np.zeros((n_src, margs.num_hidden_layers, margs.num_experts), np.float64)
    per_src_cnt = np.zeros((n_src, margs.num_hidden_layers, margs.num_experts), np.float64)
    # AWQ input statistic: mean |x| per input channel of the shared MoE latent.
    # K3's LatentMoE routes every expert in a layer through the SAME 3584-d
    # latent, so one vector per layer covers all 896 experts -- 1.3 MB instead of
    # the 1.2 GB a per-expert table would need.
    lat_dim = margs.routed_expert_hidden_size or margs.hidden_size
    act_mag = np.zeros((margs.num_hidden_layers, lat_dim), np.float64)
    act_n = np.zeros((margs.num_hidden_layers,), np.float64)

    t0 = time.time()
    for i in range(n_layers):
        lt = time.time()
        layer = streamer.build(i)

        NE_ = margs.num_experts
        sal = mx.zeros((NE_,), mx.float32)
        cnt = mx.zeros((NE_,), mx.float32)
        sal_s = mx.zeros((n_src * NE_,), mx.float32)
        cnt_s = mx.zeros((n_src * NE_,), mx.float32)
        cur_src = {"ids": None}
        amag = mx.zeros((margs.routed_expert_hidden_size or margs.hidden_size,), mx.float32)
        an = [0]

        if layer.is_moe:
            def hook(inds, weights, y):
                nonlocal sal, cnt, sal_s, cnt_s
                # ||e_j(x)||_2 over the expert's output dim, in fp32
                norms = mx.sqrt(mx.sum(y.astype(mx.float32) ** 2, axis=-1))
                contrib = (weights.astype(mx.float32) * norms).reshape(-1)
                idx = inds.reshape(-1)
                sal = sal.at[idx].add(contrib)
                cnt = cnt.at[idx].add(mx.ones_like(contrib))
                si = cur_src["ids"]
                if si is not None:
                    # Fold (source, expert) into one flat index so the per-source
                    # split costs a single scatter, not one pass per source.
                    s_exp = mx.repeat(si.reshape(-1), inds.shape[-1])
                    comb = s_exp * NE_ + idx
                    sal_s = sal_s.at[comb].add(contrib)
                    cnt_s = cnt_s.at[comb].add(mx.ones_like(contrib))

            def latent_hook(x, _l=layer):
                nonlocal amag
                amag = amag + mx.sum(mx.abs(x.astype(mx.float32)).reshape(-1, x.shape[-1]), axis=0)
                an[0] += x.size // x.shape[-1]

            layer.block_sparse_moe.latent_stats_hook = latent_hook
            layer.block_sparse_moe.expert_stats_hook = hook

        for ci in range(len(chunks)):
            cur_src["ids"] = src_chunks[ci]
            h, br = layer(
                H[ci],
                ssm_masks[ci] if layer.is_linear else attn_masks[ci],
                None,
                BR[ci],
            )
            mx.eval(h, br if br is not None else h, sal, cnt, sal_s, cnt_s)
            H[ci], BR[ci] = h, br

        if layer.is_moe:
            layer.block_sparse_moe.expert_stats_hook = None
            layer.block_sparse_moe.latent_stats_hook = None
            act_mag[i] = np.array(amag, copy=True)
            act_n[i] = an[0]
            saliency[i] = np.array(sal, copy=True)
            counts[i] = np.array(cnt, copy=True)
            if labels:
                per_src[:, i, :] = np.array(sal_s, copy=True).reshape(n_src, NE_)
                per_src_cnt[:, i, :] = np.array(cnt_s, copy=True).reshape(n_src, NE_)
            nz = int((counts[i] > 0).sum())
            print(f"[reap] layer {i:3d}/{n_layers-1} MoE  {time.time()-lt:6.1f}s  "
                  f"experts routed {nz}/{margs.num_experts}  "
                  f"S mean {saliency[i].mean()/n_tokens:.4e}  "
                  f"elapsed {(time.time()-t0)/60:5.1f}m", flush=True)
        else:
            print(f"[reap] layer {i:3d}/{n_layers-1} dense {time.time()-lt:6.1f}s", flush=True)

        del layer
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(
        args.out,
        saliency=saliency,
        counts=counts,
        n_tokens=np.array(n_tokens),
        moe_layers=np.array(moe_layers),
        num_experts=np.array(margs.num_experts),
        top_k=np.array(margs.num_experts_per_token),
        num_hidden_layers=np.array(margs.num_hidden_layers),
        per_source=per_src,
        per_source_counts=per_src_cnt,
        source_labels=np.array(labels if labels else ["all"]),
        act_mag=act_mag, act_n=act_n,
        source_tokens=np.array(
            [int((source_ids == i).sum()) for i in range(n_src)]
            if source_ids is not None else [n_tokens]
        ),
    )
    print(f"[reap] wrote {args.out}  ({(time.time()-t0)/60:.1f} min)")
    return saliency, counts, n_tokens


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--calib-text", default=None, help="plain-text calibration corpus")
    p.add_argument("--calib-tokens", default=None, help="pre-tokenized int .npy (seqs, len)")
    p.add_argument("--seqs", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8, help="sequences per forward chunk")
    p.add_argument("--layers", type=int, default=None, help="limit layers (smoke test)")
    calibrate(p.parse_args())
