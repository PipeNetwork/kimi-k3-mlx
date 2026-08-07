#!/usr/bin/env python3
"""Settle what fla actually does with K3's length-128 A_log.

kimi_k3.py assumes the decay is PER-CHANNEL (A_log broadcast against the
trailing head_dim axis) because the released weights ship A_log as [128] while
b_proj is [96, H]. tests/test_kimi_k3.py asserts that reading -- but asserting
an inference is not the same as checking it against the kernel that defines it.

This drives fla's own kernel three ways on identical real inputs:
  ref       kernel computes the gate itself from raw logits (what Moonshot runs)
  channel   gate precomputed with A_log broadcast on head_dim  (kimi_k3.py)
  head      gate precomputed with A_log broadcast on heads     (the stale init)

Whichever precomputed form reproduces `ref` is the true convention.

  .venv/bin/python parity_probe_gate.py --bundle layer13_mlx.safetensors --config config.json
"""
import argparse
import json

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from fla.ops.kda import chunk_kda


def stats(name, a, ref):
    a, ref = a.float().flatten(), ref.float().flatten()
    cos = F.cosine_similarity(a, ref, dim=0).item()
    print(f"  {name:<10} cosine={cos:.8f}  maxdiff={(a-ref).abs().max().item():.3e}  "
          f"mean|y|={a.abs().mean().item():.6f}")
    return cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    text = cfg.get("text_config", cfg)
    lac = text["linear_attn_config"]
    H, D = lac["num_heads"], lac["head_dim"]
    lower = lac.get("gate_lower_bound")
    hidden = text["hidden_size"]
    print(f"num_heads={H} head_dim={D} gate_lower_bound={lower}")

    b = load_file(args.bundle)
    w = {k[len("__w_self_attn."):]: v.cuda() for k, v in b.items()
         if k.startswith("__w_self_attn.")}
    x = b["__in_x"].cuda().float()
    B, T, _ = x.shape

    A_log = w["A_log"].float()
    dt_bias = w["dt_bias"].float()
    print(f"A_log {tuple(A_log.shape)}  dt_bias {tuple(dt_bias.shape)}")

    a_raw = F.linear(F.linear(x, w["f_a_proj.weight"].float()),
                     w["f_b_proj.weight"].float()).view(B, T, H, D)
    beta_raw = F.linear(x, w["b_proj.weight"].float())

    kernel_size = w["q_conv1d.weight"].shape[-1]

    # q/k/v exactly as both implementations build them (proj -> silu(conv) -> heads)
    def conv(name, proj):
        pad = F.pad(proj.transpose(1, 2), (kernel_size - 1, 0))  # causal left pad
        c = F.conv1d(pad, w[name].float(), groups=proj.shape[-1])
        return F.silu(c.transpose(1, 2))

    q = conv("q_conv1d.weight", F.linear(x, w["q_proj.weight"].float())).view(B, T, H, D)
    k = conv("k_conv1d.weight", F.linear(x, w["k_proj.weight"].float())).view(B, T, H, D)
    v = conv("v_conv1d.weight", F.linear(x, w["v_proj.weight"].float())).view(B, T, H, D)

    common = dict(q=q.bfloat16(), k=k.bfloat16(), v=v.bfloat16(),
                  initial_state=None, output_final_state=False,
                  use_qk_l2norm_in_kernel=True, transpose_state_layout=True,
                  cu_seqlens=None)

    # 1. what Moonshot actually runs: kernel owns the gate
    ref, _ = chunk_kda(**common, g=a_raw.bfloat16(), beta=beta_raw,
                       A_log=A_log, dt_bias=dt_bias,
                       use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                       safe_gate=lower is not None, lower_bound=lower)
    torch.cuda.synchronize()
    print(f"\nreference (kernel-computed gate)  mean|y|={ref.float().abs().mean().item():.6f}")

    beta = torch.sigmoid(beta_raw)
    sp = F.softplus(a_raw + dt_bias.view(H, D))

    forms = {
        "channel": -torch.exp(A_log.view(1, D)) * sp,   # kimi_k3.py's reading
        "head": -torch.exp(A_log.view(D, 1)[:H]) * sp,  # decay indexed by head
    }
    print("\nprecomputed gate, LOG space:")
    log_res = {}
    for name, log_g in forms.items():
        lg = torch.clamp(log_g, min=lower) if lower is not None else log_g
        o, _ = chunk_kda(**common, g=lg.bfloat16(), beta=beta,
                         A_log=None, dt_bias=None, use_gate_in_kernel=False,
                         use_beta_sigmoid_in_kernel=False)
        torch.cuda.synchronize()
        log_res[name] = stats(name, o, ref)

    print("\nprecomputed gate, EXP space (what kimi_k3.py hands its kernel):")
    for name, log_g in forms.items():
        lg = torch.clamp(log_g, min=lower) if lower is not None else log_g
        o, _ = chunk_kda(**common, g=torch.exp(lg).bfloat16(), beta=beta,
                         A_log=None, dt_bias=None, use_gate_in_kernel=False,
                         use_beta_sigmoid_in_kernel=False)
        torch.cuda.synchronize()
        stats(name, o, ref)

    best = max(log_res, key=log_res.get)
    print(f"\n-> fla's log-space match: '{best}' (cosine {log_res[best]:.6f})")


if __name__ == "__main__":
    main()
