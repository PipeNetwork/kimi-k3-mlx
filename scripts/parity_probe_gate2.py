#!/usr/bin/env python3
"""Pin down K3's gate formula against fla's own kernel.

fla/ops/kda/gate.py indexes A_log by HEAD (`tl.load(A_log + i_h)`, grid dim H =
g.shape[-2] = 96) and broadcasts it as a scalar over all 128 channels; the
checkpoint's [128] tensor is 96 real values zero-padded. It also takes a
completely different branch when lower_bound is set:

    USE_LOWER_BOUND: g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))
    else:            g = -exp(A_log) * softplus(a + dt_bias)

K3 sets gate_lower_bound=-5.0, so the sigmoid branch is the live one -- there is
no softplus and no clamp anywhere in it.

This reproduces the kernel's gate in plain torch and scores each candidate.

  .venv/bin/python parity_probe_gate2.py --bundle layer13_mlx.safetensors --config config.json
"""
import argparse
import json

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from fla.ops.kda import chunk_kda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    text = json.load(open(args.config))
    text = text.get("text_config", text)
    lac = text["linear_attn_config"]
    H, D, lower = lac["num_heads"], lac["head_dim"], lac.get("gate_lower_bound")

    b = load_file(args.bundle)
    w = {k[len("__w_self_attn."):]: v.cuda() for k, v in b.items()
         if k.startswith("__w_self_attn.")}
    x = b["__in_x"].cuda().float()
    B, T, _ = x.shape

    A_log = w["A_log"].float()
    dt_bias = w["dt_bias"].float()
    print(f"H={H} D={D} lower_bound={lower}")
    print(f"A_log{tuple(A_log.shape)}  nonzero tail [{H}:]: "
          f"{bool((A_log[H:] != 0).any().item())}")

    a_raw = F.linear(F.linear(x, w["f_a_proj.weight"].float()),
                     w["f_b_proj.weight"].float()).view(B, T, H, D)
    beta = torch.sigmoid(F.linear(x, w["b_proj.weight"].float()))
    ks = w["q_conv1d.weight"].shape[-1]

    def conv(name, proj):
        c = F.conv1d(F.pad(proj.transpose(1, 2), (ks - 1, 0)),
                     w[name].float(), groups=proj.shape[-1])
        return F.silu(c.transpose(1, 2))

    q = conv("q_conv1d.weight", F.linear(x, w["q_proj.weight"].float())).view(B, T, H, D)
    k = conv("k_conv1d.weight", F.linear(x, w["k_proj.weight"].float())).view(B, T, H, D)
    v = conv("v_conv1d.weight", F.linear(x, w["v_proj.weight"].float())).view(B, T, H, D)

    common = dict(q=q.bfloat16(), k=k.bfloat16(), v=v.bfloat16(),
                  initial_state=None, output_final_state=False,
                  use_qk_l2norm_in_kernel=True, transpose_state_layout=True,
                  cu_seqlens=None)

    ref, _ = chunk_kda(**common, g=a_raw.bfloat16(), beta=beta,
                       A_log=A_log, dt_bias=dt_bias, use_gate_in_kernel=True,
                       use_beta_sigmoid_in_kernel=False,
                       safe_gate=lower is not None, lower_bound=lower)
    torch.cuda.synchronize()

    s = a_raw + dt_bias.view(H, D)
    A_head = A_log[:H].view(H, 1)          # fla: per-head scalar
    A_chan = A_log.view(1, D)              # kimi_k3.py: per-channel

    cands = {
        "sigmoid/per-head  (fla)": lower * torch.sigmoid(torch.exp(A_head) * s),
        "sigmoid/per-chan": lower * torch.sigmoid(torch.exp(A_chan) * s),
        "softplus+clamp/per-head": torch.clamp(-torch.exp(A_head) * F.softplus(s), min=lower),
        "softplus+clamp/per-chan  (kimi_k3.py)":
            torch.clamp(-torch.exp(A_chan) * F.softplus(s), min=lower),
    }

    print()
    for name, log_g in cands.items():
        o, _ = chunk_kda(**common, g=log_g.bfloat16(), beta=beta,
                         A_log=None, dt_bias=None, use_gate_in_kernel=False,
                         use_beta_sigmoid_in_kernel=False)
        torch.cuda.synchronize()
        a, r = o.float().flatten(), ref.float().flatten()
        cos = F.cosine_similarity(a, r, dim=0).item()
        print(f"  {name:<40} cosine={cos:.8f}  maxdiff={(a-r).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
