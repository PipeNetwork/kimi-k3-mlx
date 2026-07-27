# MLX port of Kimi-K3's vision tower — a 3D (video-capable) MoonViT variant plus
# the patchmergerv2 projector.
#
# 0.49 B params, 0.02% of the model, and the only reason K3 is multimodal.
#
# Closest existing MLX code is mlx-vlm's `kimi_vl/vision.py` (MoonViT for
# Kimi-VL). This shares its complex64 2D-rope approach but differs in six ways,
# each marked [K3-V*] below:
#
#   [K3-V1] 3D grids (t,h,w): a 1D sin-cos *time* embedding is added on top of
#           the learnable 2D grid, and rope frequencies repeat per frame.
#   [K3-V2] bilinear position-embedding interpolation (Kimi-VL used bicubic);
#           mlx-vlm ships bicubic/nearest kernels only, so it is written here.
#   [K3-V3] qkv_hidden_size 1536 != hidden 1024 — attention runs wider than the
#           residual stream, so wo is [1024, 1536] and is NOT square.
#   [K3-V4] RMSNorm, not LayerNorm.
#   [K3-V5] no biases anywhere (attn_bias / linear_bias / patch_embed_proj_bias
#           are all false).
#   [K3-V6] sd2_tpool merge: mean-pool over frames, then 2x2 spatial merge, then
#           a 2-layer MLP projector with *exact* (erf) GELU and a post-RMSNorm.
#
# The ViT blocks use tanh-approximate GELU (`PytorchGELUTanh` in the reference)
# while the projector uses exact `nn.GELU()`. They are different functions and
# the reference deliberately uses both; do not unify them.

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# RMSNorm eps for the ViT blocks. The reference constructs `nn.RMSNorm(dim)`
# with no eps, so PyTorch falls back to finfo(dtype).eps — 1.19e-7 in fp32 but
# 7.8e-3 in bf16, which are materially different normalizers. We use the fp32
# value, matching an upcast implementation; `projector_ln_eps` (1e-5) is given
# explicitly by the config and is used only for the projector's post_norm.
VIT_NORM_EPS = 1.1920929e-07


@dataclass
class VisionConfig:
    vt_hidden_size: int = 1024
    vt_intermediate_size: int = 4096
    vt_num_hidden_layers: int = 27
    vt_num_attention_heads: int = 12
    qkv_hidden_size: int = 1536
    patch_size: int = 14
    num_channels: int = 3
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    init_pos_emb_time: int = 4
    pos_emb_type: str = "divided_fixed"
    pos_emb_interpolation_mode: str = "bilinear"
    merge_kernel_size: Tuple[int, int] = (2, 2)
    merge_type: str = "sd2_tpool"
    norm_type: str = "rmsnorm"
    mlp_type: str = "mlp2"
    mm_projector_type: str = "patchmergerv2"
    mm_hidden_size: int = 1024
    text_hidden_size: int = 7168
    projector_hidden_act: str = "gelu"
    projector_ln_eps: float = 1e-5
    attn_bias: bool = False
    linear_bias: bool = False
    patch_embed_proj_bias: bool = False
    rope_theta_base: float = 10000.0
    rope_max_size: int = 512

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisionConfig":
        known = {f for f in cls.__dataclass_fields__}
        out = {k: v for k, v in d.items() if k in known}
        if "merge_kernel_size" in out:
            out["merge_kernel_size"] = tuple(out["merge_kernel_size"])
        return cls(**out)

    @property
    def head_dim(self) -> int:
        return self.qkv_hidden_size // self.vt_num_attention_heads


# ------------------------------------------------------- [K3-V2] bilinear


def bilinear_interpolate(x: mx.array, out_h: int, out_w: int) -> mx.array:
    """Resize (H, W, C) -> (out_h, out_w, C), matching torch's bilinear
    `F.interpolate(..., align_corners=False)`.

    torch maps destination index d to source coordinate
    `max(0, scale * (d + 0.5) - 0.5)` and lerps between the two neighbours,
    clamping the upper index at the edge. Getting the half-pixel offset wrong
    shifts every position embedding by half a patch, which is the kind of error
    that degrades quality without ever looking obviously broken.
    """
    in_h, in_w = x.shape[0], x.shape[1]
    if (in_h, in_w) == (out_h, out_w):
        return x

    def axis_index(in_n: int, out_n: int):
        scale = in_n / out_n
        src = mx.maximum(
            (mx.arange(out_n, dtype=mx.float32) + 0.5) * scale - 0.5,
            mx.array(0.0, mx.float32),
        )
        i0 = mx.floor(src)
        w1 = src - i0
        i0 = i0.astype(mx.int32)
        i1 = mx.minimum(i0 + 1, mx.array(in_n - 1, mx.int32))
        return i0, i1, w1

    h0, h1, wh = axis_index(in_h, out_h)
    w0, w1i, ww = axis_index(in_w, out_w)

    xf = x.astype(mx.float32)
    top = mx.take(xf, h0, axis=0)      # (out_h, in_w, C)
    bot = mx.take(xf, h1, axis=0)
    wh = wh.reshape(out_h, 1, 1)
    rows = top * (1.0 - wh) + bot * wh

    left = mx.take(rows, w0, axis=1)   # (out_h, out_w, C)
    right = mx.take(rows, w1i, axis=1)
    ww = ww.reshape(1, out_w, 1)
    return left * (1.0 - ww) + right * ww


def sincos_1d(dim: int, length: int) -> mx.array:
    """(length, dim) sin-cos table; matches get_1d_sincos_pos_embed."""
    omega = mx.arange(dim // 2, dtype=mx.float32) / (dim / 2.0)
    omega = 1.0 / (10000.0**omega)
    out = mx.outer(mx.arange(length, dtype=mx.float32), omega)
    return mx.concatenate([mx.sin(out), mx.cos(out)], axis=1)


# ------------------------------------------------ [K3-V1] 3D position embed


class Learnable2DInterpPosEmbDivided(nn.Module):
    """Learnable 2D grid, bilinearly resized per image, plus a sin-cos time term."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        # reference does nn.init.normal_(weight); a zero init would make an
        # unloaded tower emit exactly zero, which hides a failed weight load
        self.weight = mx.random.normal(
            (cfg.init_pos_emb_height, cfg.init_pos_emb_width, cfg.vt_hidden_size)
        )
        self._time = sincos_1d(cfg.vt_hidden_size, cfg.init_pos_emb_time)

    def __call__(self, x: mx.array, grid_thws: List[Tuple[int, int, int]]) -> mx.array:
        embs = []
        for t, h, w in grid_thws:
            if (h, w) == self.weight.shape[:2]:
                emb2d = self.weight.reshape(-1, self.weight.shape[-1])
            else:
                emb2d = bilinear_interpolate(self.weight, h, w).reshape(
                    -1, self.weight.shape[-1]
                )
            if t == 1:
                emb = emb2d
            else:
                emb = mx.expand_dims(emb2d, 0) + mx.expand_dims(self._time[:t], 1)
                emb = emb.reshape(-1, emb.shape[-1])
            embs.append(emb)
        return x + mx.concatenate(embs, axis=0).astype(x.dtype)


class MoonVision3dPatchEmbed(nn.Module):
    """Patch projection. The reference is a Conv2d whose kernel equals the patch
    size applied to pre-extracted (L, 3, 14, 14) patches, which is exactly a
    linear map over the flattened patch — done as a matmul here to sidestep
    MLX's NHWC conv weight layout entirely."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        fan_in = cfg.num_channels * cfg.patch_size * cfg.patch_size
        # torch Conv2d default: uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
        bound = 1.0 / math.sqrt(fan_in)
        self.proj_weight = mx.random.uniform(
            low=-bound, high=bound, shape=(cfg.vt_hidden_size, fan_in)
        )
        self.pos_emb = Learnable2DInterpPosEmbDivided(cfg)

    def __call__(self, patches: mx.array, grid_thws) -> mx.array:
        x = patches.reshape(patches.shape[0], -1).astype(self.proj_weight.dtype)
        x = x @ self.proj_weight.T
        return self.pos_emb(x, grid_thws)


# --------------------------------------------------------- [K3-V1] 2D rope


class Rope2DPosEmbRepeated(nn.Module):
    """freqs_cis[h, w] interleaves x (width) and y (height) frequencies:
    entry 2i is cis(w * theta^(-4i/dim)), entry 2i+1 is cis(h * ...).
    Repeated t times so every frame shares the same spatial rope."""

    def __init__(self, dim: int, max_h: int, max_w: int, theta_base: float = 10000.0):
        super().__init__()
        assert dim % 4 == 0, "rope dim must be divisible by 4"
        self.dim, self.max_h, self.max_w, self.theta_base = dim, max_h, max_w, theta_base
        self._cache: Optional[mx.array] = None

    def _full(self) -> mx.array:
        if self._cache is None:
            n = self.max_h * self.max_w
            flat = mx.arange(0, n, dtype=mx.float32)
            x_pos = flat % self.max_w
            y_pos = mx.floor(flat / self.max_w)
            dim_range = mx.arange(0, self.dim, 4)[: self.dim // 4].astype(mx.float32)
            freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
            xf, yf = mx.outer(x_pos, freqs), mx.outer(y_pos, freqs)
            x_cis = mx.cos(xf) + 1j * mx.sin(xf)
            y_cis = mx.cos(yf) + 1j * mx.sin(yf)
            self._cache = mx.stack([x_cis, y_cis], axis=-1).reshape(
                self.max_h, self.max_w, -1
            )
        return self._cache

    def __call__(self, grid_thws: List[Tuple[int, int, int]]) -> mx.array:
        full = self._full()
        parts = []
        for t, h, w in grid_thws:
            assert 1 <= h <= self.max_h and 1 <= w <= self.max_w, (h, w)
            base = full[:h, :w].reshape(-1, self.dim // 2)
            parts.append(mx.tile(base, (t, 1)) if t > 1 else base)
        return mx.concatenate(parts, axis=0)


def apply_rope(q: mx.array, k: mx.array, freqs_cis: mx.array):
    """q, k: (L, heads, head_dim); freqs_cis: (L, head_dim/2) complex64."""
    f = mx.expand_dims(freqs_cis, axis=-2)

    def rot(t):
        pair = t.astype(mx.float32).reshape(*t.shape[:-1], -1, 2)
        c = (pair[..., 0] + 1j * pair[..., 1]) * f
        out = mx.stack([mx.real(c), mx.imag(c)], axis=-1)
        return out.reshape(*t.shape).astype(t.dtype)

    return rot(q), rot(k)


# ----------------------------------------------------------- encoder


class MLP2(nn.Module):
    def __init__(self, dims: List[int], bias: bool = False):
        super().__init__()
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        # PytorchGELUTanh -- the tanh approximation, not exact erf GELU.
        return self.fc1(nn.gelu_approx(self.fc0(x)))


class MoonViTEncoderLayer(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        h, qkv = cfg.vt_hidden_size, cfg.qkv_hidden_size
        self.num_heads = cfg.vt_num_attention_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5

        self.norm0 = nn.RMSNorm(h, eps=VIT_NORM_EPS)
        self.norm1 = nn.RMSNorm(h, eps=VIT_NORM_EPS)
        self.wqkv = nn.Linear(h, qkv * 3, bias=cfg.attn_bias)
        self.wo = nn.Linear(qkv, h, bias=cfg.attn_bias)  # [K3-V3] not square
        self.mlp = MLP2([h, cfg.vt_intermediate_size, h], bias=cfg.linear_bias)

    def _attend(self, x, freqs_cis, spans):
        L = x.shape[0]
        qkv = self.wqkv(x).reshape(L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = apply_rope(q, k, freqs_cis)

        # Images are packed into one sequence but must not attend across each
        # other. Looping the (few) images costs sum(L_i^2) instead of the
        # (sum L_i)^2 an explicit block-diagonal mask would.
        outs = []
        for s, e in spans:
            qi = q[s:e].transpose(1, 0, 2)[None]
            ki = k[s:e].transpose(1, 0, 2)[None]
            vi = v[s:e].transpose(1, 0, 2)[None]
            o = mx.fast.scaled_dot_product_attention(qi, ki, vi, scale=self.scale)
            outs.append(o[0].transpose(1, 0, 2).reshape(e - s, -1))
        return self.wo(mx.concatenate(outs, axis=0))

    def __call__(self, x, freqs_cis, spans):
        x = x + self._attend(self.norm0(x), freqs_cis, spans)
        return x + self.mlp(self.norm1(x))


class MoonViT3dEncoder(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.rope_2d = Rope2DPosEmbRepeated(
            cfg.head_dim, cfg.rope_max_size, cfg.rope_max_size, cfg.rope_theta_base
        )
        self.blocks = [MoonViTEncoderLayer(cfg) for _ in range(cfg.vt_num_hidden_layers)]
        self.final_layernorm = nn.RMSNorm(cfg.vt_hidden_size, eps=VIT_NORM_EPS)

    def __call__(self, x: mx.array, grid_thws) -> mx.array:
        freqs_cis = self.rope_2d(grid_thws)
        spans, pos = [], 0
        for t, h, w in grid_thws:
            spans.append((pos, pos + t * h * w))
            pos += t * h * w
        for blk in self.blocks:
            x = blk(x, freqs_cis, spans)
        return self.final_layernorm(x)


# ---------------------------------------------------- [K3-V6] merge + project


def tpool_patch_merger(x: mx.array, grid_thws, merge_kernel_size=(2, 2)):
    """-> list of (h/2 * w/2, 4, C): mean over frames, then 2x2 spatial merge."""
    d = x.shape[-1]
    kh, kw = merge_kernel_size
    out, pos = [], 0
    for t, h, w in grid_thws:
        seq = x[pos : pos + t * h * w]
        pos += t * h * w
        nh, nw = h // kh, w // kw
        s = seq.reshape(t, nh, kh, nw, kw, d).transpose(0, 1, 3, 2, 4, 5)
        s = mx.mean(s, axis=0)                      # temporal pooling
        out.append(s.reshape(nh * nw, kh * kw, d))
    return out


class PatchMergerMLPV2(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        inner = cfg.mm_hidden_size * cfg.merge_kernel_size[0] * cfg.merge_kernel_size[1]
        self.inner = inner
        # nn.Sequential(Linear, GELU, Linear) in the reference -> proj.0 / proj.2
        self.proj_0 = nn.Linear(inner, inner, bias=False)
        self.proj_2 = nn.Linear(inner, cfg.text_hidden_size, bias=False)
        self.post_norm = nn.RMSNorm(cfg.text_hidden_size, eps=cfg.projector_ln_eps)

    def __call__(self, feats: List[mx.array]) -> List[mx.array]:
        lengths = [f.shape[0] for f in feats]
        x = mx.concatenate([f.reshape(f.shape[0], -1) for f in feats], axis=0)
        # exact erf GELU here -- deliberately different from the ViT's tanh GELU
        x = self.post_norm(self.proj_2(nn.gelu(self.proj_0(x))))
        out, pos = [], 0
        for n in lengths:
            out.append(x[pos : pos + n])
            pos += n
        return out


class VisionModel(nn.Module):
    """vision_tower + mm_projector: pixels -> text-space embeddings."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.pos_emb_type != "divided_fixed":
            raise NotImplementedError(cfg.pos_emb_type)
        if cfg.merge_type != "sd2_tpool":
            raise NotImplementedError(cfg.merge_type)
        if cfg.mm_projector_type != "patchmergerv2":
            raise NotImplementedError(cfg.mm_projector_type)
        if cfg.norm_type != "rmsnorm":
            raise NotImplementedError(cfg.norm_type)
        self.patch_embed = MoonVision3dPatchEmbed(cfg)
        self.encoder = MoonViT3dEncoder(cfg)
        self.mm_projector = PatchMergerMLPV2(cfg)

    def __call__(self, pixel_values: mx.array, grid_thws) -> List[mx.array]:
        grid_thws = [tuple(int(v) for v in g) for g in grid_thws]
        x = self.patch_embed(pixel_values, grid_thws)
        x = self.encoder(x, grid_thws)
        merged = tpool_patch_merger(x, grid_thws, self.cfg.merge_kernel_size)
        return self.mm_projector(merged)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        out: Dict[str, mx.array] = {}
        for k, v in weights.items():
            if k.startswith("vision_tower."):
                k = k[len("vision_tower.") :]
                if k == "patch_embed.proj.weight":
                    # Conv2d (out, in, kH, kW) -> flat linear (out, in*kH*kW).
                    # torch sums over (c, i, j) in that order, so a plain
                    # reshape is the correct flattening.
                    out["patch_embed.proj_weight"] = v.reshape(v.shape[0], -1)
                elif k == "patch_embed.pos_emb.weight":
                    out["patch_embed.pos_emb.weight"] = v
                else:
                    out[k] = v
            elif k.startswith("mm_projector."):
                k = k[len("mm_projector.") :]
                if k == "proj.0.weight":
                    out["mm_projector.proj_0.weight"] = v
                elif k == "proj.2.weight":
                    out["mm_projector.proj_2.weight"] = v
                else:
                    out["mm_projector." + k] = v
        return out
