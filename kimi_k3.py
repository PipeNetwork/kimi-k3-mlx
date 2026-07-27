# MLX port of moonshotai/Kimi-K3 (2.78T-total / 104B-active multimodal MoE).
#
# Text tower only — the vision tower rides along in the checkpoint and is handled
# separately (see scripts/convert.py, which passes it through untouched).
#
# Derived from mlx-lm's `kimi_linear.py` (Kimi-Linear-48B-A3B), which already
# covers KDA + gated MLA + DeepSeek-style sparse MoE. K3 adds five things that
# do not exist anywhere in MLX; each is marked [K3] below:
#
#   [K3-1] SiTU-GLU activation           beta=4.0, linear_beta=25.0
#   [K3-2] Attention Residuals (AttnRes) rank-1 [1,H] score projections, block 12
#   [K3-3] Stable LatentMoE              routed experts live in a 3584-d latent
#   [K3-4] q-LoRA + output-gated MLA     q_a/q_b rank 1536, sigmoid g_proj
#   [K3-5] per-channel KDA decay         A_log is [head_dim], not [num_heads]
#
# NOTE on [K3-5]: the reference `modeling_kimi_linear.py` allocates
# `A_log = Parameter(log(empty(num_heads)))` — i.e. [96] — but every checkpoint
# shard ships A_log as [128] == head_dim, while b_proj is [96, H] and dt_bias is
# [12288] == 96*128. fla's kernel broadcasts A_log against g of shape
# (B,T,96,128), so a length-128 vector can only align with the trailing head-dim
# axis. The decay rate is therefore per-channel and shared across heads. The
# reference init code is simply stale w.r.t. the released weights; the shapes are
# authoritative. tests/test_kimi_k3.py asserts this against the real shard.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_kernel, gated_delta_ops
from .mla import MultiLinear
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rms_norm_eps: float
    linear_attn_config: Dict[str, Any]
    num_experts: int
    moe_intermediate_size: int
    kv_lora_rank: int
    q_lora_rank: Optional[int] = None
    tie_word_embeddings: bool = False
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mla_use_nope: bool = True
    mla_use_output_gate: bool = False
    num_experts_per_token: int = 1
    num_shared_experts: int = 0
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    num_expert_group: int = 1
    topk_group: int = 1
    max_position_embeddings: int = 1048576
    # [K3-1]
    hidden_act: str = "situ"
    activation_situ_beta: float = 1.0
    activation_situ_linear_beta: Optional[float] = None
    # [K3-2]
    attn_res_block_size: Optional[int] = None
    # [K3-3]
    routed_expert_hidden_size: Optional[int] = None
    latent_moe_use_norm: bool = False
    # REAP-pruned checkpoints keep a different number of experts per layer
    # (0 for dense layers). None => every MoE layer has `num_experts`.
    expert_counts: Optional[List[int]] = None
    # multimodal wrapper fields (ignored by the text tower)
    vision_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelArgs":
        # K3's config.json nests the language model under `text_config`.
        if "text_config" in params:
            merged = dict(params["text_config"])
            merged["model_type"] = params.get("model_type", "kimi_k3")
            merged["vision_config"] = params.get("vision_config", {})
            for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
                merged.setdefault(k, params.get(k))
            params = merged
        return super().from_dict(params)


# ---------------------------------------------------------------- [K3-1] SiTU


class SituGLU(nn.Module):
    """beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta).

    Matches SituAndMul in the reference, including the float32 upcast — the
    tanh saturation points (beta=4, linear_beta=25) are far enough out that
    bf16 rounding visibly changes the result.
    """

    def __init__(self, beta: float = 1.0, linear_beta: Optional[float] = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def __call__(self, x_up: mx.array, x_gate: mx.array) -> mx.array:
        dtype = x_up.dtype
        gate = x_gate.astype(mx.float32)
        up = x_up.astype(mx.float32)
        out = self.beta * mx.tanh(gate / self.beta) * mx.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * mx.tanh(up / self.linear_beta)
        return (out * up).astype(dtype)


def _situ_from_args(args: ModelArgs) -> SituGLU:
    return SituGLU(args.activation_situ_beta or 1.0, args.activation_situ_linear_beta)


class KimiMLP(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        h = hidden_size or args.hidden_size
        i = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        self.act = _situ_from_args(args)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act(self.up_proj(x), self.gate_proj(x)))


# ------------------------------------------------------------- [K3-2] AttnRes


def _apply_attn_res(
    prefix_sum: mx.array,
    block_residual: mx.array,
    proj_weight: mx.array,
    norm_weight: mx.array,
    eps: float,
) -> mx.array:
    """Softmax mixture over [block residuals..., prefix_sum].

    prefix_sum:     (N, H)
    block_residual: (N, K, H)  ->  returns (N, H)

    Each candidate is RMS-normed, scored against a single learned direction
    (norm.weight * proj.weight, both length H), and the candidates are combined
    by a softmax over K+1. All in float32, as in the reference.
    """
    v = mx.concatenate([block_residual, mx.expand_dims(prefix_sum, 1)], axis=1)
    vf = v.astype(mx.float32)
    variance = mx.mean(vf * vf, axis=-1, keepdims=True)
    k = vf * mx.rsqrt(variance + eps)
    score_weight = norm_weight.astype(mx.float32) * proj_weight.reshape(-1).astype(
        mx.float32
    )
    scores = mx.sum(k * score_weight, axis=-1)
    probs = mx.expand_dims(mx.softmax(scores, axis=-1), 1)
    return (probs @ vf).squeeze(1).astype(v.dtype)


class AttnResProj(nn.Module):
    """Holds the rank-1 [1, H] score projection and its paired RMSNorm gain.

    Kept as plain arrays rather than nn.Linear/nn.RMSNorm so the quantizer skips
    them — they are 7168 floats each and feed a softmax, so quantizing them is
    all downside.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj_weight = mx.zeros((1, hidden_size))
        self.norm_weight = mx.ones((hidden_size,))

    def __call__(
        self, prefix_sum: mx.array, block_residual: mx.array, eps: float
    ) -> mx.array:
        return _apply_attn_res(
            prefix_sum, block_residual, self.proj_weight, self.norm_weight, eps
        )


# ------------------------------------------------------ [K3-4] q-LoRA gated MLA


class KimiMLAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim or 128
        self.qk_rope_head_dim = args.qk_rope_head_dim or 0
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim or 128
        self.kv_lora_rank = args.kv_lora_rank
        self.q_lora_rank = args.q_lora_rank
        self.scale = self.q_head_dim**-0.5

        h = args.hidden_size
        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(h, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(h, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            h, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)

        # kv_b_proj is absorbed into these two at load time (see sanitize), so the
        # KV cache holds the 512-d latent instead of 96 full heads. At 1M context
        # that is the difference between ~110 GB and ~1.2 GB of cache per layer.
        self.embed_q = MultiLinear(self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(self.kv_lora_rank, self.v_head_dim, self.num_heads)

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, h, bias=False)

        self.use_output_gate = args.mla_use_output_gate
        if self.use_output_gate:
            self.g_proj = nn.Linear(h, self.num_heads * self.v_head_dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = mx.expand_dims(self.kv_a_layernorm(compressed_kv), axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        # NoPE: q_pe/k_pe carry no rotary embedding at all (the reference asserts
        # use_nope and leaves rotary_emb=None). They are a plain extra 64 dims,
        # folded in as an additive score term alongside the causal mask.
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        out = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            out = self.unembed_out(out)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.use_output_gate:
            out = out * mx.sigmoid(self.g_proj(x))
        return self.o_proj(out)


# ------------------------------------------------------------- [K3-5] KDA


class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=0,
            bias=False,
        )

    def __call__(self, x, state, mask=None, lengths=None):
        if mask is not None and state is not None:
            x = x * mx.expand_dims(mask, -1) if mask.ndim == 2 else x
        conv_input = mx.concatenate([state, x], axis=1)
        new_state = conv_input[:, -(self.kernel_size - 1) :, :]
        return nn.silu(self.conv(conv_input)), new_state


class KimiDeltaAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        cfg = args.linear_attn_config
        self.layer_idx = layer_idx
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.conv_kernel = cfg.get("short_conv_kernel_size", 4)
        self.gate_lower_bound = cfg.get("gate_lower_bound", None)
        self.use_full_rank_gate = cfg.get("use_full_rank_gate", False)

        self.projection_dim = self.num_heads * self.head_dim
        h = args.hidden_size
        self.scale = float(self.head_dim) ** -0.5

        self.q_proj = nn.Linear(h, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(h, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(h, self.projection_dim, bias=False)

        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(h, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(h, self.num_heads, bias=False)

        # [K3] full-rank output gate (use_full_rank_gate=True) replaces the
        # low-rank g_a/g_b pair used by Kimi-Linear-48B.
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(h, self.projection_dim, bias=False)
        else:
            self.g_a_proj = nn.Linear(h, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        # [K3-5] per-channel, shared across heads -> broadcast as (1, head_dim).
        self.A_log = mx.zeros((self.head_dim,))
        self.dt_bias = mx.zeros((self.projection_dim,))

        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, h, bias=False)

    def _compute_g(self, a: mx.array) -> mx.array:
        # log-decay, then the safe-gate clamp (fla's `lower_bound`), then exp.
        log_g = -mx.exp(self.A_log.reshape(1, self.head_dim).astype(mx.float32)) * nn.softplus(
            a + self.dt_bias.reshape(self.num_heads, self.head_dim)
        )
        if self.gate_lower_bound is not None:
            log_g = mx.maximum(log_g, self.gate_lower_bound)
        return mx.exp(log_g)

    def __call__(self, x, mask=None, cache=None):
        B, T, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, ssm_state = cache
            lengths = cache.lengths
        else:
            q_state = k_state = v_state = ssm_state = None
            lengths = None

        if q_state is None:
            s = mx.zeros((B, self.conv_kernel - 1, self.projection_dim), dtype=dtype)
            q_state = k_state = v_state = s

        q_conv, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k_conv, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v_conv, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)

        if cache is not None:
            cache[0], cache[1], cache[2] = q_state, k_state, v_state

        q = q_conv.reshape(B, T, self.num_heads, self.head_dim)
        k = k_conv.reshape(B, T, self.num_heads, self.head_dim)
        v = v_conv.reshape(B, T, self.num_heads, self.head_dim)

        # use_qk_l2norm_in_kernel=True in the reference
        q = (self.scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = self.scale * mx.fast.rms_norm(k, None, 1e-6)

        a_logits = self.f_b_proj(self.f_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim
        )
        beta = mx.sigmoid(self.b_proj(x).reshape(B, T, self.num_heads))
        g = self._compute_g(a_logits)

        if ssm_state is None:
            ssm_state = mx.zeros(
                (B, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32
            )
        use_kernel = mx.default_device() == mx.gpu and mx.metal.is_available()
        fn = gated_delta_kernel if use_kernel else gated_delta_ops
        out, ssm_state = fn(q, k, v, g, beta, ssm_state, mask)

        if cache is not None:
            cache[3] = ssm_state
            cache.advance(T)

        if self.use_full_rank_gate:
            gate = self.g_proj(x)
        else:
            gate = self.g_b_proj(self.g_a_proj(x))
        gate = gate.reshape(B, T, self.num_heads, self.head_dim)
        out = (
            self.o_norm(out.reshape(B, T, self.num_heads, self.head_dim))
            * mx.sigmoid(gate)
        ).reshape(B, T, -1)
        return self.o_proj(out)


# --------------------------------------------------------- [K3-3] LatentMoE


def _group_expert_select(
    scores: mx.array,
    bias: mx.array,
    top_k: int,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
    activation: str,
):
    if activation == "sigmoid":
        scores = mx.sigmoid(scores.astype(mx.float32))
    else:
        scores = mx.softmax(scores.astype(mx.float32), axis=-1)

    orig = scores
    scores = scores + bias
    if num_expert_group > 1 and num_expert_group > topk_group:
        k = scores.shape[-1] // num_expert_group
        scores = mx.unflatten(scores, -1, (num_expert_group, k))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        gate = mx.argpartition(-group_scores, kth=topk_group - 1, axis=-2)
        gate = mx.put_along_axis(
            mx.ones_like(group_scores),
            gate[..., :topk_group, :],
            mx.zeros((1,), scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores - gate * mx.finfo(scores.dtype).max, -2, -1)

    inds = mx.stop_gradient(mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k])
    weights = mx.take_along_axis(orig, inds, axis=-1)
    if top_k > 1 and renormalize:
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights.astype(orig.dtype)


class KimiSparseMoE(nn.Module):
    """LatentMoE: 896 experts run in a 3584-d latent, not the 7168-d residual.

    hidden -> down_proj -> [top-16 of 896 experts] -> norm -> up_proj -> hidden
    plus 2 shared experts applied to the *undown-projected* input.
    """

    def __init__(self, args: ModelArgs, num_experts: Optional[int] = None):
        super().__init__()
        self.args = args
        h = args.hidden_size
        self.num_experts = num_experts if num_experts is not None else args.num_experts
        self.use_latent = args.routed_expert_hidden_size is not None
        self.moe_hidden = args.routed_expert_hidden_size if self.use_latent else h
        self.use_norm = args.latent_moe_use_norm

        self.gate = nn.Linear(h, self.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((self.num_experts,), dtype=mx.float32)
        self.switch_mlp = SwitchGLU(
            self.moe_hidden,
            args.moe_intermediate_size,
            self.num_experts,
            activation=_situ_from_args(args),
        )

        if self.use_latent:
            self.routed_expert_down_proj = nn.Linear(h, self.moe_hidden, bias=False)
            self.routed_expert_up_proj = nn.Linear(self.moe_hidden, h, bias=False)
            if self.use_norm:
                self.routed_expert_norm = nn.RMSNorm(
                    self.moe_hidden, eps=args.rms_norm_eps
                )

        if args.num_shared_experts:
            self.shared_experts = KimiMLP(
                args, intermediate_size=args.moe_intermediate_size * args.num_shared_experts
            )
        else:
            self.shared_experts = None

        # Optional REAP saliency tap (scripts/reap_calibrate.py). Left as None in
        # normal use and costs nothing. It exists so the calibration harness reads
        # gates and per-expert outputs from *this* forward pass rather than a
        # reimplementation that could drift from it. MLX keeps non-array
        # attributes out of parameters()/children(), so this never touches
        # quantization, saving or loading.
        self.expert_stats_hook = None

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        inds, weights = _group_expert_select(
            self.gate(x),
            self.e_score_correction_bias,
            min(self.args.num_experts_per_token, self.num_experts),
            self.args.num_expert_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.moe_renormalize,
            self.args.moe_router_activation_func,
        )
        y = self.routed_expert_down_proj(x) if self.use_latent else x
        y = self.switch_mlp(y, inds)
        if self.expert_stats_hook is not None:
            # y here is the per-expert output BEFORE gate weighting, shape
            # (..., top_k, moe_hidden) -- exactly what REAP saliency needs.
            self.expert_stats_hook(inds, weights, y)
        y = (y * weights[..., None]).sum(axis=-2)
        if self.use_latent:
            if self.use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


# ------------------------------------------------- checkpoint key remapping
#
# Single source of truth, shared by Model.sanitize (whole-checkpoint, used when
# mlx-lm loads a converted repo) and scripts/convert.py (per-layer, used when
# streaming the 1.56 TB source). Keeping one implementation is the point: the
# streaming converter never instantiates a Model, so a second copy of these
# rules would be free to drift out of sync and silently mis-place weights.


def is_kda_layer(args: ModelArgs, idx: int) -> bool:
    # config lists kda_layers/full_attn_layers 1-BASED: config n -> layer n-1.
    return (idx + 1) in args.linear_attn_config["kda_layers"]


def is_moe_layer(args: ModelArgs, idx: int) -> bool:
    return (
        args.num_experts > 0
        and idx >= args.first_k_dense_replace
        and idx % args.moe_layer_freq == 0
    )


def experts_in_layer(args: ModelArgs, idx: int) -> int:
    """Expert count for one layer — varies after REAP pruning."""
    if args.expert_counts:
        return int(args.expert_counts[idx])
    return args.num_experts


def remap_checkpoint(
    weights: Dict[str, mx.array],
    args: ModelArgs,
    layer_indices: Optional[List[int]] = None,
    stack_experts: bool = True,
) -> Dict[str, mx.array]:
    """Map Kimi-K3 checkpoint keys onto this module tree.

    `layer_indices` restricts the layer loop (the converter passes the single
    layer it is holding). `stack_experts=False` leaves per-expert tensors alone
    so the converter can quantize each expert before stacking, which keeps peak
    memory at ~6 GB per layer instead of ~59 GB.
    """
    out: Dict[str, mx.array] = {}
    for k, v in weights.items():
        # K3 wraps the text tower as `language_model.*`; the vision tower and
        # projector are handled outside mlx-lm.
        if k.startswith("vision_tower.") or k.startswith("mm_projector."):
            continue
        out[k[len("language_model.") :] if k.startswith("language_model.") else k] = v
    weights = out

    if args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    if layer_indices is None:
        layer_indices = range(args.num_hidden_layers)

    for i in layer_indices:
        p = f"model.layers.{i}"

        # [K3-2] AttnRes: fold `*_res_proj` / `*_res_norm` into the AttnResProj
        # containers so the quantizer leaves them alone.
        for src in ("self_attention_res", "mlp_res"):
            for suffix, field_name in (("proj", "proj_weight"), ("norm", "norm_weight")):
                key = f"{p}.{src}_{suffix}.weight"
                if key in weights:
                    weights[f"{p}.{src}.{field_name}"] = weights.pop(key)

        if is_moe_layer(args, i):
            src = f"{p}.block_sparse_moe"
            if stack_experts:
                for w, dst in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
                    if f"{src}.experts.0.{w}.weight" in weights:
                        weights[f"{src}.switch_mlp.{dst}.weight"] = mx.stack(
                            [
                                weights.pop(f"{src}.experts.{e}.{w}.weight")
                                for e in range(experts_in_layer(args, i))
                            ]
                        )
            bias_key = f"{src}.gate.e_score_correction_bias"
            if bias_key in weights:
                weights[f"{src}.e_score_correction_bias"] = weights.pop(bias_key)

        attn = f"{p}.self_attn"
        if is_kda_layer(args, i):
            for src, dst in (("q_conv1d", "q_conv"), ("k_conv1d", "k_conv"), ("v_conv1d", "v_conv")):
                key = f"{attn}.{src}.weight"
                if key in weights:
                    w = weights.pop(key)
                    if w.ndim == 3:
                        w = w.moveaxis(2, 1)
                    weights[f"{attn}.{dst}.conv.weight"] = w
            dt = f"{attn}.dt_bias"
            if dt in weights and weights[dt].ndim > 1:
                weights[dt] = weights[dt].reshape(-1)
        else:
            # [K3-4] absorb kv_b_proj -> embed_q / unembed_out so the KV cache
            # stores the 512-d latent rather than 96 materialised heads.
            kv_b = f"{attn}.kv_b_proj.weight"
            if kv_b in weights:
                qk_nope = args.qk_nope_head_dim or 128
                v_head = args.v_head_dim or 128
                v = weights.pop(kv_b).reshape(args.num_attention_heads, qk_nope + v_head, -1)
                weights[f"{attn}.embed_q.weight"] = mx.contiguous(
                    v[:, :qk_nope, :].swapaxes(-1, -2)
                )
                weights[f"{attn}.unembed_out.weight"] = mx.contiguous(v[:, qk_nope:, :])

    for suffix, field_name in (("proj", "proj_weight"), ("norm", "norm_weight")):
        key = f"model.output_attn_res_{suffix}.weight"
        if key in weights:
            weights[f"model.output_attn_res.{field_name}"] = weights.pop(key)

    return weights


# ------------------------------------------------------------------- layers


class KimiDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        # config lists full_attn_layers/kda_layers 1-BASED: config n -> layer n-1.
        kda_layers = args.linear_attn_config["kda_layers"]
        self.is_linear = (layer_idx + 1) in kda_layers

        if self.is_linear:
            self.self_attn = KimiDeltaAttention(args, layer_idx)
        else:
            self.self_attn = KimiMLAAttention(args)

        is_moe = (
            args.num_experts > 0
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        )
        if is_moe:
            self.block_sparse_moe = KimiSparseMoE(args, experts_in_layer(args, layer_idx))
        else:
            self.mlp = KimiMLP(args)
        self.is_moe = is_moe

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        self.use_attn_res = args.attn_res_block_size is not None
        if self.use_attn_res:
            self.attn_res_block_size = args.attn_res_block_size
            self.self_attention_res = AttnResProj(args.hidden_size)
            self.mlp_res = AttnResProj(args.hidden_size)

    def _ffn(self, x: mx.array) -> mx.array:
        return self.block_sparse_moe(x) if self.is_moe else self.mlp(x)

    def __call__(self, x, mask=None, cache=None, block_residual=None):
        if not self.use_attn_res:
            h = x + self.self_attn(self.input_layernorm(x), mask, cache)
            return h + self._ffn(self.post_attention_layernorm(h)), block_residual

        B, T, H = x.shape
        eps = self.args.rms_norm_eps
        prefix_sum = x

        if block_residual is not None and block_residual.shape[1] > 0:
            x = self.self_attention_res(
                prefix_sum.reshape(-1, H), block_residual, eps
            ).reshape(B, T, H)

        pushed = self.layer_idx % self.attn_res_block_size == 0
        if pushed:
            block_residual = mx.concatenate(
                [block_residual, mx.expand_dims(prefix_sum.reshape(-1, H), 1)], axis=1
            )

        y = self.self_attn(self.input_layernorm(x), mask, cache)
        prefix_sum = y if pushed else prefix_sum + y

        x = self.mlp_res(prefix_sum.reshape(-1, H), block_residual, eps).reshape(B, T, H)
        z = self._ffn(self.post_attention_layernorm(x))
        return prefix_sum + z, block_residual


class KimiK3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [KimiDecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.use_attn_res = args.attn_res_block_size is not None
        if self.use_attn_res:
            self.output_attn_res = AttnResProj(args.hidden_size)

        kda_layers = args.linear_attn_config["kda_layers"]
        self.ssm_idx = kda_layers[0] - 1
        self.attn_idx = next(
            i for i in range(len(self.layers)) if (i + 1) not in kda_layers
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        # inputs_embeds lets the multimodal wrapper splice image features in
        # place of the <|media_pad|> placeholder before the stack runs.
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)

        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        attn_mask = create_attention_mask(h, cache[self.attn_idx], return_array=True)

        B, T, H = h.shape
        block_residual = mx.zeros((B * T, 0, H), dtype=h.dtype) if self.use_attn_res else None

        for layer, layer_cache in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else attn_mask
            h, block_residual = layer(h, mask, layer_cache, block_residual)

        if self.use_attn_res:
            h = self.output_attn_res(
                h.reshape(-1, H), block_residual, self.args.rms_norm_eps
            ).reshape(B, T, H)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = KimiK3Model(args)
        self.lm_head = (
            None
            if args.tie_word_embeddings
            else nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, inputs_embeds=inputs_embeds)
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            ArraysCache(size=4) if layer.is_linear else KVCache()
            for layer in self.layers
        ]

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return remap_checkpoint(weights, self.args)

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # router gate drives top-16-of-896 selection; 8 bits is cheap insurance
            if path.endswith("block_sparse_moe.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
