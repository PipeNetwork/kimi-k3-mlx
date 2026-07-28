"""Kimi-K3 multimodal glue for mlx-vlm: vision tower -> prompt -> text tower.

The only genuinely interesting part is `merge_image_features`.

K3's processor rewrites each `<|kimi_image_placeholder|>` in the raw text into

    <|media_begin|>image {W}x{H}<|media_content|><|media_pad|><|media_end|>

so the tokenized prompt carries **exactly one** `<|media_pad|>` (163605) per
image, which must then EXPAND into that image's full token count. That differs
from the more common LLaVA-style arrangement (and from mlx-vlm's existing
Kimi-VL glue), where the processor has already emitted one placeholder per image
token and merging is a same-length scatter.

Getting this wrong is quiet rather than loud: a same-length scatter would write
one image token and silently drop the rest, leaving a model that still generates
fluent text while being effectively blind to most of the picture.
"""

import os
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel

try:
    from mlx_vlm.models.base import InputEmbeddingsFeatures
except ImportError:  # allow importing the glue without mlx-vlm present
    from dataclasses import dataclass

    @dataclass
    class InputEmbeddingsFeatures:  # type: ignore[no-redef]
        inputs_embeds: mx.array
        attention_mask: Optional[mx.array] = None
        position_ids: Optional[mx.array] = None


def merge_image_features(
    inputs_embeds: mx.array,
    input_ids: mx.array,
    image_features: List[mx.array],
    placeholder_id: int,
    pad_token_id: Optional[int] = None,
):
    """Expand each placeholder token into its image's feature block.

    inputs_embeds : (B, S, D)   token embeddings
    input_ids     : (B, S)
    image_features: list of (n_i, D), one per image, in prompt order
    returns       : (B, S', D) merged embeddings and a (B, S') attention mask,
                    where S' = S - K + sum(n_i) for K placeholders

    Rows are built by concatenating the spans between placeholders, which is
    O(images) concatenations rather than O(sequence length) scatter indices, and
    is far easier to check than the reference's index arithmetic. With B > 1 the
    rows are LEFT-padded to the longest merged length, matching the reference's
    `left_padding` branch (it is what generation with a KV cache expects).
    """
    B, S = input_ids.shape
    ids = input_ids.tolist()

    n_ph_total = sum(1 for row in ids for t in row if t == placeholder_id)
    if n_ph_total != len(image_features):
        raise ValueError(
            f"{n_ph_total} <|media_pad|> placeholder(s) in the prompt but "
            f"{len(image_features)} image feature block(s). K3 uses exactly one "
            f"placeholder per image."
        )

    rows, masks, taken = [], [], 0
    for b in range(B):
        parts, mask_parts, prev = [], [], 0
        for j, tok in enumerate(ids[b]):
            if tok != placeholder_id:
                continue
            if j > prev:
                parts.append(inputs_embeds[b, prev:j])
                mask_parts.append(mx.ones((j - prev,), dtype=mx.int32))
            feat = image_features[taken].astype(inputs_embeds.dtype)
            taken += 1
            parts.append(feat)
            mask_parts.append(mx.ones((feat.shape[0],), dtype=mx.int32))
            prev = j + 1
        if prev < S:
            parts.append(inputs_embeds[b, prev:])
            mask_parts.append(mx.ones((S - prev,), dtype=mx.int32))
        rows.append(mx.concatenate(parts, axis=0))
        masks.append(mx.concatenate(mask_parts, axis=0))

    if B == 1:
        merged = rows[0][None]
        mask = masks[0][None]
    else:
        longest = max(r.shape[0] for r in rows)
        padded, padded_masks = [], []
        for r, m in zip(rows, masks):
            gap = longest - r.shape[0]
            if gap:
                r = mx.concatenate([mx.zeros((gap, r.shape[-1]), r.dtype), r], axis=0)
                m = mx.concatenate([mx.zeros((gap,), mx.int32), m], axis=0)
            padded.append(r)
            padded_masks.append(m)
        merged = mx.stack(padded)
        mask = mx.stack(padded_masks)

    # zero out embeddings at pad positions, as the reference does
    if pad_token_id is not None:
        merged = mx.where(mask[..., None] == 0, mx.zeros_like(merged), merged)
    return merged, mask


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

    # ------------------------------------------------------------------ api

    @property
    def layers(self):
        return self.language_model.model.model.layers

    def make_cache(self):
        return self.language_model.model.make_cache()

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        embed = self.language_model.model.model.embed_tokens
        inputs_embeds = embed(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        grids = (
            kwargs.pop("grid_thws", None)
            or kwargs.pop("image_grid_thw", None)
            or kwargs.pop("image_grid_hws", None)
        )
        if grids is None:
            raise ValueError("grid_thws is required alongside pixel_values")
        if isinstance(grids, mx.array):
            grids = grids.tolist()
        grids = [tuple(int(v) for v in g) for g in grids]

        feats = kwargs.get("cached_image_features")
        if feats is None:
            feats = self.vision_tower(pixel_values, grids)

        placeholder = kwargs.pop("image_token_id", None) or self.config.media_placeholder_token_id
        merged, mask = merge_image_features(
            inputs_embeds, input_ids, feats, placeholder, self.config.pad_token_id
        )
        return InputEmbeddingsFeatures(inputs_embeds=merged, attention_mask=mask)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        **kwargs,
    ) -> mx.array:
        feats = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        embeds = feats.inputs_embeds
        # After merging, the token axis no longer lines up with input_ids, so the
        # text tower must be driven by the embeddings' own length.
        ids = mx.zeros(embeds.shape[:2], dtype=mx.int32)
        return self.language_model(ids, cache=cache, inputs_embeds=embeds)

    @classmethod
    def from_pretrained(cls, path):
        """Build a multimodal model from a converted repo, quantization included.

        mlx_lm.load_model handles quantization for the text tower, but it only
        ever builds a text model. Doing it here keeps the VL wrapper able to load
        the same artifacts: without this it constructs an unquantized 2.8T-shaped
        model and the weights refuse to load.
        """
        import glob
        import json as _json

        import mlx.nn as _nn

        cfg = _json.load(open(os.path.join(path, "config.json")))
        model = cls(ModelConfig.from_dict(cfg))

        weights = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            weights.update(mx.load(f))
        weights = model.sanitize(weights)

        q = cfg.get("quantization")
        if q:
            # config keys are text-tower-relative ("model.layers.0...."); inside
            # the wrapper the same module is "language_model.model.model.layers.0..."
            prefix = "language_model.model."

            def predicate(p, m):
                key = p[len(prefix):] if p.startswith(prefix) else p
                if key in q:
                    return q[key]
                if not hasattr(m, "to_quantized"):
                    return False
                return f"{p}.scales" in weights

            _nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                         mode=q.get("mode", "affine"), class_predicate=predicate)

        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        return model

    def sanitize(self, weights):
        """Route source keys to the two towers.

        `vision_tower.*` / `mm_projector.*` are consumed by the vision tower;
        `language_model.*` by the text tower. Both delegate to the same sanitize
        functions the standalone models use, so there is one implementation of
        the key mapping rather than a wrapper-specific copy that could drift.
        """
        vis = {k: v for k, v in weights.items()
               if k.startswith(("vision_tower.", "mm_projector."))}
        txt = {k: v for k, v in weights.items() if not k.startswith(("vision_tower.", "mm_projector."))}
        out = {f"vision_tower.{k}": v for k, v in self.vision_tower.sanitize(vis).items()}
        out.update({f"language_model.{k}": v for k, v in self.language_model.sanitize(txt).items()})
        return out
