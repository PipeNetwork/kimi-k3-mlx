"""Config plumbing for the Kimi-K3 mlx-vlm wrapper."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BaseCfg:
    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in params.items() if k in known})


@dataclass
class TextConfig(BaseCfg):
    """Thin holder: the real parsing lives in kimi_k3.ModelArgs, which already
    understands K3's nested `text_config`. Kept so mlx-vlm's config machinery
    has something with the shape it expects."""

    model_type: str = "kimi_linear"
    hidden_size: int = 7168
    num_hidden_layers: int = 93
    vocab_size: int = 163840
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        return cls(
            model_type=params.get("model_type", "kimi_linear"),
            hidden_size=params.get("hidden_size", 7168),
            num_hidden_layers=params.get("num_hidden_layers", 93),
            vocab_size=params.get("vocab_size", 163840),
            raw=dict(params),
        )


@dataclass
class VisionConfig(BaseCfg):
    model_type: str = "moonvit3d"
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        return cls(model_type=params.get("model_type", "moonvit3d"), raw=dict(params))


@dataclass
class ModelConfig(BaseCfg):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "kimi_k3"
    # <|media_pad|>. Exactly ONE of these per image in the prompt; it expands to
    # that image's full token count. See kimi_k3_vl.merge_image_features.
    media_placeholder_token_id: int = 163605
    image_token_index: Optional[int] = None
    ignore_index: int = -100
    pad_token_id: int = 163839
    bos_token_id: int = 163584
    eos_token_id: Optional[Any] = 163586
    vocab_size: int = 163840
    quantization: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        # TextConfig.raw carries the *whole* config: kimi_k3.ModelArgs.from_dict
        # already knows how to unwrap K3's nested `text_config`, so reusing that
        # path avoids a second, divergent copy of the same unwrapping rules.
        text = TextConfig.from_dict(params.get("text_config", {}))
        text.raw = dict(params)
        return cls(
            text_config=text,
            vision_config=VisionConfig.from_dict(params.get("vision_config", {})),
            model_type=params.get("model_type", "kimi_k3"),
            media_placeholder_token_id=params.get("media_placeholder_token_id", 163605),
            image_token_index=params.get("image_token_index"),
            ignore_index=params.get("ignore_index", -100),
            pad_token_id=params.get("pad_token_id", 163839),
            bos_token_id=params.get("bos_token_id", 163584),
            eos_token_id=params.get("eos_token_id", 163586),
            vocab_size=params.get("text_config", {}).get("vocab_size", 163840),
            quantization=params.get("quantization"),
            raw=dict(params),
        )
