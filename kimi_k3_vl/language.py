"""Text-tower adapter: wraps kimi_k3.Model for mlx-vlm."""

from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import TextConfig

try:
    from mlx_lm.models import kimi_k3 as _k3
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "kimi_k3 is not registered with mlx-lm. It uses mlx-lm-relative imports "
        "(.base, .cache, .switch_layers) so it must live in mlx_lm/models/. Run "
        "`scripts/install_model.sh` once, or copy kimi_k3.py there."
    ) from e


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        args = _k3.ModelArgs.from_dict(config.raw)
        self.args = args
        self.model = _k3.Model(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        return self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)

    @property
    def layers(self):
        return self.model.model.layers

    def make_cache(self):
        return self.model.make_cache()

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # kimi_k3.Model.sanitize already strips `language_model.` and drops the
        # vision keys; prefix the result to sit under this module.
        return {f"model.{k}": v for k, v in self.model.sanitize(weights).items()}
