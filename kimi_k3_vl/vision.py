"""Vision-tower adapter: wraps kimi_k3_vision.VisionModel for mlx-vlm."""

import importlib.util
import os
import sys
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig as WrapperVisionConfig


def _load_vision_module():
    """kimi_k3_vision.py ships alongside the weights; import it from wherever
    this package lives so a downloaded repo works without installation."""
    if "kimi_k3_vision" in sys.modules:
        return sys.modules["kimi_k3_vision"]
    try:
        import kimi_k3_vision  # noqa: F401

        return sys.modules["kimi_k3_vision"]
    except ImportError:
        pass
    for d in (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
              os.path.dirname(os.path.abspath(__file__))):
        p = os.path.join(d, "kimi_k3_vision.py")
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("kimi_k3_vision", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["kimi_k3_vision"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("kimi_k3_vision.py not found next to kimi_k3_vl/")


_v = _load_vision_module()


class VisionModel(nn.Module):
    def __init__(self, config: WrapperVisionConfig):
        super().__init__()
        self.cfg = _v.VisionConfig.from_dict(config.raw)
        self.tower = _v.VisionModel(self.cfg)

    def __call__(self, pixel_values: mx.array, grid_thws) -> List[mx.array]:
        """pixel_values: (L, 3, patch, patch) pre-extracted patches.
        Returns one (n_i, text_hidden) array per image."""
        return self.tower(pixel_values, grid_thws)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return {f"tower.{k}": v for k, v in self.tower.sanitize(weights).items()}
