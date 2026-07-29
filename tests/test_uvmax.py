import unittest
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlx_lm.models import kimi_k3_uvmax
from scripts.prepare_uvmax_stage import pipeline_bounds, select_stage_files


def tiny_config(num_layers=5):
    kda_layers = list(range(1, num_layers))
    return {
        "model_type": "kimi_k3_uvmax",
        "text_config": {
            "model_type": "kimi_linear",
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 96,
            "rms_norm_eps": 1e-5,
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "linear_attn_config": {
                "kda_layers": kda_layers,
                "full_attn_layers": [num_layers],
                "num_heads": 2,
                "head_dim": 32,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": True,
            },
            "num_experts": 8,
            "moe_intermediate_size": 32,
            "q_lora_rank": 24,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "first_k_dense_replace": 1,
            "routed_expert_hidden_size": 32,
            "latent_moe_use_norm": True,
            "attn_res_block_size": 2,
        },
    }


class FakeGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class TestUvmaxStage(unittest.TestCase):
    def test_file_selection_keeps_text_common_and_local_layers(self):
        weight_map = {
            "language_model.model.embed_tokens.weight": "begin.safetensors",
            "language_model.model.layers.0.mlp.weight": "l0.safetensors",
            "language_model.model.layers.1.mlp.weight": "boundary.safetensors",
            "language_model.model.layers.2.mlp.weight": "boundary.safetensors",
            "language_model.model.layers.3.mlp.weight": "l3.safetensors",
            "language_model.model.norm.weight": "end.safetensors",
            "language_model.lm_head.weight": "end.safetensors",
            "vision_tower.weight": "vision.safetensors",
        }
        self.assertEqual(
            select_stage_files(weight_map, 0, 2),
            {"begin.safetensors", "l0.safetensors", "boundary.safetensors", "end.safetensors"},
        )
        self.assertEqual(
            select_stage_files(weight_map, 2, 4),
            {"begin.safetensors", "boundary.safetensors", "l3.safetensors", "end.safetensors"},
        )

    def test_odd_pipeline_has_no_gap(self):
        self.assertEqual(pipeline_bounds(93, 2, 1), (0, 47))
        self.assertEqual(pipeline_bounds(93, 2, 0), (47, 93))

        models = []
        for rank in (0, 1):
            args = kimi_k3_uvmax.ModelArgs.from_dict(tiny_config())
            model = kimi_k3_uvmax.Model(args)
            model.model.pipeline(FakeGroup(rank, 2))
            models.append(model)
        self.assertEqual((models[1].model.start_idx, models[1].model.end_idx), (0, 3))
        self.assertEqual((models[0].model.start_idx, models[0].model.end_idx), (3, 5))
        self.assertEqual(len(models[1].layers), 3)
        self.assertEqual(len(models[0].layers), 2)
        self.assertEqual(len(models[1].make_cache()), 3)
        self.assertEqual(len(models[0].make_cache()), 2)

    def test_stable_release_forward_compatibility(self):
        args = kimi_k3_uvmax.ModelArgs.from_dict(tiny_config(num_layers=4))
        model = kimi_k3_uvmax.Model(args)
        logits = model(mx.array([[1, 2]], dtype=mx.int32))
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 2, 128))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())


if __name__ == "__main__":
    unittest.main()
