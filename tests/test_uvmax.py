import unittest
import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlx_lm.models import kimi_k3_uvmax
from mlx_lm.models import kimi_k3
from scripts.distributed_generate import (
    benchmark_cases,
    load_manifest,
    parse_active_rdma_ports,
)
from scripts.prepare_uvmax_stage import pipeline_bounds, select_stage_files, sha256_file
from scripts.summarize_benchmark import validate_pair


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
    def test_uvmax_gate_matches_upstream_corrected_formula(self):
        """The live TP loader must retain PipeNetwork's fla parity correction."""
        args = kimi_k3.ModelArgs.from_dict(tiny_config(num_layers=4))
        attention = kimi_k3.KimiDeltaAttention(args, 0)
        attention.A_log = mx.linspace(-0.4, 0.8, attention.head_dim)
        attention.dt_bias = mx.linspace(-1.0, 1.0, attention.projection_dim)
        activations = mx.random.normal(
            (2, 3, attention.num_heads, attention.head_dim)
        )
        got = attention._compute_g(activations)
        expected = kimi_k3_uvmax._compute_g_safe(
            attention.A_log[: attention.num_heads].reshape(attention.num_heads, 1),
            activations,
            attention.dt_bias.reshape(attention.num_heads, attention.head_dim),
            attention.gate_lower_bound,
        )
        self.assertLess(float(mx.max(mx.abs(got - expected))), 1e-7)

    def test_runner_requires_sha256_verified_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            manifest = {
                "complete": True,
                "pipeline": {"rank": 0, "world_size": 2},
                "weights": {"files": [], "sha256_verified": False},
            }
            (stage / "stage-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                load_manifest(stage, 0, 2)
            manifest["weights"]["sha256_verified"] = True
            (stage / "stage-manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(load_manifest(stage, 0, 2), manifest)

    def test_stage_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weight.safetensors"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_paired_benchmark_validation(self):
        base = {
            "run_id": "test-factual-r1",
            "suite_run_id": "test",
            "case_id": "factual",
            "repetition": 1,
            "world_size": 2,
            "backend": "jaccl",
            "transport": "thunderbolt-rdma",
            "rdma": {"active_ports": ["rdma_en3"]},
            "code": {"commit": "abc", "dirty": False},
            "versions": {"mlx": "0.32.0", "mlx_lm": "0.31.3"},
            "checkpoint": {"revision": "immutable"},
            "jaccl_hostfile_sha256": "1234",
            "mlx_metal_fast_synch": "1",
            "hardware": {
                "device_name": "Apple M3 Ultra",
                "memory_size": 549755813888,
            },
            "prompt": "Hello",
            "seed": 0,
            "sampling": {"temperature": 0.0},
            "max_tokens": 8,
            "prompt_tokens": 2,
            "generation_tokens": 8,
            "text": "world",
            "prompt_tps": 5.0,
            "generation_tps": 3.0,
            "peak_memory_gb": 400.0,
        }
        rank0 = {**base, "rank": 0, "host": "beast1"}
        rank1 = {
            **base,
            "rank": 1,
            "host": "beast2",
            "prompt_tps": 4.5,
            "generation_tps": 2.9,
            "peak_memory_gb": 401.0,
        }
        result = validate_pair(rank0, rank1)
        self.assertEqual(result["generation_tps"], 2.9)
        self.assertEqual(result["prompt_tps"], 4.5)
        self.assertEqual(result["peak_memory_gb"], 401.0)

        rank1["text"] = "different"
        with self.assertRaisesRegex(ValueError, "text"):
            validate_pair(rank0, rank1)

    def test_active_rdma_port_parser(self):
        output = """
        hca_id:\trdma_en2
            state:\t\t\tPORT_DOWN (1)
        hca_id:\trdma_en3
            state:\t\t\tPORT_ACTIVE (4)
        hca_id:\trdma_en4
            state:\t\t\tPORT_ACTIVE (4)
        """
        self.assertEqual(parse_active_rdma_ports(output), ["rdma_en3", "rdma_en4"])

    def test_benchmark_suite_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "suite.json"
            path.write_text(
                json.dumps(
                    {
                        "prompts": [
                            {"id": "factual", "prompt": "Hello"},
                            {"id": "code-1", "prompt": "Write code", "max_tokens": 64},
                        ]
                    }
                )
            )
            cases = benchmark_cases(
                SimpleNamespace(
                    suite=path,
                    prompt="unused",
                    max_tokens=256,
                )
            )
            self.assertEqual(
                cases,
                [
                    {"id": "factual", "prompt": "Hello", "max_tokens": 256},
                    {"id": "code-1", "prompt": "Write code", "max_tokens": 64},
                ],
            )

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
