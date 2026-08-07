#!/usr/bin/env python3
"""Build a tiny deterministic Kimi-K3 checkpoint for distributed CI/smoke tests."""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx_lm.models import kimi_k3_uvmax
from mlx_lm.utils import save_model


def config() -> dict:
    return {
        "model_type": "kimi_k3_uvmax",
        "eos_token_id": 127,
        "text_config": {
            "model_type": "kimi_linear",
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 5,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 96,
            "rms_norm_eps": 1e-5,
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "linear_attn_config": {
                "kda_layers": [1, 2, 3, 4],
                "full_attn_layers": [5],
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("work/tiny-uvmax-pipeline-2")
    )
    args = parser.parse_args()
    output = args.output.resolve()
    source = output / "source"

    mx.random.seed(7)
    cfg = config()
    model = kimi_k3_uvmax.Model(kimi_k3_uvmax.ModelArgs.from_dict(cfg))
    mx.eval(model.parameters())
    save_model(source, model)
    (source / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    for rank, bounds in ((0, (3, 5)), (1, (0, 3))):
        stage = output / f"rank{rank}"
        stage.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, stage / path.name)
        manifest = {
            "format": 1,
            "source": {"repo": "local/tiny-uvmax", "revision": "deterministic-7"},
            "pipeline": {
                "rank": rank,
                "world_size": 2,
                "layer_start": bounds[0],
                "layer_end": bounds[1],
            },
            "complete": True,
        }
        (stage / "stage-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
