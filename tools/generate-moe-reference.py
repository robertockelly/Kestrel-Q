#!/usr/bin/env python3
"""Generate Task 2.8 Class-C MoE evidence from pinned Transformers source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_REVISION = "805a9e939fa8c1bff8d8ffdf041c051b71a914aa"
CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
SOURCE_HASHES = {
    "transformers/activations.py":
        "5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665",
    "transformers/integrations/moe.py":
        "2c8894f6d1392980a61ff265f90f7a99fa90678e0eec461d958b4d32fda9628c",
    "transformers/models/qwen4_exp/configuration_qwen4_exp.py":
        "26b47995740e3bc596b44b2011ee6c3d971d46136438b00dd5fad9557bec4254",
    "transformers/models/qwen4_exp/modeling_qwen4_exp.py":
        "91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f",
}
ORACLE_FILES = (
    "moe-contract.json",
    "moe-calibration.json",
    "moe-holdout.json",
    "moe-routing-vectors.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True,
                                 ensure_ascii=False) + "\n").encode("utf-8"))


def f32_hex(value: float) -> str:
    return struct.pack("<f", float(value)).hex()


def tensor_record(torch: Any, tensor: Any) -> dict[str, Any]:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {
        "dtype": "float32",
        "shape": list(value.shape),
        "f32_le_hex": [f32_hex(item) for item in value.flatten().tolist()],
    }


def lcg_values(torch: Any, count: int, seed: int,
               divisor: float = 8192.0) -> Any:
    state = seed & 0xFFFFFFFF
    values: list[float] = []
    for _ in range(count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        signed = ((state >> 16) & 0xFFFF) - 32768
        values.append(signed / divisor)
    return torch.tensor(values, dtype=torch.float32)


def base_config(Config: Any, hidden: int, experts: int, top_k: int,
                intermediate: int) -> Any:
    config = Config(
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=hidden,
        layer_types=["linear_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=hidden,
        linear_value_head_dim=hidden,
        linear_conv_kernel_dim=4,
        num_experts=experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=intermediate,
        shared_expert_intermediate_size=intermediate,
        norm_topk_prob=True,
        hidden_act="silu",
        hc_count=2,
        hc_lowrank=max(1, hidden),
        max_position_embeddings=64,
    )
    config._experts_implementation = "eager"
    return config


def fill_parameters(torch: Any, module: Any) -> dict[str, Any]:
    records: dict[str, Any] = {}
    with torch.no_grad():
        for index, (name, parameter) in enumerate(module.named_parameters()):
            values = lcg_values(torch, parameter.numel(),
                                0x28A00000 + index * 0x101, 16384.0)
            parameter.copy_(values.reshape(parameter.shape))
            records[name] = tensor_record(torch, parameter)
    return records


def make_input(torch: Any, length: int, hidden: int, seed: int,
               mode: str) -> Any:
    if mode == "zero":
        return torch.zeros((1, length, hidden), dtype=torch.float32)
    if mode == "repeated":
        row = lcg_values(torch, hidden, seed, 8192.0)
        return row.reshape(1, 1, hidden).repeat(1, length, 1)
    if mode == "alternating":
        first = lcg_values(torch, hidden, seed, 8192.0)
        second = -first + torch.arange(hidden, dtype=torch.float32) / 128.0
        rows = [first if index % 2 == 0 else second
                for index in range(length)]
        return torch.stack(rows).reshape(1, length, hidden)
    return lcg_values(torch, length * hidden, seed, 8192.0).reshape(
        1, length, hidden)


def selected_checkpoint(torch: Any, module: Any, hidden: Any,
                        selected_ids: Any, selected_weights: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for token in range(hidden.shape[0]):
        for position in range(selected_ids.shape[1]):
            expert_id = int(selected_ids[token, position])
            gate_up = torch.nn.functional.linear(
                hidden[token], module.experts.gate_up_proj[expert_id])
            gate, up = gate_up.chunk(2, dim=-1)
            activated = module.experts.act_fn(gate) * up
            output = torch.nn.functional.linear(
                activated, module.experts.down_proj[expert_id])
            weighted = output * selected_weights[token, position]
            records.append({
                "token": token,
                "top_k_position": position,
                "expert_id": expert_id,
                "gate": tensor_record(torch, gate),
                "up": tensor_record(torch, up),
                "activated": tensor_record(torch, activated),
                "output": tensor_record(torch, output),
                "weighted_output": tensor_record(torch, weighted),
            })
    return records


def run_full_case(torch: Any, module: Any, case_id: str, value: Any) -> dict[str, Any]:
    flat = value.reshape(-1, value.shape[-1])
    with torch.no_grad():
        router_logits, selected_weights, selected_ids = module.gate(flat)
        router_probabilities = torch.nn.functional.softmax(
            router_logits, dtype=torch.float32, dim=-1)
        routed = module.experts(flat, selected_ids, selected_weights)
        shared_gate_projection = module.shared_expert.gate_proj(flat)
        shared_up_projection = module.shared_expert.up_proj(flat)
        shared_activated = module.shared_expert.act_fn(
            shared_gate_projection) * shared_up_projection
        shared_output = module.shared_expert.down_proj(shared_activated)
        shared_gate_logit = module.shared_expert_gate(flat)
        shared_gate = torch.sigmoid(shared_gate_logit)
        gated_shared = shared_gate * shared_output
        final = module(value).reshape_as(flat)
        reconstructed = routed + gated_shared
        if not torch.equal(final, reconstructed):
            raise RuntimeError(f"canonical checkpoint reconstruction differs: {case_id}")
    return {
        "id": case_id,
        "input": tensor_record(torch, value),
        "expected": {
            "router_logits": tensor_record(torch, router_logits),
            "router_probabilities": tensor_record(torch, router_probabilities),
            "selected_expert_ids": selected_ids.detach().cpu().tolist(),
            "selected_weights": tensor_record(torch, selected_weights),
            "selected_experts": selected_checkpoint(
                torch, module, flat, selected_ids, selected_weights),
            "routed_weighted_sum": tensor_record(torch, routed),
            "shared_gate_projection": tensor_record(torch, shared_gate_projection),
            "shared_up_projection": tensor_record(torch, shared_up_projection),
            "shared_activated": tensor_record(torch, shared_activated),
            "shared_output": tensor_record(torch, shared_output),
            "shared_gate_logit": tensor_record(torch, shared_gate_logit),
            "shared_gate": tensor_record(torch, shared_gate),
            "gated_shared_output": tensor_record(torch, gated_shared),
            "final_output": tensor_record(torch, final),
        },
    }


def tier_b_router(torch: Any, modeling: Any, Config: Any,
                  case_id: str, logits: list[float], repeat: int = 1) -> dict[str, Any]:
    if len(logits) != 512:
        raise ValueError("Tier-B logits must contain exactly 512 experts")
    config = base_config(Config, 4, 512, 10, 2)
    router = modeling.Qwen4ExpTextTopKRouter(config).eval()
    with torch.no_grad():
        router.weight.zero_()
        router.weight[:, 0] = torch.tensor(logits, dtype=torch.float32)
    hidden = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(
        repeat, 1)
    with torch.no_grad():
        router_logits, selected_weights, selected_ids = router(hidden)
        probabilities = torch.nn.functional.softmax(
            router_logits, dtype=torch.float32, dim=-1)
    return {
        "id": case_id,
        "input": tensor_record(torch, hidden),
        "router_weight": tensor_record(torch, router.weight),
        "expected": {
            "router_logits": tensor_record(torch, router_logits),
            "router_probabilities": tensor_record(torch, probabilities),
            "selected_expert_ids": selected_ids.detach().cpu().tolist(),
            "selected_weights": tensor_record(torch, selected_weights),
        },
    }


def routing_cases(torch: Any, modeling: Any, Config: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    equal = [0.0] * 512
    cases.append(tier_b_router(torch, modeling, Config,
                               "tier-b-equal-512", equal))

    boundary = [0.0] * 512
    for expert in range(9):
        boundary[expert] = 3.0
    boundary[9] = 2.0
    boundary[10] = 2.0
    cases.append(tier_b_router(torch, modeling, Config,
                               "tier-b-kth-boundary-tie", boundary))

    near = [-2.0] * 512
    for expert in range(1, 10):
        near[expert] = 2.0 + expert / 64.0
    near[0] = 1.0000000
    near[511] = 1.0000002
    cases.append(tier_b_router(torch, modeling, Config,
                               "tier-b-near-tie-endpoints", near))

    one = [-4.0] * 512
    one[511] = 8.0
    cases.append(tier_b_router(torch, modeling, Config,
                               "tier-b-one-dominant-511", one))

    exact_ten = [-3.0] * 512
    dominant = [0, 7, 31, 63, 127, 255, 383, 447, 510, 511]
    for rank, expert in enumerate(dominant):
        exact_ten[expert] = 6.0 - rank / 16.0
    cases.append(tier_b_router(torch, modeling, Config,
                               "tier-b-exactly-ten-dominant", exact_ten))

    more = [-5.0] * 512
    for rank, expert in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 511]):
        more[expert] = 5.0 - rank / 32.0
    cases.append(tier_b_router(torch, modeling, Config,
                               "tier-b-more-than-ten-plausible", more))

    repeated = [-1.0 + index / 512.0 for index in range(512)]
    cases.append(tier_b_router(torch, modeling, Config,
                               "tier-b-repeated-input", repeated, repeat=3))
    return cases


def generate(output_dir: Path, checkout: Path) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(checkout / "src"))
    import torch
    import tokenizers
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    if tokenizers.__version__ != "0.23.1":
        raise RuntimeError(
            "Task 2.8 oracle requires the governed tokenizers 0.23.1 environment")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = base_config(Qwen4ExpTextConfig, 6, 4, 2, 4)
    module = modeling.Qwen4ExpTextSparseMoeBlock(config).eval()
    weights = fill_parameters(torch, module)

    contract = {
        "schema": "kq-moe-contract-v1",
        "authority": {
            "model_revision": MODEL_REVISION,
            "transformers_revision": TRANSFORMERS_REVISION,
            "model_config_sha256": CONFIG_SHA256,
            "source_sha256": SOURCE_HASHES,
            "license": "Apache-2.0",
        },
        "oracle_environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tokenizers": tokenizers.__version__,
            "device": "cpu",
            "deterministic_algorithms": True,
            "thread_count": 1,
        },
        "tier_a_config": {
            "hidden_size": 6,
            "expert_count": 4,
            "top_k": 2,
            "routed_intermediate_size": 4,
            "shared_intermediate_size": 4,
            "activation": "silu",
            "normalize_selected_weights": True,
            "activation_dtype": "float32",
        },
        "tier_b_config": {
            "hidden_size": 4,
            "expert_count": 512,
            "top_k": 10,
            "routed_intermediate_size": 2,
            "shared_intermediate_size": 2,
            "activation_dtype": "float32",
        },
        "target_config": {
            "layer_count": 48,
            "hidden_size": 2560,
            "expert_count": 512,
            "top_k": 10,
            "routed_intermediate_size": 640,
            "shared_intermediate_size": 640,
            "activation_dtype": "bfloat16",
        },
        "routing_contract": {
            "softmax_dtype": "float32",
            "tie_order_observed": (
                "pinned CPU partial selection: first-k seed; strictly greater "
                "later values replace the lowest selected value; lowest ID is "
                "evicted among a tied lowest set; equal later values do not "
                "replace; final order is descending value then ascending ID"),
            "selected_weight_normalization": "divide selected top-k probabilities by their selected sum",
            "routed_accumulation_order": "ascending expert ID in eager implementation",
        },
        "weights": weights,
    }
    write_json(output_dir / "moe-contract.json", contract)

    calibration_specs = [
        ("cal-length-1-random", 1, 0xCA280001, "random"),
        ("cal-length-3-repeated", 3, 0xCA280002, "repeated"),
        ("cal-length-4-alternating", 4, 0xCA280003, "alternating"),
        ("cal-length-2-zero", 2, 0xCA280004, "zero"),
        ("cal-length-5-random", 5, 0xCA280005, "random"),
    ]
    calibration = {
        "schema": "kq-moe-calibration-v1",
        "comparison_classes": {
            "routing": "EXACT_DISCRETE",
            "floating": "CALIBRATED_FLOAT",
        },
        "cases": [run_full_case(
            torch, module, case_id,
            make_input(torch, length, 6, seed, mode))
            for case_id, length, seed, mode in calibration_specs],
    }
    write_json(output_dir / "moe-calibration.json", calibration)

    holdout_specs = [
        ("holdout-length-2-random", 2, 0xB0280001, "random"),
        ("holdout-length-5-alternating", 5, 0xB0280002, "alternating"),
        ("holdout-length-4-repeated", 4, 0xB0280003, "repeated"),
        ("holdout-length-3-random", 3, 0xB0280004, "random"),
    ]
    holdout = {
        "schema": "kq-moe-holdout-v1",
        "comparison_classes": {
            "routing": "EXACT_DISCRETE",
            "floating": "CALIBRATED_FLOAT",
        },
        "cases": [run_full_case(
            torch, module, case_id,
            make_input(torch, length, 6, seed, mode))
            for case_id, length, seed, mode in holdout_specs],
    }
    write_json(output_dir / "moe-holdout.json", holdout)

    routing = {
        "schema": "kq-moe-routing-vectors-v1",
        "comparison_class": "EXACT_DISCRETE",
        "config": contract["tier_b_config"],
        "cases": routing_cases(torch, modeling, Qwen4ExpTextConfig),
    }
    write_json(output_dir / "moe-routing-vectors.json", routing)

    files = {name: {"sha256": sha256(output_dir / name),
                    "bytes": (output_dir / name).stat().st_size}
             for name in ORACLE_FILES}
    manifest = {
        "schema": "kq-moe-evidence-manifest-v1",
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "generation_tool": "tools/generate-moe-reference.py",
        "generation_command": (
            "python tools/generate-moe-reference.py --checkout "
            ".research-cache/task-1.4/transformers --config "
            ".research-cache/model-baseline/" + MODEL_REVISION +
            "/config.json --output-dir research/operators/"
            "Qwen3.8-Flash-Next/" + MODEL_REVISION),
        "offline": True,
        "full_model_weights_downloaded": False,
        "expected_values_from_kestrel_q": False,
        "files": files,
    }
    write_json(output_dir / "moe-manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    config = args.config.resolve()
    output = args.output_dir.resolve()
    if not (checkout / ".git").exists():
        raise SystemExit("pinned Transformers checkout is unavailable")
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    if revision != TRANSFORMERS_REVISION:
        raise SystemExit(f"unexpected Transformers revision: {revision}")
    if not config.is_file() or sha256(config) != CONFIG_SHA256:
        raise SystemExit("pinned model config hash mismatch")
    for relative, expected in SOURCE_HASHES.items():
        path = checkout / "src" / relative
        if sha256(path) != expected:
            raise SystemExit(f"source hash mismatch: {relative}")
    if args.verify:
        with tempfile.TemporaryDirectory(prefix="kq-moe-verify-") as directory:
            candidate = Path(directory)
            generate(candidate, checkout)
            mismatches = [name for name in ORACLE_FILES
                          if not (output / name).exists() or
                          (output / name).read_bytes() !=
                          (candidate / name).read_bytes()]
            if mismatches:
                raise SystemExit("non-deterministic MoE evidence: " +
                                 ", ".join(mismatches))
        print("MoE evidence deterministic regeneration: PASS")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    for name in (*ORACLE_FILES, "moe-native-validation.json", "moe-manifest.json"):
        path = output / name
        if path.exists():
            path.unlink()
    generate(output, checkout)
    print(f"MoE Class-C evidence written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
