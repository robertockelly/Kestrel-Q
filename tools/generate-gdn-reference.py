#!/usr/bin/env python3
"""Generate Task 2.6 GDN expectations from the pinned Transformers module.

This is a research-only Class-C oracle.  It imports the pinned offline
Transformers checkout, runs the real Qwen4-Exp GDN module with reduced
dimensions, and never imports or invokes Kestrel-Q.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import sys
from pathlib import Path
from typing import Any


MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_REVISION = "805a9e939fa8c1bff8d8ffdf041c051b71a914aa"
CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
SOURCE_HASHES = {
    "transformers/cache_utils.py": "4b284431cb3a881b6e6f8b8c6430df6f2efdcb3366a2484c7984ae88c612c61a",
    "transformers/masking_utils.py": "c159cd91c2a7fcafce04a8b6cbca55c320ce904b8ebf634383c97da5d9313ce3",
    "transformers/models/qwen4_exp/configuration_qwen4_exp.py":
        "26b47995740e3bc596b44b2011ee6c3d971d46136438b00dd5fad9557bec4254",
    "transformers/models/qwen4_exp/modeling_qwen4_exp.py":
        "91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f",
}
EVIDENCE_FILES = (
    "gdn-contract.json",
    "gdn-calibration.json",
    "gdn-holdout.json",
    "gdn-state-vectors.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_bytes((data + "\n").encode("utf-8"))


def bits_of_f32(value: float) -> str:
    return struct.pack("<f", float(value)).hex()


def tensor_record(torch: Any, tensor: Any) -> dict[str, Any]:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {
        "dtype": "float32",
        "shape": list(value.shape),
        "f32_le_hex": [bits_of_f32(item) for item in value.flatten().tolist()],
    }


def lcg_values(torch: Any, count: int, seed: int, divisor: float = 4096.0) -> Any:
    state = seed & 0xFFFFFFFF
    values: list[float] = []
    for _ in range(count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        signed = ((state >> 16) & 0xFFFF) - 32768
        values.append(signed / divisor)
    return torch.tensor(values, dtype=torch.float32)


def fill_parameters(torch: Any, module: Any) -> dict[str, Any]:
    records: dict[str, Any] = {}
    with torch.no_grad():
        for index, (name, parameter) in enumerate(module.named_parameters()):
            if name == "A_log":
                values = torch.linspace(-1.75, 0.125, parameter.numel(), dtype=torch.float32)
            elif name == "dt_bias":
                values = torch.linspace(-0.5, 0.375, parameter.numel(), dtype=torch.float32)
            elif name == "norm.weight":
                values = torch.linspace(0.75, 1.125, parameter.numel(), dtype=torch.float32)
            elif name == "conv1d.weight":
                values = lcg_values(torch, parameter.numel(), 0xC0110000 + index, 32768.0)
            else:
                values = lcg_values(torch, parameter.numel(), 0x6D2B0000 + index, 65536.0)
            parameter.copy_(values.reshape(parameter.shape))
            records[name] = tensor_record(torch, parameter)
    return records


def input_tensor(torch: Any, length: int, seed: int, mode: str) -> Any:
    if mode == "repeated":
        row = lcg_values(torch, 8, seed, 8192.0)
        return row.reshape(1, 1, 8).repeat(1, length, 1)
    if mode == "alternating":
        first = lcg_values(torch, 8, seed, 8192.0)
        second = -first + torch.tensor([i / 64.0 for i in range(8)], dtype=torch.float32)
        rows = [first if index % 2 == 0 else second for index in range(length)]
        return torch.stack(rows).reshape(1, length, 8)
    return lcg_values(torch, length * 8, seed, 8192.0).reshape(1, length, 8)


def initial_state(torch: Any, seed: int) -> tuple[Any, Any]:
    conv = lcg_values(torch, 32 * 4, seed, 32768.0).reshape(1, 32, 4)
    recurrent = lcg_values(torch, 4 * 4 * 4, seed ^ 0xA5A5A5A5, 65536.0).reshape(1, 4, 4, 4)
    return conv, recurrent


def install_capture(torch: Any, modeling: Any, module: Any, captured: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    handles = []
    originals = {
        "causal_conv1d_fn": modeling.causal_conv1d_fn,
        "causal_conv1d_update": modeling.causal_conv1d_update,
        "torch_chunk_gated_delta_rule": modeling.torch_chunk_gated_delta_rule,
        "torch_recurrent_gated_delta_rule": modeling.torch_recurrent_gated_delta_rule,
    }

    def projection_hook(name: str):
        def hook(_module: Any, _args: Any, output: Any) -> None:
            captured[name] = output.detach().clone()
        return hook

    handles.append(module.in_proj_qkv.register_forward_hook(projection_hook("projected_qkv")))
    handles.append(module.in_proj_z.register_forward_hook(projection_hook("projected_gate")))
    handles.append(module.in_proj_b.register_forward_hook(projection_hook("projected_beta")))
    handles.append(module.in_proj_a.register_forward_hook(projection_hook("projected_alpha")))

    def norm_hook(_module: Any, args: Any, output: Any) -> None:
        captured["recurrent_output_before_norm"] = args[0].detach().clone()
        captured["gate_before_activation"] = args[1].detach().clone()
        captured["gated_norm_output"] = output.detach().clone()

    handles.append(module.norm.register_forward_hook(norm_hook))
    handles.append(module.out_proj.register_forward_hook(projection_hook("operator_output")))

    def conv_prefill(hidden_states: Any, weight: Any, bias: Any = None, activation: Any = None, **kwargs: Any) -> Any:
        captured["conv_input"] = hidden_states.detach().clone()
        result = originals["causal_conv1d_fn"](
            hidden_states, weight, bias, activation=activation, **kwargs
        )
        captured["conv_output"] = result.detach().clone()
        captured["conv_path"] = "prefill"
        return result

    def conv_decode(hidden_states: Any, conv_state: Any, weight: Any, bias: Any = None, activation: Any = None) -> Any:
        captured["conv_input"] = hidden_states.detach().clone()
        result = originals["causal_conv1d_update"](
            hidden_states, conv_state, weight, bias, activation=activation
        )
        captured["conv_output"] = result.detach().clone()
        captured["conv_path"] = "decode"
        return result

    def recurrence_wrapper(kind: str, original: Any):
        def wrapper(query: Any, key: Any, value: Any, g: Any, beta: Any, **kwargs: Any) -> Any:
            captured["query_before_norm"] = query.detach().clone()
            captured["key_before_norm"] = key.detach().clone()
            captured["value"] = value.detach().clone()
            captured["log_decay"] = g.detach().clone()
            captured["beta"] = beta.detach().clone()
            q_norm = modeling.l2norm(query.to(torch.float32), dim=-1, eps=1e-6)
            k_norm = modeling.l2norm(key.to(torch.float32), dim=-1, eps=1e-6)
            captured["normalized_scaled_query"] = q_norm / math.sqrt(query.shape[-1])
            captured["normalized_key"] = k_norm
            q_scan = captured["normalized_scaled_query"].transpose(1, 2).contiguous()
            k_scan = captured["normalized_key"].transpose(1, 2).contiguous()
            v_scan = value.to(torch.float32).transpose(1, 2).contiguous()
            b_scan = beta.to(torch.float32).transpose(1, 2).contiguous()
            g_scan = g.to(torch.float32).transpose(1, 2).contiguous()
            scan_state = kwargs.get("initial_state")
            if scan_state is None:
                scan_state = torch.zeros(
                    query.shape[0], value.shape[2], query.shape[-1], value.shape[-1],
                    dtype=torch.float32, device=query.device
                )
            else:
                scan_state = scan_state.to(torch.float32).clone()
            scan_read = []
            scan_delta = []
            scan_output = []
            scan_states = []
            for token_index in range(query.shape[1]):
                scan_state = scan_state * g_scan[:, :, token_index].exp()[..., None, None]
                read = (scan_state * k_scan[:, :, token_index].unsqueeze(-1)).sum(dim=-2)
                delta = (v_scan[:, :, token_index] - read) * b_scan[:, :, token_index].unsqueeze(-1)
                scan_state = scan_state + k_scan[:, :, token_index].unsqueeze(-1) * delta.unsqueeze(-2)
                core = (scan_state * q_scan[:, :, token_index].unsqueeze(-1)).sum(dim=-2)
                scan_read.append(read)
                scan_delta.append(delta)
                scan_output.append(core)
                scan_states.append(scan_state.clone())
            captured["sequential_read"] = torch.stack(scan_read, dim=1)
            captured["sequential_delta"] = torch.stack(scan_delta, dim=1)
            captured["sequential_core_output"] = torch.stack(scan_output, dim=1)
            captured["sequential_state_by_token"] = torch.stack(scan_states, dim=0)
            result = original(query, key, value, g=g, beta=beta, **kwargs)
            captured["recurrent_core_output"] = result[0].detach().clone()
            if result[1] is not None:
                captured["recurrent_final_state"] = result[1].detach().clone()
            captured["recurrence_path"] = kind
            return result
        return wrapper

    modeling.causal_conv1d_fn = conv_prefill
    modeling.causal_conv1d_update = conv_decode
    modeling.torch_chunk_gated_delta_rule = recurrence_wrapper(
        "chunk", originals["torch_chunk_gated_delta_rule"]
    )
    modeling.torch_recurrent_gated_delta_rule = recurrence_wrapper(
        "recurrent", originals["torch_recurrent_gated_delta_rule"]
    )
    return handles, originals


def remove_capture(modeling: Any, handles: list[Any], originals: dict[str, Any]) -> None:
    for handle in handles:
        handle.remove()
    for name, value in originals.items():
        setattr(modeling, name, value)


def run_case(torch: Any, DynamicCache: Any, modeling: Any, config: Any, module: Any,
             case_id: str, hidden: Any, mask: Any = None,
             initial: tuple[Any, Any] | None = None) -> tuple[dict[str, Any], Any]:
    cache = DynamicCache(config=config)
    if initial is not None:
        conv, recurrent = initial
        layer_cache = cache.layers[0]
        layer_cache.lazy_initialization(
            conv_states=conv, recurrent_states=recurrent, state_idx=0,
            conv_kernel_size=config.linear_conv_kernel_dim
        )
        layer_cache.conv_states[0].copy_(conv)
        layer_cache.recurrent_states[0].copy_(recurrent)
        layer_cache.has_previous_state[0] = True

    captured: dict[str, Any] = {}
    handles, originals = install_capture(torch, modeling, module, captured)
    try:
        with torch.no_grad():
            output = module(hidden, cache_params=cache, attention_mask=mask)
    finally:
        remove_capture(modeling, handles, originals)

    if mask is None:
        masked = hidden
        mask_values = None
    else:
        masked = hidden * mask[:, :, None]
        mask_values = mask.to(torch.int32).flatten().tolist()

    projected_alpha = captured["projected_alpha"].to(torch.float32)
    projected_beta = captured["projected_beta"].to(torch.float32)
    decay = -module.A_log.float().exp() * torch.nn.functional.softplus(
        projected_alpha + module.dt_bias.float()
    )
    beta = projected_beta.sigmoid()

    checkpoint_names = (
        "projected_qkv", "projected_gate", "projected_beta", "projected_alpha",
        "conv_input", "conv_output", "query_before_norm", "key_before_norm", "value",
        "normalized_scaled_query", "normalized_key", "log_decay", "beta",
        "recurrent_core_output", "recurrent_output_before_norm",
        "gate_before_activation", "gated_norm_output", "operator_output",
        "sequential_read", "sequential_delta", "sequential_core_output",
        "sequential_state_by_token",
    )
    checkpoints = {
        name: tensor_record(torch, captured[name])
        for name in checkpoint_names if name in captured
    }
    conv_scan = (initial[0].detach().clone() if initial is not None
                 else torch.zeros(1, 32, 4, dtype=torch.float32))
    conv_states = []
    projected_qkv = captured["projected_qkv"].to(torch.float32)
    for token_index in range(hidden.shape[1]):
        conv_scan = torch.cat(
            [conv_scan[:, :, 1:], projected_qkv[:, token_index, :, None]],
            dim=-1,
        )
        conv_states.append(conv_scan.clone())
    checkpoints["sequential_conv_state_by_token"] = tensor_record(
        torch, torch.stack(conv_states, dim=0)
    )
    checkpoints["derived_log_decay"] = tensor_record(torch, decay)
    checkpoints["derived_beta"] = tensor_record(torch, beta)

    layer_cache = cache.layers[0]
    record = {
        "id": case_id,
        "input": tensor_record(torch, hidden),
        "padding_mask": mask_values,
        "initial_state": {
            "initialized": initial is not None,
            "conv": tensor_record(torch, initial[0] if initial is not None else torch.zeros(1, 32, 4)),
            "recurrent": tensor_record(torch, initial[1] if initial is not None else torch.zeros(1, 4, 4, 4)),
        },
        "expected": {
            "masked_input": tensor_record(torch, masked),
            "output": tensor_record(torch, output),
            "conv_state": tensor_record(torch, layer_cache.conv_states[0]),
            "recurrent_state": tensor_record(torch, layer_cache.recurrent_states[0]),
            "checkpoints": checkpoints,
            "conv_path": captured["conv_path"],
            "recurrence_path": captured["recurrence_path"],
        },
        "comparison_class": "CALIBRATED_FLOAT",
    }
    return record, cache


def run_continuation(torch: Any, modeling: Any, module: Any, cache: Any,
                     hidden: Any, case_id: str) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    handles, originals = install_capture(torch, modeling, module, captured)
    try:
        before_conv = cache.layers[0].conv_states[0].detach().clone()
        before_recurrent = cache.layers[0].recurrent_states[0].detach().clone()
        with torch.no_grad():
            output = module(hidden, cache_params=cache)
    finally:
        remove_capture(modeling, handles, originals)
    checkpoint_names = (
        "projected_qkv", "projected_gate", "projected_beta", "projected_alpha",
        "conv_input", "conv_output", "query_before_norm", "key_before_norm", "value",
        "normalized_scaled_query", "normalized_key", "log_decay", "beta",
        "recurrent_core_output", "recurrent_output_before_norm",
        "gate_before_activation", "gated_norm_output", "operator_output",
        "sequential_read", "sequential_delta", "sequential_core_output",
        "sequential_state_by_token",
    )
    checkpoints = {
        name: tensor_record(torch, captured[name])
        for name in checkpoint_names if name in captured
    }
    conv_scan = before_conv.detach().clone()
    conv_states = []
    projected_qkv = captured["projected_qkv"].to(torch.float32)
    for token_index in range(hidden.shape[1]):
        conv_scan = torch.cat(
            [conv_scan[:, :, 1:], projected_qkv[:, token_index, :, None]],
            dim=-1,
        )
        conv_states.append(conv_scan.clone())
    checkpoints["sequential_conv_state_by_token"] = tensor_record(
        torch, torch.stack(conv_states, dim=0)
    )
    return {
        "id": case_id,
        "input": tensor_record(torch, hidden),
        "before_conv_state": tensor_record(torch, before_conv),
        "before_recurrent_state": tensor_record(torch, before_recurrent),
        "expected_output": tensor_record(torch, output),
        "expected_conv_state": tensor_record(torch, cache.layers[0].conv_states[0]),
        "expected_recurrent_state": tensor_record(torch, cache.layers[0].recurrent_states[0]),
        "checkpoints": checkpoints,
        "conv_path": captured["conv_path"],
        "recurrence_path": captured["recurrence_path"],
        "comparison_class": "CALIBRATED_FLOAT",
    }


def verify_sources(source_root: Path, config_path: Path) -> None:
    if sha256(config_path) != CONFIG_SHA256:
        raise SystemExit("pinned config.json hash mismatch")
    for relative, expected in SOURCE_HASHES.items():
        actual = sha256(source_root / relative)
        if actual != expected:
            raise SystemExit(f"pinned source hash mismatch: {relative}: {actual}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers-source", type=Path, required=True,
                        help="pinned Transformers src directory")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.transformers_source.resolve()
    config_path = args.config.resolve()
    verify_sources(source_root, config_path)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import numpy
    import torch
    import transformers
    from transformers import DynamicCache
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    if getattr(transformers, "__version__", "") != "5.16.0.dev0":
        raise SystemExit(f"unexpected Transformers version {transformers.__version__}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)

    config = Qwen4ExpTextConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"],
        output_gate_type="sigmoid",
        hidden_act="silu",
        rms_norm_eps=1e-6,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=2,
        vocab_size=32,
    )
    module = modeling.Qwen4ExpTextGatedDeltaNet(config, 0).to(dtype=torch.float32).eval()
    weights = fill_parameters(torch, module)

    contract = {
        "schema": "KQ-GDN-CONTRACT-v1",
        "class": "C",
        "status": "PINNED_ORACLE_GENERATED",
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "oracle": {
            "module": "Qwen4ExpTextGatedDeltaNet",
            "license": "Apache-2.0",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "numpy_version": numpy.__version__,
            "python_version": platform.python_version(),
            "source_sha256": SOURCE_HASHES,
            "config_sha256": CONFIG_SHA256,
        },
        "reduced_config": {
            "activation_dtype": "float32",
            "batch_size": 1,
            "hidden_size": 8,
            "key_heads": 2,
            "key_head_dim": 4,
            "value_heads": 4,
            "value_head_dim": 4,
            "key_value_head_repeat": 2,
            "conv_channels": 32,
            "conv_kernel": 4,
            "rms_norm_epsilon": 1e-6,
            "hidden_activation": "silu",
            "output_gate_activation": "sigmoid",
            "recurrent_dtype": "float32",
            "conv_state_dtype": "float32",
        },
        "determinism": {
            "parameter_generation": "LCG32 integer samples divided by powers of two",
            "torch_deterministic_algorithms": True,
            "torch_threads": 1,
            "network_access": False,
        },
        "weights": weights,
    }

    calibration_specs = (
        ("KQ-GDN-CAL-SEQ1-ZERO", 1, 0x1001, "random", None, None),
        ("KQ-GDN-CAL-SEQ3-REPEATED", 3, 0x1002, "repeated", None, None),
        ("KQ-GDN-CAL-SEQ4-ALTERNATING", 4, 0x1003, "alternating", None, None),
        ("KQ-GDN-CAL-SEQ5-NONZERO", 5, 0x1004, "random", None, 0x51004),
        ("KQ-GDN-CAL-SEQ7-MASKED", 7, 0x1005, "random", [1, 1, 0, 1, 0, 1, 1], None),
    )
    holdout_specs = (
        ("KQ-GDN-HOLD-SEQ2", 2, 0x2001, "random", None, None),
        ("KQ-GDN-HOLD-SEQ4-REPEATED", 4, 0x2002, "repeated", None, None),
        ("KQ-GDN-HOLD-SEQ6-ALTERNATING", 6, 0x2003, "alternating", None, None),
        ("KQ-GDN-HOLD-SEQ8-NONZERO", 8, 0x2004, "random", None, 0x52004),
        ("KQ-GDN-HOLD-SEQ9-MASKED", 9, 0x2005, "random", [1, 0, 1, 1, 1, 0, 1, 1, 1], None),
    )

    def generate_cases(specs: Any) -> list[dict[str, Any]]:
        records = []
        for case_id, length, seed, mode, mask_values, initial_seed in specs:
            hidden = input_tensor(torch, length, seed, mode)
            mask = None if mask_values is None else torch.tensor([mask_values], dtype=torch.float32)
            initial = None if initial_seed is None else initial_state(torch, initial_seed)
            record, _ = run_case(
                torch, DynamicCache, modeling, config, module,
                case_id, hidden, mask=mask, initial=initial
            )
            records.append(record)
        return records

    calibration = {
        "schema": "KQ-GDN-CALIBRATION-v1",
        "oracle": "pinned Transformers Qwen4ExpTextGatedDeltaNet",
        "expected_values_source": "independent Class-C oracle; never Kestrel-Q",
        "cases": generate_cases(calibration_specs),
    }
    holdout = {
        "schema": "KQ-GDN-HOLDOUT-v1",
        "oracle": "pinned Transformers Qwen4ExpTextGatedDeltaNet",
        "disjoint_from_calibration": True,
        "expected_values_source": "independent Class-C oracle; never Kestrel-Q",
        "cases": generate_cases(holdout_specs),
    }

    prefix = input_tensor(torch, 3, 0x3001, "random")
    prefix_record, cache = run_case(
        torch, DynamicCache, modeling, config, module,
        "KQ-GDN-STATE-PREFIX3", prefix
    )
    decode1_input = input_tensor(torch, 1, 0x3002, "random")
    decode2_input = input_tensor(torch, 1, 0x3003, "random")
    decode1 = run_continuation(
        torch, modeling, module, cache, decode1_input, "KQ-GDN-STATE-DECODE1"
    )
    decode2 = run_continuation(
        torch, modeling, module, cache, decode2_input, "KQ-GDN-STATE-DECODE2"
    )
    full_input = torch.cat([prefix, decode1_input, decode2_input], dim=1)
    full_record, _ = run_case(
        torch, DynamicCache, modeling, config, module,
        "KQ-GDN-STATE-FULL5", full_input
    )
    replay_record, _ = run_case(
        torch, DynamicCache, modeling, config, module,
        "KQ-GDN-STATE-RESET-REPLAY", prefix
    )

    states = {
        "schema": "KQ-GDN-STATE-VECTORS-v1",
        "oracle": "pinned Transformers Qwen4ExpTextGatedDeltaNet and DynamicCache",
        "expected_values_source": "independent Class-C oracle; never Kestrel-Q",
        "prefill": prefix_record,
        "decode_steps": [decode1, decode2],
        "full_recomputation": full_record,
        "reset_replay": replay_record,
        "reset_replay_exact_bits": (
            prefix_record["expected"]["output"] == replay_record["expected"]["output"]
            and prefix_record["expected"]["conv_state"] == replay_record["expected"]["conv_state"]
            and prefix_record["expected"]["recurrent_state"] == replay_record["expected"]["recurrent_state"]
        ),
        "consistency_contract": "CALIBRATED_FLOAT",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "gdn-contract.json", contract)
    write_json(args.output_dir / "gdn-calibration.json", calibration)
    write_json(args.output_dir / "gdn-holdout.json", holdout)
    write_json(args.output_dir / "gdn-state-vectors.json", states)

    manifest = {
        "schema": "KQ-GDN-MANIFEST-v1",
        "status": "ORACLE_GENERATED_NATIVE_VALIDATION_PENDING",
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "oracle_class": "C",
        "oracle_license": "Apache-2.0",
        "assets": [
            {"path": name, "sha256": sha256(args.output_dir / name)}
            for name in EVIDENCE_FILES
        ],
        "safety": {
            "full_bf16_checkpoint_downloaded": False,
            "real_model_weight_bytes_used": 0,
            "kestrel_q_used_as_expected_value_source": False,
        },
    }
    write_json(args.output_dir / "gdn-manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
