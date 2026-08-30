#!/usr/bin/env python3
"""Generate independent Qwen3.8 PLE-value Class-C evidence."""

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
    "transformers/models/qwen4_exp/modeling_qwen4_exp.py":
        "91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f",
    "transformers/models/qwen4_exp/configuration_qwen4_exp.py":
        "26b47995740e3bc596b44b2011ee6c3d971d46136438b00dd5fad9557bec4254",
}
ORACLE_FILES = (
    "ple-value-contract.json", "ple-value-calibration.json",
    "ple-value-holdout.json", "ple-value-state-vectors.json",
    "ple-value-address-integration.json",
)
MULTIPLIERS = (23703573157769, 20109073645365, 8052911324071)
OFFSETS = (0, 20000003, 40000026, 60000059, 80000106, 100000165,
           120000228, 140000297, 160000374, 180000455, 200000548,
           220000655, 240000802, 260000955, 280001114, 300001275)
MODULI = (20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
          20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
          20000153, 20000159, 20000161, 20000171)
MEMBER_ROWS = 2_500_012
EOS = 248_044


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True,
                                 ensure_ascii=False) + "\n").encode("utf-8"))


def tensor_record(torch: Any, value: Any) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous().to(torch.float32)
    return {
        "shape": list(tensor.shape), "dtype": "float32",
        "f32_le_hex": [struct.pack("<f", float(item)).hex()
                       for item in tensor.flatten().tolist()],
    }


def lcg_values(torch: Any, count: int, seed: int, divisor: float) -> Any:
    state = seed & 0xFFFFFFFF
    values = []
    for _ in range(count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        signed = ((state >> 8) & 0xFFFF) - 32768
        values.append(signed / divisor)
    return torch.tensor(values, dtype=torch.float32)


def reduced_config(Config: Any, heads: int = 2, embed: int = 8) -> Any:
    return Config(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=1,
        num_key_value_heads=1, head_dim=8, layer_types=["linear_attention"],
        hc_count=2, ple_layer_ids=[1], ple_embed_dim=embed, ngram_size=3,
        heads_per_ngram=heads, ngram_vocab_size_base=11,
        make_ngram_vocab_size_divisible_by=4, ple_conv_kernel_size=4,
        vocab_size=64, eos_token_id=63, num_experts=2,
        num_experts_per_tok=1, moe_intermediate_size=4,
        shared_expert_intermediate_size=4, linear_num_key_heads=1,
        linear_num_value_heads=1, linear_key_head_dim=4,
        linear_value_head_dim=4, linear_conv_kernel_dim=4,
    )


def fill_parameters(torch: Any, module: Any, seed: int) -> dict[str, Any]:
    records = {}
    with torch.no_grad():
        for index, (name, parameter) in enumerate(module.named_parameters()):
            divisor = 32768.0 if "norm" not in name else 131072.0
            values = lcg_values(torch, parameter.numel(),
                                seed + index * 0x101, divisor)
            parameter.copy_(values.reshape(parameter.shape))
            records[name] = tensor_record(torch, parameter)
    return records


class Capture:
    def __init__(self, torch: Any, module: Any):
        self.torch = torch
        self.records: dict[str, Any] = {}
        self.handles = []
        names = {
            "ple_embedding.ngram_embedding": "lookup",
            "ple_embedding": "embedding", "key_proj": "key_projection",
            "value_proj": "value_projection", "norm_key": "key_norm",
            "norm_query": "query_norm", "norm_conv": "conv_norm",
            "conv1d": "conv_pre_activation",
        }
        modules = dict(module.named_modules())
        for name, label in names.items():
            if name not in modules:
                continue
            self.handles.append(modules[name].register_forward_hook(
                lambda _m, inp, out, key=label: self._record(key, inp, out)))

    def _record(self, key: str, inputs: Any, output: Any) -> None:
        self.records[key] = tensor_record(self.torch, output)
        if key == "lookup":
            self.records["global_addresses"] = inputs[0].detach().cpu().tolist()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def make_hidden(torch: Any, length: int, seed: int, mode: str) -> Any:
    if mode == "repeated":
        row = lcg_values(torch, 16, seed, 8192.0).reshape(1, 1, 16)
        return row.repeat(1, length, 1)
    values = lcg_values(torch, length * 16, seed, 8192.0).reshape(1, length, 16)
    if mode == "alternating":
        values[:, 1::2] = -values[:, 1::2]
    return values


def cache_state(torch: Any, cache: Any) -> dict[str, Any]:
    layer = cache.layers[0]
    state = layer.conv_states[1]
    return tensor_record(torch, state)


def derive_checkpoints(torch: Any, module: Any, capture: dict[str, Any]) -> None:
    key = torch.tensor([struct.unpack("<f", bytes.fromhex(v))[0]
                        for v in capture["key_norm"]["f32_le_hex"]], dtype=torch.float32)
    query = torch.tensor([struct.unpack("<f", bytes.fromhex(v))[0]
                          for v in capture["query_norm"]["f32_le_hex"]], dtype=torch.float32)
    value = torch.tensor([struct.unpack("<f", bytes.fromhex(v))[0]
                          for v in capture["value_projection"]["f32_le_hex"]], dtype=torch.float32)
    shape = capture["key_norm"]["shape"]
    key = key.reshape(shape[0], shape[1], module.hc_count, module.hidden_size)
    query = query.reshape_as(key)
    value = value.reshape(shape[0], shape[1], module.hidden_size)
    gate_raw = (key * query).sum(-1, keepdim=True) / module.hidden_size ** 0.5
    gate = gate_raw.abs().clamp_min(1e-6).sqrt() * gate_raw.sign()
    gated = torch.sigmoid(gate) * value.unsqueeze(-2)
    capture["gate_raw"] = tensor_record(torch, gate_raw)
    capture["gate_transformed"] = tensor_record(torch, gate)
    capture["gated_value"] = tensor_record(torch, gated)
    conv_shape = capture["conv_pre_activation"]["shape"]
    conv_values = torch.tensor([
        struct.unpack("<f", bytes.fromhex(v))[0]
        for v in capture["conv_pre_activation"]["f32_le_hex"]
    ], dtype=torch.float32).reshape(conv_shape)
    capture["conv_output"] = tensor_record(
        torch, torch.nn.functional.silu(conv_values).transpose(1, 2))


def run_case(torch: Any, DynamicCache: Any, module: Any, config: Any,
             case_id: str, hidden: Any, ids: Any,
             initial_state: Any | None = None) -> dict[str, Any]:
    cache = DynamicCache(config=config)
    if initial_state is not None:
        cache.update_conv_state(initial_state, 0, state_idx=1,
                                conv_kernel_size=9)
    capture = Capture(torch, module)
    with torch.no_grad():
        output = module(hidden, ids, cache)
    capture.close()
    derive_checkpoints(torch, module, capture.records)
    capture.records["operator_output"] = tensor_record(torch, output)
    return {
        "id": case_id, "input": tensor_record(torch, hidden),
        "token_ids": ids.detach().cpu().tolist()[0],
        "initial_value_state": tensor_record(
            torch, initial_state if initial_state is not None else
            torch.zeros(1, module.hc_count * module.hidden_size, 9)),
        "expected": {"checkpoints": capture.records,
                     "final_value_state": cache_state(torch, cache)},
    }


def intents(tokens: list[int]) -> list[dict[str, int]]:
    out = []
    history = [EOS, EOS]
    for position, token in enumerate(tokens):
        newest, older = history[1], history[0]
        x1 = newest if newest != EOS else EOS
        x2 = older if newest != EOS and older != EOS else EOS
        shifted = (token, x1, x2)
        for order in (2, 3):
            mixed = shifted[0] * MULTIPLIERS[0]
            for at in range(1, order):
                mixed ^= shifted[at] * MULTIPLIERS[at]
            for local in range(8):
                head = (order - 2) * 8 + local
                global_address = OFFSETS[head] + mixed % MODULI[head]
                out.append({
                    "position": position, "order": order,
                    "local_head": local, "global_head": head,
                    "logical_member": global_address // MEMBER_ROWS,
                    "member_row": global_address % MEMBER_ROWS,
                    "global_address": global_address,
                })
        history = [newest, token]
    return out


def synthetic_rows(torch: Any, records: list[dict[str, int]], width: int) -> Any:
    rows = []
    for item in records:
        seed = ((item["logical_member"] + 1) * 0x1F123BB5 +
                item["member_row"] * 0x9E3779B1 +
                item["global_head"] * 0x101) & 0xFFFFFFFF
        rows.append(lcg_values(torch, width, seed, 16384.0))
    return torch.stack(rows)


class AddressEmbedding:
    def __init__(self, torch: Any, nn: Any, tokens: list[int], width: int):
        class Impl(nn.Module):
            def __init__(self, outer: Any):
                super().__init__(); self.outer = outer
            def forward(self, input_ids: Any, _cache: Any) -> Any:
                if input_ids.detach().cpu().tolist()[0] != self.outer.tokens:
                    raise RuntimeError("unexpected Tier-B token stream")
                rows = synthetic_rows(self.outer.torch, self.outer.records,
                                      self.outer.width)
                return rows.reshape(1, len(self.outer.tokens), -1)
        self.torch, self.tokens, self.width = torch, tokens, width
        self.records = intents(tokens)
        self.module = Impl(self)


def run_address_case(torch: Any, DynamicCache: Any, module: Any, config: Any,
                     case_id: str, tokens: list[int], seed: int) -> dict[str, Any]:
    provider = AddressEmbedding(torch, torch.nn, tokens, 2)
    module.ple_embedding = provider.module
    hidden = make_hidden(torch, len(tokens), seed, "alternating")
    ids = torch.tensor([tokens], dtype=torch.long)
    cache = DynamicCache(config=config)
    capture = Capture(torch, module)
    with torch.no_grad():
        output = module(hidden, ids, cache)
    capture.close()
    rows = synthetic_rows(torch, provider.records, 2)
    capture.records["lookup"] = tensor_record(
        torch, rows.reshape(1, len(tokens), 16, 2))
    capture.records["embedding"] = tensor_record(
        torch, rows.reshape(1, len(tokens), 32))
    derive_checkpoints(torch, module, capture.records)
    capture.records["operator_output"] = tensor_record(torch, output)
    return {
        "id": case_id, "token_ids": tokens,
        "input": tensor_record(torch, hidden), "intents": provider.records,
        "lookup_rows": tensor_record(torch, rows),
        "initial_value_state": tensor_record(torch, torch.zeros(1, 16, 9)),
        "expected": {"checkpoints": capture.records,
                     "final_value_state": cache_state(torch, cache)},
    }


def generate(output: Path, checkout: Path) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(checkout / "src"))
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = reduced_config(Qwen4ExpTextConfig)
    module = Qwen4ExpTextPLELayer(config, 0, 0).eval()
    weights = fill_parameters(torch, module, 0x29100000)

    contract = {
        "schema": "kq-ple-value-contract-v1",
        "authority": {"model_revision": MODEL_REVISION,
                      "transformers_revision": TRANSFORMERS_REVISION,
                      "model_config_sha256": CONFIG_SHA256,
                      "source_sha256": SOURCE_HASHES,
                      "license": "Apache-2.0"},
        "oracle_environment": {"python": platform.python_version(),
                               "torch": torch.__version__, "device": "cpu",
                               "deterministic_algorithms": True,
                               "thread_count": 1},
        "tier_a_config": {"hidden_size": 8, "residual_branches": 2,
                          "heads_per_order": 2, "head_count": 4,
                          "row_width": 2, "embedding_width": 8,
                          "conv_kernel": 4, "conv_dilation": 3,
                          "value_history": 9, "rms_epsilon": 1.0e-6},
        "target_config": {"layer_id": 1, "hidden_size": 2560,
                          "residual_branches": 4, "head_count": 16,
                          "row_width": 160, "embedding_width": 2560,
                          "logical_members": 128, "member_rows": MEMBER_ROWS,
                          "conv_kernel": 4, "conv_dilation": 3,
                          "value_history": 9,
                          "semantic_state_bytes_bf16": 184320,
                          "reference_state_bytes_f32": 368640},
        "tier_a_weights": weights,
    }

    config_b = reduced_config(Qwen4ExpTextConfig, heads=8, embed=32)
    module_b = Qwen4ExpTextPLELayer(config_b, 0, 0).eval()
    contract["tier_b_config"] = {
        "hidden_size": 8, "residual_branches": 2,
        "heads_per_order": 8, "head_count": 16, "row_width": 2,
        "embedding_width": 32, "conv_kernel": 4,
        "conv_dilation": 3, "value_history": 9, "rms_epsilon": 1.0e-6,
    }
    contract["tier_b_weights"] = fill_parameters(torch, module_b, 0x29B00000)
    write_json(output / "ple-value-contract.json", contract)

    calibration_specs = [("cal-length-1", 1, 0x291001, "random"),
                         ("cal-length-3", 3, 0x291002, "alternating"),
                         ("cal-length-9", 9, 0x291003, "repeated"),
                         ("cal-length-10", 10, 0x291004, "random")]
    calibration = {"schema": "kq-ple-value-calibration-v1",
                   "comparison_class": "CALIBRATED_FLOAT", "cases": []}
    for name, length, seed, mode in calibration_specs:
        ids = torch.tensor([[1 + (index % 31) for index in range(length)]])
        calibration["cases"].append(run_case(
            torch, DynamicCache, module, config, name,
            make_hidden(torch, length, seed, mode), ids))
    write_json(output / "ple-value-calibration.json", calibration)

    initial = lcg_values(torch, 16 * 9, 0x291100, 16384.0).reshape(1, 16, 9)
    holdout_specs = [("holdout-length-2", 2, 0x292001, "random", None),
                     ("holdout-length-4", 4, 0x292002, "alternating", initial),
                     ("holdout-length-11", 11, 0x292003, "random", None)]
    holdout = {"schema": "kq-ple-value-holdout-v1",
               "comparison_class": "CALIBRATED_FLOAT", "cases": []}
    for name, length, seed, mode, state in holdout_specs:
        ids = torch.tensor([[7 + (index % 23) for index in range(length)]])
        holdout["cases"].append(run_case(
            torch, DynamicCache, module, config, name,
            make_hidden(torch, length, seed, mode), ids, state))
    write_json(output / "ple-value-holdout.json", holdout)

    prefix_ids = torch.tensor([[3, 5, 7, 9, 11]])
    prefix_hidden = make_hidden(torch, 5, 0x293001, "alternating")
    cache = DynamicCache(config=config)
    capture = Capture(torch, module)
    with torch.no_grad():
        prefix_output = module(prefix_hidden, prefix_ids, cache)
    capture.close()
    prefix_state = cache_state(torch, cache)
    steps = []
    for step, token in enumerate((13, 15, 17)):
        hidden = make_hidden(torch, 1, 0x293100 + step, "random")
        cap = Capture(torch, module)
        with torch.no_grad():
            step_output = module(hidden, torch.tensor([[token]]), cache)
        cap.close()
        steps.append({"step": step, "token_id": token,
                      "input": tensor_record(torch, hidden),
                      "output": tensor_record(torch, step_output),
                      "value_state": cache_state(torch, cache)})
    state_vectors = {
        "schema": "kq-ple-value-state-vectors-v1",
        "comparison_classes": {"state": "CALIBRATED_FLOAT",
                               "position": "EXACT_DISCRETE"},
        "cases": [{"id": "prefill-three-decode", "prefix_ids": [3,5,7,9,11],
                   "prefix_input": tensor_record(torch, prefix_hidden),
                   "prefix_output": tensor_record(torch, prefix_output),
                   "prefix_state": prefix_state,
                   "decode_steps": steps, "reset_replay_required": True}],
    }
    write_json(output / "ple-value-state-vectors.json", state_vectors)

    address_cases = [
        run_address_case(torch, DynamicCache, module_b, config_b,
                         "address-length-1", [1], 0x294001),
        run_address_case(torch, DynamicCache, module_b, config_b,
                         "address-history-10", [1,2,3,4,5,6,7,8,9,10], 0x294002),
        run_address_case(torch, DynamicCache, module_b, config_b,
                         "address-eos-boundary", [41, EOS, 42, 42], 0x294003),
    ]
    address = {"schema": "kq-ple-value-address-integration-v1",
               "comparison_classes": {"intents": "EXACT_DISCRETE",
                                      "values": "CALIBRATED_FLOAT"},
               "provider": "deterministic synthetic member+row function",
               "cases": address_cases}
    write_json(output / "ple-value-address-integration.json", address)

    files = {name: {"sha256": sha256(output / name),
                    "bytes": (output / name).stat().st_size}
             for name in ORACLE_FILES}
    manifest = {
        "schema": "kq-ple-value-evidence-manifest-v1",
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "generation_tool": "tools/generate-ple-value-reference.py",
        "offline": True, "full_model_weights_downloaded": False,
        "expected_values_from_kestrel_q": False, "files": files,
    }
    write_json(output / "ple-value-manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    checkout, config, output = args.checkout.resolve(), args.config.resolve(), args.output_dir.resolve()
    revision = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    if revision != TRANSFORMERS_REVISION:
        raise SystemExit(f"unexpected Transformers revision: {revision}")
    if not config.is_file() or sha256(config) != CONFIG_SHA256:
        raise SystemExit("pinned model config hash mismatch")
    for relative, expected in SOURCE_HASHES.items():
        if sha256(checkout / "src" / relative) != expected:
            raise SystemExit(f"source hash mismatch: {relative}")
    if args.verify:
        with tempfile.TemporaryDirectory(prefix="kq-ple-value-verify-") as directory:
            candidate = Path(directory)
            generate(candidate, checkout)
            mismatch = [name for name in ORACLE_FILES if
                        not (output / name).is_file() or
                        (output / name).read_bytes() != (candidate / name).read_bytes()]
            if mismatch:
                raise SystemExit("non-deterministic PLE value evidence: " + ", ".join(mismatch))
        print("PLE value evidence deterministic regeneration: PASS")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    generate(output, checkout)
    print(f"PLE value Class-C evidence written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
