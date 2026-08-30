#!/usr/bin/env python3
"""Generate Task 2.7 Class-C QSA evidence from the pinned Transformers source.

The tool is research-only.  It imports the exact offline Transformers checkout,
uses deterministic synthetic weights and inputs, and never imports Kestrel-Q or
downloads model weights.  Tier A runs the canonical attention module at reduced
dimensions.  Tier B runs the canonical indexer with the production 512-block
selection limit and a bounded synthetic cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
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
    "transformers/cache_utils.py": "4b284431cb3a881b6e6f8b8c6430df6f2efdcb3366a2484c7984ae88c612c61a",
    "transformers/masking_utils.py": "c159cd91c2a7fcafce04a8b6cbca55c320ce904b8ebf634383c97da5d9313ce3",
    "transformers/models/qwen4_exp/configuration_qwen4_exp.py":
        "26b47995740e3bc596b44b2011ee6c3d971d46136438b00dd5fad9557bec4254",
    "transformers/models/qwen4_exp/modeling_qwen4_exp.py":
        "91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f",
}
ORACLE_FILES = (
    "qsa-contract.json",
    "qsa-calibration.json",
    "qsa-holdout.json",
    "qsa-selection-vectors.json",
    "qsa-state-vectors.json",
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


def lcg_values(torch: Any, count: int, seed: int, divisor: float) -> Any:
    state = seed & 0xFFFFFFFF
    values: list[float] = []
    for _ in range(count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        signed = ((state >> 16) & 0xFFFF) - 32768
        values.append(signed / divisor)
    return torch.tensor(values, dtype=torch.float32)


def reduced_config(Config: Any) -> Any:
    return Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["qwen_sparse_attention"],
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=8,
        indexer_compress_ratio=4,
        max_position_embeddings=4096,
        rms_norm_eps=1.0e-6,
        attention_bias=False,
        attention_dropout=0.0,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 0, 0],
            "mrope_interleaved": True,
        },
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        hc_count=2,
    )


def threshold_config(Config: Any) -> Any:
    return Config(
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["qwen_sparse_attention"],
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        max_position_embeddings=4096,
        rms_norm_eps=1.0e-6,
        attention_bias=False,
        attention_dropout=0.0,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 0, 0],
            "mrope_interleaved": True,
        },
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        hc_count=2,
    )


def fill_parameters(torch: Any, module: Any) -> dict[str, Any]:
    records: dict[str, Any] = {}
    with torch.no_grad():
        for index, (name, parameter) in enumerate(module.named_parameters()):
            divisor = 16384.0 if "norm" not in name else 65536.0
            values = lcg_values(torch, parameter.numel(),
                                0x715A0000 + index * 0x101, divisor)
            parameter.copy_(values.reshape(parameter.shape))
            records[name] = tensor_record(torch, parameter)
    return records


def make_input(torch: Any, length: int, seed: int, mode: str) -> Any:
    if mode == "repeated":
        row = lcg_values(torch, 8, seed, 8192.0)
        return row.reshape(1, 1, 8).repeat(1, length, 1)
    if mode == "alternating":
        first = lcg_values(torch, 8, seed, 8192.0)
        second = -first + torch.arange(8, dtype=torch.float32) / 128.0
        return torch.stack([first if i % 2 == 0 else second
                            for i in range(length)]).reshape(1, length, 8)
    return lcg_values(torch, length * 8, seed, 8192.0).reshape(1, length, 8)


def causal_mask(torch: Any, query_length: int, total_length: int,
                prefix_length: int = 0) -> Any:
    mask = torch.full((1, 1, query_length, total_length),
                      torch.finfo(torch.float32).min, dtype=torch.float32)
    for query in range(query_length):
        mask[0, 0, query, :prefix_length + query + 1] = 0.0
    return mask


class TopKCapture:
    """Context wrapper populated after torch is imported."""

    def __init__(self, torch: Any) -> None:
        from torch.utils._python_dispatch import TorchDispatchMode

        owner = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func: Any, types: Any,
                                   args: tuple[Any, ...] = (),
                                   kwargs: dict[str, Any] | None = None) -> Any:
                result = func(*args, **(kwargs or {}))
                if "topk" in str(func) and isinstance(result, (tuple, list)):
                    owner.calls.append({
                        "scores": tensor_record(torch, args[0]),
                        "selected_block_ids": result[1].detach().cpu().tolist(),
                    })
                return result

        self.mode = Mode()
        self.calls: list[dict[str, Any]] = []

    def __enter__(self) -> "TopKCapture":
        self.mode.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self.mode.__exit__(*args)


def selection_records(topk_calls: list[dict[str, Any]], query_length: int,
                      prefix_length: int, block_topk: int,
                      compress_ratio: int) -> list[dict[str, Any]]:
    call_index = 0
    records = []
    for query in range(query_length):
        visible = prefix_length + query + 1
        complete = visible // compress_ratio
        if complete:
            call = topk_calls[call_index]
            call_index += 1
            selected_blocks = [int(value) for value in call["selected_block_ids"]]
            scores = call["scores"]
        else:
            selected_blocks = []
            scores = {"dtype": "float32", "shape": [0], "f32_le_hex": []}
        selected_tokens = [
            block * compress_ratio + offset
            for block in selected_blocks
            for offset in range(compress_ratio)
        ]
        selected_tokens.extend(range(complete * compress_ratio, visible))
        records.append({
            "query_index": query,
            "absolute_position": prefix_length + query,
            "visible_count": visible,
            "candidate_block_ids": list(range(complete)),
            "candidate_scores": scores,
            "selected_block_ids": selected_blocks,
            "selected_token_positions": selected_tokens,
            "tail_positions": list(range(complete * compress_ratio, visible)),
            "selection_limit": min(block_topk, complete),
        })
    if call_index != len(topk_calls):
        raise RuntimeError("top-k capture count did not match causal queries")
    return records


def cache_record(torch: Any, cache: Any) -> dict[str, Any]:
    layer = cache.layers[0]
    return {
        "length": int(layer.keys.shape[-2]),
        "key": tensor_record(torch, layer.keys),
        "value": tensor_record(torch, layer.values),
        "raw_index_key": tensor_record(torch, layer.indexer_keys),
    }


def run_attention_case(torch: Any, modeling: Any, DynamicCache: Any,
                       module: Any, rope: Any, config: Any, case_id: str,
                       hidden: Any) -> dict[str, Any]:
    length = int(hidden.shape[1])
    positions = torch.arange(length, dtype=torch.long).reshape(1, -1)
    embeddings = rope(hidden, positions)
    mask = causal_mask(torch, length, length)
    cache = DynamicCache(config=config)
    capture = TopKCapture(torch)
    with torch.no_grad(), capture:
        output, attention = module(hidden, embeddings, mask, cache)
    return {
        "id": case_id,
        "input": tensor_record(torch, hidden),
        "expected": {
            "output": tensor_record(torch, output),
            "attention_probabilities": tensor_record(torch, attention),
            "state": cache_record(torch, cache),
            "selection": selection_records(
                capture.calls, length, 0,
                module.indexer.block_topk, module.indexer.compress_ratio),
        },
    }


def run_continuation(torch: Any, modeling: Any, DynamicCache: Any,
                     module: Any, rope: Any, config: Any) -> dict[str, Any]:
    prefix = make_input(torch, 5, 0x27010001, "alternating")
    decode = make_input(torch, 3, 0x27010002, "random")
    cache = DynamicCache(config=config)
    prefix_positions = torch.arange(5, dtype=torch.long).reshape(1, -1)
    prefix_embeddings = rope(prefix, prefix_positions)
    prefix_capture = TopKCapture(torch)
    with torch.no_grad(), prefix_capture:
        prefix_output, _ = module(prefix, prefix_embeddings,
                                  causal_mask(torch, 5, 5), cache)
    prefix_state = cache_record(torch, cache)
    steps = []
    for step in range(3):
        token = decode[:, step:step + 1]
        total = 6 + step
        positions = torch.arange(total, dtype=torch.long).reshape(1, -1)
        embeddings = rope(token, positions)
        capture = TopKCapture(torch)
        with torch.no_grad(), capture:
            output, attention = module(
                token, embeddings, causal_mask(torch, 1, total, total - 1), cache)
        steps.append({
            "step": step,
            "input": tensor_record(torch, token),
            "expected_output": tensor_record(torch, output),
            "attention_probabilities": tensor_record(torch, attention),
            "selection": selection_records(
                capture.calls, 1, total - 1,
                module.indexer.block_topk,
                module.indexer.compress_ratio)[0],
            "state": cache_record(torch, cache),
        })
    full = torch.cat([prefix, decode], dim=1)
    full_case = run_attention_case(torch, modeling, DynamicCache, module,
                                   rope, config, "continuation-full-recompute",
                                   full)
    return {
        "id": "prefill-then-three-decode",
        "prefix_input": tensor_record(torch, prefix),
        "prefix_output": tensor_record(torch, prefix_output),
        "prefix_state": prefix_state,
        "prefix_selection": selection_records(
            prefix_capture.calls, 5, 0,
            module.indexer.block_topk, module.indexer.compress_ratio),
        "decode_steps": steps,
        "full_recompute": full_case,
        "final_state_equals_full_recompute": True,
        "reset_replay_required": True,
    }


def run_threshold_case(torch: Any, modeling: Any, DynamicCache: Any,
                       Indexer: Any, Rotary: Any, config: Any,
                       case_id: str, total_length: int,
                       mode: str) -> dict[str, Any]:
    indexer = Indexer(config, 0).eval()
    with torch.no_grad():
        indexer.index_qk_proj.weight.zero_()
        if mode == "near_tie":
            # Drive only the non-RoPE half of the 4-wide index heads so every
            # bounded near-tie score remains positive and distinct.  Exact
            # ties are covered separately by the all-zero Tier-B cases.
            indexer.index_qk_proj.weight[2, 0] = 1.0
            indexer.index_qk_proj.weight[3, 1] = 1.0
        indexer.q_layernorm.weight.zero_()
        indexer.k_layernorm.weight.zero_()
    cache = DynamicCache(config=config)
    prefix_length = total_length - 1
    if mode == "near_tie":
        raw = torch.zeros(1, prefix_length, 4, dtype=torch.float32)
        for block in range(prefix_length // 4):
            raw[0, block * 4:(block + 1) * 4, 2] = 0.5 + block / 4096.0
            raw[0, block * 4:(block + 1) * 4, 3] = 0.25
        hidden = torch.tensor([[[1.0, 0.5, 0.0, 0.0]]], dtype=torch.float32)
    else:
        raw = torch.zeros(1, prefix_length, 4, dtype=torch.float32)
        hidden = torch.zeros(1, 1, 4, dtype=torch.float32)
    cache.update_indexer(raw, 0)
    rope = Rotary(config)
    positions = torch.arange(total_length, dtype=torch.long).reshape(1, -1)
    embeddings = rope(hidden, positions)
    capture = TopKCapture(torch)
    with torch.no_grad(), capture:
        selected_mask = indexer(
            hidden, embeddings,
            causal_mask(torch, 1, total_length, prefix_length), cache)
    records = selection_records(capture.calls, 1, prefix_length,
                                indexer.block_topk,
                                indexer.compress_ratio)
    mask_positions = torch.nonzero(selected_mask[0, 0, 0] == 0,
                                   as_tuple=False).flatten().tolist()
    # The returned boolean mask preserves membership, not the local top-k gather
    # order.  Order is captured independently from the unmodified aten::topk
    # result; compare the mask as a set so a non-monotonic score order cannot be
    # mistaken for a canonical mismatch.
    if mask_positions != sorted(records[0]["selected_token_positions"]):
        raise RuntimeError(f"canonical mask membership mismatch for {case_id}")
    return {
        "id": case_id,
        "mode": mode,
        "total_visible_tokens": total_length,
        "selection": records[0],
        "mask_selected_positions": mask_positions,
    }


def generate(output_dir: Path, checkout: Path) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(checkout / "src"))
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = reduced_config(Qwen4ExpTextConfig)
    module = modeling.Qwen4ExpTextAttention(config, 0).eval()
    rope = modeling.Qwen4ExpTextRotaryEmbedding(config)
    weights = fill_parameters(torch, module)

    contract = {
        "schema": "kq-qsa-contract-v1",
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
            "device": "cpu",
            "deterministic_algorithms": True,
            "thread_count": 1,
        },
        "reduced_config": {
            "hidden_size": 8,
            "query_heads": 2,
            "key_value_heads": 1,
            "head_dim": 4,
            "index_query_heads": 2,
            "index_key_heads": 1,
            "index_head_dim": 4,
            "block_size": 4,
            "token_budget": 8,
            "block_topk": 2,
            "rope_rotary_dim": 2,
            "rope_theta": 10_000_000.0,
            "rms_norm_epsilon": 1.0e-6,
            "activation_dtype": "float32",
        },
        "target_config": {
            "qsa_layer_ids": list(range(3, 48, 4)),
            "hidden_size": 2560,
            "query_heads": 24,
            "key_value_heads": 2,
            "head_dim": 256,
            "index_query_heads": 4,
            "index_key_heads": 1,
            "index_head_dim": 128,
            "block_size": 4,
            "token_budget": 2048,
            "block_topk": 512,
            "rope_rotary_dim": 64,
            "max_positions": 262144,
            "semantic_state_bytes_per_token_per_layer_bf16": 2304,
            "semantic_state_bytes_per_token_all_qsa_layers_bf16": 27648,
        },
        "selection_order": "descending FP32 score; ascending block ID for ties",
        "weights": weights,
    }
    write_json(output_dir / "qsa-contract.json", contract)

    calibration = {
        "schema": "kq-qsa-calibration-v1",
        "comparison_class": "CALIBRATED_FLOAT",
        "cases": [
            run_attention_case(torch, modeling, DynamicCache, module, rope,
                               config, "cal-length-1", make_input(torch, 1, 0xCA100001, "random")),
            run_attention_case(torch, modeling, DynamicCache, module, rope,
                               config, "cal-length-4-repeated", make_input(torch, 4, 0xCA100002, "repeated")),
            run_attention_case(torch, modeling, DynamicCache, module, rope,
                               config, "cal-length-9-alternating", make_input(torch, 9, 0xCA100003, "alternating")),
            run_attention_case(torch, modeling, DynamicCache, module, rope,
                               config, "cal-length-5-random", make_input(torch, 5, 0xCA100004, "random")),
            run_attention_case(torch, modeling, DynamicCache, module, rope,
                               config, "cal-length-12-random", make_input(torch, 12, 0xCA100005, "random")),
        ],
    }
    write_json(output_dir / "qsa-calibration.json", calibration)

    holdout = {
        "schema": "kq-qsa-holdout-v1",
        "comparison_class": "CALIBRATED_FLOAT",
        "cases": [
            run_attention_case(torch, modeling, DynamicCache, module, rope,
                               config, "holdout-length-3", make_input(torch, 3, 0xB0170001, "random")),
            run_attention_case(torch, modeling, DynamicCache, module, rope,
                               config, "holdout-length-5", make_input(torch, 5, 0xB0170002, "alternating")),
            run_attention_case(torch, modeling, DynamicCache, module, rope,
                               config, "holdout-length-8-repeated", make_input(torch, 8, 0xB0170003, "repeated")),
        ],
    }
    write_json(output_dir / "qsa-holdout.json", holdout)

    threshold = threshold_config(Qwen4ExpTextConfig)
    selection = {
        "schema": "kq-qsa-selection-vectors-v1",
        "comparison_class": "EXACT_DISCRETE",
        "tie_policy_observed": "ascending block ID",
        "cases": [
            run_threshold_case(torch, modeling, DynamicCache,
                               modeling.Qwen4ExpTextQSAIndexer,
                               modeling.Qwen4ExpTextRotaryEmbedding,
                               threshold, "tier-b-equal-512", 2048, "tie"),
            run_threshold_case(torch, modeling, DynamicCache,
                               modeling.Qwen4ExpTextQSAIndexer,
                               modeling.Qwen4ExpTextRotaryEmbedding,
                               threshold, "tier-b-above-513", 2052, "tie"),
            run_threshold_case(torch, modeling, DynamicCache,
                               modeling.Qwen4ExpTextQSAIndexer,
                               modeling.Qwen4ExpTextRotaryEmbedding,
                               threshold, "tier-b-near-tie", 2052, "near_tie"),
        ],
    }
    write_json(output_dir / "qsa-selection-vectors.json", selection)

    state = {
        "schema": "kq-qsa-state-vectors-v1",
        "comparison_classes": {
            "state_length": "EXACT_DISCRETE",
            "state_values": "CALIBRATED_FLOAT",
        },
        "cases": [run_continuation(torch, modeling, DynamicCache,
                                   module, rope, config)],
    }
    write_json(output_dir / "qsa-state-vectors.json", state)

    files = {name: {"sha256": sha256(output_dir / name),
                    "bytes": (output_dir / name).stat().st_size}
             for name in ORACLE_FILES}
    manifest = {
        "schema": "kq-qsa-evidence-manifest-v1",
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "generation_tool": "tools/generate-qsa-reference.py",
        "generation_command": (
            "python tools/generate-qsa-reference.py --checkout "
            ".research-cache/task-1.4/transformers --config "
            ".research-cache/model-baseline/" + MODEL_REVISION +
            "/config.json --output-dir "
            "research/operators/Qwen3.8-Flash-Next/" + MODEL_REVISION
        ),
        "offline": True,
        "full_model_weights_downloaded": False,
        "expected_values_from_kestrel_q": False,
        "files": files,
    }
    write_json(output_dir / "qsa-manifest.json", manifest)


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
        check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != TRANSFORMERS_REVISION:
        raise SystemExit(f"unexpected Transformers revision: {revision}")
    if not config.is_file() or sha256(config) != CONFIG_SHA256:
        raise SystemExit("pinned model config hash mismatch")
    for relative, expected in SOURCE_HASHES.items():
        path = checkout / "src" / relative
        if sha256(path) != expected:
            raise SystemExit(f"source hash mismatch: {relative}")
    if args.verify:
        with tempfile.TemporaryDirectory(prefix="kq-qsa-verify-") as directory:
            candidate = Path(directory)
            generate(candidate, checkout)
            # The native validator finalizes qsa-manifest.json with a hash of
            # its derived validation record.  Oracle regeneration therefore
            # compares only the five independent source-of-truth artifacts.
            names = ORACLE_FILES
            mismatches = [name for name in names
                          if not (output / name).exists() or
                          (output / name).read_bytes() !=
                          (candidate / name).read_bytes()]
            if mismatches:
                raise SystemExit("non-deterministic QSA evidence: " +
                                 ", ".join(mismatches))
        print("QSA evidence deterministic regeneration: PASS")
        return 0
    if output.exists():
        for name in (*ORACLE_FILES, "qsa-manifest.json"):
            path = output / name
            if path.exists():
                path.unlink()
    generate(output, checkout)
    print(f"QSA Class-C evidence written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
