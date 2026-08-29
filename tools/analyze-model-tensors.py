#!/usr/bin/env python3
"""Reconcile and classify the pinned Qwen Safetensors header inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPOSITORY_ID = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
EXPECTED_SHARDS = 131
EXPECTED_TENSORS = 1658
CLASSIFICATION_RULE_VERSION = "KQ-TENSOR-CLASSIFICATION-v1"
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
CSV_FIELDS = [
    "tensor_name",
    "shard",
    "dtype",
    "shape",
    "rank",
    "parameter_count",
    "payload_bytes",
    "subsystem",
    "component",
    "layer_id",
    "expert_id",
    "initial_text_scope",
    "classification_rule",
    "placement_candidate",
    "notes",
]


class AnalysisError(RuntimeError):
    """A deterministic reconciliation or classification error."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_totals(target: dict[str, dict[str, int]], key: str, parameters: int, payload: int) -> None:
    item = target.setdefault(key, {"parameter_count": 0, "payload_bytes": 0, "tensor_count": 0})
    item["parameter_count"] += parameters
    item["payload_bytes"] += payload
    item["tensor_count"] += 1


def quantization_floors(parameters: int) -> dict[str, dict[str, int]]:
    return {
        f"Q{bits}": {
            "payload_bits": parameters * bits,
            "payload_bytes_ceiling": (parameters * bits + 7) // 8,
        }
        for bits in (8, 6, 5, 4, 3)
    }


def classify(name: str) -> dict[str, str]:
    layer_match = re.search(r"(?:model\.language_model|mtp)\.layers\.(\d+)\.", name)
    layer_id = layer_match.group(1) if layer_match else ""
    expert_id = ""
    notes = ""

    if name.startswith("model.visual."):
        subsystem = "vision"
        scope = "EXCLUDED_INITIAL_VISION"
        placement = "EXCLUDED_INITIAL_SCOPE"
        if name.startswith("model.visual.patch_embed."):
            component = "vision_patch_embedding"
        elif name.startswith("model.visual.pos_embed."):
            component = "vision_position_embedding"
        elif name.startswith("model.visual.blocks."):
            component = "vision_block"
            block_match = re.search(r"model\.visual\.blocks\.(\d+)\.", name)
            layer_id = block_match.group(1) if block_match else ""
        elif name.startswith("model.visual.merger."):
            component = "multimodal_merger"
        else:
            raise AnalysisError(f"unclassified vision tensor: {name}")
    elif name.startswith("mtp."):
        subsystem = "mtp"
        scope = "EXCLUDED_INITIAL_MTP"
        placement = "EXCLUDED_INITIAL_SCOPE"
        if ".mlp.experts." in name:
            component = "mtp_routed_experts"
            expert_id = "STACKED_0_511"
            notes = "canonical tensor stacks all 512 routed experts"
        elif ".mlp.shared_expert" in name:
            component = "mtp_shared_expert"
        elif ".mlp.gate.weight" in name:
            component = "mtp_router"
        elif ".self_attn.indexer." in name:
            component = "mtp_qsa_indexer"
        elif ".self_attn." in name:
            component = "mtp_qsa_attention"
        elif "hyper_connection" in name:
            component = "mtp_gated_residual"
        elif name.startswith(("mtp.fc_", "mtp.pre_fc_norm_")):
            component = "mtp_input_fusion"
        else:
            raise AnalysisError(f"unclassified MTP tensor: {name}")
    else:
        scope = "REQUIRED_INITIAL_TEXT"
        if name == "model.language_model.embed_tokens.weight":
            subsystem, component, placement = "text", "token_embedding", "ALWAYS_NEEDED_CANDIDATE"
        elif name == "lm_head.weight":
            subsystem, component, placement = "text", "lm_head", "ALWAYS_NEEDED_CANDIDATE"
        elif name.startswith("model.language_model.hyper_connection_mixer."):
            subsystem, component, placement = "text", "final_gated_residual", "ALWAYS_NEEDED_CANDIDATE"
        elif ".ple." in name:
            subsystem, placement = "ple_ngram", "PLE_PREFETCH_CANDIDATE"
            if ".ngram_embedding.shard_" in name:
                component = "ple_embedding_table"
            elif name.endswith(("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")):
                component = "ple_address_metadata"
            else:
                component = "ple_dense"
        elif ".linear_attn." in name:
            subsystem, component, placement = "text", "gdn", "ALWAYS_NEEDED_CANDIDATE"
        elif ".self_attn.indexer." in name:
            subsystem, component, placement = "text", "qsa_indexer", "ALWAYS_NEEDED_CANDIDATE"
        elif ".self_attn." in name:
            subsystem, component, placement = "text", "qsa_attention", "ALWAYS_NEEDED_CANDIDATE"
        elif ".mlp.experts." in name:
            subsystem, component, placement = "text", "routed_experts", "ROUTED_EXPERT_CACHE_CANDIDATE"
            expert_id = "STACKED_0_511"
            notes = "canonical tensor stacks all 512 routed experts"
        elif ".mlp.shared_expert" in name:
            subsystem, component, placement = "text", "shared_expert", "ALWAYS_NEEDED_CANDIDATE"
        elif ".mlp.gate.weight" in name:
            subsystem, component, placement = "text", "router", "ALWAYS_NEEDED_CANDIDATE"
        elif "hyper_connection" in name:
            subsystem, component, placement = "text", "gated_residual", "ALWAYS_NEEDED_CANDIDATE"
        else:
            raise AnalysisError(f"unclassified initial-text tensor: {name}")

    return {
        "subsystem": subsystem,
        "component": component,
        "layer_id": layer_id,
        "expert_id": expert_id,
        "initial_text_scope": scope,
        "classification_rule": f"{CLASSIFICATION_RULE_VERSION}:{component}",
        "placement_candidate": placement,
        "notes": notes,
    }


def runtime_state() -> dict[str, Any]:
    gdn_recurrent = 36 * 48 * 128 * 128 * 4
    gdn_conv = 36 * 10240 * 4 * 2
    ple_history = 2 * 8
    ple_conv = 10240 * 9 * 2
    position_scalar = 8
    qsa_k_per_token = 12 * 2 * 256 * 2
    qsa_v_per_token = 12 * 2 * 256 * 2
    qsa_index_per_token = 12 * 128 * 2
    reference_position_ids_per_token = 3 * 8
    fixed = gdn_recurrent + gdn_conv + ple_history + ple_conv + position_scalar
    contexts = []
    for context in (1, 4096, 16384, 65536, 262144):
        semantic_context = context * (qsa_k_per_token + qsa_v_per_token + qsa_index_per_token)
        reference_positions = context * reference_position_ids_per_token
        contexts.append(
            {
                "context_tokens": context,
                "fixed_state_bytes": fixed,
                "qsa_k_bytes": context * qsa_k_per_token,
                "qsa_v_bytes": context * qsa_v_per_token,
                "qsa_raw_index_key_bytes": context * qsa_index_per_token,
                "qsa_semantic_context_bytes": semantic_context,
                "reference_cached_position_ids_bytes": reference_positions,
                "semantic_total_bytes": fixed + semantic_context,
                "pinned_reference_container_total_bytes": fixed + semantic_context + reference_positions,
            }
        )
    return {
        "batch": 1,
        "dtype_assumptions": {
            "gdn_recurrent": "F32",
            "gdn_conv": "BF16",
            "qsa_k_v": "BF16",
            "qsa_raw_index_keys": "BF16",
            "ple_history": "I64",
            "ple_conv": "BF16",
            "position_scalar": "I64",
            "reference_cached_position_ids": "I64",
        },
        "fixed_components": {
            "gdn_recurrent_matrix_bytes": gdn_recurrent,
            "gdn_convolution_state_bytes": gdn_conv,
            "ple_token_history_bytes": ple_history,
            "ple_dilated_convolution_state_bytes": ple_conv,
            "position_bookkeeping_bytes": position_scalar,
            "fixed_total_bytes": fixed,
        },
        "per_context_token": {
            "qsa_k_bytes": qsa_k_per_token,
            "qsa_v_bytes": qsa_v_per_token,
            "qsa_raw_index_key_bytes": qsa_index_per_token,
            "qsa_semantic_total_bytes": qsa_k_per_token + qsa_v_per_token + qsa_index_per_token,
            "reference_cached_position_ids_bytes": reference_position_ids_per_token,
        },
        "contexts": contexts,
        "notes": [
            "GR branches are current-forward activations and are excluded from persistent context state.",
            "The pinned reference stores three equivalent text position-ID axes; a native runtime may reconstruct them from scalar position state after proving equivalence.",
            "Optional vision and MTP state is outside the ADR-0005 initial scope.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--header-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.revision != REVISION:
        raise AnalysisError(f"revision must be exactly {REVISION}")
    index = json.loads(args.index.read_text(encoding="utf-8"))
    manifest = json.loads(args.header_manifest.read_text(encoding="utf-8"))
    if manifest.get("repository_id") != REPOSITORY_ID or manifest.get("revision") != REVISION:
        raise AnalysisError("header manifest repository/revision mismatch")
    if manifest.get("weight_payload_bytes_fetched") != 0:
        raise AnalysisError("header manifest does not prove zero weight-payload bytes")
    if manifest.get("shard_count") != EXPECTED_SHARDS or len(manifest.get("shards", [])) != EXPECTED_SHARDS:
        raise AnalysisError("header manifest does not contain exactly 131 shards")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != EXPECTED_TENSORS:
        raise AnalysisError("index does not contain exactly 1,658 tensors")

    remote_by_shard = {item["filename"]: item for item in manifest["shards"]}
    if len(remote_by_shard) != EXPECTED_SHARDS:
        raise AnalysisError("duplicate shard records")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    dtype_counts: Counter[str] = Counter()
    shard_payload_total = 0
    shard_file_total = 0
    header_region_total = 0
    shard_checks = []
    for shard_name in sorted(remote_by_shard):
        shard = remote_by_shard[shard_name]
        descriptors = shard.get("tensors")
        if not isinstance(descriptors, list) or len(descriptors) != shard.get("tensor_count"):
            raise AnalysisError(f"{shard_name}: invalid tensor descriptor list")
        ranges = []
        shard_payload = 0
        for descriptor in descriptors:
            name = descriptor.get("name")
            dtype = descriptor.get("dtype")
            shape = descriptor.get("shape")
            offsets = descriptor.get("data_offsets")
            if not isinstance(name, str) or name in seen:
                raise AnalysisError(f"duplicate/invalid tensor name: {name!r}")
            if weight_map.get(name) != shard_name:
                raise AnalysisError(f"{name}: index/header shard mismatch")
            if dtype not in DTYPE_BYTES:
                raise AnalysisError(f"{name}: unsupported dtype {dtype!r}")
            if not isinstance(shape, list) or any(not isinstance(dim, int) or dim < 0 for dim in shape):
                raise AnalysisError(f"{name}: invalid shape")
            if not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(value, int) for value in offsets):
                raise AnalysisError(f"{name}: invalid offsets")
            start, end = offsets
            if start < 0 or end < start:
                raise AnalysisError(f"{name}: invalid data interval")
            parameters = math.prod(shape)
            payload = end - start
            if payload != parameters * DTYPE_BYTES[dtype]:
                raise AnalysisError(f"{name}: dtype/shape payload mismatch")
            classification = classify(name)
            row = {
                "tensor_name": name,
                "shard": shard_name,
                "dtype": dtype,
                "shape": "x".join(str(dim) for dim in shape),
                "rank": len(shape),
                "parameter_count": parameters,
                "payload_bytes": payload,
                **classification,
            }
            rows.append(row)
            seen.add(name)
            dtype_counts[dtype] += 1
            shard_payload += payload
            ranges.append((start, end, name))
        ranges.sort()
        cursor = 0
        for start, end, name in ranges:
            if start != cursor:
                raise AnalysisError(f"{shard_name}: gap/overlap before {name}: {cursor} -> {start}")
            cursor = end
        header_region = 8 + shard["header_length_bytes"]
        expected_remote = header_region + cursor
        if expected_remote != shard["remote_size_bytes"]:
            raise AnalysisError(
                f"{shard_name}: header+payload {expected_remote} != remote size {shard['remote_size_bytes']}"
            )
        shard_payload_total += shard_payload
        shard_file_total += shard["remote_size_bytes"]
        header_region_total += header_region
        shard_checks.append(
            {
                "filename": shard_name,
                "tensor_count": len(descriptors),
                "payload_bytes": shard_payload,
                "header_region_bytes": header_region,
                "remote_size_bytes": shard["remote_size_bytes"],
                "reconciliation": "PASS",
            }
        )

    rows.sort(key=lambda item: item["tensor_name"])
    if seen != set(weight_map):
        raise AnalysisError("header tensor names do not exactly equal index tensor names")
    if len(rows) != EXPECTED_TENSORS:
        raise AnalysisError("inventory does not contain exactly 1,658 tensors")
    index_total = index.get("metadata", {}).get("total_size")
    if shard_payload_total != index_total:
        raise AnalysisError(f"payload total {shard_payload_total} != index total_size {index_total}")

    by_scope: dict[str, dict[str, int]] = {}
    by_subsystem: dict[str, dict[str, int]] = {}
    by_component: dict[str, dict[str, int]] = {}
    for row in rows:
        add_totals(by_scope, row["initial_text_scope"], row["parameter_count"], row["payload_bytes"])
        add_totals(by_subsystem, row["subsystem"], row["parameter_count"], row["payload_bytes"])
        add_totals(by_component, row["component"], row["parameter_count"], row["payload_bytes"])
    if set(by_scope) != {"REQUIRED_INITIAL_TEXT", "EXCLUDED_INITIAL_VISION", "EXCLUDED_INITIAL_MTP"}:
        raise AnalysisError(f"unexpected scope coverage: {sorted(by_scope)}")

    per_layer = []
    expert_sizes = []
    for layer in range(48):
        layer_rows = [row for row in rows if row["subsystem"] in {"text", "ple_ngram"} and row["layer_id"] == str(layer)]
        if not layer_rows:
            raise AnalysisError(f"text layer {layer} has no tensors")
        bucket = defaultdict(lambda: {"parameter_count": 0, "payload_bytes": 0})
        for row in layer_rows:
            if row["component"] == "routed_experts":
                key = "routed_experts"
            elif row["component"] == "shared_expert":
                key = "shared_expert"
            elif row["component"] == "router":
                key = "router"
            else:
                key = "non_expert"
            bucket[key]["parameter_count"] += row["parameter_count"]
            bucket[key]["payload_bytes"] += row["payload_bytes"]
        routed = bucket["routed_experts"]
        if routed["parameter_count"] % 512 or routed["payload_bytes"] % 512:
            raise AnalysisError(f"layer {layer}: routed-expert stack is not divisible by 512")
        per_expert_parameters = routed["parameter_count"] // 512
        per_expert_bytes = routed["payload_bytes"] // 512
        expert_sizes.append(per_expert_bytes)
        total_parameters = sum(item["parameter_count"] for item in bucket.values())
        total_bytes = sum(item["payload_bytes"] for item in bucket.values())
        per_layer.append(
            {
                "layer_id": layer,
                "layer_number": layer + 1,
                "layer_type": "QSA" if (layer + 1) % 4 == 0 else ("GDN+PLE" if layer == 1 else "GDN"),
                "non_expert": bucket["non_expert"],
                "router": bucket["router"],
                "routed_experts": routed,
                "shared_expert": bucket["shared_expert"],
                "per_routed_expert_parameter_count": per_expert_parameters,
                "per_routed_expert_payload_bytes": per_expert_bytes,
                "top10_selected_routed_expert_parameter_count": per_expert_parameters * 10,
                "top10_selected_routed_expert_payload_bytes": per_expert_bytes * 10,
                "total_parameter_count": total_parameters,
                "total_payload_bytes": total_bytes,
            }
        )
    selected_parameters = sum(item["top10_selected_routed_expert_parameter_count"] for item in per_layer)
    selected_bytes = sum(item["top10_selected_routed_expert_payload_bytes"] for item in per_layer)

    quant_families = {
        "full_checkpoint": {"parameter_count": sum(row["parameter_count"] for row in rows)},
        "initial_text_only": {"parameter_count": by_scope["REQUIRED_INITIAL_TEXT"]["parameter_count"]},
        "excluded_vision": {"parameter_count": by_scope["EXCLUDED_INITIAL_VISION"]["parameter_count"]},
        "excluded_mtp": {"parameter_count": by_scope["EXCLUDED_INITIAL_MTP"]["parameter_count"]},
        "ple_ngram": {"parameter_count": by_subsystem["ple_ngram"]["parameter_count"]},
        "routed_experts": {"parameter_count": by_component["routed_experts"]["parameter_count"]},
        "shared_experts": {"parameter_count": by_component["shared_expert"]["parameter_count"]},
        "dense_shared_non_routed_text": {
            "parameter_count": by_scope["REQUIRED_INITIAL_TEXT"]["parameter_count"]
            - by_subsystem["ple_ngram"]["parameter_count"]
            - by_component["routed_experts"]["parameter_count"]
        },
        "dense_non_routed_excluding_shared": {
            "parameter_count": by_scope["REQUIRED_INITIAL_TEXT"]["parameter_count"]
            - by_subsystem["ple_ngram"]["parameter_count"]
            - by_component["routed_experts"]["parameter_count"]
            - by_component["shared_expert"]["parameter_count"]
        },
    }
    for value in quant_families.values():
        value["idealized_lower_bounds"] = quantization_floors(value["parameter_count"])

    summary = {
        "schema_version": 1,
        "generated_by": "tools/analyze-model-tensors.py",
        "repository_id": REPOSITORY_ID,
        "revision": REVISION,
        "classification_rule_version": CLASSIFICATION_RULE_VERSION,
        "source_index_sha256": sha256_path(args.index),
        "source_header_manifest_sha256": sha256_path(args.header_manifest),
        "reconciliation": {
            "status": "PASS",
            "shard_count": EXPECTED_SHARDS,
            "tensor_count": len(rows),
            "unique_tensor_count": len(seen),
            "missing_index_tensors": 0,
            "extra_header_tensors": 0,
            "duplicate_tensors": 0,
            "unknown_or_review_tensors": 0,
            "index_metadata_total_size": index_total,
            "aggregate_payload_bytes": shard_payload_total,
            "aggregate_header_region_bytes": header_region_total,
            "aggregate_remote_shard_bytes": shard_file_total,
            "shards": shard_checks,
        },
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "totals": {
            "full_checkpoint": {
                "parameter_count": sum(row["parameter_count"] for row in rows),
                "payload_bytes": shard_payload_total,
                "tensor_count": len(rows),
            },
            "by_initial_text_scope": by_scope,
            "by_subsystem": by_subsystem,
            "by_component": by_component,
        },
        "per_layer": per_layer,
        "routed_expert_analysis": {
            "experts_per_layer": 512,
            "selected_experts_per_token_per_layer": 10,
            "per_expert_payload_bytes_min": min(expert_sizes),
            "per_expert_payload_bytes_max": max(expert_sizes),
            "per_expert_payload_bytes_median": int(statistics.median(expert_sizes)),
            "per_expert_parameter_count_min": min(
                item["per_routed_expert_parameter_count"] for item in per_layer
            ),
            "per_expert_parameter_count_max": max(
                item["per_routed_expert_parameter_count"] for item in per_layer
            ),
            "per_expert_parameter_count_median": int(
                statistics.median(item["per_routed_expert_parameter_count"] for item in per_layer)
            ),
            "uniform_across_layers": len(set(expert_sizes)) == 1,
            "top10_selected_across_48_layers_parameter_count": selected_parameters,
            "top10_selected_across_48_layers_payload_bytes": selected_bytes,
            "interpretation": "selected parameter payload only; not measured I/O and not throughput",
        },
        "idealized_quantization": quant_families,
        "runtime_state_batch1": runtime_state(),
    }

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tensor-inventory.csv").write_text(output.getvalue(), encoding="utf-8", newline="")
    (args.output_dir / "tensor-summary.json").write_bytes(canonical_json_bytes(summary))
    print(f"reconciled {EXPECTED_SHARDS} shards and {len(rows)} tensors")
    print(f"payload_bytes={shard_payload_total}")
    print("unknown_or_review_tensors=0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AnalysisError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
