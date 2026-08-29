#!/usr/bin/env python3
"""Deterministically reconcile canonical Qwen3.8 tensors with one pinned GGUF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CANONICAL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
UNSLOTH_REVISION = "c8b5954a88c2775c546b92593eda40ea041d3176"
LLAMA_CPP_REVISION = "90c26fcd4b2114b4aa39d09d69318cb8f438d27a"
EXPECTED_CANONICAL = 1658
EXPECTED_GGUF = 1224
EXPECTED_PACKED = 111_323_630_080

CANONICAL_FIELDS = [
    "canonical_tensor_name", "canonical_dtype", "canonical_shape", "canonical_parameter_count",
    "canonical_payload_bytes", "subsystem", "component", "layer_id", "expert_id",
    "initial_text_scope", "mapping_status", "gguf_tensor_names", "gguf_parameter_count",
    "gguf_packed_bytes_allocated", "mapping_rule", "evidence_ids", "confidence", "notes",
]
GGUF_EXTRA_FIELDS = [
    "canonical_tensor_names", "canonical_mapping_status", "canonical_component", "canonical_layer_id",
    "initial_text_scope", "packed_family", "mapping_rule", "evidence_ids",
]
BASE_GGUF_FIELDS = [
    "gguf_tensor_name", "rank", "dimensions", "parameter_count", "type_id", "type_name",
    "block_size", "bytes_per_block", "nominal_bits_per_parameter", "relative_offset",
    "absolute_offset", "packed_bytes", "padding_after_bytes", "payload_end_absolute",
]


class MappingError(RuntimeError):
    """Fail-closed mapping or reconciliation error."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.write_text(csv_text(rows, fields), encoding="utf-8", newline="")


def layer_id(name: str) -> int | None:
    match = re.search(r"model\.language_model\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def gguf_names_for(row: dict[str, str]) -> tuple[str, list[str], str, str, str]:
    name = row["tensor_name"]
    component = row["component"]
    layer = layer_id(name)
    if row["initial_text_scope"] in ("EXCLUDED_INITIAL_VISION", "EXCLUDED_INITIAL_MTP"):
        return "OMITTED_INITIAL_SCOPE_VALID", [], "KQ-GGUF-RULE-SCOPE-OMISSION-v1", "KQ-GGUF-EVIDENCE-005", "PROVEN_CANONICAL"
    if component == "ple_address_metadata":
        return "OMITTED_FORMAT_DERIVED", [], "KQ-GGUF-RULE-PLE-KV-v1", "KQ-GGUF-EVIDENCE-003", "PROVEN_CONVERTER"
    if component == "ple_embedding_table":
        return "FUSED_INTO_GGUF", ["per_layer_token_embd.weight"], "KQ-GGUF-RULE-PLE-FUSION-v1", "KQ-GGUF-EVIDENCE-003", "PROVEN_CONVERTER"
    if layer is None:
        top = {
            "lm_head.weight": "output.weight",
            "model.language_model.embed_tokens.weight": "token_embd.weight",
            "model.language_model.hyper_connection_mixer.hc_norm.weight": "output_hc_norm.weight",
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": "output_hc_down.weight",
            "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": "output_hc_up.weight",
        }
        if name not in top:
            raise MappingError(f"unmapped top-level canonical tensor: {name}")
        return "RENAMED_ONE_TO_ONE", [top[name]], "KQ-GGUF-RULE-TENSOR-NAME-v1", "KQ-GGUF-EVIDENCE-002", "PROVEN_CONVERTER"

    prefix = f"model.language_model.layers.{layer}."
    suffix = name.removeprefix(prefix)
    target_prefix = f"blk.{layer}."
    mapping = {
        "attn_hyper_connection.block_inject_weight.weight": "hc_attn_inject.weight",
        "attn_hyper_connection.hc_norm.weight": "hc_attn_norm.weight",
        "attn_hyper_connection.input_mix_weight_down.weight": "hc_attn_down.weight",
        "attn_hyper_connection.input_mix_weight_up.weight": "hc_attn_up.weight",
        "mlp_hyper_connection.block_inject_weight.weight": "hc_ffn_inject.weight",
        "mlp_hyper_connection.hc_norm.weight": "hc_ffn_norm.weight",
        "mlp_hyper_connection.input_mix_weight_down.weight": "hc_ffn_down.weight",
        "mlp_hyper_connection.input_mix_weight_up.weight": "hc_ffn_up.weight",
        "linear_attn.A_log": "ssm_a",
        "linear_attn.conv1d.weight": "ssm_conv1d.weight",
        "linear_attn.dt_bias": "ssm_dt.bias",
        "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
        "linear_attn.in_proj_b.weight": "ssm_beta.weight",
        "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
        "linear_attn.in_proj_z.weight": "attn_gate.weight",
        "linear_attn.norm.weight": "ssm_norm.weight",
        "linear_attn.out_proj.weight": "ssm_out.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.indexer.k_layernorm.weight": "indexer.k_norm.weight",
        "self_attn.indexer.q_layernorm.weight": "indexer.q_norm.weight",
        "mlp.experts.down_proj": "ffn_down_exps.weight",
        "mlp.gate.weight": "ffn_gate_inp.weight",
        "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
        "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
        "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
        "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
        "ple.conv1d.weight": "ple_conv1d.weight",
        "ple.key_proj.weight": "ple_key.weight",
        "ple.norm_conv.weight": "ple_norm_conv.weight",
        "ple.norm_key.weight": "ple_norm_key.weight",
        "ple.norm_query.weight": "ple_norm_query.weight",
        "ple.value_proj.weight": "ple_value.weight",
    }
    if suffix == "mlp.experts.gate_up_proj":
        return "SPLIT_IN_GGUF", [target_prefix + "ffn_gate_exps.weight", target_prefix + "ffn_up_exps.weight"], "KQ-GGUF-RULE-MOE-SPLIT-v1", "KQ-GGUF-EVIDENCE-004", "PROVEN_CONVERTER"
    if suffix == "self_attn.indexer.index_qk_proj.weight":
        return "SPLIT_IN_GGUF", [target_prefix + "indexer.q_proj.weight", target_prefix + "indexer.k_proj.weight"], "KQ-GGUF-RULE-QSA-SPLIT-v1", "KQ-GGUF-EVIDENCE-003", "PROVEN_CONVERTER"
    if suffix not in mapping:
        raise MappingError(f"unmapped layer tensor: {name}")

    transformed = False
    if suffix in {
        "linear_attn.A_log", "linear_attn.conv1d.weight", "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight", "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
        "linear_attn.out_proj.weight", "ple.conv1d.weight", "ple.norm_conv.weight",
        "ple.norm_key.weight", "ple.norm_query.weight",
    }:
        transformed = True
    status = "TRANSFORMED_LAYOUT" if transformed else "RENAMED_ONE_TO_ONE"
    rule = "KQ-GGUF-RULE-CONVERTER-TRANSFORM-v1" if transformed else "KQ-GGUF-RULE-TENSOR-NAME-v1"
    evidence = "KQ-GGUF-EVIDENCE-003" if transformed else "KQ-GGUF-EVIDENCE-002"
    confidence = "PROVEN_CONVERTER" if transformed else "PROVEN_CONVERTER"
    return status, [target_prefix + mapping[suffix]], rule, evidence, confidence


def family_for(component: str) -> str:
    return {
        "token_embedding": "token_embedding",
        "lm_head": "lm_head",
        "final_gated_residual": "gated_residual",
        "gated_residual": "gated_residual",
        "gdn": "gdn",
        "qsa_attention": "qsa_attention",
        "qsa_indexer": "qsa_indexer",
        "ple_address_metadata": "ple",
        "ple_embedding_table": "ple",
        "ple_dense": "ple",
        "router": "router",
        "routed_experts": "routed_experts",
        "shared_expert": "shared_experts",
        "vision_block": "vision",
        "vision_patch_embedding": "vision",
        "vision_position_embedding": "vision",
        "multimodal_merger": "vision",
        "mtp_gated_residual": "mtp",
        "mtp_input_fusion": "mtp",
        "mtp_qsa_attention": "mtp",
        "mtp_qsa_indexer": "mtp",
        "mtp_routed_experts": "mtp",
        "mtp_router": "mtp",
        "mtp_shared_expert": "mtp",
    }[component]


def transformation_note(canonical_name: str, status: str, has_targets: bool) -> str:
    if status == "OMITTED_INITIAL_SCOPE_VALID":
        return "Excluded by the accepted ADR-0005 initial text-only scope"
    if status == "OMITTED_FORMAT_DERIVED":
        return "Exact int64 PLE address constants are represented as GGUF metadata"
    if status == "FUSED_INTO_GGUF":
        return "One of 128 canonical PLE table shards concatenated in numeric shard order"
    if status == "SPLIT_IN_GGUF":
        return "Canonical projection is split along the converter-defined projection dimension"
    if canonical_name.endswith("linear_attn.A_log"):
        return "Converter stores -exp(A_log) in the one-to-one GGUF tensor"
    if canonical_name.endswith(("linear_attn.conv1d.weight", "ple.conv1d.weight")):
        return "Converter removes the singleton convolution dimension"
    if canonical_name.endswith((
        "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
        "linear_attn.out_proj.weight",
    )):
        return "Converter reorders grouped GDN value-head rows or columns to tiled GGML order"
    if canonical_name.endswith(("ple.norm_conv.weight", "ple.norm_key.weight", "ple.norm_query.weight")):
        return "Converter changes zero-centered norm gamma to stored gamma by adding one"
    if has_targets:
        return "GGUF dimensions use reversed physical dimension order"
    return "No standalone GGUF tensor"


def aggregate_family(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    canonical_parameters = sum(int(row["canonical_parameter_count"]) for row in rows)
    gguf_parameters = sum(int(row["parameter_count"]) for row in rows if "parameter_count" in row)
    packed_bytes = sum(int(row["packed_bytes"]) for row in rows if "packed_bytes" in row)
    return {
        "family": name,
        "canonical_parameter_count": canonical_parameters,
        "gguf_tensor_parameter_count": gguf_parameters,
        "packed_bytes": packed_bytes,
        "effective_bits_per_gguf_tensor_parameter": (packed_bytes * 8 / gguf_parameters) if gguf_parameters else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-inventory", required=True, type=Path)
    parser.add_argument("--gguf-metadata", required=True, type=Path)
    parser.add_argument("--gguf-inventory", required=True, type=Path)
    parser.add_argument("--upstream-split-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    canonical = read_csv(args.canonical_inventory)
    gguf = read_csv(args.gguf_inventory)
    metadata = json.loads(args.gguf_metadata.read_text(encoding="utf-8"))
    split_audit = json.loads(args.upstream_split_audit.read_text(encoding="utf-8"))
    if len(canonical) != EXPECTED_CANONICAL or len({r["tensor_name"] for r in canonical}) != EXPECTED_CANONICAL:
        raise MappingError("canonical inventory identity/count mismatch")
    if len(gguf) != EXPECTED_GGUF or len({r["gguf_tensor_name"] for r in gguf}) != EXPECTED_GGUF:
        raise MappingError("GGUF inventory identity/count mismatch")
    if any(field not in gguf[0] for field in BASE_GGUF_FIELDS):
        raise MappingError("GGUF inventory lacks required structural fields")
    normalized_base_gguf_sha256 = hashlib.sha256(
        csv_text(gguf, BASE_GGUF_FIELDS).encode("utf-8")
    ).hexdigest()
    if metadata["sha256_verification"] != "PASS_FULL_FILE":
        raise MappingError("canonical outputs require a full-file GGUF SHA-256 verification")
    if split_audit["revision"] != UNSLOTH_REVISION or split_audit["audit"]["tensor_payload_bytes_fetched"] != 0:
        raise MappingError("upstream split audit identity/safety mismatch")

    gguf_by_name = {row["gguf_tensor_name"]: row for row in gguf}
    canonical_rows: list[dict[str, Any]] = []
    reverse: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sorted(canonical, key=lambda item: item["tensor_name"]):
        status, names, rule, evidence, confidence = gguf_names_for(source)
        targets = []
        for name in names:
            target = gguf_by_name.get(name)
            if target is None:
                raise MappingError(f"mapped GGUF tensor absent: {source['tensor_name']} -> {name}")
            targets.append(target)
        source_parameters = int(source["parameter_count"])
        target_parameters = sum(int(target["parameter_count"]) for target in targets)
        if status in ("RENAMED_ONE_TO_ONE", "TRANSFORMED_LAYOUT", "SPLIT_IN_GGUF") and target_parameters != source_parameters:
            raise MappingError(f"parameter reconciliation failed: {source['tensor_name']}")
        allocated = sum(int(target["packed_bytes"]) for target in targets)
        if status == "FUSED_INTO_GGUF":
            allocated = source_parameters * int(targets[0]["packed_bytes"]) // int(targets[0]["parameter_count"])
        out = {
            "canonical_tensor_name": source["tensor_name"],
            "canonical_dtype": source["dtype"],
            "canonical_shape": source["shape"],
            "canonical_parameter_count": source_parameters,
            "canonical_payload_bytes": int(source["payload_bytes"]),
            "subsystem": source["subsystem"],
            "component": source["component"],
            "layer_id": source["layer_id"],
            "expert_id": source["expert_id"],
            "initial_text_scope": source["initial_text_scope"],
            "mapping_status": status,
            "gguf_tensor_names": ";".join(names),
            "gguf_parameter_count": target_parameters,
            "gguf_packed_bytes_allocated": allocated,
            "mapping_rule": rule,
            "evidence_ids": evidence,
            "confidence": confidence,
            "notes": transformation_note(source["tensor_name"], status, bool(names)),
        }
        canonical_rows.append(out)
        for name in names:
            reverse[name].append(out)

    ple_sources = [row for row in canonical_rows if row["mapping_status"] == "FUSED_INTO_GGUF"]
    ple_target = gguf_by_name["per_layer_token_embd.weight"]
    if sum(row["canonical_parameter_count"] for row in ple_sources) != int(ple_target["parameter_count"]):
        raise MappingError("PLE fusion parameters do not reconcile")
    if sum(row["gguf_packed_bytes_allocated"] for row in ple_sources) != int(ple_target["packed_bytes"]):
        raise MappingError("PLE fusion packed-byte allocation does not reconcile")
    if set(reverse) != set(gguf_by_name):
        missing = sorted(set(gguf_by_name) - set(reverse))
        raise MappingError(f"unexplained GGUF tensors: {missing[:5]}")

    enriched = []
    for base in sorted(gguf, key=lambda item: item["gguf_tensor_name"]):
        mapped = reverse[base["gguf_tensor_name"]]
        components = sorted({row["component"] for row in mapped})
        families = sorted({family_for(component) for component in components})
        if len(families) != 1:
            raise MappingError(f"ambiguous GGUF family: {base['gguf_tensor_name']}")
        enriched.append(
            {
                **base,
                "canonical_tensor_names": ";".join(row["canonical_tensor_name"] for row in mapped),
                "canonical_mapping_status": ";".join(sorted({row["mapping_status"] for row in mapped})),
                "canonical_component": ";".join(components),
                "canonical_layer_id": ";".join(sorted({row["layer_id"] for row in mapped if row["layer_id"]}, key=int)),
                "initial_text_scope": ";".join(sorted({row["initial_text_scope"] for row in mapped})),
                "packed_family": families[0],
                "mapping_rule": ";".join(sorted({row["mapping_rule"] for row in mapped})),
                "evidence_ids": ";".join(sorted({row["evidence_ids"] for row in mapped})),
            }
        )

    statuses = Counter(row["mapping_status"] for row in canonical_rows)
    if statuses["UNRESOLVED"] or len(canonical_rows) != EXPECTED_CANONICAL:
        raise MappingError("canonical mapping coverage failed")
    packed_total = sum(int(row["packed_bytes"]) for row in enriched)
    if packed_total != EXPECTED_PACKED:
        raise MappingError(f"packed byte mismatch: {packed_total}")

    canonical_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_rows:
        canonical_by_family[family_for(row["component"])].append(row)
    gguf_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        gguf_by_family[row["packed_family"]].append(row)
    family_names = ["token_embedding", "lm_head", "ple", "gdn", "qsa_attention", "qsa_indexer", "gated_residual", "router", "routed_experts", "shared_experts", "vision", "mtp"]
    families = {}
    for family in family_names:
        canonical_parameters = sum(row["canonical_parameter_count"] for row in canonical_by_family[family])
        canonical_stored_bytes = sum(row["canonical_payload_bytes"] for row in canonical_by_family[family])
        gguf_parameters = sum(int(row["parameter_count"]) for row in gguf_by_family[family])
        packed_bytes = sum(int(row["packed_bytes"]) for row in gguf_by_family[family])
        types = Counter(row["type_name"] for row in gguf_by_family[family])
        if not types:
            precision_policy = "NOT_PRESENT_INITIAL_SCOPE"
        elif set(types).issubset({"F32", "BF16"}):
            precision_policy = "UNQUANTIZED"
        elif len(types) == 1:
            precision_policy = "OBSERVED_LOWER_PRECISION"
        else:
            precision_policy = "MIXED"
        families[family] = {
            "canonical_parameter_count": canonical_parameters,
            "canonical_stored_payload_bytes": canonical_stored_bytes,
            "idealized_q4_payload_floor_bytes": (canonical_parameters * 4 + 7) // 8,
            "gguf_tensor_parameter_count": gguf_parameters,
            "packed_bytes": packed_bytes,
            "effective_bits_per_gguf_tensor_parameter": packed_bytes * 8 / gguf_parameters if gguf_parameters else None,
            "gguf_tensor_count": len(gguf_by_family[family]),
            "gguf_tensor_count_by_type": dict(sorted(types.items())),
            "packed_share_of_all_tensors": packed_bytes / EXPECTED_PACKED,
            "precision_policy_observation": precision_policy,
        }
    dense_families = {"token_embedding", "lm_head", "gdn", "qsa_attention", "qsa_indexer", "gated_residual", "router"}
    families["dense_non_routed_excluding_shared"] = {
        "canonical_parameter_count": sum(families[name]["canonical_parameter_count"] for name in dense_families),
        "canonical_stored_payload_bytes": sum(families[name]["canonical_stored_payload_bytes"] for name in dense_families),
        "idealized_q4_payload_floor_bytes": sum(families[name]["idealized_q4_payload_floor_bytes"] for name in dense_families),
        "gguf_tensor_parameter_count": sum(families[name]["gguf_tensor_parameter_count"] for name in dense_families),
        "packed_bytes": sum(families[name]["packed_bytes"] for name in dense_families),
        "effective_bits_per_gguf_tensor_parameter": None,
        "gguf_tensor_count": sum(families[name]["gguf_tensor_count"] for name in dense_families),
        "packed_share_of_all_tensors": None,
        "precision_policy_observation": "MIXED",
    }
    dense = families["dense_non_routed_excluding_shared"]
    dense["effective_bits_per_gguf_tensor_parameter"] = dense["packed_bytes"] * 8 / dense["gguf_tensor_parameter_count"]
    dense["packed_share_of_all_tensors"] = dense["packed_bytes"] / EXPECTED_PACKED
    families["dense_shared_non_routed_text"] = {
        "canonical_parameter_count": dense["canonical_parameter_count"] + families["shared_experts"]["canonical_parameter_count"],
        "canonical_stored_payload_bytes": dense["canonical_stored_payload_bytes"] + families["shared_experts"]["canonical_stored_payload_bytes"],
        "idealized_q4_payload_floor_bytes": dense["idealized_q4_payload_floor_bytes"] + families["shared_experts"]["idealized_q4_payload_floor_bytes"],
        "gguf_tensor_parameter_count": dense["gguf_tensor_parameter_count"] + families["shared_experts"]["gguf_tensor_parameter_count"],
        "packed_bytes": dense["packed_bytes"] + families["shared_experts"]["packed_bytes"],
        "gguf_tensor_count": dense["gguf_tensor_count"] + families["shared_experts"]["gguf_tensor_count"],
        "packed_share_of_all_tensors": None,
        "precision_policy_observation": "MIXED",
    }
    dense_shared = families["dense_shared_non_routed_text"]
    dense_shared["effective_bits_per_gguf_tensor_parameter"] = dense_shared["packed_bytes"] * 8 / dense_shared["gguf_tensor_parameter_count"]
    dense_shared["packed_share_of_all_tensors"] = dense_shared["packed_bytes"] / EXPECTED_PACKED

    type_summary: dict[str, dict[str, Any]] = {}
    for type_name in sorted({row["type_name"] for row in enriched}):
        rows = [row for row in enriched if row["type_name"] == type_name]
        parameters = sum(int(row["parameter_count"]) for row in rows)
        packed = sum(int(row["packed_bytes"]) for row in rows)
        type_summary[type_name] = {
            "tensor_count": len(rows), "parameter_count": parameters, "packed_bytes": packed,
            "effective_bits_per_parameter": packed * 8 / parameters,
        }

    per_layer = {}
    for layer in range(48):
        rows = [row for row in enriched if row["canonical_layer_id"] == str(layer)]
        per_layer[str(layer)] = {
            "gguf_tensor_count": len(rows),
            "parameter_count": sum(int(row["parameter_count"]) for row in rows),
            "packed_bytes": sum(int(row["packed_bytes"]) for row in rows),
            "packed_bytes_by_family": {
                family: sum(int(row["packed_bytes"]) for row in rows if row["packed_family"] == family)
                for family in sorted({row["packed_family"] for row in rows})
            },
        }

    routed_params_per_expert = families["routed_experts"]["gguf_tensor_parameter_count"] // (48 * 512)
    routed_layer_selection = {}
    for layer in range(48):
        layer_packed = per_layer[str(layer)]["packed_bytes_by_family"]["routed_experts"]
        if layer_packed % 512:
            raise MappingError(f"layer {layer} routed packed bytes are not expert-divisible")
        bytes_per_expert = layer_packed // 512
        routed_layer_selection[str(layer)] = {
            "parameter_count_per_expert": routed_params_per_expert,
            "packed_bytes_per_expert": bytes_per_expert,
            "effective_bits_per_parameter": bytes_per_expert * 8 / routed_params_per_expert,
            "top10_selected_parameter_count": routed_params_per_expert * 10,
            "top10_selected_packed_bytes": bytes_per_expert * 10,
        }
    routed_bytes_across_layers_per_expert = families["routed_experts"]["packed_bytes"] // 512

    physical = sorted(enriched, key=lambda row: int(row["relative_offset"]))
    locality = {}
    for family in family_names:
        family_rows = [row for row in physical if row["packed_family"] == family]
        runs = 0
        previous_index = None
        for index, row in enumerate(physical):
            if row["packed_family"] != family:
                continue
            if previous_index is None or index != previous_index + 1:
                runs += 1
            previous_index = index
        locality[family] = {
            "physical_runs": runs,
            "first_absolute_offset": int(family_rows[0]["absolute_offset"]) if family_rows else None,
            "last_payload_end_absolute": int(family_rows[-1]["payload_end_absolute"]) if family_rows else None,
        }
    split_counts = [item["tensor_count"] for item in split_audit["shards"]]
    if split_counts != [0, 297, 752, 175]:
        raise MappingError(f"unexpected upstream split tensor partition: {split_counts}")
    cursor = 0
    split_layout = []
    for shard, count in zip(split_audit["shards"], split_counts):
        group = physical[cursor:cursor + count]
        packed = sum(int(row["packed_bytes"]) for row in group)
        overhead = shard["file_size_bytes"] - packed
        split_layout.append(
            {
                "filename": shard["filename"], "tensor_count": count,
                "first_merged_tensor": group[0]["gguf_tensor_name"] if group else None,
                "last_merged_tensor": group[-1]["gguf_tensor_name"] if group else None,
                "packed_tensor_bytes": packed, "file_size_bytes": shard["file_size_bytes"],
                "header_directory_overhead_bytes": overhead,
            }
        )
        cursor += count
    upstream_total = sum(item["file_size_bytes"] for item in split_audit["shards"])
    upstream_overhead = upstream_total - packed_total
    local_overhead = metadata["structural_summary"]["file_overhead_bytes"]
    delta = upstream_total - metadata["file_size_bytes"]
    if cursor != EXPECTED_GGUF or delta != 384 or upstream_overhead - local_overhead != 384:
        raise MappingError("split/merged 384-byte reconciliation failed")

    count_reconciliation = [
        {"cause": "canonical total", "delta": 1658, "running_total": 1658},
        {"cause": "vision omitted by ADR-0005 initial scope", "delta": -333, "running_total": 1325},
        {"cause": "MTP omitted by ADR-0005 initial scope", "delta": -31, "running_total": 1294},
        {"cause": "three PLE int64 address tensors represented as GGUF metadata", "delta": -3, "running_total": 1291},
        {"cause": "128 PLE table shards fused to one GGUF tensor", "delta": -127, "running_total": 1164},
        {"cause": "48 MoE gate_up tensors split into gate and up tensors", "delta": 48, "running_total": 1212},
        {"cause": "12 QSA index_qk tensors split into q and k tensors", "delta": 12, "running_total": 1224},
    ]

    metadata_values = {item["key"]: item["value"] for item in metadata["metadata"]}
    initial_canonical = [row for row in canonical_rows if row["initial_text_scope"] == "REQUIRED_INITIAL_TEXT"]
    artifact_footprints = {
        "canonical_full_checkpoint": {
            "parameter_count": sum(row["canonical_parameter_count"] for row in canonical_rows),
            "stored_payload_bytes": sum(row["canonical_payload_bytes"] for row in canonical_rows),
        },
        "canonical_initial_text_scope": {
            "parameter_count": sum(row["canonical_parameter_count"] for row in initial_canonical),
            "stored_payload_bytes": sum(row["canonical_payload_bytes"] for row in initial_canonical),
        },
        "gguf_initial_text_scope": {
            "tensor_parameter_count": sum(int(row["parameter_count"]) for row in enriched),
            "tensor_packed_bytes": packed_total,
            "file_overhead_bytes": metadata["structural_summary"]["file_overhead_bytes"],
            "total_file_bytes": metadata["file_size_bytes"],
        },
        "excluded_vision": families["vision"],
        "excluded_mtp": families["mtp"],
    }
    summary = {
        "schema_version": 1,
        "generated_by": "tools/map-canonical-to-gguf.py",
        "canonical_revision": CANONICAL_REVISION,
        "unsloth_revision": UNSLOTH_REVISION,
        "llama_cpp_revision": LLAMA_CPP_REVISION,
        "artifact": {
            "artifact_id": metadata["artifact_id"], "filename": metadata["filename"],
            "file_size_bytes": metadata["file_size_bytes"], "sha256": metadata["sha256"],
            "version": metadata["version"], "architecture": metadata["architecture"],
            "metadata_count": metadata["metadata_count"], "tensor_count": metadata["tensor_count"],
        },
        "structure": metadata["structural_summary"],
        "artifact_footprints": artifact_footprints,
        "mapping_coverage": {
            "canonical_total": len(canonical_rows), "gguf_total": len(enriched),
            "canonical_unresolved": statuses["UNRESOLVED"], "unexplained_gguf": 0,
            "canonical_status_counts": dict(sorted(statuses.items())),
        },
        "tensor_count_reconciliation": count_reconciliation,
        "family_footprints": families,
        "family_physical_locality": locality,
        "type_summary": type_summary,
        "per_layer": per_layer,
        "routed_expert_selection": {
            "experts_per_layer": 512, "selected_experts_per_token": 10,
            "parameter_count_per_expert_per_layer": routed_params_per_expert,
            "per_layer": routed_layer_selection,
            "top10_selected_parameter_count_across_48_layers": routed_params_per_expert * 10 * 48,
            "top10_selected_packed_bytes_across_48_layers": routed_bytes_across_layers_per_expert * 10,
            "interpretation": "selected parameter payload only; not measured I/O, residency, cache traffic, or throughput",
        },
        "quantization_provenance": {
            "quantized_by": metadata_values["general.quantized_by"],
            "quantization_version": metadata_values["general.quantization_version"],
            "general_file_type": metadata_values["general.file_type"],
            "imatrix_file": metadata_values["quantize.imatrix.file"],
            "imatrix_dataset": metadata_values["quantize.imatrix.dataset"],
            "imatrix_entries_count": metadata_values["quantize.imatrix.entries_count"],
            "imatrix_chunks_count": metadata_values["quantize.imatrix.chunks_count"],
            "limitation": "metadata establishes calibration provenance, not per-family sensitivity or quality",
        },
        "split_merged_investigation": {
            "upstream_shard_total_bytes": upstream_total,
            "local_merged_bytes": metadata["file_size_bytes"],
            "difference_bytes": delta,
            "upstream_aggregate_header_directory_overhead_bytes": upstream_overhead,
            "local_header_directory_overhead_bytes": local_overhead,
            "packed_tensor_bytes_both_representations": packed_total,
            "result": "PROVEN_FORMAT_OVERHEAD_ONLY",
            "shards": split_layout,
            "network_tensor_payload_bytes_fetched": split_audit["audit"]["tensor_payload_bytes_fetched"],
        },
    }

    evidence = {
        "schema_version": 1,
        "generated_by": "tools/map-canonical-to-gguf.py",
        "format_and_converter_sources": {
            "repository": "ggml-org/llama.cpp",
            "revision": LLAMA_CPP_REVISION,
            "license": "MIT",
            "license_file_sha256": "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d",
            "files": [
                {"path": "gguf-py/gguf/constants.py", "sha256": "425024d367b166444f39ed724bd01b3ef9276775a68e6c71519d6866c7701ab8", "evidence_role": "GGUF/GGML type IDs, block geometry, Qwen4Exp tensor symbols"},
                {"path": "gguf-py/gguf/tensor_mapping.py", "sha256": "0bff4f10215bbf76dd0ab5762e3548f4929da6a8f0cdb51bfc8c675801ee0101", "evidence_role": "canonical converter name patterns to GGUF tensor symbols"},
                {"path": "conversion/qwen4exp.py", "sha256": "12a0a5aea7877fbb8fe35af041a9c34f8b57b05278871b22c24c650b9760dfc3", "evidence_role": "PLE fusion/KV conversion, QSA split, PLE transforms, initial-scope MTP omission"},
                {"path": "conversion/qwen.py", "sha256": "3e3ea6be268c65915f8f5b8419edc15c291f0adf374b97216f436191d23456b4", "evidence_role": "aggregated MoE gate_up split and GDN layout transforms"},
                {"path": "tools/gguf-split/gguf-split.cpp", "sha256": "86484d1c25843ef5abd01f9d1523b707fa60ad13905a5912d949201791681347", "evidence_role": "split metadata and payload-preserving merge semantics"}
            ]
        },
        "unsloth_source": {
            "repository": "unsloth/Qwen3.8-Flash-Next-GGUF",
            "revision": UNSLOTH_REVISION,
            "readme_sha256": "abe002ccf7cefcfcae3587c7f863ad6287147c6ebedf3ccf967faf97035baaab",
            "license_boundary": "model artifact remains under the upstream Qwen Community License 1.0 boundary documented by Task 1.0"
        },
        "claims": [
            {"id": "KQ-GGUF-EVIDENCE-001", "claim": "Canonical identity and composition come from the pinned Safetensors inventory.", "source": str(args.canonical_inventory).replace("\\", "/"), "sha256": file_sha256(args.canonical_inventory)},
            {"id": "KQ-GGUF-EVIDENCE-002", "claim": "GGUF type, name, block-size and tensor-name mappings are pinned to llama.cpp.", "source": "ggml-org/llama.cpp gguf-py/gguf/constants.py and tensor_mapping.py", "revision": LLAMA_CPP_REVISION},
            {"id": "KQ-GGUF-EVIDENCE-003", "claim": "Qwen4Exp converter defines PLE fusion/KV representation, QSA split and converter transforms.", "source": "ggml-org/llama.cpp conversion/qwen4exp.py", "revision": LLAMA_CPP_REVISION, "sha256": "12a0a5aea7877fbb8fe35af041a9c34f8b57b05278871b22c24c650b9760dfc3"},
            {"id": "KQ-GGUF-EVIDENCE-004", "claim": "Qwen MoE converter splits canonical aggregated gate_up expert weights.", "source": "ggml-org/llama.cpp conversion/qwen.py", "revision": LLAMA_CPP_REVISION, "sha256": "3e3ea6be268c65915f8f5b8419edc15c291f0adf374b97216f436191d23456b4"},
            {"id": "KQ-GGUF-EVIDENCE-005", "claim": "ADR-0005 defines vision and MTP as outside the initial text runtime scope.", "source": "docs/adr/0005-initial-text-only-runtime-scope.md"},
            {"id": "KQ-GGUF-EVIDENCE-006", "claim": "The pinned local artifact supplies observed metadata, tensor directory, types and offsets.", "source": "KQ-MODEL-ARTIFACT-001", "sha256": metadata["sha256"]},
            {"id": "KQ-GGUF-EVIDENCE-007", "claim": "Pinned upstream shard headers and llama.cpp merge semantics reconcile the 384-byte difference as format overhead.", "source": "unsloth/Qwen3.8-Flash-Next-GGUF and llama.cpp tools/gguf-split/gguf-split.cpp", "unsloth_revision": UNSLOTH_REVISION, "llama_cpp_revision": LLAMA_CPP_REVISION, "llama_cpp_source_sha256": "86484d1c25843ef5abd01f9d1523b707fa60ad13905a5912d949201791681347"},
        ],
        "input_hashes": {
            "canonical_inventory_sha256": file_sha256(args.canonical_inventory),
            "gguf_metadata_sha256": file_sha256(args.gguf_metadata),
            "normalized_base_gguf_inventory_sha256": normalized_base_gguf_sha256,
            "upstream_split_audit_sha256": file_sha256(args.upstream_split_audit),
        },
        "upstream_split_range_audit": split_audit,
        "safety": {
            "local_tensor_payload_bytes_read_by_inspector": 0,
            "upstream_tensor_payload_bytes_fetched": 0,
            "model_artifacts_written_to_repository": 0,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "canonical-gguf-mapping.csv", canonical_rows, CANONICAL_FIELDS)
    write_csv(args.output_dir / "gguf-tensor-inventory.csv", enriched, BASE_GGUF_FIELDS + GGUF_EXTRA_FIELDS)
    (args.output_dir / "gguf-summary.json").write_bytes(canonical_json(summary))
    (args.output_dir / "mapping-evidence.json").write_bytes(canonical_json(evidence))
    print(
        f"canonical={len(canonical_rows)} gguf={len(enriched)} unresolved={statuses['UNRESOLVED']} "
        f"unexplained_gguf=0 packed_bytes={packed_total} split_delta={delta}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MappingError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        raise SystemExit(1)
