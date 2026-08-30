#!/usr/bin/env python3
"""Compare the native scalar GDN with independent pinned Class-C evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_REVISION = "805a9e939fa8c1bff8d8ffdf041c051b71a914aa"
ORACLE_FILES = (
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


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json(value))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_bits(record: dict[str, Any]) -> list[int]:
    return [int.from_bytes(bytes.fromhex(value), "little") for value in record["f32_le_hex"]]


def record_from_standard_bits(bits: list[int], shape: list[int]) -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": shape,
        "f32_le_hex": [
            int(value).to_bytes(4, "little").hex() for value in bits
        ],
    }


def bits_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def ordered_bits(bits: int) -> int:
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else (bits | 0x80000000)


def append_array(lines: list[str], label: str, bits: Iterable[int]) -> None:
    values = list(bits)
    lines.append(f"{label} {len(values)} " + " ".join(f"{value:08x}" for value in values))


def standard_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case["id"],
        "input": case["input"],
        "mask": case.get("padding_mask"),
        "initial_conv": case["initial_state"]["conv"],
        "initial_recurrent": case["initial_state"]["recurrent"],
        "initialized": bool(case["initial_state"]["initialized"]),
        "expected_output": case["expected"]["output"],
        "expected_conv": case["expected"]["conv_state"],
        "expected_recurrent": case["expected"]["recurrent_state"],
        "checkpoints": case["expected"]["checkpoints"],
    }


def continuation_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case["id"],
        "input": case["input"],
        "mask": None,
        "initial_conv": case["before_conv_state"],
        "initial_recurrent": case["before_recurrent_state"],
        "initialized": True,
        "expected_output": case["expected_output"],
        "expected_conv": case["expected_conv_state"],
        "expected_recurrent": case["expected_recurrent_state"],
        "checkpoints": case["checkpoints"],
    }


def make_request(contract: dict[str, Any], case: dict[str, Any]) -> str:
    config = contract["reduced_config"]
    sequence = case["input"]["shape"][1]
    lines = [
        "KQGDN1",
        "CONFIG {hidden_size} {key_heads} {value_heads} {key_head_dim} "
        "{value_head_dim} {conv_kernel} {sequence} {initialized}".format(
            **config, sequence=sequence, initialized=int(case["initialized"])
        ),
    ]
    weights = contract["weights"]
    append_array(lines, "A_LOG", evidence_bits(weights["A_log"]))
    append_array(lines, "CONV", evidence_bits(weights["conv1d.weight"]))
    append_array(lines, "DT_BIAS", evidence_bits(weights["dt_bias"]))
    append_array(lines, "ALPHA", evidence_bits(weights["in_proj_a.weight"]))
    append_array(lines, "BETA", evidence_bits(weights["in_proj_b.weight"]))
    append_array(lines, "QKV", evidence_bits(weights["in_proj_qkv.weight"]))
    append_array(lines, "GATE", evidence_bits(weights["in_proj_z.weight"]))
    append_array(lines, "NORM", evidence_bits(weights["norm.weight"]))
    append_array(lines, "OUTPUT_WEIGHT", evidence_bits(weights["out_proj.weight"]))
    append_array(lines, "INITIAL_CONV", evidence_bits(case["initial_conv"]))
    append_array(lines, "INITIAL_RECURRENT", evidence_bits(case["initial_recurrent"]))
    append_array(lines, "INPUT", evidence_bits(case["input"]))
    if case["mask"] is None:
        lines.append("MASK 0")
    else:
        lines.append(f"MASK {len(case['mask'])} " + " ".join(str(value) for value in case["mask"]))
    return "\n".join(lines) + "\n"


def run_probe(probe: Path, contract: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [str(probe)], input=make_request(contract, case), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"native GDN probe failed for {case['id']}: rc={result.returncode}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    parsed: dict[str, Any] = {"traces": {}}
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "TRACE":
            kind = fields[1]
            token = int(fields[2])
            rank = int(fields[3])
            dimensions = [int(value) for value in fields[4:4 + rank]]
            count_index = 4 + rank
            count = int(fields[count_index])
            values = [int(value, 16) for value in fields[count_index + 1:]]
            if len(values) != count:
                raise RuntimeError(f"malformed trace count for {case['id']} {kind}")
            parsed["traces"][(kind, token)] = {
                "dimensions": dimensions,
                "bits": values,
            }
        elif fields[0] == "RESULT":
            if fields[1] != "OK":
                raise RuntimeError(f"native probe returned error for {case['id']}: {line}")
            parsed["result"] = "OK"
        elif fields[0] in {"OUTPUT", "FINAL_CONV", "FINAL_RECURRENT"}:
            count = int(fields[1])
            values = [int(value, 16) for value in fields[2:]]
            if len(values) != count:
                raise RuntimeError(f"malformed {fields[0]} count for {case['id']}")
            parsed[fields[0].lower()] = values
        elif fields[0] == "INITIALIZED":
            parsed["initialized"] = int(fields[1])
        elif fields[0] == "METRICS":
            parsed["metrics"] = {
                "config_owned_bytes": int(fields[1]),
                "state_owned_bytes": int(fields[2]),
                "scratch_bytes": int(fields[3]),
                "token_scratch_bytes": int(fields[4]),
                "dequant_scratch_bytes": int(fields[5]),
            }
        elif fields[0] == "TIMING":
            parsed["execution_nanoseconds"] = int(fields[1])
    required = {"result", "output", "final_conv", "final_recurrent", "initialized", "metrics", "execution_nanoseconds"}
    missing = required - parsed.keys()
    if missing:
        raise RuntimeError(f"native probe output missing {sorted(missing)} for {case['id']}")
    return parsed


def slice_axis(record: dict[str, Any], axis: int, index: int) -> list[int]:
    shape = record["shape"]
    values = evidence_bits(record)
    before = math.prod(shape[:axis])
    after = math.prod(shape[axis + 1:])
    axis_size = shape[axis]
    if before != 1:
        raise ValueError(f"unsupported leading product {before} for shape {shape}")
    if index < 0 or index >= axis_size:
        raise IndexError(index)
    return values[index * after:(index + 1) * after]


def extract_conv_output(record: dict[str, Any], token: int, sequence: int) -> list[int]:
    shape = record["shape"]
    values = evidence_bits(record)
    channels = shape[1]
    total_sequence = shape[2]
    source_token = total_sequence - sequence + token
    return [values[channel * total_sequence + source_token] for channel in range(channels)]


def expected_trace(case: dict[str, Any], kind: str, token: int) -> list[int]:
    checkpoints = case["checkpoints"]
    sequence = case["input"]["shape"][1]
    direct = {
        "MASKED_INPUT": None,
        "PROJECTED_QKV": "projected_qkv",
        "PROJECTED_GATE": "projected_gate",
        "PROJECTED_BETA": "projected_beta",
        "PROJECTED_ALPHA": "projected_alpha",
        "QUERY_BEFORE_NORM": "query_before_norm",
        "KEY_BEFORE_NORM": "key_before_norm",
        "VALUE": "value",
        "NORMALIZED_SCALED_QUERY": "normalized_scaled_query",
        "NORMALIZED_KEY": "normalized_key",
        "LOG_DECAY": "log_decay",
        "BETA": "beta",
        "RECURRENT_READ": "sequential_read",
        "RECURRENT_DELTA": "sequential_delta",
        "RECURRENT_OUTPUT": "sequential_core_output",
        "GATED_NORM_OUTPUT": "gated_norm_output",
        "OPERATOR_OUTPUT": "operator_output",
    }
    if kind == "MASKED_INPUT":
        values = evidence_bits(case["input"])
        width = case["input"]["shape"][2]
        token_values = values[token * width:(token + 1) * width]
        if case["mask"] is not None and case["mask"][token] == 0:
            return [float_bits(0.0)] * width
        return token_values
    if kind == "CONV_INPUT":
        return slice_axis(checkpoints["projected_qkv"], 1, token)
    if kind == "CONV_OUTPUT":
        return extract_conv_output(checkpoints["conv_output"], token, sequence)
    if kind == "RECURRENT_STATE":
        return slice_axis(checkpoints["sequential_state_by_token"], 0, token)
    if kind == "CONV_STATE":
        return slice_axis(checkpoints["sequential_conv_state_by_token"], 0, token)
    name = direct.get(kind)
    if name is None:
        raise KeyError(kind)
    record = checkpoints[name]
    shape = record["shape"]
    if name == "gated_norm_output":
        heads = 4
        width = shape[-1]
        values = evidence_bits(record)
        start = token * heads * width
        return values[start:start + heads * width]
    return slice_axis(record, 1, token)


def comparison_stats(expected: list[int], actual: list[int]) -> dict[str, Any]:
    if len(expected) != len(actual):
        raise ValueError(f"comparison length mismatch: {len(expected)} != {len(actual)}")
    max_abs = 0.0
    max_rel = 0.0
    max_ulp = 0
    max_expected = 0.0
    exact = True
    for expected_bits, actual_bits in zip(expected, actual):
        expected_value = bits_float(expected_bits)
        actual_value = bits_float(actual_bits)
        if not math.isfinite(expected_value) or not math.isfinite(actual_value):
            raise ValueError("non-finite value in GDN comparison")
        exact = exact and expected_bits == actual_bits
        absolute = abs(actual_value - expected_value)
        relative = absolute / max(abs(expected_value), 1.0e-30)
        ulp = abs(ordered_bits(actual_bits) - ordered_bits(expected_bits))
        max_abs = max(max_abs, absolute)
        max_rel = max(max_rel, relative)
        max_ulp = max(max_ulp, ulp)
        max_expected = max(max_expected, abs(expected_value))
    return {
        "count": len(expected),
        "exact_bits": exact,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "max_ulp": max_ulp,
        "max_expected_abs": max_expected,
    }


def merge_stats(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    target["count"] += incoming["count"]
    target["exact_bits"] = target["exact_bits"] and incoming["exact_bits"]
    target["max_abs"] = max(target["max_abs"], incoming["max_abs"])
    target["max_rel"] = max(target["max_rel"], incoming["max_rel"])
    target["max_ulp"] = max(target["max_ulp"], incoming["max_ulp"])
    target["max_expected_abs"] = max(target["max_expected_abs"], incoming["max_expected_abs"])


def empty_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "exact_bits": True,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "max_ulp": 0,
        "max_expected_abs": 0.0,
    }


def compare_case(case: dict[str, Any], native: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    observed["FINAL_OUTPUT"] = comparison_stats(
        evidence_bits(case["expected_output"]), native["output"]
    )
    observed["FINAL_CONV_STATE"] = comparison_stats(
        evidence_bits(case["expected_conv"]), native["final_conv"]
    )
    observed["FINAL_RECURRENT_STATE"] = comparison_stats(
        evidence_bits(case["expected_recurrent"]), native["final_recurrent"]
    )
    sequence = case["input"]["shape"][1]
    expected_kinds = {
        "MASKED_INPUT", "PROJECTED_QKV", "PROJECTED_GATE",
        "PROJECTED_BETA", "PROJECTED_ALPHA", "CONV_INPUT", "CONV_OUTPUT",
        "QUERY_BEFORE_NORM", "KEY_BEFORE_NORM", "VALUE",
        "NORMALIZED_SCALED_QUERY", "NORMALIZED_KEY", "LOG_DECAY", "BETA",
        "RECURRENT_READ", "RECURRENT_DELTA", "RECURRENT_OUTPUT",
        "RECURRENT_STATE", "CONV_STATE", "GATED_NORM_OUTPUT",
        "OPERATOR_OUTPUT",
    }
    for token in range(sequence):
        for kind in expected_kinds:
            trace = native["traces"].get((kind, token))
            if trace is None:
                raise RuntimeError(f"missing native trace {kind}/{token} for {case['id']}")
            stats = comparison_stats(expected_trace(case, kind, token), trace["bits"])
            if kind not in observed:
                observed[kind] = empty_stats()
            merge_stats(observed[kind], stats)
    return observed


def derive_contract(stats: dict[str, Any]) -> dict[str, Any]:
    if stats["exact_bits"]:
        return {
            "comparison_class": "EXACT_BITS",
            "abs_limit": 0.0,
            "rel_limit": 0.0,
            "ulp_limit": 0,
            "calibration_observed": stats,
        }
    return {
        "comparison_class": "CALIBRATED_FLOAT",
        "abs_limit": stats["max_abs"] * 4.0 +
                     math.ldexp(max(1.0, stats["max_expected_abs"]), -21),
        "rel_limit": stats["max_rel"] * 4.0 + math.ldexp(1.0, -20),
        "ulp_limit": stats["max_ulp"] * 4 + 8,
        "calibration_observed": stats,
        "margin_rationale": (
            "four-times the disjoint calibration maximum plus four F32 ULP-scale "
            "guard for scalar C versus pinned PyTorch reduction/libm ordering"
        ),
    }


def contract_accepts(contract: dict[str, Any], stats: dict[str, Any]) -> bool:
    if contract["comparison_class"] == "EXACT_BITS":
        return stats["exact_bits"]
    return (
        stats["max_abs"] <= contract["abs_limit"] or
        stats["max_rel"] <= contract["rel_limit"] or
        stats["max_ulp"] <= contract["ulp_limit"]
    )


def aggregate_cases(probe: Path, contract: dict[str, Any], cases: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, int]]:
    aggregate: dict[str, dict[str, Any]] = defaultdict(empty_stats)
    metrics = None
    ids = []
    timings: dict[str, int] = {}
    for case in cases:
        native = run_probe(probe, contract, case)
        ids.append(case["id"])
        timings[case["id"]] = native["execution_nanoseconds"]
        if native["initialized"] != 1:
            raise RuntimeError(f"native state not initialized after {case['id']}")
        if metrics is None:
            metrics = native["metrics"]
        elif metrics != native["metrics"]:
            raise RuntimeError("native GDN metrics changed across identical configs")
        for category, stats in compare_case(case, native).items():
            merge_stats(aggregate[category], stats)
    return dict(sorted(aggregate.items())), metrics or {}, ids, timings


def validate_native_chain(probe: Path, contract: dict[str, Any],
                          state_json: dict[str, Any]) -> dict[str, Any]:
    prefix = standard_case(state_json["prefill"])
    full = standard_case(state_json["full_recomputation"])
    prefix_native = run_probe(probe, contract, prefix)
    combined_output = list(prefix_native["output"])
    previous = prefix_native
    for raw_step in state_json["decode_steps"]:
        step = continuation_case(raw_step)
        step["initial_conv"] = record_from_standard_bits(
            previous["final_conv"], raw_step["before_conv_state"]["shape"]
        )
        step["initial_recurrent"] = record_from_standard_bits(
            previous["final_recurrent"], raw_step["before_recurrent_state"]["shape"]
        )
        previous = run_probe(probe, contract, step)
        combined_output.extend(previous["output"])
    full_native = run_probe(probe, contract, full)
    replay_native = run_probe(probe, contract, prefix)
    exact_prefill_decode = (
        combined_output == full_native["output"] and
        previous["final_conv"] == full_native["final_conv"] and
        previous["final_recurrent"] == full_native["final_recurrent"]
    )
    exact_reset_replay = (
        prefix_native["output"] == replay_native["output"] and
        prefix_native["final_conv"] == replay_native["final_conv"] and
        prefix_native["final_recurrent"] == replay_native["final_recurrent"]
    )
    if not exact_prefill_decode or not exact_reset_replay:
        raise RuntimeError("native GDN split-prefill/decode or reset/replay is not bit-identical")
    return {
        "split_prefill_plus_decode_equals_full_recomputation": "EXACT_BITS_PASS",
        "reset_replay": "EXACT_BITS_PASS",
        "decoded_steps": len(state_json["decode_steps"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--write-validation", action="store_true")
    args = parser.parse_args()

    evidence_dir = args.evidence_dir.resolve()
    probe = args.probe.resolve()
    oracle_manifest = load_json(evidence_dir / "gdn-manifest.json")
    if oracle_manifest["model_revision"] != MODEL_REVISION or \
       oracle_manifest["transformers_revision"] != TRANSFORMERS_REVISION:
        raise SystemExit("GDN evidence revision mismatch")
    recorded = {item["path"]: item["sha256"] for item in oracle_manifest["assets"]}
    for name in ORACLE_FILES:
        if recorded.get(name) != sha256(evidence_dir / name):
            raise SystemExit(f"GDN oracle evidence hash mismatch: {name}")

    contract = load_json(evidence_dir / "gdn-contract.json")
    calibration_json = load_json(evidence_dir / "gdn-calibration.json")
    holdout_json = load_json(evidence_dir / "gdn-holdout.json")
    state_json = load_json(evidence_dir / "gdn-state-vectors.json")
    calibration_cases = [standard_case(case) for case in calibration_json["cases"]]
    holdout_cases = [standard_case(case) for case in holdout_json["cases"]]
    state_cases = [standard_case(state_json["prefill"])]
    state_cases.extend(continuation_case(case) for case in state_json["decode_steps"])
    state_cases.append(standard_case(state_json["full_recomputation"]))
    state_cases.append(standard_case(state_json["reset_replay"]))

    calibration_stats, metrics, calibration_ids, calibration_timings = aggregate_cases(
        probe, contract, calibration_cases
    )
    contracts = {
        category: derive_contract(stats)
        for category, stats in calibration_stats.items()
    }
    holdout_stats, holdout_metrics, holdout_ids, holdout_timings = aggregate_cases(
        probe, contract, holdout_cases
    )
    state_stats, state_metrics, state_ids, state_timings = aggregate_cases(
        probe, contract, state_cases
    )
    native_chain = validate_native_chain(probe, contract, state_json)
    if metrics != holdout_metrics or metrics != state_metrics:
        raise SystemExit("native GDN metrics are inconsistent")
    failed = [
        f"holdout:{category}" for category, stats in holdout_stats.items()
        if not contract_accepts(contracts[category], stats)
    ]
    failed.extend(
        f"state:{category}" for category, stats in state_stats.items()
        if not contract_accepts(contracts[category], stats)
    )
    if failed:
        raise SystemExit("GDN calibrated contract failures: " + ", ".join(failed))

    validation = {
        "schema": "KQ-GDN-NATIVE-VALIDATION-v1",
        "status": "PASS",
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "expected_values_source": "independent pinned Class-C oracle; never Kestrel-Q",
        "contracts": contracts,
        "calibration": {
            "case_ids": calibration_ids,
            "result": "PASS",
        },
        "holdout": {
            "case_ids": holdout_ids,
            "observed": holdout_stats,
            "result": "PASS",
        },
        "state_transitions": {
            "case_ids": state_ids,
            "observed": state_stats,
            "native_consistency": native_chain,
            "result": "PASS",
        },
        "metrics": metrics,
        "coverage": {
            "calibration_cases": len(calibration_cases),
            "holdout_cases": len(holdout_cases),
            "state_cases": len(state_cases),
            "checkpoint_classes": len(contracts),
        },
        "safety": {
            "full_bf16_checkpoint_downloaded": False,
            "real_model_payload_bytes_touched": 0,
            "kestrel_q_used_as_expected_value_source": False,
        },
    }
    validation_path = evidence_dir / "gdn-native-validation.json"
    if args.write_validation:
        write_json(validation_path, validation)
        manifest = {
            "schema": "KQ-GDN-MANIFEST-v1",
            "status": "COMPLETE_PASS",
            "model_revision": MODEL_REVISION,
            "transformers_revision": TRANSFORMERS_REVISION,
            "oracle_class": "C",
            "oracle_license": "Apache-2.0",
            "assets": [
                {"path": name, "sha256": sha256(evidence_dir / name)}
                for name in (*ORACLE_FILES, "gdn-native-validation.json")
            ],
            "safety": validation["safety"],
        }
        write_json(evidence_dir / "gdn-manifest.json", manifest)
    else:
        if not validation_path.exists() or validation_path.read_bytes() != canonical_json(validation):
            raise SystemExit("committed native GDN validation is absent or not byte-identical")
        final_manifest = load_json(evidence_dir / "gdn-manifest.json")
        final_recorded = {item["path"]: item["sha256"] for item in final_manifest["assets"]}
        if final_manifest.get("status") != "COMPLETE_PASS" or \
           final_recorded.get("gdn-native-validation.json") != sha256(validation_path):
            raise SystemExit("final GDN manifest is incomplete")

    print(
        f"GDN native validation PASS: calibration={len(calibration_cases)} "
        f"holdout={len(holdout_cases)} state={len(state_cases)} "
        f"checkpoints={len(contracts)}"
    )
    print("GDN observed execution ns (characterization only): " + json.dumps(
        {"calibration": calibration_timings,
         "holdout": holdout_timings,
         "state": state_timings}, sort_keys=True
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
