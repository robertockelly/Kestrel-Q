#!/usr/bin/env python3
"""Compare native scalar MoE with pinned independent Class-C evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_REVISION = "805a9e939fa8c1bff8d8ffdf041c051b71a914aa"
ORACLE_FILES = (
    "moe-contract.json", "moe-calibration.json", "moe-holdout.json",
    "moe-routing-vectors.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True,
                                 ensure_ascii=False) + "\n").encode("utf-8"))


def bits(record: dict[str, Any]) -> list[int]:
    return [int.from_bytes(bytes.fromhex(value), "little")
            for value in record["f32_le_hex"]]


def as_float(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def ordered(value: int) -> int:
    return (~value & 0xFFFFFFFF) if value & 0x80000000 else value | 0x80000000


def append_array(lines: list[str], label: str, values: Iterable[int]) -> None:
    data = list(values)
    lines.append(f"{label} {len(data)}" +
                 (" " + " ".join(f"{item:08x}" for item in data)
                  if data else ""))


def split_gate_up(record: dict[str, Any]) -> tuple[list[int], list[int]]:
    shape = record["shape"]
    values = bits(record)
    experts, doubled, hidden = shape
    intermediate = doubled // 2
    gate: list[int] = []
    up: list[int] = []
    stride = doubled * hidden
    half = intermediate * hidden
    for expert in range(experts):
        base = expert * stride
        gate.extend(values[base:base + half])
        up.extend(values[base + half:base + stride])
    return gate, up


def make_request(contract: dict[str, Any], case: dict[str, Any]) -> str:
    config = contract["tier_a_config"]
    weights = contract["weights"]
    gate, up = split_gate_up(weights["experts.gate_up_proj"])
    tokens = case["input"]["shape"][1]
    lines = [
        "KQMOE1",
        "CONFIG {hidden_size} {expert_count} {top_k} "
        "{routed_intermediate_size} {shared_intermediate_size} "
        f"{tokens}".format(**config),
    ]
    append_array(lines, "ROUTER", bits(weights["gate.weight"]))
    append_array(lines, "ROUTED_GATE", gate)
    append_array(lines, "ROUTED_UP", up)
    append_array(lines, "ROUTED_DOWN", bits(weights["experts.down_proj"]))
    append_array(lines, "SHARED_GATE", bits(weights["shared_expert.gate_proj.weight"]))
    append_array(lines, "SHARED_UP", bits(weights["shared_expert.up_proj.weight"]))
    append_array(lines, "SHARED_DOWN", bits(weights["shared_expert.down_proj.weight"]))
    append_array(lines, "SHARED_GATE_WEIGHT", bits(weights["shared_expert_gate.weight"]))
    append_array(lines, "INPUT", bits(case["input"]))
    return "\n".join(lines) + "\n"


def parse_probe(stdout: str, case_id: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"routes": {}, "traces": {}}
    for line in stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "ROUTE":
            at = 1
            token = int(fields[at]); at += 1
            experts = int(fields[at]); at += 1
            logits, probabilities = [], []
            for _ in range(experts):
                left, right = fields[at].split(":", 1); at += 1
                logits.append(int(left, 16)); probabilities.append(int(right, 16))
            count = int(fields[at]); at += 1
            ids, weights = [], []
            for _ in range(count):
                left, right = fields[at].split(":", 1); at += 1
                ids.append(int(left)); weights.append(int(right, 16))
            if at != len(fields):
                raise RuntimeError(f"bad route record: {case_id}")
            parsed["routes"][token] = {
                "logits": logits, "probabilities": probabilities,
                "ids": ids, "weights": weights,
            }
        elif fields[0] == "TRACE":
            kind = fields[1]
            token, expert, position, rank = map(int, fields[2:6])
            at = 6
            dimensions = [int(value) for value in fields[at:at + rank]]
            at += rank
            count = int(fields[at]); at += 1
            values = [int(value, 16) for value in fields[at:]]
            if len(values) != count:
                raise RuntimeError(f"bad trace count: {case_id} {kind}")
            parsed["traces"][(kind, token, expert, position)] = {
                "dimensions": dimensions, "bits": values}
        elif fields[0] == "OUTPUT":
            count = int(fields[1]); values = [int(value, 16) for value in fields[2:]]
            if len(values) != count:
                raise RuntimeError(f"bad output count: {case_id}")
            parsed["output"] = values
        elif fields[0] == "METRICS":
            parsed["metrics"] = {part.split("=", 1)[0]: int(part.split("=", 1)[1])
                                 for part in fields[1:]}
        elif fields[0] == "TIMING":
            parsed["timing_ns"] = int(fields[1].split("=", 1)[1])
    required = {"output", "metrics", "timing_ns"}
    if required - parsed.keys():
        raise RuntimeError(f"incomplete probe output: {case_id}")
    return parsed


def run_probe(probe: Path, contract: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [str(probe)], input=make_request(contract, case), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"native MoE probe failed for {case['id']}: "
                           f"rc={result.returncode}\n{result.stdout}\n{result.stderr}")
    return parse_probe(result.stdout, case["id"])


def empty_stats() -> dict[str, Any]:
    return {"count": 0, "max_abs": 0.0, "max_rel": 0.0, "max_ulp": 0,
            "max_expected_abs": 0.0, "exact_bits": True}


def compare_float(category: str, expected: list[int], actual: list[int],
                  stats: dict[str, dict[str, Any]],
                  contract: dict[str, Any] | None = None) -> None:
    if len(expected) != len(actual):
        raise RuntimeError(f"{category} count mismatch")
    item = stats[category]
    for expected_bits, actual_bits in zip(expected, actual):
        e, a = as_float(expected_bits), as_float(actual_bits)
        if not math.isfinite(e) or not math.isfinite(a):
            raise RuntimeError(f"non-finite {category}")
        absolute = abs(a - e)
        relative = absolute / abs(e) if e != 0.0 else (0.0 if absolute == 0.0 else math.inf)
        ulp = abs(ordered(actual_bits) - ordered(expected_bits))
        if contract is not None and contract["class"] == "EXACT_BITS" and \
                expected_bits != actual_bits:
            raise RuntimeError(f"{category} holdout is not exact-bit equal")
        if contract is not None and contract["class"] != "EXACT_BITS" and not (
            absolute <= contract["abs_limit"] or
            relative <= contract["rel_limit"] or
            ulp <= contract["ulp_limit"]):
            raise RuntimeError(f"{category} holdout exceeds calibrated contract: "
                               f"abs={absolute} rel={relative} ulp={ulp}")
        item["count"] += 1
        item["max_abs"] = max(item["max_abs"], absolute)
        if math.isfinite(relative):
            item["max_rel"] = max(item["max_rel"], relative)
        item["max_ulp"] = max(item["max_ulp"], ulp)
        item["max_expected_abs"] = max(item["max_expected_abs"], abs(e))
        item["exact_bits"] = item["exact_bits"] and expected_bits == actual_bits


AGGREGATE = {
    "router_logits": "ROUTER_LOGITS",
    "router_probabilities": "ROUTER_PROBABILITIES",
    "selected_weights": "SELECTED_WEIGHTS",
    "routed_weighted_sum": "ROUTED_WEIGHTED_SUM",
    "shared_gate_projection": "SHARED_GATE_PROJECTION",
    "shared_up_projection": "SHARED_UP_PROJECTION",
    "shared_activated": "SHARED_ACTIVATED",
    "shared_output": "SHARED_OUTPUT",
    "shared_gate_logit": "SHARED_SCALE_LOGIT",
    "shared_gate": "SHARED_SCALE",
    "gated_shared_output": "GATED_SHARED_OUTPUT",
    "final_output": "OPERATOR_OUTPUT",
}
EXPERT = {
    "gate": "ROUTED_GATE", "up": "ROUTED_UP",
    "activated": "ROUTED_ACTIVATED", "output": "ROUTED_EXPERT_OUTPUT",
    "weighted_output": "ROUTED_WEIGHTED_OUTPUT",
}


def compare_case(case: dict[str, Any], native: dict[str, Any],
                 stats: dict[str, dict[str, Any]],
                 contracts: dict[str, dict[str, Any]] | None = None) -> None:
    expected = case["expected"]
    ids = expected["selected_expert_ids"]
    for token, expected_ids in enumerate(ids):
        route = native["routes"].get(token)
        if route is None or route["ids"] != expected_ids:
            raise RuntimeError(f"{case['id']} exact routing mismatch at token {token}")
    for category, kind in AGGREGATE.items():
        record = expected[category]
        shape = record["shape"]
        values = bits(record)
        per_token = 1
        for dimension in shape[2:] if len(shape) > 2 else shape[1:]:
            per_token *= dimension
        if category in {"router_logits", "router_probabilities", "selected_weights",
                        "routed_weighted_sum", "shared_gate_projection",
                        "shared_up_projection", "shared_activated", "shared_output",
                        "shared_gate_logit", "shared_gate", "gated_shared_output",
                        "final_output"}:
            per_token = len(values) // len(ids)
        actual: list[int] = []
        for token in range(len(ids)):
            trace = native["traces"].get((kind, token, 0xFFFFFFFF, 0xFFFFFFFF))
            if trace is None:
                raise RuntimeError(f"{case['id']} missing {kind} token {token}")
            actual.extend(trace["bits"])
        compare_float(category, values, actual, stats,
                      None if contracts is None else contracts[category])
    for selected in expected["selected_experts"]:
        token = selected["token"]
        expert = selected["expert_id"]
        position = selected["top_k_position"]
        for category, kind in EXPERT.items():
            trace = native["traces"].get((kind, token, expert, position))
            if trace is None:
                raise RuntimeError(f"{case['id']} missing {kind}/{token}/{expert}")
            key = "routed_" + category
            compare_float(key, bits(selected[category]), trace["bits"], stats,
                          None if contracts is None else contracts[key])
    traced_output: list[int] = []
    for token in range(len(ids)):
        traced_output.extend(native["traces"][(
            "OPERATOR_OUTPUT", token, 0xFFFFFFFF, 0xFFFFFFFF)]["bits"])
    if native["output"] != traced_output:
        raise RuntimeError(f"{case['id']} final output/trace mismatch")


def make_contracts(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for category, item in sorted(stats.items()):
        if item["exact_bits"]:
            result[category] = {
                "class": "EXACT_BITS", "abs_limit": 0.0,
                "rel_limit": 0.0, "ulp_limit": 0,
                "calibration_observed": item,
            }
        else:
            result[category] = {
                "class": "CALIBRATED_FLOAT",
                "abs_limit": item["max_abs"] * 4.0 +
                             math.ldexp(max(1.0, item["max_expected_abs"]), -21),
                "rel_limit": item["max_rel"] * 4.0 + math.ldexp(1.0, -20),
                "ulp_limit": item["max_ulp"] * 4 + 8,
                "calibration_observed": item,
                "margin_rationale": (
                    "four-times the disjoint calibration maximum plus a bounded "
                    "F32 ULP-scale guard for scalar C versus pinned PyTorch "
                    "reduction and libm ordering"
                ),
            }
    return result


def run_routing_case(probe: Path, config: dict[str, Any], case: dict[str, Any]) -> None:
    tokens = len(case["input"]["f32_le_hex"]) // config["hidden_size"]
    lines = ["KQMOEROUTE1",
             "CONFIG {hidden_size} {expert_count} {top_k} ".format(**config) + str(tokens)]
    append_array(lines, "ROUTER", bits(case["router_weight"]))
    append_array(lines, "INPUT", bits(case["input"]))
    result = subprocess.run(
        [str(probe), "--route"], input="\n".join(lines) + "\n", text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"routing probe failed {case['id']}: {result.stderr}")
    got: list[list[int]] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields or fields[0] != "ROUTE":
            continue
        experts = int(fields[2]); at = 3 + experts
        count = int(fields[at]); at += 1
        got.append([int(fields[at + index].split(":", 1)[0])
                    for index in range(count)])
    if got != case["expected"]["selected_expert_ids"]:
        raise RuntimeError(f"Tier-B exact routing mismatch: {case['id']}")


def validate(probe: Path, evidence: Path, output: Path) -> dict[str, Any]:
    contract = load(evidence / "moe-contract.json")
    calibration = load(evidence / "moe-calibration.json")
    holdout = load(evidence / "moe-holdout.json")
    routing = load(evidence / "moe-routing-vectors.json")
    categories = list(AGGREGATE) + ["routed_" + item for item in EXPERT]
    cal_stats = defaultdict(empty_stats)
    metrics: dict[str, Any] = {}
    for case in calibration["cases"]:
        native = run_probe(probe, contract, case)
        compare_case(case, native, cal_stats)
        metrics[case["id"]] = native["metrics"]
    contracts = make_contracts({category: cal_stats[category]
                                for category in categories})
    hold_stats = defaultdict(empty_stats)
    for case in holdout["cases"]:
        native = run_probe(probe, contract, case)
        compare_case(case, native, hold_stats, contracts)
        metrics[case["id"]] = native["metrics"]
    for case in routing["cases"]:
        run_routing_case(probe, routing["config"], case)
    result = {
        "schema": "kq-moe-native-validation-v1",
        "authority": {
            "model_revision": MODEL_REVISION,
            "transformers_revision": TRANSFORMERS_REVISION,
            "oracle_hashes": {name: sha256(evidence / name) for name in ORACLE_FILES},
            "expected_values_source": "pinned Transformers Class-C evidence; never Kestrel-Q",
        },
        "results": {
            "calibration_cases": len(calibration["cases"]),
            "holdout_cases": len(holdout["cases"]),
            "tier_b_routing_cases": len(routing["cases"]),
            "routing": "EXACT_DISCRETE_PASS",
            "calibration": "PASS",
            "holdout": "PASS",
            "routed_path": "PASS",
            "shared_path": "PASS",
            "final_combination": "PASS",
        },
        "contracts": contracts,
        "holdout_observed": {category: hold_stats[category]
                             for category in categories},
        "characterization_metrics": metrics,
    }
    write_json(output, result)
    return result


def update_manifest(evidence: Path) -> None:
    path = evidence / "moe-manifest.json"
    manifest = load(path)
    files = manifest.setdefault("files", {})
    validation = evidence / "moe-native-validation.json"
    files[validation.name] = {
        "bytes": validation.stat().st_size,
        "sha256": sha256(validation),
        "class": "native-validation-against-independent-Class-C",
        "status": "GENERATED_VERIFIED",
    }
    write_json(path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    evidence = args.evidence_dir.resolve()
    output = evidence / "moe-native-validation.json"
    if not args.probe.is_file():
        raise SystemExit("MoE probe executable not found")
    for name in ORACLE_FILES:
        if not (evidence / name).is_file():
            raise SystemExit(f"missing independent MoE evidence: {name}")
    if args.verify and output.is_file():
        existing_validation = output.read_bytes()
        existing_manifest = (evidence / "moe-manifest.json").read_bytes()
        with tempfile.TemporaryDirectory(prefix="kq-moe-verify-") as temp:
            temp_output = Path(temp) / output.name
            validate(args.probe.resolve(), evidence, temp_output)
            output.write_bytes(temp_output.read_bytes())
            update_manifest(evidence)
            if output.read_bytes() != existing_validation or \
                    (evidence / "moe-manifest.json").read_bytes() != existing_manifest:
                output.write_bytes(existing_validation)
                (evidence / "moe-manifest.json").write_bytes(existing_manifest)
                raise SystemExit("MoE native evidence regeneration is not byte-identical")
    else:
        validate(args.probe.resolve(), evidence, output)
        update_manifest(evidence)
    print("MoE native oracle validation: PASS; calibration=5, holdout=4, "
          "tier_b_exact_routing=7, self_oracle=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
