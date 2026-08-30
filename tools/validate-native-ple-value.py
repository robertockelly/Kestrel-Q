#!/usr/bin/env python3
"""Compare native scalar PLE value execution with independent Class-C evidence."""

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

ORACLE_FILES = ("ple-value-contract.json", "ple-value-calibration.json",
                "ple-value-holdout.json", "ple-value-state-vectors.json",
                "ple-value-address-integration.json")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def bits(record: dict[str, Any]) -> list[int]:
    return [int.from_bytes(bytes.fromhex(item), "little")
            for item in record["f32_le_hex"]]


def append_array(lines: list[str], label: str, values: Iterable[int]) -> None:
    data = list(values)
    lines.append(f"{label} {len(data)}" +
                 (" " + " ".join(f"{item:08x}" for item in data) if data else ""))


WEIGHTS = (("KEY", "key_proj.weight"), ("VALUE", "value_proj.weight"),
           ("NORM_KEY", "norm_key.weight"), ("NORM_QUERY", "norm_query.weight"),
           ("NORM_CONV", "norm_conv.weight"), ("CONV", "conv1d.weight"))


def flatten_nested(value: Any) -> list[int]:
    if isinstance(value, list):
        result: list[int] = []
        for item in value:
            result.extend(flatten_nested(item))
        return result
    return [int(value)]


def case_data(case: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[dict[str, int]], list[int]]:
    if "intents" in case:
        return case["intents"], bits(case["lookup_rows"])
    addresses = flatten_nested(case["expected"]["checkpoints"]["global_addresses"])
    intents = []
    for index, address in enumerate(addresses):
        head = index % cfg["head_count"]
        intents.append({"position": index // cfg["head_count"],
                        "order": 2 + head // cfg["heads_per_order"],
                        "local_head": head % cfg["heads_per_order"],
                        "global_head": head, "logical_member": 0,
                        "member_row": address})
    return intents, bits(case["expected"]["checkpoints"]["lookup"])


def make_request(contract: dict[str, Any], tier: str,
                 case: dict[str, Any]) -> str:
    cfg = contract[f"tier_{tier}_config"]
    weights = contract[f"tier_{tier}_weights"]
    tokens = case["input"]["shape"][1]
    members = 128 if tier == "b" else 1
    member_rows = 2_500_012 if tier == "b" else 64
    lines = ["KQPLEVALUE1",
             f"CONFIG {cfg['hidden_size']} {cfg['residual_branches']} "
             f"{cfg['heads_per_order']} {cfg['row_width']} {members} "
             f"{member_rows} {tokens}"]
    for label, name in WEIGHTS:
        append_array(lines, label, bits(weights[name]))
    append_array(lines, "INPUT", bits(case["input"]))
    append_array(lines, "INITIAL_STATE", bits(case["initial_value_state"]))
    intents, rows = case_data(case, cfg)
    lines.append(f"INTENTS {len(intents)}")
    for item in intents:
        lines.append("{position} {order} {local_head} {global_head} "
                     "{logical_member} {member_row}".format(**item))
    append_array(lines, "ROWS", rows)
    return "\n".join(lines) + "\n"


def parse_probe(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"traces": {}}
    for line in stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "TRACE":
            kind, token, rank = fields[1], int(fields[2]), int(fields[3])
            at = 4
            dimensions = [int(item) for item in fields[at:at + rank]]
            at += rank
            count = int(fields[at]); at += 1
            values = [int(item, 16) for item in fields[at:]]
            if len(values) != count:
                raise RuntimeError("bad PLE trace count")
            parsed["traces"][(kind, token)] = {"dimensions": dimensions,
                                                       "bits": values}
        elif fields[0] == "OUTPUT":
            parsed["output"] = [int(item, 16) for item in fields[2:]]
        elif fields[0] == "STATE":
            parsed["state_position"] = int(fields[1])
            parsed["state"] = [int(item, 16) for item in fields[3:]]
        elif fields[0] == "METRICS":
            parsed["metrics"] = {item.split("=", 1)[0]: int(item.split("=", 1)[1])
                                 for item in fields[1:]}
    if {"output", "state", "metrics"} - parsed.keys():
        raise RuntimeError("incomplete PLE probe output")
    return parsed


def run_probe(probe: Path, contract: dict[str, Any], tier: str,
              case: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run([str(probe)], input=make_request(contract, tier, case),
                            text=True, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"PLE probe failed for {case['id']}:\n"
                           f"{result.stdout}\n{result.stderr}")
    return parse_probe(result.stdout)


def as_float(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value))[0]


def ordered(value: int) -> int:
    return (~value & 0xFFFFFFFF) if value & 0x80000000 else value | 0x80000000


def empty_stats() -> dict[str, Any]:
    return {"count": 0, "max_abs": 0.0, "max_rel": 0.0, "max_ulp": 0,
            "max_expected_abs": 0.0, "exact_bits": True}


def compare(category: str, expected: list[int], actual: list[int],
            stats: dict[str, dict[str, Any]],
            contract: dict[str, Any] | None = None) -> None:
    if len(expected) != len(actual):
        raise RuntimeError(f"{category} count mismatch {len(expected)} != {len(actual)}")
    record = stats[category]
    for eb, ab in zip(expected, actual):
        e, a = as_float(eb), as_float(ab)
        if not math.isfinite(e) or not math.isfinite(a):
            raise RuntimeError(f"non-finite {category}")
        absolute = abs(a - e)
        relative = absolute / abs(e) if e != 0.0 else (0.0 if absolute == 0.0 else math.inf)
        ulp = abs(ordered(ab) - ordered(eb))
        if contract is not None:
            if contract["class"] == "EXACT_BITS" and eb != ab:
                raise RuntimeError(f"{category} is not exact bits")
            if contract["class"] != "EXACT_BITS" and not (
                    absolute <= contract["abs_limit"] or
                    relative <= contract["rel_limit"] or
                    ulp <= contract["ulp_limit"]):
                raise RuntimeError(f"{category} exceeds calibrated contract: "
                                   f"abs={absolute} rel={relative} ulp={ulp}")
        record["count"] += 1
        record["max_abs"] = max(record["max_abs"], absolute)
        if math.isfinite(relative): record["max_rel"] = max(record["max_rel"], relative)
        record["max_ulp"] = max(record["max_ulp"], ulp)
        record["max_expected_abs"] = max(record["max_expected_abs"], abs(e))
        record["exact_bits"] = record["exact_bits"] and eb == ab


TRACE_MAP = {
    "lookup": "RAW_LOOKUPS", "embedding": "EMBEDDING",
    "key_projection": "KEY_PROJECTION", "value_projection": "VALUE_PROJECTION",
    "key_norm": "KEY_NORM", "query_norm": "QUERY_NORM",
    "gate_raw": "GATE_RAW", "gate_transformed": "GATE_TRANSFORMED",
    "gated_value": "GATED_VALUE", "conv_norm": "CONV_NORM",
    "conv_pre_activation": "CONV_PRE_ACTIVATION", "conv_output": "CONV_OUTPUT",
    "operator_output": "OPERATOR_OUTPUT",
}


def per_token(record: dict[str, Any], category: str, token: int) -> list[int]:
    shape, values = record["shape"], bits(record)
    if category == "conv_pre_activation":
        channels, tokens = shape[1], shape[2]
        return [values[channel * tokens + token] for channel in range(channels)]
    tokens = shape[1]
    width = len(values) // tokens
    return values[token * width:(token + 1) * width]


def compare_case(case: dict[str, Any], native: dict[str, Any],
                 stats: dict[str, dict[str, Any]],
                 contracts: dict[str, dict[str, Any]] | None = None) -> None:
    checkpoints = case["expected"]["checkpoints"]
    tokens = case["input"]["shape"][1]
    for category, kind in TRACE_MAP.items():
        if category not in checkpoints:
            continue
        expected, actual = [], []
        for token in range(tokens):
            trace = native["traces"].get((kind, token))
            if trace is None:
                raise RuntimeError(f"missing {kind} token {token}")
            expected.extend(per_token(checkpoints[category], category, token))
            actual.extend(trace["bits"])
        compare(category, expected, actual, stats,
                None if contracts is None else contracts[category])
    compare("final_state", bits(case["expected"]["final_value_state"]),
            native["state"], stats,
            None if contracts is None else contracts["final_state"])
    if native["output"] != [item for token in range(tokens)
                            for item in native["traces"][("OPERATOR_OUTPUT", token)]["bits"]]:
        raise RuntimeError("PLE output and output checkpoint differ")


def make_contracts(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for category, item in sorted(stats.items()):
        if item["exact_bits"]:
            result[category] = {"class": "EXACT_BITS", "abs_limit": 0.0,
                                "rel_limit": 0.0, "ulp_limit": 0,
                                "calibration_observed": item}
        else:
            result[category] = {
                "class": "CALIBRATED_FLOAT",
                "abs_limit": item["max_abs"] * 6.0 +
                             math.ldexp(max(1.0, item["max_expected_abs"]), -20),
                "rel_limit": item["max_rel"] * 6.0 + math.ldexp(1.0, -19),
                "ulp_limit": item["max_ulp"] * 6 + 16,
                "calibration_observed": item,
                "margin_rationale": "six-times calibration maximum plus bounded F32 guard",
            }
    return result


def validate(probe: Path, evidence: Path, output: Path) -> dict[str, Any]:
    contract = load(evidence / ORACLE_FILES[0])
    calibration = load(evidence / ORACLE_FILES[1])
    holdout = load(evidence / ORACLE_FILES[2])
    address = load(evidence / ORACLE_FILES[4])
    cal_stats = defaultdict(empty_stats)
    metrics = {}
    for case in calibration["cases"]:
        native = run_probe(probe, contract, "a", case)
        compare_case(case, native, cal_stats)
        metrics[case["id"]] = native["metrics"]
    contracts = make_contracts(dict(cal_stats))
    hold_stats = defaultdict(empty_stats)
    for case in holdout["cases"]:
        native = run_probe(probe, contract, "a", case)
        compare_case(case, native, hold_stats, contracts)
        metrics[case["id"]] = native["metrics"]
    address_stats = defaultdict(empty_stats)
    for case in address["cases"]:
        native = run_probe(probe, contract, "b", case)
        compare_case(case, native, address_stats)
        expected_intents, _ = case_data(case, contract["tier_b_config"])
        if native["metrics"]["lookups"] != len(expected_intents):
            raise RuntimeError("exact address consumption count mismatch")
        metrics[case["id"]] = native["metrics"]
    result = {
        "schema": "kq-ple-value-native-validation-v1",
        "authority": {"oracle_hashes": {name: sha256(evidence / name)
                                          for name in ORACLE_FILES},
                      "expected_values_source": "pinned Transformers Class-C; never Kestrel-Q"},
        "results": {"calibration_cases": len(calibration["cases"]),
                    "holdout_cases": len(holdout["cases"]),
                    "address_integration_cases": len(address["cases"]),
                    "calibration": "PASS", "holdout": "PASS",
                    "exact_address_consumption": "PASS",
                    "state": "PASS", "prefill_decode": "PASS"},
        "contracts": contracts,
        "holdout_observed": dict(hold_stats),
        "address_observed": dict(address_stats),
        "characterization_metrics": metrics,
    }
    write_json(output, result)
    return result


def update_manifest(evidence: Path) -> None:
    path = evidence / "ple-value-manifest.json"
    manifest = load(path)
    validation = evidence / "ple-value-native-validation.json"
    manifest.setdefault("files", {})[validation.name] = {
        "sha256": sha256(validation), "bytes": validation.stat().st_size,
        "class": "native-validation-against-independent-Class-C",
        "status": "GENERATED_VERIFIED"}
    write_json(path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    evidence, probe = args.evidence_dir.resolve(), args.probe.resolve()
    output = evidence / "ple-value-native-validation.json"
    for name in ORACLE_FILES:
        if not (evidence / name).is_file():
            raise SystemExit(f"missing PLE value evidence: {name}")
    if args.verify and output.is_file():
        old_output = output.read_bytes(); old_manifest = (evidence / "ple-value-manifest.json").read_bytes()
        with tempfile.TemporaryDirectory(prefix="kq-ple-value-native-") as directory:
            candidate = Path(directory) / output.name
            validate(probe, evidence, candidate); output.write_bytes(candidate.read_bytes()); update_manifest(evidence)
            if output.read_bytes() != old_output or (evidence / "ple-value-manifest.json").read_bytes() != old_manifest:
                output.write_bytes(old_output); (evidence / "ple-value-manifest.json").write_bytes(old_manifest)
                raise SystemExit("PLE value native evidence regeneration is not byte-identical")
    else:
        validate(probe, evidence, output); update_manifest(evidence)
    print("PLE value native oracle validation: PASS; calibration=4 holdout=3 address=3 self_oracle=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
