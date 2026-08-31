#!/usr/bin/env python3
"""Validate native Task 2.11 output against independent target evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
from pathlib import Path
from typing import Any


def write(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def from_bits(text: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(text, 16)))[0]


def from_hex(text: str) -> float:
    return struct.unpack("<f", bytes.fromhex(text))[0]


def ordered_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return 0x80000000 - bits if bits & 0x80000000 else bits + 0x80000000


def run_probe(probe: Path, profile: str) -> dict[str, Any]:
    command = [str(probe), "--dump-values"]
    if profile == "holdout":
        command.append("--holdout")
    elif profile == "calibration-b":
        command.append("--calibration-secondary")
    elif profile == "calibration-c":
        command.append("--calibration-tertiary")
    elif profile == "calibration-d":
        command.append("--calibration-permuted")
    run = subprocess.run(command, capture_output=True, text=True)
    if run.returncode == 77:
        raise SystemExit(77)
    if run.returncode != 0:
        raise RuntimeError(run.stderr or "native target probe failed")
    outputs: dict[tuple[str, str], list[float]] = {}
    routes: list[tuple[int, int]] = []
    experts: list[tuple[int, int]] = []
    ple: list[tuple[int, int]] = []
    metrics = ""
    for line in run.stdout.splitlines():
        fields = line.split()
        if line.startswith("VALUES "):
            family = fields[1].split("=", 1)[1]
            phase = fields[2].split("=", 1)[1]
            count = int(fields[3].split("=", 1)[1])
            values = [from_bits(item) for item in fields[4:]]
            if len(values) != count:
                raise RuntimeError("native target value count mismatch")
            outputs[(family, phase)] = values
        elif line.startswith("ROUTE_TRACE "):
            routes = [tuple(map(int, item.split(":"))) for item in fields[2:]]
        elif line.startswith("EXPERT_TRACE "):
            experts = [tuple(map(int, item.split(":"))) for item in fields[2:]]
        elif line.startswith("PLE_TRACE "):
            ple = [tuple(map(int, item.split(":"))) for item in fields[2:]]
        elif line.startswith("target quantized layer integration:"):
            metrics = re.sub(r" linear_elapsed_ns=\d+", "", line)
    if len(outputs) != 6 or len(routes) != 60 or len(experts) != 60 or len(ple) != 32:
        raise RuntimeError("native target trace is incomplete")
    return {"outputs": outputs, "routes": routes, "experts": experts,
            "ple": ple, "metrics": metrics}


def compare(expected: list[float], actual: list[float]) -> dict[str, Any]:
    if len(expected) != len(actual):
        raise RuntimeError("target output shape mismatch")
    absolute = [abs(left - right) for left, right in zip(expected, actual)]
    relative = [difference / max(abs(left), 1.0e-30)
                for left, difference in zip(expected, absolute)]
    ulp = [abs(ordered_bits(left) - ordered_bits(right))
           for left, right in zip(expected, actual)]
    return {
        "count": len(expected),
        "max_abs": max(absolute, default=0.0),
        "max_rel": max(relative, default=0.0),
        "max_ulp": max(ulp, default=0),
        "finite": all(math.isfinite(value) for value in actual),
    }


def expected_trace(document: dict[str, Any]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    routes = []
    ple = []
    for case in document["cases"]:
        layer = int(case["layer_id"])
        for step in case["steps"]:
            routes.extend((layer, int(expert)) for expert in step["selected_expert_ids"])
        ple.extend((int(item["member"]), int(item["row"]))
                   for item in case.get("ple_requests", []))
    return routes, ple


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.evidence_dir.resolve()
    calibration = json.loads((root / "target-layer-calibration.json").read_text())
    holdout = json.loads((root / "target-layer-holdout.json").read_text())
    native_profiles = {
        "calibration-a": run_probe(args.probe.resolve(), "calibration-a"),
        "calibration-b": run_probe(args.probe.resolve(), "calibration-b"),
        "calibration-c": run_probe(args.probe.resolve(), "calibration-c"),
        "calibration-d": run_probe(args.probe.resolve(), "calibration-d"),
        "holdout": run_probe(args.probe.resolve(), "holdout"),
    }
    records = []
    limits: dict[tuple[str, str], dict[str, float]] = {}
    for case in calibration["cases"]:
        profile = case["profile"]
        for step in case["steps"]:
            key = (case["family"], step["phase"])
            expected = [from_hex(item) for item in step["output"]["f32_le_hex"]]
            result = compare(expected, native_profiles[profile]["outputs"][key])
            previous = limits.get(key, {"abs": 0.0, "rel": 0.0})
            limits[key] = {"abs": max(previous["abs"], result["max_abs"]),
                           "rel": max(previous["rel"], result["max_rel"])}
            result.update({"family": key[0], "phase": key[1],
                           "corpus": profile,
                           "contract": "CALIBRATED_FLOAT",
                           "pass": result["finite"]})
            records.append(result)
    for key in limits:
        limits[key]["abs"] = limits[key]["abs"] * 1.25 + 1.0e-9
        limits[key]["rel"] = limits[key]["rel"] * 1.25 + 1.0e-7
    for result in records:
        key = (result["family"], result["phase"])
        result["abs_limit"] = limits[key]["abs"]
        result["rel_limit"] = limits[key]["rel"]
    for case in holdout["cases"]:
        for step in case["steps"]:
            key = (case["family"], step["phase"])
            expected = [from_hex(item) for item in step["output"]["f32_le_hex"]]
            result = compare(expected, native_profiles["holdout"]["outputs"][key])
            limit = limits[key]
            result.update({"family": key[0], "phase": key[1],
                           "corpus": "holdout",
                           "contract": "CALIBRATED_FLOAT",
                           "abs_limit": limit["abs"], "rel_limit": limit["rel"],
                           "pass": result["finite"] and
                                   (result["max_abs"] <= limit["abs"] or
                                    result["max_rel"] <= limit["rel"])})
            records.append(result)
    calibration_a = {"cases": [case for case in calibration["cases"]
                                if case["profile"] == "calibration-a"]}
    calibration_b = {"cases": [case for case in calibration["cases"]
                                if case["profile"] == "calibration-b"]}
    calibration_c = {"cases": [case for case in calibration["cases"]
                                if case["profile"] == "calibration-c"]}
    calibration_d = {"cases": [case for case in calibration["cases"]
                                if case["profile"] == "calibration-d"]}
    cal_a_routes, cal_a_ple = expected_trace(calibration_a)
    cal_b_routes, cal_b_ple = expected_trace(calibration_b)
    cal_c_routes, cal_c_ple = expected_trace(calibration_c)
    cal_d_routes, cal_d_ple = expected_trace(calibration_d)
    hold_routes, hold_ple = expected_trace(holdout)
    discrete = {
        "calibration_a_route_order_exact": native_profiles["calibration-a"]["routes"] == cal_a_routes,
        "calibration_b_route_order_exact": native_profiles["calibration-b"]["routes"] == cal_b_routes,
        "calibration_c_route_order_exact": native_profiles["calibration-c"]["routes"] == cal_c_routes,
        "calibration_d_route_order_exact": native_profiles["calibration-d"]["routes"] == cal_d_routes,
        "holdout_route_order_exact": native_profiles["holdout"]["routes"] == hold_routes,
        "calibration_a_selected_expert_access_exact":
            sorted(native_profiles["calibration-a"]["experts"]) == sorted(cal_a_routes),
        "calibration_b_selected_expert_access_exact":
            sorted(native_profiles["calibration-b"]["experts"]) == sorted(cal_b_routes),
        "calibration_c_selected_expert_access_exact":
            sorted(native_profiles["calibration-c"]["experts"]) == sorted(cal_c_routes),
        "calibration_d_selected_expert_access_exact":
            sorted(native_profiles["calibration-d"]["experts"]) == sorted(cal_d_routes),
        "holdout_selected_expert_access_exact":
            sorted(native_profiles["holdout"]["experts"]) == sorted(hold_routes),
        "calibration_a_ple_requests_exact": native_profiles["calibration-a"]["ple"] == cal_a_ple,
        "calibration_b_ple_requests_exact": native_profiles["calibration-b"]["ple"] == cal_b_ple,
        "calibration_c_ple_requests_exact": native_profiles["calibration-c"]["ple"] == cal_c_ple,
        "calibration_d_ple_requests_exact": native_profiles["calibration-d"]["ple"] == cal_d_ple,
        "holdout_ple_requests_exact": native_profiles["holdout"]["ple"] == hold_ple,
        "qsa_prefill_decode_selection": "asserted by native probe: 0 candidate blocks; tail tokens 1 then 2",
    }
    passed = all(record["pass"] for record in records) and all(
        value for value in discrete.values() if isinstance(value, bool))
    result = {
        "schema": "kq-target-layer-native-validation-v1",
        "status": "PASS" if passed else "FAIL",
        "records": records,
        "discrete": discrete,
        "native_metrics": {
            key: value["metrics"] for key, value in native_profiles.items()
        },
        "self_oracle": False,
    }
    validation = root / "target-layer-native-validation.json"
    if args.verify:
        if not validation.exists() or json.loads(validation.read_text()) != result:
            raise SystemExit("target layer native validation is not deterministic")
    else:
        write(validation, result)
        manifest_path = root / "target-layer-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        files = ["target-layer-contract.json", "target-layer-calibration.json",
                 "target-layer-holdout.json", "target-layer-state-vectors.json",
                 "target-layer-native-validation.json"]
        manifest["files"] = {name: sha256(root / name) for name in files}
        manifest["status"] = result["status"]
        write(manifest_path, manifest)
    if not passed:
        for record in records:
            if not record["pass"]:
                print("FAIL", record)
        print(discrete)
        raise SystemExit(1)
    case_count = len(calibration["cases"]) + len(holdout["cases"])
    print(f"target layer independent validation: PASS; cases={case_count} "
          "exact_discrete=PASS")


if __name__ == "__main__":
    main()
