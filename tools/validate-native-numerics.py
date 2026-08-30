#!/usr/bin/env python3
"""Validate native Task 2.5 output against committed independent evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True,
                          text=True).stdout.strip()


def bits_line(text: str) -> list[str]:
    if not text.startswith("bits="):
        raise RuntimeError(f"unexpected native probe output: {text!r}")
    return text[5:].split(",")


def ordered(bits: int) -> int:
    return 0x80000000 - (bits & 0x7FFFFFFF) if bits & 0x80000000 else 0x80000000 + bits


def maximum_ulp(expected: list[str], observed: list[str]) -> int:
    if len(expected) != len(observed):
        raise RuntimeError("native output length mismatch")
    return max((abs(ordered(int(a, 16)) - ordered(int(b, 16)))
                for a, b in zip(expected, observed, strict=True)), default=0)


def validate_primitive_file(probe: Path, data: dict[str, object]) -> int:
    checked = 0
    for case in data["cases"]:
        operation = case["operation"]
        command = [str(probe), operation, ",".join(case["input_left_f32_bits"])]
        if operation in ("add", "multiply", "dot", "scale", "swiglu"):
            command.append(",".join(case["input_right_f32_bits"]))
        elif operation == "rmsnorm":
            command.extend((",".join(case["input_right_f32_bits"]), case["epsilon_f32_bits"]))
        elif operation == "topk":
            command.append(str(case["k"]))
        output = run(command)
        observed_indices = None
        if operation == "topk":
            prefix, output = output.split(";", 1)
            observed_indices = [int(value) for value in prefix.removeprefix("indices=").split(",")]
            if observed_indices != case["expected_indices"]:
                raise RuntimeError(f"top-k indices differ for {case['id']}")
        observed = bits_line(output)
        if observed != case["observed_native_f32_bits"]:
            raise RuntimeError(f"native reproducibility changed for {case['id']}")
        contract = data["contracts"][operation]
        if maximum_ulp(case["expected_f32_bits"], observed) > contract["acceptance"]["ulp_limit"]:
            raise RuntimeError(f"numeric contract failed for {case['id']}")
        checked += 1
    return checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    dequant = json.loads((args.evidence_dir / "dequant-vectors.json").read_text(encoding="utf-8"))
    calibration = json.loads((args.evidence_dir / "primitive-calibration.json").read_text(encoding="utf-8"))
    holdout = json.loads((args.evidence_dir / "primitive-holdout.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.evidence_dir / "manifest.json").read_text(encoding="utf-8"))

    for asset in manifest["assets"]:
        path = args.evidence_dir / asset["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise RuntimeError(f"manifest hash mismatch for {asset['path']}")

    dequant_checked = 0
    for case in dequant["cases"]:
        observed = bits_line(run([str(args.probe), "dequant", case["type"], case["packed_hex"]]))
        if observed != case["expected_f32_bits"]:
            raise RuntimeError(f"dequant mismatch for {case['id']}")
        dequant_checked += 1
    for case in dequant["row_dot_cases"]:
        source = next(item for item in dequant["cases"] if item["id"] == case["source_case_id"])
        observed = bits_line(run([str(args.probe), "rowdot", case["type"], source["packed_hex"],
                                  ",".join(case["activation_f32_bits"])]))
        if observed != case["expected_f32_bits"]:
            raise RuntimeError(f"row-dot mismatch for {case['id']}")
    calibration_checked = validate_primitive_file(args.probe, calibration)
    holdout_checked = validate_primitive_file(args.probe, holdout)
    print(f"numeric oracle: PASS; dequant={dequant_checked}, row_dot={len(dequant['row_dot_cases'])}, "
          f"calibration={calibration_checked}, holdout={holdout_checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
