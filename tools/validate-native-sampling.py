#!/usr/bin/env python3
"""Compare native Task 3.0 sampling with independent pinned evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ORACLE_FILES = (
    "sampling-contract.json",
    "sampling-calibration.json",
    "sampling-holdout.json",
    "sampling-rng-vectors.json",
    "sampling-statistical.json",
)
STATUS_OK = 0
STATUS_INVALID_TOKEN_ID = 31


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


def bits_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def ordered(bits: int) -> int:
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000


def difference(expected_bits: int, actual_bits: int) -> dict[str, Any]:
    expected = bits_float(expected_bits)
    actual = bits_float(actual_bits)
    absolute = abs(actual - expected)
    relative = absolute / abs(expected) if expected != 0.0 else absolute
    return {
        "expected_f32_le_hex": struct.pack("<I", expected_bits).hex(),
        "actual_f32_le_hex": struct.pack("<I", actual_bits).hex(),
        "ulp": abs(ordered(actual_bits) - ordered(expected_bits)),
        "absolute": absolute,
        "relative": relative,
    }


def run(probe: Path, arguments: list[str]) -> list[str]:
    completed = subprocess.run([str(probe), *arguments], check=True,
                               capture_output=True, text=True)
    return [line for line in completed.stdout.splitlines() if line]


def parse_sample(lines: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {"tokens": {}}
    for line in lines:
        fields = line.split()
        if fields[0] == "STATUS":
            parsed["status"] = int(fields[1])
        elif fields[0] == "STATE":
            parsed["state_before"] = fields[1]
            parsed["state_after"] = fields[2]
            parsed["draws_after"] = int(fields[3])
        elif fields[0] == "RESULT":
            parsed["result"] = {
                "selected_token_id": int(fields[1]),
                "top_k_retained_count": int(fields[2]),
                "retained_count": int(fields[3]),
                "rng_word": fields[4],
                "selected_probability_bits": int(fields[5], 16),
                "maximum_probability_bits": int(fields[6], 16),
                "normalized_sum_bits": int(fields[7], 16),
                "uniform_hex": fields[8],
            }
        elif fields[0] == "TOKEN":
            parsed["tokens"][int(fields[1])] = {
                "score_bits": int(fields[2], 16),
                "probability_bits": int(fields[3], 16),
            }
    if "status" not in parsed or "state_before" not in parsed:
        raise RuntimeError("native sample probe output is incomplete")
    return parsed


def case_arguments(case: dict[str, Any]) -> list[str]:
    policy = case["policy"]
    arguments = [
        "sample", repr(policy["temperature"]), str(policy["top_k"]),
        repr(policy["top_p"]), str(case["seed"]), str(case["stream"]),
        repr(case["base_logit"]),
    ]
    arguments.extend(f"{token_id}={repr(value)}"
                     for token_id, value in case["overrides"].items())
    return arguments


def validate_case(probe: Path, case: dict[str, Any]) -> dict[str, Any]:
    expected = case["oracle"]
    actual = parse_sample(run(probe, case_arguments(case)))
    expected_status = (STATUS_OK if expected["expected_status"] == "OK"
                       else STATUS_INVALID_TOKEN_ID)
    if actual["status"] != expected_status:
        raise RuntimeError(
            f"{case['id']}: status {actual['status']} != {expected_status}")
    if actual["state_before"] != expected["rng_state_before"]:
        raise RuntimeError(f"{case['id']}: RNG-before mismatch")
    if expected_status != STATUS_OK:
        if (actual["state_after"] != expected["rng_state_before"] or
                actual["draws_after"] != 0):
            raise RuntimeError(f"{case['id']}: failure consumed RNG state")
        return {
            "id": case["id"], "status": expected["expected_status"],
            "transactional_rng_rollback": True,
        }
    result = actual["result"]
    if (result["selected_token_id"] != expected["selected_token_id"] or
            result["top_k_retained_count"] != expected["top_k_retained_count"] or
            result["retained_count"] != expected["retained_count"] or
            result["rng_word"] != expected["rng_word"] or
            actual["state_after"] != expected["rng_state_after"] or
            actual["draws_after"] != 1):
        raise RuntimeError(f"{case['id']}: exact native result mismatch")
    expected_ids = expected["retained_ids"]
    actual_ids = sorted(actual["tokens"])
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"{case['id']}: retained IDs differ: {actual_ids} != {expected_ids}")
    expected_records = {item["token_id"]: item
                        for item in expected["retained"]}
    logit_differences = []
    probability_differences = []
    for token_id in expected_ids:
        expected_record = expected_records[token_id]
        actual_record = actual["tokens"][token_id]
        logit_differences.append(difference(
            int.from_bytes(bytes.fromhex(
                expected_record["processed_logit_f32_le_hex"]), "little"),
            actual_record["score_bits"]))
        probability_differences.append(difference(
            int.from_bytes(bytes.fromhex(
                expected_record["probability_f32_le_hex"]), "little"),
            actual_record["probability_bits"]))
    expected_uniform = int(expected["rng_word"], 16) / 4294967296.0
    if float.fromhex(result["uniform_hex"]) != expected_uniform:
        raise RuntimeError(f"{case['id']}: uniform mapping mismatch")
    return {
        "id": case["id"],
        "status": "OK",
        "selected_token_id": result["selected_token_id"],
        "top_k_retained_count": result["top_k_retained_count"],
        "retained_count": result["retained_count"],
        "rng_word": result["rng_word"],
        "logit_differences": logit_differences,
        "probability_differences": probability_differences,
        "normalized_sum_f32_le_hex": struct.pack(
            "<I", result["normalized_sum_bits"]).hex(),
    }


def maxima(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [value for record in records for value in record.get(field, [])]
    if not values:
        return {"max_ulp": 0, "max_absolute": 0.0, "max_relative": 0.0}
    return {
        "max_ulp": max(value["ulp"] for value in values),
        "max_absolute": max(value["absolute"] for value in values),
        "max_relative": max(value["relative"] for value in values),
    }


def contract_from_calibration(calibration: list[dict[str, Any]]) -> dict[str, Any]:
    logit = maxima(calibration, "logit_differences")
    probability = maxima(calibration, "probability_differences")
    return {
        "processed_logit": {
            "comparison": ("EXACT_BITS" if logit["max_ulp"] == 0
                           else "CALIBRATED_FLOAT"),
            "calibration_observed": logit,
            "maximum_ulp": logit["max_ulp"],
            "maximum_absolute": logit["max_absolute"],
        },
        "probability": {
            "comparison": "CALIBRATED_FLOAT",
            "calibration_observed": probability,
            "maximum_ulp": probability["max_ulp"] + 2,
            "maximum_absolute": max(probability["max_absolute"] * 1.25,
                                    2.0 ** -24),
        },
    }


def check_holdout_contract(records: list[dict[str, Any]],
                           contract: dict[str, Any]) -> None:
    for record in records:
        for field, name in (("logit_differences", "processed_logit"),
                            ("probability_differences", "probability")):
            policy = contract[name]
            for value in record.get(field, []):
                if (value["ulp"] > policy["maximum_ulp"] or
                        value["absolute"] > policy["maximum_absolute"]):
                    raise RuntimeError(
                        f"{record['id']}: {name} exceeds frozen calibration contract")


def validate_rng(probe: Path, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for case in evidence["cases"]:
        lines = run(probe, ["rng", str(case["seed"]), str(case["stream"]),
                            str(len(case["words"]))])
        rng = lines[0].split()
        if (rng[0] != "RNG" or rng[1] != case["seeded_state_hex"] or
                rng[2] != case["increment_hex"]):
            raise RuntimeError("native RNG seed expansion mismatch")
        words = []
        for expected, line in zip(case["words"], lines[1:-1]):
            fields = line.split()
            if (fields[0] != "WORD" or fields[2] != expected["word_hex"] or
                    fields[3] != expected["state_after_hex"]):
                raise RuntimeError("native RNG word/state mismatch")
            words.append(fields[2])
        if lines[-1] != "STATUS 0":
            raise RuntimeError("native RNG probe did not complete")
        records.append({"seed": case["seed"], "stream": case["stream"],
                        "words": words, "status": "EXACT_BITS_PASS"})
    return records


def validate_boundaries(probe: Path,
                        evidence: dict[str, Any]) -> list[dict[str, Any]]:
    expected_ids = [0, 1, 1]
    records = []
    for item, expected in zip(evidence["explicit_word_boundaries"], expected_ids):
        word = str(int(item["word_hex"], 16))
        fields = run(probe, ["word", word, str(0x3F000000),
                             str(0x3F000000)])[0].split()
        if (fields[0] != "WORD_RESULT" or int(fields[1]) != STATUS_OK or
                int(fields[2]) != expected):
            raise RuntimeError("native categorical boundary mismatch")
        records.append({"word_hex": item["word_hex"],
                        "selected_index": expected})
    return records


def validate_statistics(probe: Path,
                        evidence: dict[str, Any]) -> list[dict[str, Any]]:
    bound = evidence["absolute_error_bound"]
    records = []
    for case in evidence["cases"]:
        probability_bits = [int.from_bytes(bytes.fromhex(
            item["expected_probability_f32_le_hex"]), "little")
                            for item in case["outcomes"]]
        lines = run(probe, ["stat", str(case["seed"]), str(case["stream"]),
                            str(case["trials"]),
                            *(str(bits) for bits in probability_bits)])
        fields = lines[0].split()
        if fields[0] != "STAT" or int(fields[1]) != len(probability_bits):
            raise RuntimeError(f"{case['id']}: invalid native statistical output")
        counts = [int(value) for value in fields[2:]]
        outcomes = []
        passed = True
        for source, count in zip(case["outcomes"], counts):
            expected = bits_float(int.from_bytes(bytes.fromhex(
                source["expected_probability_f32_le_hex"]), "little"))
            frequency = count / case["trials"]
            error = abs(frequency - expected)
            passed = passed and error <= bound
            outcomes.append({"token_id": source["token_id"], "count": count,
                             "frequency": frequency,
                             "absolute_error": error,
                             "pass": error <= bound})
        if not passed:
            raise RuntimeError(f"{case['id']}: native statistical bound failed")
        records.append({"id": case["id"], "outcomes": outcomes,
                        "pass": True})
    return records


def info(probe: Path) -> dict[str, Any]:
    fields = run(probe, ["info"])[0].split()
    if fields[0] != "INFO":
        raise RuntimeError("native sampling info probe failed")
    return {
        "immutable_config_bytes": int(fields[1]),
        "target_scratch_bytes": int(fields[2]),
        "rng_state_bytes": int(fields[3]),
        "result_bytes": int(fields[4]),
    }


def generate_validation(probe: Path, evidence_dir: Path) -> dict[str, Any]:
    calibration_source = load(evidence_dir / "sampling-calibration.json")
    holdout_source = load(evidence_dir / "sampling-holdout.json")
    rng_source = load(evidence_dir / "sampling-rng-vectors.json")
    statistical_source = load(evidence_dir / "sampling-statistical.json")
    calibration = [validate_case(probe, case)
                   for case in calibration_source["cases"]]
    contract = contract_from_calibration(calibration)
    holdout = [validate_case(probe, case) for case in holdout_source["cases"]]
    check_holdout_contract(holdout, contract)
    return {
        "schema": "kq-sampling-native-validation-v1",
        "comparison_subject": "native Kestrel-Q C17 sampler",
        "expected_values_from_kestrel_q": False,
        "native_output_used_as_oracle": False,
        "metrics": info(probe),
        "rng": validate_rng(probe, rng_source),
        "categorical_boundaries": validate_boundaries(probe, rng_source),
        "floating_contract": contract,
        "calibration": calibration,
        "holdout": holdout,
        "calibration_maxima": {
            "processed_logit": maxima(calibration, "logit_differences"),
            "probability": maxima(calibration, "probability_differences"),
        },
        "holdout_maxima": {
            "processed_logit": maxima(holdout, "logit_differences"),
            "probability": maxima(holdout, "probability_differences"),
        },
        "statistical_method": {
            "family_alpha": statistical_source["family_alpha"],
            "multiple_test_policy": statistical_source["multiple_test_policy"],
            "absolute_error_bound": statistical_source["absolute_error_bound"],
            "threshold_predeclared": True,
        },
        "statistical": validate_statistics(probe, statistical_source),
        "exact_case_count": len(calibration) + len(holdout),
        "transactional_padded_failures": sum(
            item["status"] == "INVALID_TOKEN_ID" for item in holdout),
        "status": "COMPLETE_PASS",
    }


def final_manifest(evidence_dir: Path, native_path: Path) -> dict[str, Any]:
    manifest = load(evidence_dir / "sampling-manifest.json")
    for name in ORACLE_FILES:
        path = evidence_dir / name
        recorded = manifest["files"].get(name, {})
        if (recorded.get("sha256") != sha256(path) or
                recorded.get("bytes") != path.stat().st_size):
            raise RuntimeError(f"manifest oracle hash mismatch: {name}")
    manifest["files"]["sampling-native-validation.json"] = {
        "sha256": sha256(native_path),
        "bytes": native_path.stat().st_size,
        "status": "GENERATED_COMPARISON_ONLY",
    }
    manifest["native_validation_tool"] = "tools/validate-native-sampling.py"
    manifest["native_validation_status"] = "COMPLETE_PASS"
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    probe = args.probe.resolve()
    evidence = args.evidence_dir.resolve()
    if not probe.is_file():
        raise SystemExit("native sampling probe is unavailable")
    for name in (*ORACLE_FILES, "sampling-manifest.json"):
        if not (evidence / name).is_file():
            raise SystemExit(f"sampling evidence is missing: {name}")
    validation = generate_validation(probe, evidence)
    if args.verify:
        with tempfile.TemporaryDirectory(prefix="kq-sampling-native-") as directory:
            candidate = Path(directory) / "sampling-native-validation.json"
            write_json(candidate, validation)
            current = evidence / "sampling-native-validation.json"
            if not current.is_file() or current.read_bytes() != candidate.read_bytes():
                raise SystemExit("native sampling validation is not byte-identical")
            candidate_manifest = final_manifest(evidence, candidate)
            manifest_path = Path(directory) / "sampling-manifest.json"
            write_json(manifest_path, candidate_manifest)
            if (evidence / "sampling-manifest.json").read_bytes() != \
                    manifest_path.read_bytes():
                raise SystemExit("sampling manifest is not byte-identical")
        print("Native sampling evidence verification: PASS")
        return 0
    native_path = evidence / "sampling-native-validation.json"
    write_json(native_path, validation)
    write_json(evidence / "sampling-manifest.json",
               final_manifest(evidence, native_path))
    print("Native sampling validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
