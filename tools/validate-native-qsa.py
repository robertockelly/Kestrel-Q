#!/usr/bin/env python3
"""Compare native scalar QSA with pinned independent Class-C evidence."""

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
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True,
                                 ensure_ascii=False) + "\n").encode("utf-8"))


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def state_flat(record: dict[str, Any], name: str) -> list[int]:
    return bits(record[name])


def standard_case(case: dict[str, Any]) -> dict[str, Any]:
    length = case["input"]["shape"][1]
    return {
        "id": case["id"],
        "input": case["input"],
        "initial_state": {"length": 0, "key": [], "value": [], "raw": []},
        "expected_output": case["expected"]["output"],
        "expected_state": case["expected"]["state"],
        "selection": case["expected"]["selection"],
        "attention": case["expected"]["attention_probabilities"],
        "query_length": length,
    }


def continuation_cases(record: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    previous = record["prefix_state"]
    for step in record["decode_steps"]:
        output.append({
            "id": f"{record['id']}-step-{step['step']}",
            "input": step["input"],
            "initial_state": {
                "length": previous["length"],
                "key": state_flat(previous, "key"),
                "value": state_flat(previous, "value"),
                "raw": state_flat(previous, "raw_index_key"),
            },
            "expected_output": step["expected_output"],
            "expected_state": step["state"],
            "selection": [step["selection"]],
            "attention": step["attention_probabilities"],
            "query_length": 1,
        })
        previous = step["state"]
    return output


def make_request(contract: dict[str, Any], case: dict[str, Any]) -> str:
    config = contract["reduced_config"]
    initial = case["initial_state"]
    sequence = case["input"]["shape"][1]
    capacity = max(16, initial["length"] + sequence)
    lines = [
        "KQQSA1",
        "CONFIG {hidden_size} {query_heads} {key_value_heads} {head_dim} "
        "{index_query_heads} {index_head_dim} {block_size} {token_budget} "
        f"{sequence} {capacity} {initial['length']}".format(**config),
    ]
    weights = contract["weights"]
    qk = bits(weights["indexer.index_qk_proj.weight"])
    iq_count = config["index_query_heads"] * config["index_head_dim"] * config["hidden_size"]
    append_array(lines, "QUERY", bits(weights["q_proj.weight"]))
    append_array(lines, "KEY", bits(weights["k_proj.weight"]))
    append_array(lines, "VALUE", bits(weights["v_proj.weight"]))
    append_array(lines, "OUTPUT_WEIGHT", bits(weights["o_proj.weight"]))
    append_array(lines, "QUERY_NORM", bits(weights["q_norm.weight"]))
    append_array(lines, "KEY_NORM", bits(weights["k_norm.weight"]))
    append_array(lines, "INDEX_QUERY", qk[:iq_count])
    append_array(lines, "INDEX_KEY", qk[iq_count:])
    append_array(lines, "INDEX_QUERY_NORM", bits(weights["indexer.q_layernorm.weight"]))
    append_array(lines, "INDEX_KEY_NORM", bits(weights["indexer.k_layernorm.weight"]))
    append_array(lines, "INITIAL_KEY", initial["key"])
    append_array(lines, "INITIAL_VALUE", initial["value"])
    append_array(lines, "INITIAL_RAW", initial["raw"])
    append_array(lines, "INPUT", bits(case["input"]))
    return "\n".join(lines) + "\n"


def run_probe(probe: Path, contract: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [str(probe)], input=make_request(contract, case), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"native QSA probe failed for {case['id']}: rc={result.returncode}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}")
    parsed: dict[str, Any] = {"selections": {}, "traces": {}}
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "TRACE":
            kind = fields[1]
            token = int(fields[2])
            rank = int(fields[3])
            dimensions = [int(value) for value in fields[4:4 + rank]]
            count_at = 4 + rank
            count = int(fields[count_at])
            values = [int(value, 16) for value in fields[count_at + 1:]]
            if len(values) != count:
                raise RuntimeError(f"bad trace count: {case['id']} {kind}")
            parsed["traces"][(kind, token)] = {
                "dimensions": dimensions, "bits": values}
        elif fields[0] == "SELECT":
            at = 1
            token = int(fields[at]); at += 1
            absolute = int(fields[at]); at += 1
            candidate_count = int(fields[at]); at += 1
            candidates, score_bits = [], []
            for _ in range(candidate_count):
                candidate, value = fields[at].split(":", 1); at += 1
                candidates.append(int(candidate)); score_bits.append(int(value, 16))
            block_count = int(fields[at]); at += 1
            selected_blocks = [int(value) for value in fields[at:at + block_count]]
            at += block_count
            token_count = int(fields[at]); at += 1
            selected_tokens = [int(value) for value in fields[at:at + token_count]]
            at += token_count
            tail_count = int(fields[at]); at += 1
            if at != len(fields):
                raise RuntimeError(f"bad selection record: {case['id']}")
            parsed["selections"][token] = {
                "absolute_position": absolute,
                "candidate_block_ids": candidates,
                "candidate_score_bits": score_bits,
                "selected_block_ids": selected_blocks,
                "selected_token_positions": selected_tokens,
                "tail_count": tail_count,
            }
        elif fields[0] == "RESULT":
            if fields[1] != "OK":
                raise RuntimeError(f"native result error: {line}")
            parsed["result"] = "OK"
        elif fields[0] in {"OUTPUT", "FINAL_KEY", "FINAL_VALUE", "FINAL_RAW"}:
            count = int(fields[1]); values = [int(value, 16) for value in fields[2:]]
            if len(values) != count:
                raise RuntimeError(f"bad {fields[0]} count: {case['id']}")
            parsed[fields[0].lower()] = values
        elif fields[0] == "LENGTH":
            parsed["length"] = int(fields[1])
        elif fields[0] == "METRICS":
            parsed["metrics"] = {
                "config_owned_bytes": int(fields[1]),
                "state_owned_bytes": int(fields[2]),
                "scratch_bytes": int(fields[3]),
                "semantic_state_bytes_per_token": int(fields[4]),
            }
        elif fields[0] == "TIMING":
            parsed["timing_ns"] = int(fields[1])
    required = {"result", "output", "final_key", "final_value", "final_raw",
                "length", "metrics", "timing_ns"}
    missing = required - parsed.keys()
    if missing:
        raise RuntimeError(f"missing probe fields for {case['id']}: {sorted(missing)}")
    return parsed


def run_selection_probe(probe: Path, case: dict[str, Any]) -> None:
    selection = case["selection"]
    score_bits = bits(selection["candidate_scores"])
    limit = selection["selection_limit"]
    request = (f"KQQSASEL1 {len(score_bits)} {limit}\n" +
               " ".join(f"{value:08x}" for value in score_bits) + "\n")
    result = subprocess.run(
        [str(probe), "--selection"], input=request, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"native QSA selection probe failed for {case['id']}: "
            f"{result.stderr}{result.stdout}")
    fields = result.stdout.split()
    if not fields or fields[0] != "SELECTED":
        raise RuntimeError(f"malformed selection probe output for {case['id']}")
    count = int(fields[1])
    selected = [int(value) for value in fields[2:]]
    if count != len(selected) or selected != selection["selected_block_ids"]:
        raise RuntimeError(f"threshold selection mismatch for {case['id']}")
    block_size = 4
    tokens = [block * block_size + offset
              for block in selected for offset in range(block_size)]
    tokens.extend(selection["tail_positions"])
    if tokens != selection["selected_token_positions"]:
        raise RuntimeError(f"threshold token ordering mismatch for {case['id']}")


def empty_stats() -> dict[str, Any]:
    return {"count": 0, "max_abs": 0.0, "max_rel": 0.0, "max_ulp": 0}


def compare_float(category: str, expected: list[int], actual: list[int],
                  stats: dict[str, dict[str, Any]],
                  contract: dict[str, Any] | None = None) -> None:
    if len(expected) != len(actual):
        raise RuntimeError(f"{category} count mismatch: {len(expected)} != {len(actual)}")
    item = stats[category]
    for expected_bits, actual_bits in zip(expected, actual):
        e, a = as_float(expected_bits), as_float(actual_bits)
        if not math.isfinite(e) or not math.isfinite(a):
            raise RuntimeError(f"non-finite {category} value")
        absolute = abs(a - e)
        relative = absolute / abs(e) if e != 0.0 else (0.0 if absolute == 0.0 else math.inf)
        ulp = abs(ordered(actual_bits) - ordered(expected_bits))
        if contract is not None and not (
            absolute <= contract["atol"] + contract["rtol"] * abs(e) and
            ulp <= contract["max_ulp"]
        ):
            raise RuntimeError(
                f"{category} holdout value exceeds calibrated contract: "
                f"abs={absolute} rel={relative} ulp={ulp}")
        item["count"] += 1
        item["max_abs"] = max(item["max_abs"], absolute)
        if math.isfinite(relative):
            item["max_rel"] = max(item["max_rel"], relative)
        item["max_ulp"] = max(item["max_ulp"], ulp)


def expected_attention_bits(case: dict[str, Any], query: int,
                            selected: list[int]) -> list[int]:
    record = case["attention"]
    values = bits(record)
    shape = record["shape"]
    heads = shape[1]
    query_count = shape[2]
    total = shape[3]
    if query >= query_count:
        raise RuntimeError("attention query index out of range")
    result = []
    for head in range(heads):
        for token in selected:
            result.append(values[((head * query_count + query) * total) + token])
    return result


def compare_case(case: dict[str, Any], native: dict[str, Any],
                 stats: dict[str, dict[str, Any]],
                 contracts: dict[str, dict[str, Any]] | None = None) -> None:
    expected_state = case["expected_state"]
    if native["length"] != expected_state["length"]:
        raise RuntimeError(f"{case['id']} state length mismatch")
    compare_float("operator_output", bits(case["expected_output"]),
                  native["output"], stats,
                  None if contracts is None else contracts["operator_output"])
    compare_float("key_state", state_flat(expected_state, "key"),
                  native["final_key"], stats,
                  None if contracts is None else contracts["key_state"])
    compare_float("value_state", state_flat(expected_state, "value"),
                  native["final_value"], stats,
                  None if contracts is None else contracts["value_state"])
    compare_float("raw_index_state", state_flat(expected_state, "raw_index_key"),
                  native["final_raw"], stats,
                  None if contracts is None else contracts["raw_index_state"])
    if len(native["selections"]) != len(case["selection"]):
        raise RuntimeError(f"{case['id']} selection count mismatch")
    for query, expected in enumerate(case["selection"]):
        actual = native["selections"][query]
        exact_fields = (
            "absolute_position", "candidate_block_ids", "selected_block_ids",
            "selected_token_positions")
        for field in exact_fields:
            if actual[field] != expected[field]:
                raise RuntimeError(f"{case['id']} {field} mismatch at query {query}")
        if actual["tail_count"] != len(expected["tail_positions"]):
            raise RuntimeError(f"{case['id']} tail count mismatch at query {query}")
        compare_float("candidate_scores", bits(expected["candidate_scores"]),
                      actual["candidate_score_bits"], stats,
                      None if contracts is None else contracts["candidate_scores"])
        trace = native["traces"].get(("ATTENTION_PROBABILITIES", query))
        if trace is None:
            raise RuntimeError(f"{case['id']} missing attention trace {query}")
        expected_probs = expected_attention_bits(
            case, query, expected["selected_token_positions"])
        compare_float("attention_probabilities", expected_probs,
                      trace["bits"], stats,
                      None if contracts is None else contracts["attention_probabilities"])


def within_contract(observed: dict[str, Any], contract: dict[str, Any]) -> bool:
    if observed["count"] == 0:
        return True
    return (observed["max_abs"] <= contract["atol"] +
            contract["rtol"] and
            observed["max_ulp"] <= contract["max_ulp"])


def validate(probe: Path, evidence_dir: Path, output_path: Path) -> dict[str, Any]:
    contract = load(evidence_dir / "qsa-contract.json")
    calibration = load(evidence_dir / "qsa-calibration.json")
    holdout = load(evidence_dir / "qsa-holdout.json")
    state = load(evidence_dir / "qsa-state-vectors.json")
    selection_vectors = load(evidence_dir / "qsa-selection-vectors.json")
    categories = (
        "operator_output", "key_state", "value_state", "raw_index_state",
        "candidate_scores", "attention_probabilities")
    calibration_stats = defaultdict(empty_stats)
    holdout_stats = defaultdict(empty_stats)
    cases_run = 0
    metrics: dict[str, Any] = {}
    for selection_case in selection_vectors["cases"]:
        run_selection_probe(probe, selection_case)
    for source in calibration["cases"]:
        case = standard_case(source)
        native = run_probe(probe, contract, case)
        compare_case(case, native, calibration_stats)
        metrics = native["metrics"]
        cases_run += 1
    contracts = {
        category: {
            "comparison": "CALIBRATED_FLOAT",
            "atol": calibration_stats[category]["max_abs"],
            "rtol": calibration_stats[category]["max_rel"],
            "max_ulp": calibration_stats[category]["max_ulp"],
        }
        for category in categories
    }
    holdout_cases = [standard_case(item) for item in holdout["cases"]]
    holdout_cases.extend(continuation_cases(state["cases"][0]))
    for case in holdout_cases:
        native = run_probe(probe, contract, case)
        compare_case(case, native, holdout_stats, contracts)
        cases_run += 1
    failures = [category for category in categories
                if not within_contract(holdout_stats[category], contracts[category])]
    if failures:
        raise RuntimeError("QSA holdout exceeds calibration contract: " +
                           ", ".join(failures) +
                           "; calibration=" + json.dumps(dict(calibration_stats), sort_keys=True) +
                           "; holdout=" + json.dumps(dict(holdout_stats), sort_keys=True))
    result = {
        "schema": "kq-qsa-native-validation-v1",
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "native_probe": "kq_qsa_probe",
        "cases_run": cases_run + len(selection_vectors["cases"]),
        "calibration_case_count": len(calibration["cases"]),
        "holdout_case_count": len(holdout_cases),
        "threshold_selection_case_count": len(selection_vectors["cases"]),
        "selection_comparison": "EXACT_DISCRETE",
        "selection_result": "PASS",
        "state_transition_result": "PASS",
        "prefill_decode_result": "PASS",
        "floating_contracts": contracts,
        "calibration_observed": dict(calibration_stats),
        "holdout_observed": dict(holdout_stats),
        "metrics": metrics,
        "result": "PASS",
    }
    write_json(output_path, result)
    manifest = {
        "schema": "kq-qsa-evidence-manifest-v1",
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "oracle_generation_tool": "tools/generate-qsa-reference.py",
        "native_validation_tool": "tools/validate-native-qsa.py",
        "offline": True,
        "full_model_weights_downloaded": False,
        "expected_values_from_kestrel_q": False,
        "files": {
            name: {"sha256": sha256(evidence_dir / name),
                   "bytes": (evidence_dir / name).stat().st_size}
            for name in (*ORACLE_FILES, output_path.name)
        },
    }
    write_json(evidence_dir / "qsa-manifest.json", manifest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    evidence = args.evidence_dir.resolve()
    output = evidence / "qsa-native-validation.json"
    if args.verify:
        previous = output.read_bytes()
        previous_manifest = (evidence / "qsa-manifest.json").read_bytes()
        with tempfile.TemporaryDirectory(prefix="kq-qsa-native-") as directory:
            candidate = Path(directory) / "qsa-native-validation.json"
            validate(args.probe.resolve(), evidence, candidate)
            generated = candidate.read_bytes()
        if generated != previous:
            raise SystemExit("native QSA validation is not byte-identical")
        if (evidence / "qsa-manifest.json").read_bytes() != previous_manifest:
            raise SystemExit("QSA evidence manifest is not byte-identical")
        print("Native QSA validation deterministic regeneration: PASS")
        return 0
    result = validate(args.probe.resolve(), evidence, output)
    print(f"Native QSA independent oracle validation: {result['result']}; "
          f"cases={result['cases_run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
