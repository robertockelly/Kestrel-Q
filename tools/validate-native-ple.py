#!/usr/bin/env python3
"""Compare the native Task 2.4 PLE API with independent canonical evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


GOLDEN_SHA256 = "495ef70f091e8d61caac99bb14ad8cea0fdb77940ec4dc6e8ce9811a144da3b6"
DIFFERENTIAL_SHA256 = "b9c9be4d927d59c9ac12ba2313034cda5a1857d5484fca479327c9b771cb9671"


class ValidationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checked(path: Path, expected: str) -> dict[str, Any]:
    if sha256(path) != expected:
        raise ValidationError(f"evidence hash mismatch: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_probe(probe: Path, model: Path, operation: str, *arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        [str(probe), str(model), operation, *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if completed.returncode != 0:
        raise ValidationError(
            f"native PLE probe failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    result = json.loads(completed.stdout)
    if result.get("payload_bytes_accessed") != 0:
        raise ValidationError("native PLE probe crossed the tensor payload boundary")
    return result


def compact_expected(row: dict[str, Any]) -> list[dict[str, int]]:
    return [
        {
            "position": row["position"],
            "token_id": row["token_id"],
            "ngram_order": address["ngram_order"],
            "local_head": address["local_head"],
            "global_head": (address["ngram_order"] - 2) * 8 + address["local_head"],
            "global_address": address["global_address"],
            "head_offset": address["head_offset"],
            "head_vocab_size": address["head_vocab_size"],
            "partition_id": address["partition_id"],
            "partition_row": address["partition_row"],
        }
        for address in row["addresses"]
    ]


def expected_intents(rows: list[dict[str, Any]]) -> list[dict[str, int]]:
    return [intent for row in rows for intent in compact_expected(row)]


def validate_sequence(probe: Path, model: Path, case: dict[str, Any]) -> None:
    tokens = case["input_ids"]
    actual = run_probe(probe, model, "sequence", ",".join(map(str, tokens)))
    replay = run_probe(probe, model, "sequence", ",".join(map(str, tokens)))
    if replay != actual:
        raise ValidationError(f"PLE reset/replay mismatch: {case.get('id', case.get('case_id'))}")
    if actual["tokens"] != tokens or actual["intents"] != expected_intents(case["rows"]):
        raise ValidationError(f"PLE sequence mismatch: {case.get('id', case.get('case_id'))}")


def validate_stream(probe: Path, model: Path, case: dict[str, Any]) -> None:
    prefill = case["prefill_input_ids"]
    step_ids = [step["token_id"] for step in case["steps"]]
    actual = run_probe(
        probe,
        model,
        "decode",
        ",".join(map(str, prefill)),
        ",".join(map(str, step_ids)),
    )
    if len(actual["steps"]) != len(case["steps"]):
        raise ValidationError(f"PLE decode count mismatch: {case.get('id', case.get('case_id'))}")
    for expected, observed in zip(case["steps"], actual["steps"], strict=True):
        expected_position = len(prefill) + expected["step"]
        expected_row = {
            "position": expected_position,
            "token_id": expected["token_id"],
            "addresses": expected["addresses"],
        }
        if observed["token_id"] != expected["token_id"] or \
                observed["intents"] != compact_expected(expected_row) or \
                observed["state"]["history"] != expected["history_after"] or \
                observed["state"]["position"] != expected_position + 1:
            raise ValidationError(
                f"PLE decode mismatch: {case.get('id', case.get('case_id'))} step {expected['step']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--differential", type=Path, required=True)
    args = parser.parse_args()

    model_raw = os.environ.get("KQ_GGUF_PATH")
    if not model_raw:
        print("KQ_GGUF_PATH unavailable; native PLE oracle skipped")
        return 77
    model = Path(model_raw)
    if not model.is_file() or not args.probe.is_file():
        raise ValidationError("PLE probe or registered model artifact is unavailable")

    golden = load_checked(args.golden, GOLDEN_SHA256)
    differential = load_checked(args.differential, DIFFERENTIAL_SHA256)
    for case in golden["vectors"]:
        validate_sequence(args.probe, model, case)
    validate_stream(args.probe, model, golden["decode_case"])
    for case in differential["sequence_cases"]:
        validate_sequence(args.probe, model, case)
    for case in differential["stream_cases"]:
        validate_stream(args.probe, model, case)

    text_case = differential["tokenizer_integration_case"]
    actual = run_probe(
        args.probe,
        model,
        "text",
        text_case["input_utf8"].encode("utf-8").hex(),
    )
    if actual["tokens"] != text_case["token_ids"] or \
            actual["intents"] != expected_intents(text_case["rows"]):
        raise ValidationError("tokenizer-to-PLE differential mismatch")

    print(
        json.dumps(
            {
                "status": "PASS",
                "original_sequence_vectors": len(golden["vectors"]),
                "original_decode_steps": len(golden["decode_case"]["steps"]),
                "differential_sequence_cases": len(differential["sequence_cases"]),
                "differential_stream_cases": len(differential["stream_cases"]),
                "tokenizer_integration_cases": 1,
                "PLE_payload_views_opened": 0,
                "model_tensor_payload_bytes_touched": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        raise SystemExit(1)
