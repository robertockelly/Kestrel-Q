#!/usr/bin/env python3
"""Compare native physical/view geometry with pinned Task 1.3 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys


SKIP = 77
EXPECTED_COLUMNS = (
    "physical_name",
    "rank",
    "dimensions",
    "type_id",
    "type_name",
    "block_elements",
    "bytes_per_block",
    "element_count",
    "relative_offset",
    "data_offset",
    "packed_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    return parser.parse_args()


def resolve_model(argument: Path | None) -> Path:
    if argument is not None:
        return argument
    value = os.environ.get("KQ_GGUF_PATH")
    if not value:
        print("KQ_GGUF_PATH unavailable; tensor-view oracle skipped")
        raise SystemExit(SKIP)
    return Path(value)


def run_dump(inspect: Path, model: Path) -> bytes:
    completed = subprocess.run(
        [str(inspect), "--view-geometry-dump", str(model)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "native geometry dump failed: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    return completed.stdout


def load_native(data: bytes) -> dict[str, dict[str, str]]:
    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
        raise ValueError("native geometry dump schema mismatch")
    rows: dict[str, dict[str, str]] = {}
    for row in reader:
        name = row["physical_name"]
        if not name or name in rows:
            raise ValueError(f"duplicate/empty native tensor name: {name!r}")
        rows[name] = row
    return rows


def load_evidence(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows: dict[str, dict[str, str]] = {}
        for row in csv.DictReader(stream):
            name = row["gguf_tensor_name"]
            if not name or name in rows:
                raise ValueError(f"duplicate/empty evidence tensor: {name!r}")
            rows[name] = row
    return rows


def compare(native: dict[str, dict[str, str]],
            evidence: dict[str, dict[str, str]]) -> None:
    if set(native) != set(evidence):
        missing = sorted(set(evidence) - set(native))
        extra = sorted(set(native) - set(evidence))
        raise ValueError(
            f"physical-name coverage mismatch: missing={missing[:3]} "
            f"extra={extra[:3]}"
        )
    mappings = {
        "rank": "rank",
        "dimensions": "dimensions",
        "type_id": "type_id",
        "type_name": "type_name",
        "block_elements": "block_size",
        "bytes_per_block": "bytes_per_block",
        "element_count": "parameter_count",
        "relative_offset": "relative_offset",
        "data_offset": "absolute_offset",
        "packed_bytes": "packed_bytes",
    }
    for name, native_row in native.items():
        evidence_row = evidence[name]
        for native_key, evidence_key in mappings.items():
            if native_row[native_key] != evidence_row[evidence_key]:
                raise ValueError(
                    f"{name}: {native_key}={native_row[native_key]!r} "
                    f"does not match evidence {evidence_row[evidence_key]!r}"
                )


def main() -> int:
    args = parse_args()
    model = resolve_model(args.model)
    if not args.inspect.is_file():
        raise FileNotFoundError(f"inspect executable missing: {args.inspect}")
    if not args.inventory.is_file():
        raise FileNotFoundError(f"Task 1.3 inventory missing: {args.inventory}")
    if not model.is_file():
        raise FileNotFoundError("KQ_GGUF_PATH does not identify a file")

    first = run_dump(args.inspect, model)
    second = run_dump(args.inspect, model)
    if first != second:
        raise ValueError("native geometry dump is not byte-identical")
    native = load_native(first)
    evidence = load_evidence(args.inventory)
    compare(native, evidence)
    packed_bytes = sum(int(row["packed_bytes"]) for row in native.values())
    digest = hashlib.sha256(first).hexdigest()
    print(
        "tensor-view geometry oracle: PASS; "
        f"physical={len(native)} packed_bytes={packed_bytes} "
        f"dump_sha256={digest} payload_bytes_touched_by_test=0"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"tensor-view geometry oracle: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
