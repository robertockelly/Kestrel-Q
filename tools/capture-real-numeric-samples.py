#!/usr/bin/env python3
"""Capture safe Task 2.5 real-GGUF sample evidence.

Raw packed bytes exist only in captured subprocess memory long enough to call
the pinned llama.cpp helper. They are never printed or written to evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path


MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
LLAMA_REVISION = "90c26fcd4b2114b4aa39d09d69318cb8f438d27a"
GGUF_REVISION = "c8b5954a88c2775c546b92593eda40ea041d3176"
GGUF_SHA256 = "8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2"
GGUF_SIZE = 111334654400
PAYLOAD_BUDGET = 1048576


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def run(command: list[str], environment: dict[str, str] | None = None) -> str:
    completed = subprocess.run(command, check=True, capture_output=True,
                               text=True, env=environment)
    return completed.stdout


def parse_oracle_bits(text: str) -> list[str]:
    text = text.strip()
    if not text.startswith("bits="):
        raise RuntimeError("unexpected llama oracle output")
    return text[5:].split(",")


def decoded_hash(bits: list[str]) -> str:
    payload = b"".join(struct.pack("<I", int(value, 16)) for value in bits)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-probe", type=Path, required=True)
    parser.add_argument("--llama-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gguf-path", type=Path)
    args = parser.parse_args()
    if not args.real_probe.is_file() or not args.llama_oracle.is_file():
        raise SystemExit("real probe and llama oracle executables are required")
    gguf_path = args.gguf_path or (Path(os.environ["KQ_GGUF_PATH"])
                                   if os.environ.get("KQ_GGUF_PATH") else None)
    if gguf_path is None or not gguf_path.is_file():
        raise SystemExit("KQ_GGUF_PATH or --gguf-path must identify the exact registered GGUF")
    if gguf_path.stat().st_size != GGUF_SIZE:
        raise SystemExit("registered GGUF size mismatch")

    environment = dict(os.environ)
    environment["KQ_GGUF_PATH"] = str(gguf_path)
    text = run([str(args.real_probe.resolve()), "--oracle-raw"], environment)
    samples: list[dict[str, object]] = []
    total_bytes = None
    total_blocks = None
    for line in text.splitlines():
        fields = line.split("\t")
        if fields[0] == "sample":
            if len(fields) != 11:
                raise RuntimeError("real probe raw record shape changed")
            _, sample_id, semantic_id, type_name, block_index, packed_bytes, native_hash, minimum, maximum, total, raw_hex = fields
            raw_length = len(raw_hex) // 2
            if raw_length != int(packed_bytes):
                raise RuntimeError(f"raw byte count mismatch for {sample_id}")
            expected_bits = parse_oracle_bits(run([str(args.llama_oracle.resolve()), type_name, raw_hex]))
            oracle_hash = decoded_hash(expected_bits)
            if oracle_hash != native_hash:
                raise RuntimeError(f"native/llama decoded output mismatch for {sample_id}")
            samples.append({
                "id": sample_id,
                "semantic_id": semantic_id,
                "physical_type": type_name,
                "logical_block_index": int(block_index),
                "packed_bytes_touched": int(packed_bytes),
                "decoded_element_count": len(expected_bits),
                "decoded_output_sha256": native_hash,
                "decoded_min_hexfloat": minimum,
                "decoded_max_hexfloat": maximum,
                "decoded_sum_f64_hexfloat": total,
                "oracle_comparison": "EXACT_BITS_PASS",
            })
            raw_hex = ""  # Drop the only raw representation before writing evidence.
        elif fields[0] == "summary":
            if len(fields) != 7:
                raise RuntimeError("real probe summary shape changed")
            total_bytes = int(fields[2])
            total_blocks = int(fields[4])
            if int(fields[6]) != PAYLOAD_BUDGET:
                raise RuntimeError("real probe budget guard changed")
    if total_bytes is None or total_blocks is None or len(samples) != 9:
        raise RuntimeError("real sample set is incomplete")
    if total_bytes != sum(int(sample["packed_bytes_touched"]) for sample in samples):
        raise RuntimeError("real sample byte accounting mismatch")
    if total_blocks != len(samples) or total_bytes > PAYLOAD_BUDGET:
        raise RuntimeError("real sample count/budget invariant failed")
    if ({sample["physical_type"] for sample in samples} !=
            {"F32", "BF16", "Q5_1", "Q8_0", "Q4_K", "Q5_K", "IQ4_NL"}):
        raise RuntimeError("real samples do not cover all seven types")

    evidence = {
        "asset_id": "KQ-NUMERIC-REAL-GGUF-001",
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "artifact": {
            "id": "KQ-MODEL-ARTIFACT-001",
            "filename": "Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf",
            "registered_size_bytes": GGUF_SIZE,
            "registered_sha256": GGUF_SHA256,
            "unsloth_revision": GGUF_REVISION,
            "identity_verification": "registered Task 1.0 identity; this bounded run rechecked exact file size and target model compatibility without rehashing the full file",
        },
        "oracle": {"implementation": "ggml-org/llama.cpp",
                   "revision": LLAMA_REVISION, "license": "MIT"},
        "access_path": "semantic descriptor -> Task 2.2 bounded view/member -> one packed block -> Task 2.5 scalar decode",
        "samples": samples,
        "summary": {
            "samples": len(samples),
            "types": 7,
            "real_model_payload_logical_bytes_touched": total_bytes,
            "real_model_payload_blocks_touched": total_blocks,
            "sample_budget_bytes": PAYLOAD_BUDGET,
            "within_budget": True,
            "oracle_mismatches": 0,
            "raw_model_bytes_recorded": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(evidence))
    print(f"real_samples={len(samples)}")
    print(f"real_model_payload_logical_bytes_touched={total_bytes}")
    print(f"real_model_payload_blocks_touched={total_blocks}")
    print(f"sha256.real-gguf-samples.json={hashlib.sha256(args.output.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
