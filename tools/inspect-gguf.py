#!/usr/bin/env python3
"""Read-only, metadata-only GGUF v3 inspector for KQ-MODEL-ARTIFACT-001."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO


EXPECTED_FILENAME = "Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf"
EXPECTED_SIZE = 111_334_654_400
EXPECTED_SHA256 = "8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2"
EXPECTED_VERSION = 3
EXPECTED_METADATA = 67
EXPECTED_TENSORS = 1224
EXPECTED_ARCHITECTURE = "qwen4exp"
MAX_STRING_BYTES = 128 * 1024 * 1024
MAX_ARRAY_LENGTH = 2_000_000
MAX_HEADER_DIRECTORY_BYTES = 512 * 1024 * 1024

VALUE_TYPES = {
    0: ("UINT8", "<B"),
    1: ("INT8", "<b"),
    2: ("UINT16", "<H"),
    3: ("INT16", "<h"),
    4: ("UINT32", "<I"),
    5: ("INT32", "<i"),
    6: ("FLOAT32", "<f"),
    7: ("BOOL", "<?"),
    8: ("STRING", None),
    9: ("ARRAY", None),
    10: ("UINT64", "<Q"),
    11: ("INT64", "<q"),
    12: ("FLOAT64", "<d"),
}

# Block geometry from the pinned llama.cpp GGML type definitions documented by
# Task 1.3. Only types present in KQ-MODEL-ARTIFACT-001 are accepted here.
GGML_TYPES = {
    0: ("F32", 1, 4),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    20: ("IQ4_NL", 32, 18),
    30: ("BF16", 1, 2),
}

CSV_FIELDS = [
    "gguf_tensor_name",
    "rank",
    "dimensions",
    "parameter_count",
    "type_id",
    "type_name",
    "block_size",
    "bytes_per_block",
    "nominal_bits_per_parameter",
    "relative_offset",
    "absolute_offset",
    "packed_bytes",
    "padding_after_bytes",
    "payload_end_absolute",
]


class GgufError(RuntimeError):
    """Fail-closed structural or identity error."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_exact(stream: BinaryIO, size: int) -> bytes:
    if size < 0:
        raise GgufError(f"negative read size: {size}")
    data = stream.read(size)
    if len(data) != size:
        raise GgufError(f"truncated GGUF structure: expected {size}, got {len(data)}")
    if stream.tell() > MAX_HEADER_DIRECTORY_BYTES:
        raise GgufError("GGUF header/directory exceeds defensive byte limit")
    return data


def unpack(stream: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, read_exact(stream, size))[0]


def read_string(stream: BinaryIO) -> str:
    length = unpack(stream, "<Q")
    if length > MAX_STRING_BYTES:
        raise GgufError(f"GGUF string length exceeds limit: {length}")
    try:
        return read_exact(stream, length).decode("utf-8")
    except UnicodeDecodeError as error:
        raise GgufError("invalid UTF-8 GGUF string") from error


def read_scalar(stream: BinaryIO, type_id: int) -> Any:
    if type_id not in VALUE_TYPES or type_id in (8, 9):
        raise GgufError(f"unsupported scalar metadata type {type_id}")
    return unpack(stream, VALUE_TYPES[type_id][1])


def read_value(stream: BinaryIO, type_id: int) -> tuple[Any, str | None]:
    if type_id not in VALUE_TYPES:
        raise GgufError(f"unknown GGUF metadata type {type_id}")
    if type_id == 8:
        return read_string(stream), None
    if type_id == 9:
        element_type = unpack(stream, "<I")
        if element_type not in VALUE_TYPES or element_type == 9:
            raise GgufError(f"unsupported GGUF array element type {element_type}")
        length = unpack(stream, "<Q")
        if length > MAX_ARRAY_LENGTH:
            raise GgufError(f"GGUF array length exceeds limit: {length}")
        if element_type == 8:
            values = [read_string(stream) for _ in range(length)]
        else:
            values = [read_scalar(stream, element_type) for _ in range(length)]
        return values, VALUE_TYPES[element_type][0]
    return read_scalar(stream, type_id), None


def summarize_value(value: Any, array_element_type: str | None) -> Any:
    if not isinstance(value, list):
        return value
    encoded = canonical_json_bytes(value)
    summary: dict[str, Any] = {
        "array_element_type": array_element_type,
        "length": len(value),
        "canonical_json_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    if len(value) <= 64 and len(encoded) <= 16 * 1024:
        summary["values"] = value
    else:
        summary["first_values"] = value[:4]
        summary["last_values"] = value[-4:]
    return summary


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-full-sha256",
        action="store_true",
        help="development only; permitted only when output remains under .research-cache",
    )
    args = parser.parse_args()

    raw_path = os.environ.get("KQ_GGUF_PATH")
    if not raw_path:
        raise GgufError("KQ_GGUF_PATH is required")
    path = Path(raw_path)
    if not path.is_file():
        raise GgufError("KQ_GGUF_PATH does not identify a file")
    if path.name != EXPECTED_FILENAME:
        raise GgufError(f"unexpected GGUF filename: {path.name}")
    file_size = path.stat().st_size
    if file_size != EXPECTED_SIZE:
        raise GgufError(f"unexpected GGUF size: {file_size}")
    if args.skip_full_sha256:
        resolved_output = args.output_dir.resolve()
        cache_root = (Path.cwd() / ".research-cache").resolve()
        if cache_root not in resolved_output.parents and resolved_output != cache_root:
            raise GgufError("unverified development output must remain under .research-cache")
        observed_sha256 = EXPECTED_SHA256
        sha256_verification = "SKIPPED_DEVELOPMENT_ONLY"
    else:
        observed_sha256 = sha256_file(path)
        if observed_sha256 != EXPECTED_SHA256:
            raise GgufError(f"GGUF SHA-256 mismatch: {observed_sha256}")
        sha256_verification = "PASS_FULL_FILE"

    with path.open("rb") as stream:
        if read_exact(stream, 4) != b"GGUF":
            raise GgufError("invalid GGUF magic")
        version = unpack(stream, "<I")
        tensor_count = unpack(stream, "<Q")
        metadata_count = unpack(stream, "<Q")
        if version != EXPECTED_VERSION:
            raise GgufError(f"expected GGUF v3, got {version}")
        if tensor_count != EXPECTED_TENSORS or metadata_count != EXPECTED_METADATA:
            raise GgufError(
                f"unexpected directory counts: tensors={tensor_count}, metadata={metadata_count}"
            )

        metadata_entries = []
        metadata_values: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = read_string(stream)
            if key in metadata_values:
                raise GgufError(f"duplicate metadata key: {key}")
            type_id = unpack(stream, "<I")
            value, array_element_type = read_value(stream, type_id)
            metadata_values[key] = value
            metadata_entries.append(
                {
                    "key": key,
                    "type_id": type_id,
                    "type_name": VALUE_TYPES[type_id][0],
                    "value": summarize_value(value, array_element_type),
                }
            )
        if metadata_values.get("general.architecture") != EXPECTED_ARCHITECTURE:
            raise GgufError("GGUF architecture mismatch")
        alignment = metadata_values.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment <= 0 or alignment > 4096 or alignment & (alignment - 1):
            raise GgufError(f"unsafe GGUF alignment: {alignment!r}")

        tensors = []
        names: set[str] = set()
        for _ in range(tensor_count):
            name = read_string(stream)
            if name in names:
                raise GgufError(f"duplicate GGUF tensor name: {name}")
            names.add(name)
            rank = unpack(stream, "<I")
            if rank < 1 or rank > 8:
                raise GgufError(f"{name}: invalid rank {rank}")
            dimensions = [unpack(stream, "<Q") for _ in range(rank)]
            if any(dimension <= 0 for dimension in dimensions):
                raise GgufError(f"{name}: non-positive dimension")
            type_id = unpack(stream, "<I")
            if type_id not in GGML_TYPES:
                raise GgufError(f"{name}: unsupported GGML type {type_id}")
            relative_offset = unpack(stream, "<Q")
            type_name, block_size, bytes_per_block = GGML_TYPES[type_id]
            parameters = math.prod(dimensions)
            if dimensions[0] % block_size:
                raise GgufError(
                    f"{name}: first dimension {dimensions[0]} not divisible by block {block_size}"
                )
            packed_bytes = parameters // block_size * bytes_per_block
            tensors.append(
                {
                    "gguf_tensor_name": name,
                    "rank": rank,
                    "dimensions": dimensions,
                    "parameter_count": parameters,
                    "type_id": type_id,
                    "type_name": type_name,
                    "block_size": block_size,
                    "bytes_per_block": bytes_per_block,
                    "nominal_bits_per_parameter": bytes_per_block * 8 / block_size,
                    "relative_offset": relative_offset,
                    "packed_bytes": packed_bytes,
                }
            )
        directory_end = stream.tell()
        data_section_offset = align_up(directory_end, alignment)
        if data_section_offset > file_size:
            raise GgufError("GGUF data section begins beyond file")

    physical = sorted(tensors, key=lambda item: (item["relative_offset"], item["gguf_tensor_name"]))
    if physical[0]["relative_offset"] != 0:
        raise GgufError("first GGUF tensor does not begin at relative offset zero")
    total_packed_bytes = 0
    total_alignment_padding = 0
    for index, tensor in enumerate(physical):
        relative_offset = tensor["relative_offset"]
        if relative_offset % alignment:
            raise GgufError(f"{tensor['gguf_tensor_name']}: unaligned offset {relative_offset}")
        end = relative_offset + tensor["packed_bytes"]
        next_offset = (
            physical[index + 1]["relative_offset"]
            if index + 1 < len(physical)
            else file_size - data_section_offset
        )
        if end > next_offset:
            raise GgufError(f"{tensor['gguf_tensor_name']}: overlapping/out-of-bounds packed span")
        tensor["absolute_offset"] = data_section_offset + relative_offset
        tensor["payload_end_absolute"] = data_section_offset + end
        tensor["padding_after_bytes"] = next_offset - end
        total_packed_bytes += tensor["packed_bytes"]
        total_alignment_padding += tensor["padding_after_bytes"]
    if data_section_offset + physical[-1]["relative_offset"] + physical[-1]["packed_bytes"] > file_size:
        raise GgufError("last GGUF tensor exceeds file size")

    type_summary: dict[str, dict[str, int | float]] = {}
    for tensor in tensors:
        item = type_summary.setdefault(
            tensor["type_name"],
            {
                "type_id": tensor["type_id"],
                "block_size": tensor["block_size"],
                "bytes_per_block": tensor["bytes_per_block"],
                "nominal_bits_per_parameter": tensor["nominal_bits_per_parameter"],
                "tensor_count": 0,
                "parameter_count": 0,
                "packed_bytes": 0,
            },
        )
        item["tensor_count"] += 1
        item["parameter_count"] += tensor["parameter_count"]
        item["packed_bytes"] += tensor["packed_bytes"]

    metadata_output = {
        "schema_version": 1,
        "generated_by": "tools/inspect-gguf.py",
        "artifact_id": "KQ-MODEL-ARTIFACT-001",
        "filename": path.name,
        "file_size_bytes": file_size,
        "sha256": observed_sha256,
        "sha256_verification": sha256_verification,
        "magic": "GGUF",
        "version": version,
        "architecture": metadata_values["general.architecture"],
        "metadata_count": metadata_count,
        "tensor_count": tensor_count,
        "alignment_bytes": alignment,
        "tensor_directory_end_offset": directory_end,
        "data_section_offset": data_section_offset,
        "pre_data_alignment_padding_bytes": data_section_offset - directory_end,
        "metadata": metadata_entries,
    }
    structural_summary = {
        "file_size_bytes": file_size,
        "tensor_directory_end_offset": directory_end,
        "pre_data_alignment_padding_bytes": data_section_offset - directory_end,
        "data_section_offset": data_section_offset,
        "data_section_bytes": file_size - data_section_offset,
        "total_tensor_packed_bytes": total_packed_bytes,
        "tensor_alignment_padding_bytes": total_alignment_padding,
        "file_overhead_bytes": file_size - total_packed_bytes,
        "type_summary": dict(sorted(type_summary.items())),
    }
    metadata_output["structural_summary"] = structural_summary

    csv_rows = []
    for tensor in sorted(tensors, key=lambda item: item["gguf_tensor_name"]):
        csv_rows.append(
            {
                **tensor,
                "dimensions": "x".join(str(value) for value in tensor["dimensions"]),
            }
        )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gguf-metadata.json").write_bytes(canonical_json_bytes(metadata_output))
    (args.output_dir / "gguf-tensor-inventory.csv").write_text(
        output.getvalue(), encoding="utf-8", newline=""
    )
    print(
        f"GGUF v{version}: metadata={metadata_count} tensors={tensor_count} "
        f"data_offset={data_section_offset} packed_bytes={total_packed_bytes}"
    )
    print(f"sha256_verification={sha256_verification}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GgufError, OSError, ValueError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
