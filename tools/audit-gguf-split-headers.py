#!/usr/bin/env python3
"""Audit only bounded metadata ranges from the pinned upstream GGUF split."""

from __future__ import annotations

import argparse
import json
import re
import struct
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REVISION = "c8b5954a88c2775c546b92593eda40ea041d3176"
REPOSITORY = "unsloth/Qwen3.8-Flash-Next-GGUF"
FILES = [
    ("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", 10_946_624, "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082", 0, 67),
    ("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf", 49_859_583_136, "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9", 297, 3),
    ("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf", 49_376_141_504, "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3", 752, 3),
    ("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf", 12_087_983_520, "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a", 175, 3),
]
SCALAR_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class AuditError(RuntimeError):
    """Fail-closed network or format error."""


class ExactRangeReader:
    def __init__(self, url: str, expected_size: int, filename: str):
        self.url = url
        self.expected_size = expected_size
        self.filename = filename
        self.position = 0
        self.requests: list[dict[str, Any]] = []

    def read(self, size: int, purpose: str) -> bytes:
        if size <= 0 or size > 16 * 1024 * 1024:
            raise AuditError(f"unsafe exact range size: {size}")
        start = self.position
        end = start + size - 1
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "Kestrel-Q-Task-1.3"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read(size + 1)
            content_range = response.headers.get("Content-Range")
            expected_range = f"bytes {start}-{end}/{self.expected_size}"
            if response.status != 206 or content_range != expected_range or len(body) != size:
                raise AuditError(
                    f"Range not honored for {self.filename}: status={response.status}, "
                    f"Content-Range={content_range!r}, bytes={len(body)}"
                )
        self.requests.append(
            {
                "filename": self.filename,
                "range_start": start,
                "range_end": end,
                "response_bytes": size,
                "purpose": purpose,
            }
        )
        self.position += size
        return body

    def unpack(self, fmt: str, purpose: str) -> Any:
        return struct.unpack(fmt, self.read(struct.calcsize(fmt), purpose))[0]

    def string(self, purpose: str) -> str:
        length = self.unpack("<Q", f"{purpose} length")
        if length > 4096:
            raise AuditError(f"unexpected metadata key length: {length}")
        return self.read(length, purpose).decode("utf-8")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    api_url = f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true"
    api_request = urllib.request.Request(api_url, headers={"User-Agent": "Kestrel-Q-Task-1.3"})
    with urllib.request.urlopen(api_request, timeout=60) as response:
        if response.status != 200:
            raise AuditError(f"unexpected Hugging Face API status: {response.status}")
        api = json.load(response)
    if api.get("sha") != REVISION:
        raise AuditError(f"upstream revision mismatch: {api.get('sha')!r}")
    siblings = {item["rfilename"]: item for item in api.get("siblings", [])}

    shards = []
    requests = []
    for index, (filename, expected_size, expected_sha256, expected_tensors, expected_metadata) in enumerate(FILES):
        sibling = siblings.get(filename)
        if (
            sibling is None
            or sibling.get("size") != expected_size
            or sibling.get("lfs", {}).get("sha256") != expected_sha256
        ):
            raise AuditError(f"pinned sibling identity mismatch: {filename}")
        url = (
            f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/"
            f"{urllib.parse.quote(filename, safe='/')}?download=true"
        )
        reader = ExactRangeReader(url, expected_size, filename)
        magic, version, tensor_count, metadata_count = struct.unpack(
            "<4sIQQ", reader.read(24, "GGUF fixed header")
        )
        if magic != b"GGUF" or version != 3:
            raise AuditError(f"unexpected GGUF identity: {filename}")
        if tensor_count != expected_tensors or metadata_count != expected_metadata:
            raise AuditError(f"unexpected GGUF counts: {filename}")

        split_metadata: dict[str, Any] = {}
        # Shards 2-4 intentionally contain only the three scalar split keys.
        # Parse them exactly; never advance into their tensor directory.
        if index > 0:
            for _ in range(metadata_count):
                key = reader.string("metadata key")
                if not re.fullmatch(r"split\.(no|count|tensors\.count)", key):
                    raise AuditError(f"unexpected secondary-shard metadata key: {key}")
                type_id = reader.unpack("<I", f"{key} type")
                fmt = SCALAR_FORMATS.get(type_id)
                if fmt is None:
                    raise AuditError(f"unexpected scalar type {type_id} for {key}")
                split_metadata[key] = reader.unpack(fmt, f"{key} value")
            if split_metadata != {
                "split.no": index,
                "split.count": 4,
                "split.tensors.count": 1224,
            }:
                raise AuditError(f"unexpected split metadata: {split_metadata}")

        requests.extend(reader.requests)
        shards.append(
            {
                "filename": filename,
                "file_size_bytes": expected_size,
                "upstream_lfs_sha256": expected_sha256,
                "version": version,
                "tensor_count": tensor_count,
                "metadata_count": metadata_count,
                "split_metadata": split_metadata,
                "bytes_fetched": sum(item["response_bytes"] for item in reader.requests),
                "last_fetched_offset": reader.position - 1,
                "tensor_payload_bytes_fetched": 0,
            }
        )

    if sum(item["tensor_count"] for item in shards) != 1224:
        raise AuditError("upstream split tensor counts do not reconcile")
    output = {
        "schema_version": 1,
        "generated_by": "tools/audit-gguf-split-headers.py",
        "repository": REPOSITORY,
        "revision": REVISION,
        "api_url": api_url,
        "shards": shards,
        "audit": {
            "request_count": len(requests),
            "response_bytes": sum(item["response_bytes"] for item in requests),
            "tensor_payload_bytes_fetched": 0,
            "requests": requests,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(output))
    print(
        f"shards={len(shards)} tensors={sum(item['tensor_count'] for item in shards)} "
        f"response_bytes={output['audit']['response_bytes']} tensor_payload_bytes_fetched=0"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, OSError, UnicodeError, ValueError, struct.error) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        raise SystemExit(1)
