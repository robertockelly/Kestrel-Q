#!/usr/bin/env python3
"""Capture only bounded Safetensors JSON headers from the pinned Qwen release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REPOSITORY_ID = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
EXPECTED_SHARDS = 131
EXPECTED_TENSORS = 1658
MAX_HEADER_BYTES = 4 * 1024 * 1024
MAX_TOTAL_RESPONSE_BYTES = 64 * 1024 * 1024
SHARD_PREFIX = "model-"
SHARD_SUFFIX = "-of-00131.safetensors"
ALLOWED_HOSTS = {"huggingface.co"}
ALLOWED_HOST_SUFFIXES = (".huggingface.co", ".hf.co")


class CaptureError(RuntimeError):
    """A fail-closed capture error."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def host_allowed(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    return host in ALLOWED_HOSTS or any(host.endswith(suffix) for suffix in ALLOWED_HOST_SUFFIXES)


class RestrictedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        if not host_allowed(newurl):
            raise CaptureError(f"redirect to disallowed URL: {newurl}")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        requested_range = req.headers.get("Range") or req.unredirected_hdrs.get("Range")
        if not requested_range:
            raise CaptureError("redirect lost mandatory Range request")
        redirected.add_unredirected_header("Range", requested_range)
        redirected.add_unredirected_header("Accept-Encoding", "identity")
        redirected.add_unredirected_header("User-Agent", "Kestrel-Q-Task-1.2-header-audit/1")
        return redirected


def exact_shard_allowlist(index: dict[str, Any]) -> list[str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != EXPECTED_TENSORS:
        raise CaptureError(f"expected {EXPECTED_TENSORS} index tensors")
    shards = sorted(set(weight_map.values()))
    expected = [f"model-{number:05d}-of-00131.safetensors" for number in range(1, EXPECTED_SHARDS + 1)]
    if shards != expected:
        raise CaptureError("index shard set does not match exact 131-shard allowlist")
    return shards


def remote_sizes(baseline_manifest: dict[str, Any], shards: list[str]) -> dict[str, int]:
    files = baseline_manifest.get("files")
    if not isinstance(files, list):
        raise CaptureError("baseline manifest has no file inventory")
    sizes = {
        entry["path"]: entry["size_bytes"]
        for entry in files
        if isinstance(entry, dict) and entry.get("path") in shards
    }
    if sorted(sizes) != shards or any(not isinstance(value, int) or value <= 0 for value in sizes.values()):
        raise CaptureError("baseline manifest does not contain exact positive shard sizes")
    return sizes


def parse_content_range(value: str | None) -> tuple[int, int, int]:
    if not value or not value.startswith("bytes ") or "/" not in value:
        raise CaptureError(f"missing or invalid Content-Range: {value!r}")
    interval, total_text = value[6:].split("/", 1)
    start_text, end_text = interval.split("-", 1)
    try:
        start, end, total = int(start_text), int(end_text), int(total_text)
    except ValueError as error:
        raise CaptureError(f"non-integer Content-Range: {value!r}") from error
    return start, end, total


def fetch_exact_range(
    opener: urllib.request.OpenerDirector,
    url: str,
    shard: str,
    kind: str,
    start: int,
    end: int,
    expected_remote_size: int,
    header_length: int | None,
    audit: list[dict[str, Any]],
) -> bytes:
    expected = end - start + 1
    record: dict[str, Any] = {
        "actual_bytes": 0,
        "content_range": None,
        "expected_bytes": expected,
        "header_length": header_length,
        "http_status": None,
        "pinned_revision": REVISION,
        "request_kind": kind,
        "requested_range": f"bytes={start}-{end}",
        "result": "FAIL",
        "shard": shard,
    }
    audit.append(record)
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}",
            "User-Agent": "Kestrel-Q-Task-1.2-header-audit/1",
        },
        method="GET",
    )
    response = None
    try:
        response = opener.open(request, timeout=120)
        record["http_status"] = response.status
        record["content_range"] = response.headers.get("Content-Range")
        if response.status != 206:
            record["result"] = "DIRTY_INVALID_RANGE_NOT_HONORED"
            raise CaptureError(f"{shard} {kind}: expected HTTP 206, got {response.status}")
        content_length = response.headers.get("Content-Length")
        if content_length is None or int(content_length) != expected:
            record["result"] = "DIRTY_INVALID_CONTENT_LENGTH"
            raise CaptureError(f"{shard} {kind}: Content-Length is not exactly {expected}")
        actual_start, actual_end, actual_total = parse_content_range(record["content_range"])
        if (actual_start, actual_end) != (start, end) or actual_total != expected_remote_size:
            record["result"] = "DIRTY_INVALID_CONTENT_RANGE"
            raise CaptureError(
                f"{shard} {kind}: Content-Range mismatch "
                f"({actual_start}-{actual_end}/{actual_total})"
            )
        data = response.read(expected + 1)
        record["actual_bytes"] = len(data)
        if len(data) != expected:
            record["result"] = "DIRTY_INVALID_BODY_SIZE"
            raise CaptureError(f"{shard} {kind}: body size {len(data)} != {expected}")
        if response.read(1):
            record["result"] = "DIRTY_INVALID_TRAILING_BODY"
            raise CaptureError(f"{shard} {kind}: response contained bytes beyond requested range")
        record["result"] = "PASS"
        return data
    except urllib.error.HTTPError as error:
        record["http_status"] = error.code
        record["content_range"] = error.headers.get("Content-Range") if error.headers else None
        record["result"] = "FAIL_HTTP"
        raise CaptureError(f"{shard} {kind}: HTTP {error.code}") from error
    finally:
        if response is not None:
            response.close()


def write_failure_audit(path: Path, audit: list[dict[str, Any]], message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": 1,
        "repository_id": REPOSITORY_ID,
        "revision": REVISION,
        "status": "DIRTY_INVALID",
        "error": message,
        "requests": audit,
        "total_response_bytes": sum(item["actual_bytes"] for item in audit),
        "weight_payload_bytes_fetched": 0 if all(item["result"] == "PASS" for item in audit) else None,
    }
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-audit", type=Path, required=True)
    args = parser.parse_args()

    if args.revision != REVISION:
        raise CaptureError(f"revision must be exactly {REVISION}")
    if args.failure_audit.exists():
        raise CaptureError(f"refusing to overwrite preserved failure audit: {args.failure_audit}")
    index = json.loads(args.index.read_text(encoding="utf-8"))
    baseline_manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    if baseline_manifest.get("repository_id") != REPOSITORY_ID or baseline_manifest.get("revision") != REVISION:
        raise CaptureError("baseline manifest repository/revision mismatch")
    shards = exact_shard_allowlist(index)
    sizes = remote_sizes(baseline_manifest, shards)
    audit: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    opener = urllib.request.build_opener(RestrictedRedirectHandler())

    try:
        for shard in shards:
            if not (shard.startswith(SHARD_PREFIX) and shard.endswith(SHARD_SUFFIX)):
                raise CaptureError(f"disallowed shard name: {shard}")
            quoted_shard = urllib.parse.quote(shard, safe="")
            url = f"https://huggingface.co/{REPOSITORY_ID}/resolve/{REVISION}/{quoted_shard}"
            if not host_allowed(url):
                raise CaptureError(f"disallowed source URL: {url}")
            prefix = fetch_exact_range(opener, url, shard, "length_prefix", 0, 7, sizes[shard], None, audit)
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length == 0 or header_length > MAX_HEADER_BYTES:
                raise CaptureError(f"{shard}: unsafe header length {header_length}")
            if sum(item["actual_bytes"] for item in audit) + header_length > MAX_TOTAL_RESPONSE_BYTES:
                raise CaptureError("total response-byte safety budget would be exceeded")
            header_bytes = fetch_exact_range(
                opener,
                url,
                shard,
                "json_header",
                8,
                8 + header_length - 1,
                sizes[shard],
                header_length,
                audit,
            )
            try:
                header = json.loads(header_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CaptureError(f"{shard}: invalid UTF-8 JSON header") from error
            if not isinstance(header, dict):
                raise CaptureError(f"{shard}: Safetensors header is not an object")
            tensors = []
            metadata = header.get("__metadata__")
            for name in sorted(key for key in header if key != "__metadata__"):
                descriptor = header[name]
                if not isinstance(descriptor, dict):
                    raise CaptureError(f"{shard}: invalid descriptor for {name}")
                tensors.append(
                    {
                        "data_offsets": descriptor.get("data_offsets"),
                        "dtype": descriptor.get("dtype"),
                        "name": name,
                        "shape": descriptor.get("shape"),
                    }
                )
            shard_records.append(
                {
                    "filename": shard,
                    "header_length_bytes": header_length,
                    "header_sha256": sha256_bytes(header_bytes),
                    "metadata": metadata,
                    "remote_size_bytes": sizes[shard],
                    "tensor_count": len(tensors),
                    "tensors": tensors,
                }
            )
    except Exception as error:
        write_failure_audit(args.failure_audit, audit, str(error))
        raise

    total_response_bytes = sum(item["actual_bytes"] for item in audit)
    if len(shard_records) != EXPECTED_SHARDS or len(audit) != EXPECTED_SHARDS * 2:
        raise CaptureError("capture did not produce exactly 131 shards and 262 requests")
    if total_response_bytes > MAX_TOTAL_RESPONSE_BYTES:
        raise CaptureError("capture exceeded total response-byte safety budget")
    manifest = {
        "schema_version": 1,
        "generated_by": "tools/capture-safetensors-headers.py",
        "repository_id": REPOSITORY_ID,
        "revision": REVISION,
        "source_index_sha256": sha256_path(args.index),
        "source_baseline_manifest_sha256": sha256_path(args.baseline_manifest),
        "shard_count": len(shard_records),
        "tensor_count": sum(item["tensor_count"] for item in shard_records),
        "header_json_bytes_fetched": sum(item["header_length_bytes"] for item in shard_records),
        "total_response_bytes": total_response_bytes,
        "weight_payload_bytes_fetched": 0,
        "shards": shard_records,
    }
    network_audit = {
        "schema_version": 1,
        "generated_by": "tools/capture-safetensors-headers.py",
        "repository_id": REPOSITORY_ID,
        "revision": REVISION,
        "status": "PASS",
        "request_count": len(audit),
        "largest_response_bytes": max(item["actual_bytes"] for item in audit),
        "total_response_bytes": total_response_bytes,
        "weight_payload_bytes_fetched": 0,
        "requests": audit,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shard-header-manifest.json").write_bytes(canonical_json_bytes(manifest))
    (args.output_dir / "network-range-audit.json").write_bytes(canonical_json_bytes(network_audit))
    print(f"captured {len(shard_records)} bounded headers with {total_response_bytes} response bytes")
    print("weight_payload_bytes_fetched=0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CaptureError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
