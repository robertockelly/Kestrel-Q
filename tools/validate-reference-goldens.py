#!/usr/bin/env python3
"""Fail-closed validator for Task 1.4 machine-readable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_STATUSES = {
    "GENERATED_VERIFIED",
    "PLANNED_REFERENCE_REQUIRED",
    "DEFERRED_CAPABLE_REFERENCE_ENV",
    "NOT_APPLICABLE",
}
EXPECTED_CLASSES = {"C", "Q"}
EXPECTED_SAFE_ASSETS = {
    "KQ-GOLD-TOK-001",
    "KQ-GOLD-CHAT-001",
    "KQ-GOLD-PLE-ROOT-001",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.golden_dir.resolve()
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        fail("unsupported manifest schema")
    if manifest.get("weight_payload_downloaded_or_executed") is not False:
        fail("weight safety declaration is not false")

    suite_path = root / "prompt-suite.json"
    suite = load_json(suite_path)
    if sha256_file(suite_path) != manifest.get("prompt_suite_sha256"):
        fail("prompt-suite hash mismatch")
    prompts = suite.get("prompts", [])
    if len(prompts) != 10 or len({item["id"] for item in prompts}) != 10:
        fail("prompt suite does not contain ten unique prompts")
    for prompt in prompts:
        if not prompt["id"].startswith("KQ-PROMPT-"):
            fail("invalid prompt ID")

    safe_assets = set()
    asset_ids = set()
    for asset in manifest.get("assets", []):
        asset_id = asset.get("asset_id")
        if asset_id in asset_ids:
            fail(f"duplicate asset ID: {asset_id}")
        asset_ids.add(asset_id)
        if asset.get("class") not in EXPECTED_CLASSES:
            fail(f"invalid class for {asset_id}")
        if asset.get("status") not in ALLOWED_STATUSES:
            fail(f"invalid status for {asset_id}")
        prefix = "research/goldens/Qwen3.8-Flash-Next/"
        recorded_path = asset.get("path", "")
        if not recorded_path.startswith(prefix):
            fail(f"noncanonical asset path for {asset_id}")
        path = root / recorded_path[len(prefix):]
        if not path.is_file():
            fail(f"missing asset file for {asset_id}: {path}")
        if sha256_file(path) != asset.get("sha256"):
            fail(f"asset hash mismatch for {asset_id}")
        load_json(path)
        if asset_id in EXPECTED_SAFE_ASSETS:
            if asset.get("status") != "GENERATED_VERIFIED" or asset.get("comparison_mode") != "EXACT":
                fail(f"safe vector is not exact/generated: {asset_id}")
            safe_assets.add(asset_id)

    if safe_assets != EXPECTED_SAFE_ASSETS:
        fail(f"missing safe assets: {sorted(EXPECTED_SAFE_ASSETS - safe_assets)}")
    if len(manifest.get("oracles", [])) != 2:
        fail("manifest must contain exactly two oracle classes")
    if {oracle.get("class") for oracle in manifest["oracles"]} != EXPECTED_CLASSES:
        fail("oracle classes are incomplete")

    tokenizers = load_json(root / "canonical/tokenizer-vectors.json")
    if len(tokenizers.get("vectors", [])) != 10:
        fail("tokenizer vectors do not cover all prompts")
    chats = load_json(root / "canonical/chat-template-vectors.json")
    if len(chats.get("vectors", [])) != 4:
        fail("chat vectors must contain two prompts times two generation-prompt modes")
    ple = load_json(root / "canonical/ple-address-vectors.json")
    if len(ple.get("vectors", [])) != 7 or len(ple.get("decode_case", {}).get("steps", [])) != 4:
        fail("PLE coverage is incomplete")

    result = {
        "asset_count": len(asset_ids),
        "manifest_sha256": sha256_file(manifest_path),
        "prompt_count": len(prompts),
        "prompt_suite_sha256": sha256_file(suite_path),
        "safe_assets": sorted(safe_assets),
        "status": "PASS",
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
