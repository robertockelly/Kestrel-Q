#!/usr/bin/env python3
"""Compare the native Task 2.1 registry with the pinned Epic 1 mapping oracle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys


SKIP_RETURN_CODE = 77

RELATIONS = {
    "RENAMED_ONE_TO_ONE": "RENAMED_ONE_TO_ONE",
    "TRANSFORMED_LAYOUT": "TRANSFORMED_LAYOUT",
    "FUSED_INTO_GGUF": "MULTIPLE_CANONICAL_TO_ONE_PHYSICAL",
    "SPLIT_IN_GGUF": "ONE_CANONICAL_TO_MULTIPLE_PHYSICAL",
    "OMITTED_FORMAT_DERIVED": "METADATA_DERIVED",
}

COMPONENTS = {
    "token_embedding": "TOKEN_EMBEDDING",
    "lm_head": "LM_HEAD",
    "final_gated_residual": "FINAL_GATED_RESIDUAL",
    "gated_residual": "GATED_RESIDUAL",
    "gdn": "GDN",
    "qsa_attention": "QSA_ATTENTION",
    "qsa_indexer": "QSA_INDEXER",
    "router": "MOE_ROUTER",
    "routed_experts": "ROUTED_EXPERT_STACK",
    "shared_expert": "SHARED_EXPERT",
    "ple_embedding_table": "PLE_TABLE",
    "ple_dense": "PLE_DENSE",
    "ple_address_metadata": "PLE_ADDRESS_METADATA",
}

METADATA_KEYS = {
    "model.language_model.layers.1.ple.ple_embedding.layer_multipliers":
        "qwen4exp.ple.layer_multipliers",
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets":
        "qwen4exp.ple.head_offsets",
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes":
        "qwen4exp.ple.head_vocab_sizes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def run_dump(inspect: Path, model: Path) -> bytes:
    result = subprocess.run(
        [str(inspect), "--semantic-dump", str(model)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        fail(
            "native semantic dump failed: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    return result.stdout


def read_native(raw: bytes) -> list[dict[str, str]]:
    text = raw.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    expected_fields = [
        "semantic_id",
        "canonical_name",
        "component",
        "layer_id",
        "role",
        "relation",
        "placement",
        "physical_names",
        "metadata_key",
    ]
    if reader.fieldnames != expected_fields:
        fail(f"native dump fields differ: {reader.fieldnames!r}")
    return list(reader)


def read_oracle(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        row for row in rows
        if row["initial_text_scope"] == "REQUIRED_INITIAL_TEXT"
    ]


def expected_placement(component: str) -> str:
    if component == "routed_experts":
        return "ROUTED_EXPERT_CACHE_CANDIDATE"
    if component.startswith("ple_"):
        return "PLE_DISK_BACKED_CANDIDATE"
    return "ALWAYS_NEEDED_CANDIDATE"


def validate(native: list[dict[str, str]],
             oracle: list[dict[str, str]]) -> tuple[int, int]:
    if len(native) != 1294 or len(oracle) != 1294:
        fail(f"semantic count mismatch: native={len(native)} oracle={len(oracle)}")
    native_by_canonical = {row["canonical_name"]: row for row in native}
    if len(native_by_canonical) != len(native):
        fail("native dump contains duplicate canonical identities")
    if len({row["semantic_id"] for row in native}) != len(native):
        fail("native dump contains duplicate stable semantic identifiers")

    physical_names: set[str] = set()
    for expected in oracle:
        canonical_name = expected["canonical_tensor_name"]
        actual = native_by_canonical.get(canonical_name)
        if actual is None:
            fail(f"native registry is missing {canonical_name}")
        expected_relation = RELATIONS.get(expected["mapping_status"])
        if expected_relation is None:
            fail(f"unhandled oracle relation {expected['mapping_status']}")
        checks = {
            "component": COMPONENTS[expected["component"]],
            "layer_id": expected["layer_id"],
            "relation": expected_relation,
            "placement": expected_placement(expected["component"]),
            "physical_names": expected["gguf_tensor_names"],
            "metadata_key": METADATA_KEYS.get(canonical_name, ""),
        }
        for field, wanted in checks.items():
            if actual[field] != wanted:
                fail(
                    f"{canonical_name}: {field}={actual[field]!r}, "
                    f"expected {wanted!r}"
                )
        if actual["physical_names"]:
            physical_names.update(actual["physical_names"].split(";"))

    if set(native_by_canonical) != {
        row["canonical_tensor_name"] for row in oracle
    }:
        fail("native registry contains canonical identities outside the oracle")
    if len(physical_names) != 1224:
        fail(f"unique physical coverage is {len(physical_names)}, expected 1224")
    metadata_derived = sum(
        row["relation"] == "METADATA_DERIVED" for row in native
    )
    if metadata_derived != 3:
        fail(f"metadata-derived count is {metadata_derived}, expected 3")
    return len(physical_names), metadata_derived


def main() -> int:
    args = parse_args()
    model = args.model
    if model is None:
        value = os.environ.get("KQ_GGUF_PATH")
        if not value:
            print("KQ_GGUF_PATH is unavailable; semantic oracle skipped")
            return SKIP_RETURN_CODE
        model = Path(value)
    if not model.is_file():
        print("KQ_GGUF_PATH does not identify a file", file=sys.stderr)
        return 1
    if not args.inspect.is_file() or not args.mapping.is_file():
        print("inspect executable or mapping oracle is unavailable", file=sys.stderr)
        return 1

    first = run_dump(args.inspect, model)
    second = run_dump(args.inspect, model)
    if first != second:
        fail("native semantic dump is not byte-identical across two runs")
    native = read_native(first)
    oracle = read_oracle(args.mapping)
    physical_count, metadata_count = validate(native, oracle)
    digest = hashlib.sha256(first).hexdigest()
    print(
        "semantic registry oracle comparison: PASS; "
        f"semantics={len(native)} physical={physical_count} "
        f"metadata_derived={metadata_count} dump_sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, UnicodeError, csv.Error) as error:
        print(f"semantic registry oracle comparison failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
