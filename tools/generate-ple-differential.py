#!/usr/bin/env python3
"""Generate Task 2.4 PLE differential evidence from pinned canonical sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_REVISION = "805a9e939fa8c1bff8d8ffdf041c051b71a914aa"
TRANSFORMERS_SOURCE_SHA256 = (
    "91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f"
)
CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
TOKENIZER_VECTORS_SHA256 = (
    "cbe1290d84a7a61113cf201edaf5034893eb1e70ba5a3c4bc9f4ea50bdcaf153"
)
MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
VOCAB_SIZE = 248_320
EOS_TOKEN_ID = 248_044
NGRAM_SIZE = 3
HEADS_PER_ORDER = 8
HEAD_COUNT = 16
MEMBER_ROWS = 2_500_012
MEMBER_COUNT = 128
EXPECTED_MULTIPLIERS = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
EXPECTED_HEAD_VOCABS = [
    20_000_003, 20_000_023, 20_000_033, 20_000_047,
    20_000_059, 20_000_063, 20_000_069, 20_000_077,
    20_000_081, 20_000_093, 20_000_107, 20_000_147,
    20_000_153, 20_000_159, 20_000_161, 20_000_171,
]
EXPECTED_HEAD_OFFSETS = [
    0, 20_000_003, 40_000_026, 60_000_059,
    80_000_106, 100_000_165, 120_000_228, 140_000_297,
    160_000_374, 180_000_455, 200_000_548, 220_000_655,
    240_000_802, 260_000_955, 280_001_114, 300_001_275,
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def derive_multipliers() -> list[int]:
    bound = ((1 << 63) - 1) // VOCAB_SIZE
    half_bound = max(1, bound // 2)
    return [
        2 * (splitmix64((1234 + SPLITMIX_GAMMA * (index + 1)) & MASK64) % half_bound) + 1
        for index in range(NGRAM_SIZE)
    ]


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def derive_head_vocabs() -> list[int]:
    values: list[int] = []
    candidate = 20_000_000 - 1
    for _ in range(HEAD_COUNT):
        candidate += 1
        while not is_prime(candidate):
            candidate += 1
        values.append(candidate)
    return values


def shifted_token(tokens: list[int], position: int, shift: int) -> int:
    if shift == 0:
        return tokens[position]
    segment_start = 0
    for prior in range(position):
        if tokens[prior] == EOS_TOKEN_ID:
            segment_start = prior + 1
    source = position - shift
    if source < 0 or source < segment_start:
        return EOS_TOKEN_ID
    return tokens[source]


def rows(tokens: list[int], start: int = 0) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for position in range(start, len(tokens)):
        addresses: list[dict[str, int]] = []
        for order in (2, 3):
            mixed = shifted_token(tokens, position, 0) * EXPECTED_MULTIPLIERS[0]
            for shift in range(1, order):
                mixed ^= shifted_token(tokens, position, shift) * EXPECTED_MULTIPLIERS[shift]
            head_start = (order - 2) * HEADS_PER_ORDER
            for local_head in range(HEADS_PER_ORDER):
                global_head = head_start + local_head
                address = (
                    mixed % EXPECTED_HEAD_VOCABS[global_head]
                    + EXPECTED_HEAD_OFFSETS[global_head]
                )
                addresses.append(
                    {
                        "global_address": address,
                        "head_offset": EXPECTED_HEAD_OFFSETS[global_head],
                        "head_vocab_size": EXPECTED_HEAD_VOCABS[global_head],
                        "local_head": local_head,
                        "ngram_order": order,
                        "partition_id": address // MEMBER_ROWS,
                        "partition_row": address % MEMBER_ROWS,
                    }
                )
        result.append(
            {
                "addresses": addresses,
                "history_tokens": [
                    shifted_token(tokens, position, shift)
                    for shift in range(1, NGRAM_SIZE)
                ],
                "position": position,
                "token_id": tokens[position],
            }
        )
    return result


def make_stream(case_id: str, prefill: list[int], steps: list[int]) -> dict[str, Any]:
    complete = list(prefill)
    step_vectors: list[dict[str, Any]] = []
    for step_index, token_id in enumerate(steps):
        history_before = [
            shifted_token(complete, len(complete), shift)
            if complete else EOS_TOKEN_ID
            for shift in (2, 1)
        ]
        combined = complete + [token_id]
        row = rows(combined, len(complete))[0]
        history_after = combined[-2:]
        if len(history_after) == 1:
            history_after.insert(0, EOS_TOKEN_ID)
        step_vectors.append(
            {
                "addresses": row["addresses"],
                "history_after": history_after,
                "history_before": history_before,
                "step": step_index,
                "token_id": token_id,
            }
        )
        complete.append(token_id)
    return {
        "case_id": case_id,
        "prefill_input_ids": prefill,
        "steps": step_vectors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers-source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.transformers_source) != TRANSFORMERS_SOURCE_SHA256:
        raise RuntimeError("pinned Transformers PLE source hash mismatch")
    if sha256(args.config) != CONFIG_SHA256:
        raise RuntimeError("pinned canonical config hash mismatch")
    if sha256(args.tokenizer_vectors) != TOKENIZER_VECTORS_SHA256:
        raise RuntimeError("pinned tokenizer-vector hash mismatch")
    if derive_multipliers() != EXPECTED_MULTIPLIERS:
        raise RuntimeError("source-derived PLE multipliers differ")
    if derive_head_vocabs() != EXPECTED_HEAD_VOCABS:
        raise RuntimeError("source-derived PLE head primes differ")
    offsets: list[int] = []
    total = 0
    for size in EXPECTED_HEAD_VOCABS:
        offsets.append(total)
        total += size
    if offsets != EXPECTED_HEAD_OFFSETS:
        raise RuntimeError("source-derived PLE offsets differ")

    tokenizer = json.loads(args.tokenizer_vectors.read_text(encoding="utf-8"))
    integration_segment = next(
        segment
        for vector in tokenizer["vectors"]
        for segment in vector["segments"]
        if segment["decoded"] == "Hello, Kestrel-Q."
    )
    sequence_inputs = [
        ("KQ-PLE-DIFF-001", "shortest_zero_id", [0]),
        ("KQ-PLE-DIFF-002", "shortest_eos", [EOS_TOKEN_ID]),
        ("KQ-PLE-DIFF-003", "repeated_tokens", [7] * 10),
        ("KQ-PLE-DIFF-004", "alternating_tokens", [0, VOCAB_SIZE - 1] * 5),
        ("KQ-PLE-DIFF-005", "low_token_ids", [0, 1, 2, 3]),
        ("KQ-PLE-DIFF-006", "high_token_ids", [248_316, 248_317, 248_318, 248_319]),
        ("KQ-PLE-DIFF-007", "history_length_two_boundary", [31, 32]),
        ("KQ-PLE-DIFF-008", "history_length_three_boundary", [31, 32, 33]),
        ("KQ-PLE-DIFF-009", "future_value_dilation_span_minus_one", list(range(1, 10))),
        ("KQ-PLE-DIFF-010", "future_value_dilation_span_plus_one", list(range(1, 11))),
        ("KQ-PLE-DIFF-011", "multiple_eos_segments", [5, EOS_TOKEN_ID, 7, EOS_TOKEN_ID, 9]),
        ("KQ-PLE-DIFF-012", "bounded_long_prefill", [((index * 7919) + 17) % VOCAB_SIZE for index in range(64)]),
    ]
    long_prefill = sequence_inputs[-1][2]
    stream_cases = [
        make_stream("KQ-PLE-DIFF-STREAM-001", [], [0, 1, 2]),
        make_stream("KQ-PLE-DIFF-STREAM-002", [42, 42], [42, EOS_TOKEN_ID, 42]),
        make_stream("KQ-PLE-DIFF-STREAM-003", long_prefill, [248_319, 0, EOS_TOKEN_ID, 11]),
    ]
    evidence = {
        "address_arithmetic": (
            "nonnegative signed-int64-compatible products; XOR in order; positive remainder; "
            "then per-head offset"
        ),
        "asset_id": "KQ-PLE-DIFFERENTIAL-001",
        "class": "C",
        "comparison_mode": "EXACT",
        "coverage": {
            "decode_stream_cases": len(stream_cases),
            "reset_replay_equivalence": True,
            "sequence_cases": len(sequence_inputs),
            "tokenizer_integration_cases": 1,
            "value_dilation_boundary_note": (
                "Lengths 9 and 10 straddle the future PLE value-state span; address state "
                "remains exactly two prior tokens."
            ),
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "oracle": {
            "config_sha256": CONFIG_SHA256,
            "implementation": "pinned Transformers Qwen4-Exp PLE algorithm and official config",
            "license": "Apache-2.0",
            "revision": TRANSFORMERS_REVISION,
            "source_file_sha256": TRANSFORMERS_SOURCE_SHA256,
        },
        "sequence_cases": [
            {"case": label, "case_id": case_id, "input_ids": tokens, "rows": rows(tokens)}
            for case_id, label, tokens in sequence_inputs
        ],
        "stream_cases": stream_cases,
        "tokenizer_integration_case": {
            "case_id": "KQ-PLE-DIFF-TOKENIZER-001",
            "input_utf8": integration_segment["decoded"],
            "input_utf8_sha256": hashlib.sha256(
                integration_segment["decoded"].encode("utf-8")
            ).hexdigest(),
            "rows": rows(integration_segment["token_ids_add_special_tokens_false"]),
            "token_ids": integration_segment["token_ids_add_special_tokens_false"],
            "tokenizer_vectors_sha256": TOKENIZER_VECTORS_SHA256,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    args.output.write_bytes(payload)
    print(f"wrote {args.output} sha256={hashlib.sha256(payload).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
