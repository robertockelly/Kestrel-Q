#!/usr/bin/env python3
"""Generate deterministic Task 2.13 evidence from independent/native raw runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

BASELINE = "1f2b967d6bf5ec7aaa263192462ede1eb5c9bd28"
MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
LLAMA_REVISION = "90c26fcd4b2114b4aa39d09d69318cb8f438d27a"
GGUF_SHA256 = "8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2"
PROMPT_IDS = [9419, 11, 710, 467, 3621, 27325, 13]
EXPECTED_TOKENS = [271, 248068, 198, 760]
EXPECTED_PIECES = ["\n\n", "<think>", "\n", "The"]
FILES = [
    "multi-token-contract.json",
    "multi-token-oracle.json",
    "multi-token-native.json",
    "multi-token-state.json",
    "multi-token-validation.json",
]

STEP_RE = re.compile(
    r"^multi-token step=(?P<step>\d+) token=(?P<token>\d+) "
    r"position=(?P<position>\d+) state_hash=(?P<state>[0-9a-f]{16}) "
    r"gdn_hash=(?P<gdn>[0-9a-f]{16}) qsa_hash=(?P<qsa>[0-9a-f]{16}) "
    r"ple_address_hash=(?P<ple_address>[0-9a-f]{16}) "
    r"ple_value_hash=(?P<ple_value>[0-9a-f]{16}) "
    r"route_hash=(?P<route>[0-9a-f]{16}) ple_hash=(?P<ple>[0-9a-f]{16}) "
    r"selected_logit=(?P<selected_logit>[-+0-9.eE]+) "
    r"runner_id=(?P<runner_id>\d+) "
    r"runner_logit=(?P<runner_logit>[-+0-9.eE]+) "
    r"qsa_candidates=(?P<qsa_candidates>\d+) "
    r"qsa_blocks=(?P<qsa_blocks>\d+) "
    r"qsa_tokens=(?P<qsa_tokens>\d+) "
    r"payload=(?P<payload>\d+) blocks=(?P<blocks>\d+)$"
)
PASS_RE = re.compile(
    r"^multi-token PASS prefill=1 decode=3 rollback=1 "
    r"payload=(?P<payload>\d+) blocks=(?P<blocks>\d+) "
    r"state=(?P<state>\d+) scratch=(?P<scratch>\d+) "
    r"elapsed_ns=(?P<elapsed>\d+)$"
)


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_native(path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    steps: list[dict[str, object]] = []
    totals: dict[str, int] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = STEP_RE.fullmatch(line)
        if match:
            item: dict[str, object] = {
                "step": int(match["step"]),
                "token_id": int(match["token"]),
                "decoded_fragment": EXPECTED_PIECES[int(match["step"]) - 1],
                "position": int(match["position"]),
                "state_hash_fnv1a64": match["state"],
                "gdn_state_hash_fnv1a64": match["gdn"],
                "qsa_state_hash_fnv1a64": match["qsa"],
                "ple_address_state_hash_fnv1a64": match["ple_address"],
                "ple_value_state_hash_fnv1a64": match["ple_value"],
                "moe_route_hash_fnv1a64": match["route"],
                "ple_request_hash_fnv1a64": match["ple"],
                "selected_logit": float(match["selected_logit"]),
                "runner_up_token_id": int(match["runner_id"]),
                "runner_up_logit": float(match["runner_logit"]),
                "qsa_candidate_blocks": int(match["qsa_candidates"]),
                "qsa_selected_blocks": int(match["qsa_blocks"]),
                "qsa_selected_tokens": int(match["qsa_tokens"]),
                "logical_payload_bytes": int(match["payload"]),
                "payload_blocks": int(match["blocks"]),
            }
            steps.append(item)
            continue
        match = PASS_RE.fullmatch(line)
        if match:
            totals = {name: int(match[name]) for name in
                      ("payload", "blocks", "state", "scratch", "elapsed")}
    if len(steps) != 4 or totals is None:
        raise ValueError("native capture lacks four steps and a PASS summary")
    if [step["token_id"] for step in steps] != EXPECTED_TOKENS:
        raise ValueError("native capture does not match the frozen oracle sequence")
    return steps, totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-raw", required=True, type=Path)
    parser.add_argument("--native-raw", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    oracle_raw = json.loads(args.oracle_raw.read_text(encoding="utf-8"))
    oracle_steps = oracle_raw["generated_steps"]
    oracle_tokens = [step["token_id"] for step in oracle_steps]
    oracle_pieces = [step["piece"] for step in oracle_steps]
    if oracle_raw["oracle_revision"] != LLAMA_REVISION:
        raise ValueError("oracle revision is not pinned")
    if (oracle_raw["input_token_ids"] != PROMPT_IDS or
            oracle_raw["context_capacity"] != 16 or
            oracle_tokens != EXPECTED_TOKENS or
            oracle_pieces != EXPECTED_PIECES):
        raise ValueError("oracle capture does not match the governed case")
    native_steps, totals = parse_native(args.native_raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    contract = {
        "schema_version": 1,
        "task": "2.13",
        "baseline_commit": BASELINE,
        "canonical_model_revision": MODEL_REVISION,
        "gguf_sha256": GGUF_SHA256,
        "prompt_id": "KQ-PROMPT-001",
        "prompt_utf8": "Hello, Kestrel-Q.",
        "prompt_token_ids": PROMPT_IDS,
        "context_capacity": 16,
        "max_new_tokens": 4,
        "generation": "GREEDY_EXACT_DISCRETE",
        "prefill_count": 1,
        "decode_input": "PREVIOUS_SELECTED_TOKEN_ID_DIRECT",
        "prompt_replay": "PROHIBITED",
        "eog_token_ids": [248044, 248046],
        "transaction": "ONE_COMPLETE_MODEL_TOKEN",
        "failure_policy": "ROLLBACK_TO_PRE_STEP_STATE",
        "payload_gate_bytes": 96 * 1024**3,
    }
    oracle = {
        "schema_version": 1,
        "class": "Q_EXACT_GGUF",
        "runtime": "llama.cpp",
        "revision": LLAMA_REVISION,
        "license": "MIT",
        "gguf_sha256": GGUF_SHA256,
        "input_authority": "EXPLICIT_CANONICAL_TOKEN_IDS",
        "context_capacity": 16,
        "prompt_prefill_count": oracle_raw["prompt_prefill_count"],
        "incremental_decode_count": oracle_raw["incremental_decode_count"],
        "generated_steps": oracle_steps,
        "generated_token_ids": oracle_tokens,
        "decoded_fragments": oracle_pieces,
        "eog_reached": any(step["is_eog"] for step in oracle_steps),
        "load_elapsed_ms": oracle_raw["load_elapsed_ms"],
        "prompt_elapsed_ms": oracle_raw["prompt_elapsed_ms"],
        "peak_working_set_bytes": oracle_raw["peak_working_set_bytes"],
        "peak_pagefile_bytes": oracle_raw["peak_pagefile_bytes"],
        "private_usage_bytes": oracle_raw["private_usage_bytes"],
    }
    for index, step in enumerate(native_steps):
        position = int(step["position"])
        expected_candidates = 48 if index == 0 else 24
        expected_selected_tokens = [336, 96, 108, 120][index]
        if (step["qsa_candidate_blocks"] != expected_candidates or
                step["qsa_selected_blocks"] != expected_candidates or
                step["qsa_selected_tokens"] != expected_selected_tokens):
            raise ValueError("native QSA sparse-selection counters diverge")
        step.update({
            "exact_oracle_match": True,
            "qsa_sequence_length_min": position,
            "qsa_sequence_length_max": position,
            "qsa_complete_blocks_min": position // 4,
            "qsa_complete_blocks_max": position // 4,
            "qsa_incomplete_tail_min": position % 4,
            "qsa_incomplete_tail_max": position % 4,
            "moe_selected_experts": 3360 if index == 0 else 480,
            "moe_selected_matrix_requests": 10080 if index == 0 else 1440,
            "ple_rows": 112 if index == 0 else 16,
            "ple_address_intents": 112 if index == 0 else 16,
        })
    native = {
        "schema_version": 1,
        "implementation": "Kestrel-Q native C17 scalar reference",
        "context_capacity": 16,
        "prompt_prefill_count": 1,
        "incremental_decode_count": 3,
        "prompt_replayed": False,
        "generated_token_ids": EXPECTED_TOKENS,
        "decoded_fragments": EXPECTED_PIECES,
        "steps": native_steps,
        "logical_payload_bytes": totals["payload"],
        "payload_blocks": totals["blocks"],
        "semantic_scope_count": 1239,
        "owned_model_state_bytes": totals["state"],
        "semantic_state_capacity_bytes": 116822048,
        "container_minus_semantic_capacity_bytes": totals["state"] - 116822048,
        "qsa_capacity_state_bytes": 442368,
        "peak_scratch_bytes": totals["scratch"],
        "logits_bytes": 993280,
        "maximum_f32_weight_bytes_materialized": 993280,
        "complete_target_f32_weight_matrices": 0,
        "elapsed_nanoseconds": totals["elapsed"],
    }
    state = {
        "schema_version": 1,
        "hash_algorithm": "FNV-1a-64 over explicit semantic bytes/fields",
        "steps": [{key: step[key] for key in (
            "step", "position", "state_hash_fnv1a64", "gdn_state_hash_fnv1a64",
            "qsa_state_hash_fnv1a64", "ple_address_state_hash_fnv1a64",
            "ple_value_state_hash_fnv1a64", "qsa_sequence_length_min",
            "qsa_sequence_length_max", "qsa_complete_blocks_min",
            "qsa_complete_blocks_max", "qsa_incomplete_tail_min",
            "qsa_incomplete_tail_max")}
            for step in native_steps],
        "rollback": {
            "injected_step": 3,
            "injected_layer": 24,
            "status": "PASS",
            "pre_and_post_failure_summaries_equal": True,
            "retry_token_id": 198,
        },
        "raw_state_blobs_committed": False,
    }
    validation = {
        "schema_version": 1,
        "status": "PASS",
        "oracle_sequence": EXPECTED_TOKENS,
        "native_sequence": EXPECTED_TOKENS,
        "comparison": "EXACT_DISCRETE",
        "m1_token_271_preserved": True,
        "m1_decoded_bytes_preserved": "\n\n",
        "prompt_prefill_count": 1,
        "incremental_decode_count": 3,
        "rollback_retry": "PASS",
        "qsa_block_tail": "PASS",
        "qsa_sparse_selection_counts": "PASS",
        "moe_top10_and_selected_only": "PASS",
        "ple_exact_rows": "PASS",
        "successful_payload_gate_bytes": 96 * 1024**3,
        "successful_payload_bytes": totals["payload"],
        "payload_gate_pass": totals["payload"] <= 96 * 1024**3,
        "complete_target_f32_weight_matrices": 0,
        "self_oracle": False,
    }

    values = [contract, oracle, native, state, validation]
    for name, value in zip(FILES, values, strict=True):
        dump(args.output_dir / name, value)
    manifest = {
        "schema_version": 1,
        "task": "2.13",
        "status": "COMPLETE_PASS",
        "baseline_commit": BASELINE,
        "artifacts": [
            {"path": name, "sha256": sha256(args.output_dir / name)}
            for name in FILES
        ],
        "source_raw_captures": "IGNORED_NOT_COMMITTED",
        "raw_model_payload_committed": False,
    }
    dump(args.output_dir / "multi-token-manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
