#!/usr/bin/env python3
"""Validate Task 3.1 sampled traces from temporary full-logit captures.

The temporary F32 logits and native console transcript are inputs only.  This
tool independently applies the frozen Task 3.0 policy and PCG transition, then
writes bounded evidence without copying full logits into the repository.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import heapq
import json
import math
import platform
import re
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any


BASELINE = "9986b934d639b61575018653dfc8a4eabe2965ea"
MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
GGUF_SHA256 = "8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2"
VOCABULARY_SIZE = 248_320
CANONICAL_TOKEN_COUNT = 248_077
PROMPT_IDS = [9419, 11, 710, 467, 3621, 27325, 13]
TEMPERATURE = 1.0
TOP_K = 20
TOP_P = struct.unpack("<f", struct.pack("<f", 0.95))[0]
PCG_MULTIPLIER = 6_364_136_223_846_793_005
MASK64 = (1 << 64) - 1
FNV_ORDER_OFFSET = 14_695_981_039_346_656_037
FNV_INTEGRITY_OFFSET = 1_469_598_103_934_665_603
FNV_PRIME = 1_099_511_628_211
TRACE_NAMES = ("primary", "holdout")
EVIDENCE_FILES = (
    "sampled-generation-contract.json",
    "sampled-generation-oracle.json",
    "sampled-generation-native.json",
    "sampled-generation-state.json",
    "sampled-generation-validation.json",
)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True,
                                 ensure_ascii=False) + "\n").encode("utf-8"))


def parse_fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", line))


def parse_native(path: Path) -> dict[str, Any]:
    traces: dict[str, list[dict[str, str]]] = {name: [] for name in TRACE_NAMES}
    replay: list[dict[str, str]] = []
    controls: list[str] = []
    summaries: dict[str, dict[str, str]] = {}
    passed = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("sampled trace="):
            fields = parse_fields(line)
            name = fields.get("trace")
            if name in traces:
                traces[name].append(fields)
            elif name == "replay":
                replay.append(fields)
        elif line.startswith("sampled-control "):
            controls.append(line)
        elif line.startswith("sampled-trace "):
            fields = parse_fields(line)
            if "name" in fields:
                summaries[fields["name"]] = fields
        elif line.startswith("sampled-generation PASS"):
            passed = True
    if not passed:
        raise RuntimeError("native transcript does not contain its PASS gate")
    for name in TRACE_NAMES:
        if len(traces[name]) == 0 or len(traces[name]) > 4:
            raise RuntimeError(f"unexpected {name} trace length: {len(traces[name])}")
    if len(replay) != len(traces["primary"]):
        raise RuntimeError("native replay length differs from primary")
    required_controls = (
        "output-alias", "early-layer", "middle-layer", "late-layer",
        "corrupt-rng", "post-decode-sampler", "padded-id",
        "later-eog", "first-eog",
        "context-exhaustion", "deterministic-replay",
    )
    for name in required_controls:
        if not any(f"name={name}" in line and "PASS" in line
                   for line in controls):
            raise RuntimeError(f"missing PASS control: {name}")
    later_eog = next(parse_fields(line) for line in controls
                     if "name=later-eog" in line)
    if (later_eog.get("token") != "248046" or
            later_eog.get("position_before") != "10" or
            later_eog.get("position_after") != "11" or
            later_eog.get("draws_before") != "4" or
            later_eog.get("draws_after") != "5"):
        raise RuntimeError("later-EOG diagnostic fields are inconsistent")
    first_eog = next(parse_fields(line) for line in controls
                     if "name=first-eog" in line)
    if (first_eog.get("token") != "248046" or
            first_eog.get("position") != "7" or
            first_eog.get("prefill") != "1" or
            first_eog.get("decode") != "0" or
            first_eog.get("draws") != "1"):
        raise RuntimeError("first-EOG diagnostic fields are inconsistent")
    context = next(parse_fields(line) for line in controls
                   if "name=context-exhaustion" in line)
    if (context.get("capacity") != "8" or context.get("position") != "8"):
        raise RuntimeError("context-exhaustion diagnostic fields are inconsistent")
    return {
        "traces": traces,
        "replay": replay,
        "controls": controls,
        "summaries": summaries,
    }


def load_logits(path: Path) -> list[float]:
    expected = VOCABULARY_SIZE * 4
    data = path.read_bytes()
    if len(data) != expected:
        raise RuntimeError(f"{path}: expected {expected} bytes, got {len(data)}")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != VOCABULARY_SIZE or not all(math.isfinite(x) for x in values):
        raise RuntimeError(f"{path}: invalid full-logit capture")
    return values.tolist()


def pcg_step(state: int, increment: int) -> tuple[int, int]:
    old = state
    next_state = (old * PCG_MULTIPLIER + increment) & MASK64
    xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
    rotation = (old >> 59) & 31
    word = ((xorshifted >> rotation) |
            ((xorshifted << ((-rotation) & 31)) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return next_state, word


def pcg_seed(seed: int, stream: int) -> tuple[int, int]:
    increment = ((stream << 1) | 1) & MASK64
    state, _ = pcg_step(0, increment)
    state = (state + seed) & MASK64
    state, _ = pcg_step(state, increment)
    return state, increment


def rng_integrity(seed: int, stream: int, state: int,
                  increment: int, draws: int) -> int:
    value = FNV_INTEGRITY_OFFSET
    for field in (1, 0, seed, stream, state, increment, draws):
        value = ((value ^ field) * FNV_PRIME) & MASK64
    return value


def survivor_fnv(values: list[int]) -> int:
    value = FNV_ORDER_OFFSET
    for token_id in values:
        for shift in (0, 8, 16, 24):
            value = ((value ^ ((token_id >> shift) & 0xFF)) * FNV_PRIME) & MASK64
    return value


def independent_select(logits: list[float], native: dict[str, str]) -> dict[str, Any]:
    # The frozen policy has temperature=1.  Determine the top-k score threshold
    # independently; ties at the threshold remain eligible.
    largest = heapq.nlargest(TOP_K, logits)
    threshold = largest[-1]
    candidates = [index for index, score in enumerate(logits)
                  if score >= threshold]
    candidates.sort(key=lambda token_id: (logits[token_id], token_id))
    maximum = logits[candidates[-1]]
    weights: dict[int, float] = {}
    total = f32(0.0)
    for token_id in candidates:
        weight = f32(math.exp(f32(logits[token_id] - maximum)))
        weights[token_id] = weight
        total = f32(total + weight)
    cutoff = f32(f32(1.0) - TOP_P)
    cumulative = f32(0.0)
    retained_start = 0
    for index, token_id in enumerate(candidates):
        probability = f32(weights[token_id] / total)
        cumulative = f32(cumulative + probability)
        retained_start = index
        if index == len(candidates) - 1 or cumulative > cutoff:
            break
        weights[token_id] = f32(0.0)
        retained_start = index + 1
    retained = candidates[retained_start:]
    final_total = f32(0.0)
    final_weights: dict[int, float] = {}
    for token_id in sorted(retained):
        weight = f32(math.exp(f32(logits[token_id] - maximum)))
        final_weights[token_id] = weight
        final_total = f32(final_total + weight)
    probabilities = {
        token_id: f32(final_weights[token_id] / final_total)
        for token_id in sorted(retained)
    }
    pre_state = int(native["rng_pre_state"])
    increment = int(native["rng_pre_increment"])
    post_state, word = pcg_step(pre_state, increment)
    total_double = sum(float(probabilities[token_id])
                       for token_id in sorted(retained))
    threshold_double = (word / 4_294_967_296.0) * total_double
    cumulative_double = 0.0
    selected = -1
    for token_id in sorted(retained):
        probability = float(probabilities[token_id])
        cumulative_double += probability
        if probability > 0.0 and threshold_double < cumulative_double:
            selected = token_id
            break
    if selected < 0:
        raise RuntimeError("independent categorical interval did not resolve")
    draws_after = int(native["rng_pre_draws"]) + 1
    return {
        "top_k_retained_count": len(candidates),
        "retained_count": len(retained),
        "retained_order_fnv1a64": f"{survivor_fnv(retained):016x}",
        "rng_word": word,
        "rng_state_after": post_state,
        "rng_draws_after": draws_after,
        "rng_integrity_after": rng_integrity(
            int(native["seed"]), int(native["stream"]), post_state,
            increment, draws_after),
        "selected_token_id": selected,
        "selected_probability_f32_hex": struct.pack(
            "<f", probabilities[selected]).hex(),
        "maximum_logit_f32_hex": struct.pack("<f", maximum).hex(),
        "selected_logit_f32_hex": struct.pack("<f", logits[selected]).hex(),
        "runner_up": [
            {
                "token_id": token_id,
                "logit_f32_hex": struct.pack("<f", logits[token_id]).hex(),
            }
            for token_id in sorted(range(VOCABULARY_SIZE),
                                   key=lambda item: (-logits[item], item))[:5]
        ],
    }


def make_evidence(native_input: dict[str, Any], native_log: Path,
                  logits_dir: Path, output: Path) -> None:
    oracle_traces: dict[str, Any] = {}
    native_traces: dict[str, Any] = {}
    state_traces: dict[str, Any] = {}
    exact_steps = 0
    for name in TRACE_NAMES:
        oracle_steps = []
        native_steps = []
        state_steps = []
        seed = int(native_input["traces"][name][0]["seed"])
        stream = int(native_input["traces"][name][0]["stream"])
        expected_state, expected_increment = pcg_seed(seed, stream)
        expected_draws = 0
        for index, fields in enumerate(native_input["traces"][name], 1):
            expected_integrity = rng_integrity(
                seed, stream, expected_state, expected_increment,
                expected_draws)
            if (int(fields["rng_pre_state"]) != expected_state or
                    int(fields["rng_pre_increment"]) != expected_increment or
                    int(fields["rng_pre_draws"]) != expected_draws or
                    int(fields["rng_pre_integrity"]) != expected_integrity):
                raise RuntimeError(f"{name} step {index} pre-RNG mismatch")
            logits_path = logits_dir / f"{name}-step-{index}-logits.f32"
            oracle = independent_select(load_logits(logits_path), fields)
            comparisons = {
                "top_k_retained_count": oracle["top_k_retained_count"] == int(fields["topk_retained"]),
                "retained_count": oracle["retained_count"] == int(fields["survivors"]),
                "retained_order_hash": oracle["retained_order_fnv1a64"] == fields["survivor_hash"],
                "rng_word": oracle["rng_word"] == int(fields["rng_word"]),
                "rng_state_after": oracle["rng_state_after"] == int(fields["rng_post_state"]),
                "rng_increment_after": expected_increment == int(fields["rng_post_increment"]),
                "rng_draws_after": oracle["rng_draws_after"] == int(fields["rng_post_draws"]),
                "rng_integrity_after": oracle["rng_integrity_after"] == int(fields["rng_post_integrity"]),
                "selected_token_id": oracle["selected_token_id"] == int(fields["token"]),
            }
            if not all(comparisons.values()):
                raise RuntimeError(f"{name} step {index} independent mismatch: {comparisons}")
            exact_steps += 1
            expected_state = oracle["rng_state_after"]
            expected_draws = oracle["rng_draws_after"]
            oracle_steps.append({"step": index, **oracle,
                                 "comparison": comparisons})
            native_steps.append({
                "step": index,
                "token_id": int(fields["token"]),
                "decoded_utf8_hex": fields["decoded_hex"],
                "is_eog": bool(int(fields["eog"])),
                "rng_word": int(fields["rng_word"]),
                "retained_count": int(fields["survivors"]),
                "retained_order_fnv1a64": fields["survivor_hash"],
                "logical_payload_bytes": int(fields["payload"]),
                "payload_blocks": int(fields["blocks"]),
                "unique_semantic_tensors": int(fields["semantics"]),
                "embedding_logical_bytes": int(fields["embedding_bytes"]),
                "routed_expert_selections": int(fields["expert_selections"]),
                "selected_expert_matrix_requests": int(fields["expert_matrices"]),
                "ple_row_requests": int(fields["ple_rows"]),
                "lm_head_logical_bytes": int(fields["lm_head_bytes"]),
                "persistent_state_bytes": int(fields["persistent_state"]),
                "peak_model_scratch_bytes": int(fields["peak_scratch"]),
                "logits_bytes": int(fields["logits_bytes"]),
                "maximum_f32_weight_bytes": int(fields["max_f32_weight"]),
                "elapsed_nanoseconds": int(fields["elapsed_ns"]),
            })
            state_steps.append({
                "step": index,
                "model_position": int(fields["position"]),
                "qsa_sequence_length_min_max": [
                    int(fields["position"]), int(fields["position"])],
                "qsa_complete_blocks_min_max": [
                    int(fields["position"]) // 4,
                    int(fields["position"]) // 4],
                "qsa_incomplete_tail_min_max": [
                    int(fields["position"]) % 4,
                    int(fields["position"]) % 4],
                "structural_hash": fields["state_hash"],
                "gdn_state_hash": fields["gdn_hash"],
                "qsa_state_hash": fields["qsa_hash"],
                "ple_address_state_hash": fields["ple_address_hash"],
                "ple_value_state_hash": fields["ple_value_hash"],
                "rng_state_before": int(fields["rng_pre_state"]),
                "rng_state_after": int(fields["rng_post_state"]),
                "rng_draws_before": int(fields["rng_pre_draws"]),
                "rng_draws_after": int(fields["rng_post_draws"]),
                "qsa_candidate_blocks": int(fields["qsa_candidates"]),
                "qsa_selected_blocks": int(fields["qsa_blocks"]),
                "qsa_selected_tokens": int(fields["qsa_tokens"]),
                "moe_route_order_hash": fields["route_hash"],
                "ple_intent_order_hash": fields["ple_hash"],
            })
        oracle_traces[name] = oracle_steps
        native_traces[name] = {
            "seed": int(native_input["traces"][name][0]["seed"]),
            "stream": int(native_input["traces"][name][0]["stream"]),
            "steps": native_steps,
            "summary": native_input["summaries"][name],
        }
        state_traces[name] = state_steps

    replay_checks = []
    for index, (primary, replay) in enumerate(zip(
            native_input["traces"]["primary"], native_input["replay"]), 1):
        primary_logits = logits_dir / f"primary-step-{index}-logits.f32"
        replay_logits = logits_dir / f"replay-step-{index}-logits.f32"
        fields_equal = all(
            primary[field] == replay[field]
            for field in (
                "token", "decoded_hex", "position", "state_hash",
                "gdn_hash", "qsa_hash", "ple_address_hash",
                "ple_value_hash", "rng_pre_state", "rng_word",
                "rng_post_state", "rng_post_integrity", "survivors",
                "survivor_hash", "route_hash", "ple_hash",
            )
        )
        logits_equal = sha256(primary_logits) == sha256(replay_logits)
        if not fields_equal or not logits_equal:
            raise RuntimeError(f"replay step {index} differs from primary")
        replay_checks.append({
            "step": index,
            "bounded_fields_exact": True,
            "full_logits_sha256": sha256(replay_logits),
            "full_logits_exact_to_primary": True,
        })

    contract = {
        "schema": "kq-sampled-generation-contract-v1",
        "baseline_commit": BASELINE,
        "model_revision": MODEL_REVISION,
        "gguf_sha256": GGUF_SHA256,
        "prompt_id": "KQ-PROMPT-001",
        "prompt_token_ids": PROMPT_IDS,
        "context_capacity": 16,
        "max_new_tokens": 4,
        "policy": {"temperature": TEMPERATURE, "top_k": TOP_K,
                   "top_p_f32_hex": struct.pack("<f", TOP_P).hex()},
        "task_3_0_sampling_contract_sha256": "5a185f8b832b7de2785813bfbf4b30deb46d54f31d9208134d71562eff34112f",
        "task_3_0_sampling_manifest_sha256": "fa82837b070f81b70d3d0ff83b48cd79d157b4188f19fcd73082ff275cfe8284",
        "transaction": "model state + caller RNG state + caller outputs",
        "selection_oracle": "independent Python F32 policy/PCG processing over temporary native full logits",
        "oracle_runtime": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "binary64_mantissa_bits": sys.float_info.mant_dig,
            "f32_rounding": "explicit IEEE-754 binary32 pack/unpack after each governed operation",
        },
        "full_logits_committed": False,
    }
    oracle_doc = {
        "schema": "kq-sampled-generation-oracle-v1",
        "class": "NATIVE_LOGITS_INDEPENDENT_SELECTION",
        "implementation": "tools/generate-sampled-generation-evidence.py",
        "native_sampling_code_imported": False,
        "traces": oracle_traces,
    }
    native_doc = {
        "schema": "kq-sampled-generation-native-v1",
        "traces": native_traces,
        "prompt_prefill_count_per_trace": 1,
        "replay": native_input["replay"],
        "memory": {
            "model_state_owned_bytes": int(
                native_input["summaries"]["primary"]["model_state_owned_bytes"]),
            "qsa_capacity_semantic_bytes": 442_368,
            "rng_state_bytes": int(
                native_input["summaries"]["primary"]["rng_state_bytes"]),
            "sampling_config_owned_bytes": int(
                native_input["summaries"]["primary"]["sampling_config_bytes"]),
            "sampling_scratch_bytes": int(
                native_input["summaries"]["primary"]["sampling_scratch_bytes"]),
            "additional_transaction_heap_bytes": 0,
            "peak_model_scratch_bytes": max(
                step["peak_model_scratch_bytes"]
                for trace in native_traces.values() for step in trace["steps"]),
            "logits_bytes": VOCABULARY_SIZE * 4,
            "maximum_f32_weight_bytes": max(
                step["maximum_f32_weight_bytes"]
                for trace in native_traces.values() for step in trace["steps"]),
            "complete_target_f32_matrices": 0,
        },
    }
    state_doc = {
        "schema": "kq-sampled-generation-state-v1",
        "traces": state_traces,
        "state_blobs_committed": False,
    }
    validation_doc = {
        "schema": "kq-sampled-generation-validation-v1",
        "exact_independent_steps": exact_steps,
        "exact_independent_selection_rng": "PASS",
        "deterministic_replay": "PASS",
        "replay_checks": replay_checks,
        "controls": native_input["controls"],
        "padded_id_range": [248077, 248319],
        "padded_id_rollback": "PASS",
        "eog_control": "PASS",
        "early_middle_late_and_sampler_rollback": "PASS",
        "context_exhaustion": "PASS",
        "full_logits_files_committed": 0,
    }
    documents = {
        EVIDENCE_FILES[0]: contract,
        EVIDENCE_FILES[1]: oracle_doc,
        EVIDENCE_FILES[2]: native_doc,
        EVIDENCE_FILES[3]: state_doc,
        EVIDENCE_FILES[4]: validation_doc,
    }
    for name, document in documents.items():
        write_json(output / name, document)
    files = {name: {"sha256": sha256(output / name),
                    "bytes": (output / name).stat().st_size}
             for name in EVIDENCE_FILES}
    repository_root = Path(__file__).resolve().parent.parent
    production_sources = {
        name: sha256(repository_root / name)
        for name in (
            "include/kq_model_exec.h",
            "src/kq_model_exec.c",
            "src/kq_sampling.c",
            "src/run_main.c",
            "tests/sampled_generation_integration.c",
        )
    }
    manifest = {
        "schema": "kq-sampled-generation-manifest-v1",
        "baseline_commit": BASELINE,
        "model_revision": MODEL_REVISION,
        "generation_tool": "tools/generate-sampled-generation-evidence.py",
        "generation_tool_sha256": sha256(Path(__file__).resolve()),
        "production_source_sha256": production_sources,
        "temporary_inputs": {
            "native_transcript_sha256": sha256(native_log),
            "full_logit_files": (
                sum(len(value) for value in native_input["traces"].values()) +
                len(native_input["replay"])),
            "committed": False,
        },
        "files": files,
        "status": "PASS",
    }
    write_json(output / "sampled-generation-manifest.json", manifest)


def compare_directories(left: Path, right: Path) -> None:
    for name in (*EVIDENCE_FILES, "sampled-generation-manifest.json"):
        if (left / name).read_bytes() != (right / name).read_bytes():
            raise RuntimeError(f"non-deterministic evidence: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--logits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parsed = parse_native(args.native_log)
    if args.verify:
        with tempfile.TemporaryDirectory(prefix="kq-sampled-generation-") as temp:
            candidate = Path(temp)
            make_evidence(parsed, args.native_log, args.logits_dir, candidate)
            compare_directories(args.output_dir, candidate)
        print("Sampled generation evidence verification: PASS")
    else:
        make_evidence(parsed, args.native_log, args.logits_dir, args.output_dir)
        print("Sampled generation evidence generation: PASS")
