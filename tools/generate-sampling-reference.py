#!/usr/bin/env python3
"""Generate independent Task 3.0 sampling expectations before native code."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_REVISION = "805a9e939fa8c1bff8d8ffdf041c051b71a914aa"
PCG_REVISION = "bc39cd76ac3d541e618606bcc6e1e5ba5e5e6aa3"
GENERATION_CONFIG_SHA256 = (
    "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
)
SOURCE_HASHES = {
    "src/transformers/generation/logits_process.py":
        "c5b5d5666576b19e19a09b99b55806ea009dd8e999892f7a8d2d7876dbfc2002",
    "src/transformers/generation/utils.py":
        "c45e19eb4534a2478f8e4825dc35a28fb5efe5a2f8d09695e4b2ea532b4dbd82",
    "src/transformers/generation/configuration_utils.py":
        "427f7e57922fb58c3b321f5e5837ee782bd640cd5b22833029b16f9d4b9c4cdb",
}
PCG_HASHES = {
    "pcg_basic.c":
        "b3a77c42f9e7b57a095aa0c5f684c42a2fdade5cb823cdb483bb09a88ffc6fe0",
    "pcg_basic.h":
        "18b8752b39fe07d527b4afe7756d1349fd44871f218811ec4d57c1ea7fc1e9f5",
    "LICENSE.txt":
        "e03ba41d7fab20700769fe4118bab50d800cb74f990353a05d2f5fff1c228363",
}
VOCAB_SIZE = 248_320
CANONICAL_TOKEN_COUNT = 248_077
DEFAULT_STREAM = 0x4B515F53414D504C
PCG_MULTIPLIER = 6364136223846793005
MASK64 = (1 << 64) - 1
ORACLE_FILES = (
    "sampling-contract.json",
    "sampling-calibration.json",
    "sampling-holdout.json",
    "sampling-rng-vectors.json",
    "sampling-statistical.json",
)


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


def f32_hex(value: float) -> str:
    return struct.pack("<f", float(value)).hex()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def id_sha256(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<I", value))
    return digest.hexdigest()


def pcg_step(state: int, increment: int) -> tuple[int, int]:
    old_state = state
    state = (old_state * PCG_MULTIPLIER + increment) & MASK64
    xorshifted = (((old_state >> 18) ^ old_state) >> 27) & 0xFFFFFFFF
    rotation = (old_state >> 59) & 31
    word = ((xorshifted >> rotation) |
            ((xorshifted << ((-rotation) & 31)) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return state, word


def pcg_seed(seed: int, stream: int) -> tuple[int, int]:
    increment = ((stream << 1) | 1) & MASK64
    state, _ = pcg_step(0, increment)
    state = (state + seed) & MASK64
    state, _ = pcg_step(state, increment)
    return state, increment


def categorical(word: int, probabilities: list[tuple[int, float]]) -> int:
    total = math.fsum(probability for _, probability in probabilities)
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("invalid independent categorical mass")
    threshold = (word / 4294967296.0) * total
    cumulative = 0.0
    last = -1
    for token_id, probability in probabilities:
        if probability <= 0.0:
            continue
        last = token_id
        cumulative += probability
        if threshold < cumulative:
            return token_id
    if last >= 0 and threshold < total:
        return last
    raise RuntimeError("independent categorical interval did not resolve")


def materialize(torch: Any, base: float,
                overrides: dict[int, float]) -> Any:
    logits = torch.full((1, VOCAB_SIZE), f32(base), dtype=torch.float32)
    for token_id, value in overrides.items():
        if token_id < 0 or token_id >= VOCAB_SIZE:
            raise RuntimeError(f"invalid synthetic token ID: {token_id}")
        logits[0, token_id] = f32(value)
    return logits


def apply_oracle(torch: Any, processors: Any, case: dict[str, Any]) -> dict[str, Any]:
    policy = case["policy"]
    logits = materialize(torch, case["base_logit"], case["overrides"])
    scores = logits
    active: list[str] = []
    temperature = f32(policy["temperature"])
    top_k = int(policy["top_k"])
    top_p = f32(policy["top_p"])
    if temperature != 1.0:
        scores = processors.TemperatureLogitsWarper(float(temperature))(
            torch.zeros((1, 1), dtype=torch.long), scores)
        active.append("temperature")
    if top_k != 0:
        scores = processors.TopKLogitsWarper(top_k, min_tokens_to_keep=1)(
            torch.zeros((1, 1), dtype=torch.long), scores)
        active.append("top_k")
    top_k_ids = torch.nonzero(torch.isfinite(scores[0]), as_tuple=False).flatten()
    top_k_ids_list = [int(value) for value in top_k_ids.tolist()]
    if top_p < 1.0:
        scores = processors.TopPLogitsWarper(float(top_p), min_tokens_to_keep=1)(
            torch.zeros((1, 1), dtype=torch.long), scores)
        active.append("top_p")
    retained = torch.nonzero(torch.isfinite(scores[0]), as_tuple=False).flatten()
    retained_ids = [int(value) for value in retained.tolist()]
    probabilities = torch.nn.functional.softmax(scores, dim=-1)[0]
    records = [{
        "token_id": token_id,
        "processed_logit_f32_le_hex": f32_hex(float(scores[0, token_id])),
        "probability_f32_le_hex": f32_hex(float(probabilities[token_id])),
    } for token_id in retained_ids]
    state, increment = pcg_seed(int(case["seed"]), int(case["stream"]))
    next_state, word = pcg_step(state, increment)
    probability_pairs = [(record["token_id"],
                          f32(struct.unpack("<f", bytes.fromhex(
                              record["probability_f32_le_hex"]))[0]))
                         for record in records]
    selected = categorical(word, probability_pairs)
    expected_status = ("INVALID_TOKEN_ID" if selected >= CANONICAL_TOKEN_COUNT
                       else "OK")
    return {
        "active_processors": active,
        "top_k_retained_count": len(top_k_ids_list),
        "top_k_retained_ids": (top_k_ids_list
                               if len(top_k_ids_list) <= 256 else None),
        "top_k_retained_ids_u32le_sha256": id_sha256(top_k_ids_list),
        "retained_count": len(retained_ids),
        "retained_ids": retained_ids,
        "retained_ids_u32le_sha256": id_sha256(retained_ids),
        "retained": records,
        "rng_state_before": f"{state:016x}",
        "rng_word": f"{word:08x}",
        "rng_state_after": f"{next_state:016x}",
        "selected_token_id": selected,
        "expected_status": expected_status,
    }


def case(case_id: str, base: float, overrides: dict[int, float],
         temperature: float = 1.0, top_k: int = 20, top_p: float = 0.95,
         seed: int = 0, stream: int = DEFAULT_STREAM) -> dict[str, Any]:
    return {
        "id": case_id,
        "base_logit": f32(base),
        "overrides": {str(key): f32(value)
                      for key, value in sorted(overrides.items())},
        "policy": {
            "temperature": f32(temperature),
            "top_k": top_k,
            "top_p": f32(top_p),
        },
        "seed": seed,
        "stream": stream,
    }


def calibration_cases() -> list[dict[str, Any]]:
    official = {index * 17: 6.0 - index * 0.25 for index in range(25)}
    official[248044] = 1.5
    official[248046] = 1.25
    official[248076] = 1.0
    official[248077] = 0.75
    return [
        case("official-default", -80.0, official, seed=0),
        case("top-k-threshold-tie", -80.0,
             {5: 3.0, 6: 2.0, 7: 2.0, 8: 2.0, 9: 1.0},
             top_k=3, top_p=1.0, seed=1),
        case("top-p-below-half", -80.0, {10: 0.0, 11: 0.0},
             top_k=0, top_p=f32(0.49999997), seed=2),
        case("top-p-on-half", -80.0, {10: 0.0, 11: 0.0},
             top_k=0, top_p=0.5, seed=3),
        case("top-p-above-half", -80.0, {10: 0.0, 11: 0.0},
             top_k=0, top_p=f32(0.50000006), seed=4),
        case("temperature-half", -80.0,
             {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0},
             temperature=0.5, top_k=4, top_p=0.8, seed=5),
        case("dominant-eog", -80.0, {3: 0.0, 248046: 20.0}, seed=6),
    ]


def holdout_cases() -> list[dict[str, Any]]:
    equal_twenty = {1000 + index: 1.0 for index in range(20)}
    return [
        case("special-domain", -90.0,
             {0: 2.0, 248044: 1.5, 248046: 1.0, 248076: 0.5},
             seed=0x100),
        case("equal-twenty", -80.0, equal_twenty, seed=0x101),
        case("large-finite", -100.0,
             {20: 80.0, 21: 79.0, 22: 78.0, 23: 60.0},
             top_k=4, top_p=0.999, seed=0x102),
        case("temperature-two", -80.0,
             {30: -12.0, 31: -8.0, 32: -4.0, 33: 0.0},
             temperature=2.0, top_k=4, top_p=0.9, seed=0x103),
        case("highest-canonical-id", -80.0, {248076: 30.0}, seed=0x104),
        case("first-padded-selected", -80.0, {248077: 30.0}, seed=0x105),
        case("last-padded-selected", -80.0, {248319: 30.0}, seed=0x106),
    ]


def enrich(torch: Any, processors: Any,
           cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for specification in cases:
        normalized = dict(specification)
        normalized["overrides"] = {int(key): value
                                   for key, value in specification["overrides"].items()}
        normalized["oracle"] = apply_oracle(torch, processors, normalized)
        normalized["overrides"] = {str(key): value
                                   for key, value in sorted(normalized["overrides"].items())}
        result.append(normalized)
    return result


def rng_vectors() -> dict[str, Any]:
    cases = []
    for seed, stream in (
            (0, 0), (MASK64, 0), (0, DEFAULT_STREAM),
            (MASK64, (1 << 63) - 1)):
        state, increment = pcg_seed(seed, stream)
        initial = state
        words = []
        for draw in range(8):
            state, word = pcg_step(state, increment)
            words.append({
                "draw": draw + 1,
                "word_hex": f"{word:08x}",
                "uniform_hex": float.hex(word / 4294967296.0),
                "state_after_hex": f"{state:016x}",
            })
        cases.append({
            "seed": seed,
            "stream": stream,
            "increment_hex": f"{increment:016x}",
            "seeded_state_hex": f"{initial:016x}",
            "words": words,
        })
    return {
        "schema": "kq-sampling-rng-vectors-v1",
        "comparison_class": "EXACT_BITS",
        "algorithm": "PCG-XSH-RR-64-32",
        "pcg_revision": PCG_REVISION,
        "uniform_mapping": "binary64(word / 2^32)",
        "cases": cases,
        "explicit_word_boundaries": [
            {"word_hex": "00000000", "uniform_hex": float.hex(0.0)},
            {"word_hex": "80000000", "uniform_hex": float.hex(0.5)},
            {"word_hex": "ffffffff",
             "uniform_hex": float.hex(0xFFFFFFFF / 4294967296.0)},
        ],
    }


def statistical_evidence(torch: Any, processors: Any) -> dict[str, Any]:
    specifications = [
        case("uniform-two", -80.0, {100: 0.0, 101: 0.0},
             seed=0x300, top_k=20, top_p=0.95),
        case("uniform-four", -80.0,
             {200: 0.0, 201: 0.0, 202: 0.0, 203: 0.0},
             seed=0x301, top_k=20, top_p=0.95),
        case("official-equal-twenty", -80.0,
             {300 + index: 0.0 for index in range(20)},
             seed=0x302, top_k=20, top_p=0.95),
    ]
    prepared = enrich(torch, processors, specifications)
    trials = 100_000
    total_cells = sum(item["oracle"]["retained_count"] for item in prepared)
    family_alpha = 0.001
    epsilon = math.sqrt(math.log(2.0 * total_cells / family_alpha) /
                        (2.0 * trials))
    records = []
    for item in prepared:
        oracle = item["oracle"]
        probabilities = [
            (entry["token_id"], struct.unpack(
                "<f", bytes.fromhex(entry["probability_f32_le_hex"]))[0])
            for entry in oracle["retained"]
        ]
        state, increment = pcg_seed(item["seed"], item["stream"])
        counts = {token_id: 0 for token_id, _ in probabilities}
        for _ in range(trials):
            state, word = pcg_step(state, increment)
            counts[categorical(word, probabilities)] += 1
        outcomes = []
        passed = True
        for token_id, probability in probabilities:
            observed = counts[token_id] / trials
            error = abs(observed - probability)
            passed = passed and error <= epsilon
            outcomes.append({
                "token_id": token_id,
                "expected_probability_f32_le_hex": f32_hex(probability),
                "count": counts[token_id],
                "frequency": observed,
                "absolute_error": error,
                "pass": error <= epsilon,
            })
        records.append({
            "id": item["id"],
            "policy": item["policy"],
            "base_logit": item["base_logit"],
            "overrides": item["overrides"],
            "seed": item["seed"],
            "stream": item["stream"],
            "retained_ids": oracle["retained_ids"],
            "trials": trials,
            "outcomes": outcomes,
            "pass": passed,
        })
    return {
        "schema": "kq-sampling-statistical-v1",
        "comparison_class": "PREDECLARED_STATISTICAL",
        "method": "per-category absolute frequency Hoeffding bound",
        "family_alpha": family_alpha,
        "multiple_test_policy": "union bound over every retained category",
        "total_category_assertions": total_cells,
        "trials_per_case": trials,
        "absolute_error_bound": epsilon,
        "threshold_selected_before_native_run": True,
        "cases": records,
        "pass": all(record["pass"] for record in records),
    }


def generate(output: Path, checkout: Path, pcg_checkout: Path,
             config_path: Path) -> None:
    import torch
    import tokenizers
    import transformers
    from transformers.generation import logits_process as processors

    if torch.__version__ != "2.11.0+cpu":
        raise RuntimeError(f"unexpected PyTorch version: {torch.__version__}")
    if tokenizers.__version__ != "0.23.1":
        raise RuntimeError(f"unexpected tokenizers version: {tokenizers.__version__}")
    if not transformers.__version__.startswith("5.16.0.dev0"):
        raise RuntimeError(f"unexpected Transformers package: {transformers.__version__}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required_config = {
        "bos_token_id": 248044,
        "do_sample": True,
        "eos_token_id": [248046, 248044],
        "pad_token_id": 248044,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
    }
    if config != required_config:
        raise RuntimeError("official generation configuration changed")

    contract = {
        "schema": "kq-sampling-contract-v1",
        "status": "PINNED_BEFORE_NATIVE_VALIDATION",
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "generation_config": config,
        "vocabulary": {
            "model_capacity": VOCAB_SIZE,
            "canonical_token_count": CANONICAL_TOKEN_COUNT,
            "padded_ids": [CANONICAL_TOKEN_COUNT, VOCAB_SIZE - 1],
            "padded_policy": "participate_then_fail_if_selected",
            "eog_ids": [248046, 248044],
        },
        "processor_order": ["temperature_if_nonunit", "top_k_if_nonzero",
                            "top_p_if_below_one", "softmax", "categorical"],
        "parameter_domain": {
            "temperature": "finite F32 > 0",
            "top_k": "integer 0..248320; zero disables",
            "top_p": "finite F32 0..1; one disables",
        },
        "non_finite_policy": "FAIL_CLOSED",
        "top_k": "remove score < kth score; preserve threshold ties",
        "top_p": "ascending score/id; remove cumulative <= 1-p; keep one",
        "rng": {
            "algorithm": "PCG-XSH-RR-64-32",
            "pcg_revision": PCG_REVISION,
            "license": "Apache-2.0",
            "state_format_version": 1,
            "default_stream": DEFAULT_STREAM,
            "uniform_mapping": "binary64(word / 2^32)",
            "draws_per_successful_selection": 1,
        },
        "oracle_environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": __import__("numpy").__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "source_hashes": SOURCE_HASHES,
        "pcg_source_hashes": PCG_HASHES,
        "expected_values_from_kestrel_q": False,
        "model_weights_loaded": False,
    }
    write_json(output / "sampling-contract.json", contract)

    calibration = {
        "schema": "kq-sampling-calibration-v1",
        "comparison": {
            "membership_and_selection": "EXACT_DISCRETE",
            "rng": "EXACT_BITS",
            "logits_and_probabilities": "CALIBRATED_FLOAT",
        },
        "cases": enrich(torch, processors, calibration_cases()),
    }
    write_json(output / "sampling-calibration.json", calibration)
    holdout = {
        "schema": "kq-sampling-holdout-v1",
        "disjoint_from_calibration": True,
        "contracts_may_not_be_widened": True,
        "cases": enrich(torch, processors, holdout_cases()),
    }
    write_json(output / "sampling-holdout.json", holdout)
    write_json(output / "sampling-rng-vectors.json", rng_vectors())
    write_json(output / "sampling-statistical.json",
               statistical_evidence(torch, processors))

    files = {name: {
        "sha256": sha256(output / name),
        "bytes": (output / name).stat().st_size,
        "status": "GENERATED_INDEPENDENT_ORACLE",
    } for name in ORACLE_FILES}
    files["sampling-native-validation.json"] = {
        "status": "NOT_GENERATED_BEFORE_NATIVE_IMPLEMENTATION"
    }
    manifest = {
        "schema": "kq-sampling-evidence-manifest-v1",
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "pcg_revision": PCG_REVISION,
        "generation_tool": "tools/generate-sampling-reference.py",
        "generation_command": (
            "python tools/generate-sampling-reference.py --transformers-checkout "
            ".research-cache/task-1.4/transformers --pcg-checkout "
            ".research-cache/task-3.0/pcg-c-basic --generation-config "
            ".research-cache/model-baseline/" + MODEL_REVISION +
            "/generation_config.json --output-dir research/sampling/"
            "Qwen3.8-Flash-Next/" + MODEL_REVISION),
        "offline_generation": True,
        "model_weights_loaded": False,
        "expected_values_from_kestrel_q": False,
        "files": files,
    }
    write_json(output / "sampling-manifest.json", manifest)


def verify_checkout(path: Path, revision: str, hashes: dict[str, str]) -> None:
    actual = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            check=True, capture_output=True,
                            text=True).stdout.strip()
    if actual != revision:
        raise SystemExit(f"unexpected checkout revision at {path}: {actual}")
    for relative, expected in hashes.items():
        candidate = path / relative
        if not candidate.is_file() or sha256(candidate) != expected:
            raise SystemExit(f"pinned source hash mismatch: {candidate}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers-checkout", type=Path, required=True)
    parser.add_argument("--pcg-checkout", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    checkout = args.transformers_checkout.resolve()
    pcg_checkout = args.pcg_checkout.resolve()
    config = args.generation_config.resolve()
    output = args.output_dir.resolve()
    verify_checkout(checkout, TRANSFORMERS_REVISION, SOURCE_HASHES)
    verify_checkout(pcg_checkout, PCG_REVISION, PCG_HASHES)
    if not config.is_file() or sha256(config) != GENERATION_CONFIG_SHA256:
        raise SystemExit("official generation-config hash mismatch")
    if args.verify:
        with tempfile.TemporaryDirectory(prefix="kq-sampling-verify-") as directory:
            candidate = Path(directory)
            generate(candidate, checkout, pcg_checkout, config)
            mismatches = [name for name in ORACLE_FILES
                          if not (output / name).is_file() or
                          (output / name).read_bytes() !=
                          (candidate / name).read_bytes()]
            if mismatches:
                raise SystemExit("non-deterministic sampling evidence: " +
                                 ", ".join(mismatches))
        print("Sampling independent evidence regeneration: PASS")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    for name in (*ORACLE_FILES, "sampling-native-validation.json",
                 "sampling-manifest.json"):
        path = output / name
        if path.exists():
            path.unlink()
    generate(output, checkout, pcg_checkout, config)
    print(f"Sampling independent evidence written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
