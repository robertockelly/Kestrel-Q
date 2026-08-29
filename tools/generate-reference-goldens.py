#!/usr/bin/env python3
"""Generate Task 1.4 weight-independent reference assets.

This is research tooling, not a Kestrel-Q runtime component.  It operates only
on the allowlisted metadata captured by Task 1.0 and on pinned source checkouts.
It never resolves a remote model ID and rejects weight files in its inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
RESEARCH_REVISION = "69885871a64393807d988b27b1b5e380e8f28526"
TRANSFORMERS_REVISION = "805a9e939fa8c1bff8d8ffdf041c051b71a914aa"
LLAMA_CPP_REVISION = "90c26fcd4b2114b4aa39d09d69318cb8f438d27a"
UNSLOTH_REVISION = "c8b5954a88c2775c546b92593eda40ea041d3176"
GGUF_FILENAME = "Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf"
GGUF_SIZE = 111_334_654_400
GGUF_SHA256 = "8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2"

MODEL_FILE_HASHES = {
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "config.json": "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}

TRANSFORMERS_FILE_HASHES = {
    "LICENSE": "10fe554b4ffd57008fd1b44474b005f2c35af7312608dbd95fecad82fe87ab42",
    "src/transformers/models/qwen4_exp/configuration_qwen4_exp.py":
        "26b47995740e3bc596b44b2011ee6c3d971d46136438b00dd5fad9557bec4254",
    "src/transformers/models/qwen4_exp/modeling_qwen4_exp.py":
        "91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f",
}

LLAMA_CPP_FILE_HASHES = {
    "LICENSE": "bcd8ec749126d45cb06737d0690295d73df4b6e7e194205bcf91190368f27285",
    "src/llama-arch.cpp": "a36997a3aa3f08294a4988b9f4905dc4fe99153d8b2104b022d66d85085c7756",
    "src/llama-model.cpp": "02a058ca805b38230083c59a21e8ad42d67e753864cc56e8f5216ba6bbba61c2",
    "src/models/qwen4exp.cpp": "dbc93b44cbed2c9187d0ce8e35cdded031b986b2de31633e63f2f5687fa74619",
    "ggml/src/ggml.c": "e6489c22422ab0e3988de1898498ce31d725c2ef743e08a243a226d966a6395a",
    "ggml/src/ggml-cpu/ggml-cpu.c":
        "74ff989a94b4c1daffeeb5a0d625f658504a3131683aa72730893db1f6c683e3",
    "tools/cli/README.md": "af61c3c3013082c208ab14e923ff29e79c1d5746d1e8f462b73706ad08a99bab",
}

EOS_TOKEN_ID = 248_044
VOCAB_SIZE = 248_320
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 8
NGRAM_VOCAB_BASE = 20_000_000
PLE_PART_COUNT = 128
PLE_PART_ROWS = 2_500_012
MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PRIME_1 = 10_007
PLE_SEED = 1_234
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_source_tree(root: Path, revision: str, file_hashes: dict[str, str], role: str) -> None:
    if git_head(root) != revision:
        raise RuntimeError(f"{role} checkout is not pinned to {revision}")
    forbidden = list(root.rglob("*.safetensors")) + list(root.rglob("*.gguf"))
    if forbidden:
        raise RuntimeError(
            f"{role} source checkout contains a forbidden model/container fixture: {forbidden[0]}"
        )
    for relative, expected in file_hashes.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing {role} source file: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"{role} source hash mismatch for {relative}: {observed}")


def validate_model_metadata(model_dir: Path) -> None:
    if model_dir.name != MODEL_REVISION:
        raise RuntimeError(f"metadata directory must be the pinned revision {MODEL_REVISION}")
    forbidden = list(model_dir.rglob("*.safetensors")) + list(model_dir.rglob("*.gguf"))
    if forbidden:
        raise RuntimeError(f"weight/container files are forbidden in metadata input: {forbidden[0]}")
    for relative, expected in MODEL_FILE_HASHES.items():
        path = model_dir / relative
        if not path.is_file():
            raise RuntimeError(f"missing allowlisted model metadata: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"model metadata hash mismatch for {relative}: {observed}")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_config = config["text_config"]
    if config["model_type"] != "qwen4_exp" or text_config["vocab_size"] != VOCAB_SIZE:
        raise RuntimeError("pinned model identity/configuration mismatch")
    if text_config["eos_token_id"] != EOS_TOKEN_ID:
        raise RuntimeError("pinned PLE EOS token mismatch")

    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    if tokenizer_config.get("chat_template") != template:
        raise RuntimeError("standalone and tokenizer-config chat templates differ")


def prompt_suite() -> dict[str, Any]:
    longer = "\n".join(
        f"KQ line {index:03d}: alpha beta gamma {index * 17:05d}."
        for index in range(64)
    )
    prompts: list[dict[str, Any]] = [
        {
            "id": "KQ-PROMPT-001",
            "kind": "raw_text",
            "coverage": ["short_ascii"],
            "text": "Hello, Kestrel-Q.",
        },
        {
            "id": "KQ-PROMPT-002",
            "kind": "raw_text",
            "coverage": ["whitespace", "tab", "lf", "crlf", "trailing_spaces"],
            "text": " \tKQ\n\nline 2\r\nend  ",
        },
        {
            "id": "KQ-PROMPT-003",
            "kind": "raw_text",
            "coverage": ["punctuation", "numbers"],
            "text": "0, 1, -2, 3.14159; []{}()!? #42 / 7 = 6.",
        },
        {
            "id": "KQ-PROMPT-004",
            "kind": "raw_text",
            "coverage": ["italian", "spanish", "accented_utf8"],
            "text": "Perché l'acqua è blu? El pingüino pidió café y azúcar.",
        },
        {
            "id": "KQ-PROMPT-005",
            "kind": "raw_text",
            "coverage": ["non_latin_unicode"],
            "text": "Ελλάδα — 日本語 — العربية — नमस्ते",
        },
        {
            "id": "KQ-PROMPT-006",
            "kind": "raw_text",
            "coverage": ["c_source", "code_punctuation"],
            "text": "int sum(const int *v, size_t n) {\n    int s = 0;\n    for (size_t i = 0; i < n; ++i) s += v[i];\n    return s;\n}\n",
        },
        {
            "id": "KQ-PROMPT-007",
            "kind": "chat_messages",
            "coverage": ["system_user_chat"],
            "messages": [
                {"role": "system", "content": "Answer with one concise sentence."},
                {"role": "user", "content": "Explain why a byte has eight bits."},
            ],
        },
        {
            "id": "KQ-PROMPT-008",
            "kind": "chat_messages",
            "coverage": ["multi_turn_chat"],
            "messages": [
                {"role": "system", "content": "Keep identifiers unchanged."},
                {"role": "user", "content": "Remember KQ_VALUE = 17."},
                {"role": "assistant", "content": "KQ_VALUE is 17.", "reasoning_content": ""},
                {"role": "user", "content": "What is KQ_VALUE plus 5?"},
            ],
        },
        {
            "id": "KQ-PROMPT-009",
            "kind": "raw_text",
            "coverage": ["repeated_prefix", "token_boundary"],
            "text": "prefix prefix prefix-prefix_prefix\nprefixing prefixed prefixes",
        },
        {
            "id": "KQ-PROMPT-010",
            "kind": "raw_text",
            "coverage": ["bounded_longer_prefill"],
            "text": longer,
        },
    ]
    for prompt in prompts:
        if prompt["kind"] == "raw_text":
            identity = prompt["text"].encode("utf-8")
        else:
            identity = json.dumps(
                prompt["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        prompt["identity_serialization"] = "RAW_UTF8" if prompt["kind"] == "raw_text" else "CANONICAL_MESSAGES_JSON_UTF8"
        prompt["utf8_bytes"] = len(identity)
        prompt["utf8_sha256"] = sha256_bytes(identity)
    return {
        "schema_version": 1,
        "suite_id": "KQ-PROMPT-SUITE-001",
        "content_policy": "ORIGINAL_SYNTHETIC_ONLY",
        "prompt_count": len(prompts),
        "prompts": prompts,
    }


def tokenizer_vectors(tokenizer: Any, suite: dict[str, Any]) -> dict[str, Any]:
    vectors = []
    for prompt in suite["prompts"]:
        segments = []
        if prompt["kind"] == "raw_text":
            source_segments = [{"segment_id": "text", "text": prompt["text"]}]
        else:
            source_segments = [
                {"segment_id": f"message-{index}-{message['role']}", "text": message["content"]}
                for index, message in enumerate(prompt["messages"])
            ]
        for source in source_segments:
            text = source["text"]
            ids_without = tokenizer.encode(text, add_special_tokens=False)
            ids_with = tokenizer.encode(text, add_special_tokens=True)
            if ids_without != ids_with:
                raise RuntimeError("Qwen tokenizer unexpectedly injected special tokens")
            decoded = tokenizer.decode(ids_without, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            segments.append(
                {
                    "decoded": decoded,
                    "decoded_utf8_sha256": sha256_bytes(decoded.encode("utf-8")),
                    "round_trip_exact": decoded == text,
                    "segment_id": source["segment_id"],
                    "source_utf8_sha256": sha256_bytes(text.encode("utf-8")),
                    "token_count": len(ids_without),
                    "token_ids_add_special_tokens_false": ids_without,
                    "token_ids_add_special_tokens_true": ids_with,
                }
            )
        vectors.append({"prompt_id": prompt["id"], "segments": segments})

    special_literals = "<|endoftext|><|im_start|>assistant\n<|im_end|>"
    special_ids = tokenizer.encode(special_literals, add_special_tokens=False)
    return {
        "asset_id": "KQ-GOLD-TOK-001",
        "class": "C",
        "comparison_mode": "EXACT",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "oracle": {
            "implementation": "huggingface/transformers",
            "license": "Apache-2.0",
            "revision": TRANSFORMERS_REVISION,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "length_with_added_tokens": len(tokenizer),
            "model_vocab_size": VOCAB_SIZE,
            "tokenizer_vocab_size": tokenizer.vocab_size,
        },
        "special_token_behavior": {
            "all_special_tokens": tokenizer.all_special_tokens,
            "all_special_ids": tokenizer.all_special_ids,
            "literal": special_literals,
            "literal_token_ids": special_ids,
            "decoded_preserve": tokenizer.decode(special_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "decoded_skip": tokenizer.decode(special_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "automatic_bos_or_eos_added": False,
        },
        "vectors": vectors,
    }


def chat_vectors(tokenizer: Any, suite: dict[str, Any]) -> dict[str, Any]:
    vectors = []
    generation_suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    for prompt in suite["prompts"]:
        if prompt["kind"] != "chat_messages":
            continue
        rendered_by_flag: dict[bool, str] = {}
        for add_generation_prompt in (False, True):
            kwargs = {
                "add_generation_prompt": add_generation_prompt,
                "add_vision_id": False,
                "enable_thinking": False,
                "preserve_thinking": False,
            }
            rendered = tokenizer.apply_chat_template(prompt["messages"], tokenize=False, **kwargs)
            tokenized_template = tokenizer.apply_chat_template(prompt["messages"], tokenize=True, **kwargs)
            if not hasattr(tokenized_template, "get") or tokenized_template.get("input_ids") is None:
                raise RuntimeError("chat-template tokenizer did not return input_ids")
            token_ids = tokenized_template["input_ids"]
            independently_encoded = tokenizer.encode(rendered, add_special_tokens=False)
            if token_ids != independently_encoded:
                raise RuntimeError("chat-template tokenization disagrees with rendered-string tokenization")
            rendered_by_flag[add_generation_prompt] = rendered
            vectors.append(
                {
                    "add_generation_prompt": add_generation_prompt,
                    "messages": prompt["messages"],
                    "options": {
                        "add_vision_id": False,
                        "enable_thinking": False,
                        "preserve_thinking": False,
                        "tools": None,
                    },
                    "prompt_id": prompt["id"],
                    "rendered": rendered,
                    "rendered_utf8_bytes": len(rendered.encode("utf-8")),
                    "rendered_utf8_sha256": sha256_bytes(rendered.encode("utf-8")),
                    "special_token_positions": [
                        {"index": index, "token_id": token_id}
                        for index, token_id in enumerate(token_ids)
                        if token_id in tokenizer.all_special_ids
                    ],
                    "token_count": len(token_ids),
                    "token_ids": token_ids,
                }
            )
        if rendered_by_flag[True] != rendered_by_flag[False] + generation_suffix:
            raise RuntimeError(f"unexpected add_generation_prompt delta for {prompt['id']}")
    return {
        "asset_id": "KQ-GOLD-CHAT-001",
        "chat_template_sha256": MODEL_FILE_HASHES["chat_template.jinja"],
        "class": "C",
        "comparison_mode": "EXACT",
        "generation_prompt_suffix": generation_suffix,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "oracle": {
            "implementation": "huggingface/transformers",
            "license": "Apache-2.0",
            "revision": TRANSFORMERS_REVISION,
        },
        "vectors": vectors,
    }


def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def build_multipliers() -> list[int]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // VOCAB_SIZE
    half_bound = max(1, multiplier_max // 2)
    base_seed = PLE_SEED
    return [
        2 * (splitmix64((base_seed + SPLITMIX_GAMMA * (index + 1)) & MASK64) % half_bound) + 1
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


def head_vocab_sizes() -> list[int]:
    values = []
    candidate = NGRAM_VOCAB_BASE - 1
    for _ in range((NGRAM_SIZE - 1) * HEADS_PER_NGRAM):
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
    if source < segment_start or source < 0:
        return EOS_TOKEN_ID
    return tokens[source]


def ple_rows(tokens: list[int], include_only_last: int | None = None) -> list[dict[str, Any]]:
    multipliers = build_multipliers()
    rows = []
    start = 0 if include_only_last is None else len(tokens) - include_only_last
    for position in range(start, len(tokens)):
        addresses: list[dict[str, Any]] = []
        for order in (2, 3):
            mixed = shifted_token(tokens, position, 0) * multipliers[0]
            for shift in range(1, order):
                mixed ^= shifted_token(tokens, position, shift) * multipliers[shift]
            head_start = (order - 2) * HEADS_PER_NGRAM
            for local_head in range(HEADS_PER_NGRAM):
                head = head_start + local_head
                address = mixed % EXPECTED_HEAD_VOCABS[head] + EXPECTED_HEAD_OFFSETS[head]
                addresses.append(
                    {
                        "global_address": address,
                        "head_offset": EXPECTED_HEAD_OFFSETS[head],
                        "head_vocab_size": EXPECTED_HEAD_VOCABS[head],
                        "local_head": local_head,
                        "ngram_order": order,
                        "partition_id": address // PLE_PART_ROWS,
                        "partition_row": address % PLE_PART_ROWS,
                    }
                )
        rows.append(
            {
                "addresses": addresses,
                "history_tokens": [
                    shifted_token(tokens, position, shift) for shift in range(1, NGRAM_SIZE)
                ],
                "position": position,
                "token_id": tokens[position],
            }
        )
    return rows


def ple_vectors() -> dict[str, Any]:
    if build_multipliers() != EXPECTED_MULTIPLIERS:
        raise RuntimeError("source-derived PLE multipliers differ from pinned artifact evidence")
    if head_vocab_sizes() != EXPECTED_HEAD_VOCABS:
        raise RuntimeError("source-derived PLE head primes differ from pinned artifact evidence")
    offsets: list[int] = []
    total = 0
    for size in EXPECTED_HEAD_VOCABS:
        offsets.append(total)
        total += size
    if offsets != EXPECTED_HEAD_OFFSETS:
        raise RuntimeError("source-derived PLE offsets differ from pinned artifact evidence")

    cases = [
        ("KQ-GOLD-PLE-001", "length_1", [1]),
        ("KQ-GOLD-PLE-002", "length_2", [1, 2]),
        ("KQ-GOLD-PLE-003", "length_3", [1, 2, 3]),
        ("KQ-GOLD-PLE-004", "length_gt_3", [7, 11, 13, 17, 19]),
        ("KQ-GOLD-PLE-005", "repeated_tokens", [42, 42, 42, 42]),
        ("KQ-GOLD-PLE-006", "large_valid_token_ids", [248_319, 248_318, 248_317]),
        ("KQ-GOLD-PLE-007", "eos_segment_boundary", [5, EOS_TOKEN_ID, 7, 8]),
    ]
    vectors = [
        {"case": label, "id": vector_id, "input_ids": tokens, "rows": ple_rows(tokens)}
        for vector_id, label, tokens in cases
    ]

    decode_prefill = [101, 102, 103]
    decode_steps = [104, 105, EOS_TOKEN_ID, 106]
    history = decode_prefill[-2:]
    step_vectors = []
    complete = list(decode_prefill)
    for step_index, token in enumerate(decode_steps):
        combined = history + [token]
        address_row = ple_rows(combined, include_only_last=1)[0]
        address_row["position"] = len(complete)
        step_vectors.append(
            {
                "addresses": address_row["addresses"],
                "history_after": (history + [token])[-2:],
                "history_before": list(history),
                "step": step_index,
                "token_id": token,
            }
        )
        history = (history + [token])[-2:]
        complete.append(token)
    expected_rows = ple_rows(complete)[len(decode_prefill):]
    if [row["addresses"] for row in expected_rows] != [row["addresses"] for row in step_vectors]:
        raise RuntimeError("incremental PLE addresses differ from full-sequence reference")

    return {
        "address_arithmetic": "signed int64 products/XOR; positive remainder by per-head prime; add head offset",
        "asset_id": "KQ-GOLD-PLE-ROOT-001",
        "canonical_constants": {
            "eos_token_id": EOS_TOKEN_ID,
            "head_offsets": EXPECTED_HEAD_OFFSETS,
            "head_vocab_sizes": EXPECTED_HEAD_VOCABS,
            "heads_per_ngram": HEADS_PER_NGRAM,
            "layer_multipliers": EXPECTED_MULTIPLIERS,
            "ngram_size": NGRAM_SIZE,
            "partition_count": PLE_PART_COUNT,
            "partition_rows": PLE_PART_ROWS,
            "seed": PLE_SEED,
        },
        "class": "C",
        "comparison_mode": "EXACT",
        "decode_case": {
            "id": "KQ-GOLD-PLE-008",
            "prefill_input_ids": decode_prefill,
            "steps": step_vectors,
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "oracle": {
            "implementation": "pinned Transformers Qwen4-Exp PLE algorithm and official config",
            "license": "Apache-2.0",
            "revision": TRANSFORMERS_REVISION,
            "source_file_sha256": TRANSFORMERS_FILE_HASHES[
                "src/transformers/models/qwen4_exp/modeling_qwen4_exp.py"
            ],
        },
        "vectors": vectors,
    }


def numeric_contract(dtype: str, shape: list[Any]) -> dict[str, Any]:
    return {
        "atol": None,
        "dtype": dtype,
        "inf_policy": "MATCH_REFERENCE",
        "nan_policy": "FORBID_UNLESS_REFERENCE_HAS_NAN",
        "rtol": None,
        "shape": shape,
        "tolerance_status": "TO_BE_CALIBRATED_FROM_REFERENCE_RUNS",
    }


def operator_plan() -> dict[str, Any]:
    operators = {
        "GR": {
            "fields": ["normalized_branches", "read_gates", "branch_read_result", "write_coefficients", "resulting_branches"],
            "shapes": {"branches": [1, "N", 4, 2560], "branch_read_result": [1, "N", 2560]},
        },
        "GDN": {
            "fields": ["q", "k", "v", "alpha", "beta", "conv_state_before", "conv_state_after", "recurrent_state_before", "recurrent_state_after", "layer_output"],
            "shapes": {"recurrent_state": [1, 48, 128, 128], "conv_state": [1, 10240, 4]},
        },
        "QSA": {
            "fields": ["index_query", "raw_index_keys", "pooled_block_keys", "index_scores", "selected_block_ids", "selected_token_positions", "k_cache_before", "k_cache_after", "v_cache_before", "v_cache_after", "attention_output"],
            "shapes": {"raw_index_keys": [1, "T", 128], "kv_cache_per_kind": [1, 2, "T", 256]},
        },
        "MOE": {
            "fields": ["router_logits", "router_probabilities", "top10_expert_ids", "normalized_expert_weights", "shared_expert_gate", "routed_sum", "shared_output", "combined_output"],
            "shapes": {"router_logits": ["B*N", 512], "top10_expert_ids": ["B*N", 10]},
        },
        "FINAL": {
            "fields": ["selected_hidden_states", "final_hidden_state", "full_logits", "topk_token_ids", "topk_logits", "greedy_next_token_id", "short_greedy_sequence"],
            "shapes": {"full_logits": [1, 1, 248320], "final_hidden_state": [1, 1, 2560]},
        },
    }
    entries = []
    for name, contract in operators.items():
        entries.append(
            {
                "comparison": {
                    "discrete_fields": "EXACT",
                    "floating_fields": numeric_contract("REFERENCE_REPORTED", ["FIELD_SPECIFIC"]),
                },
                "future_capture_fields": contract["fields"],
                "operator": name,
                "reason": "No independently executed weight-free reference vector was captured in Task 1.4; capture through the pinned oracle before implementation parity is claimed.",
                "required_shapes": contract["shapes"],
                "status": "PLANNED_REFERENCE_REQUIRED",
            }
        )
    return {
        "asset_id": "KQ-GOLD-OPERATOR-PLAN-001",
        "class": "C",
        "independence_rule": "Kestrel-Q implementation output cannot create or replace its own reference vector.",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "operators": entries,
        "oracle": {
            "implementation": "huggingface/transformers",
            "license": "Apache-2.0",
            "revision": TRANSFORMERS_REVISION,
        },
        "schema_version": 1,
        "status": "PLANNED_REFERENCE_REQUIRED",
    }


def canonical_full_model_plan() -> dict[str, Any]:
    checkpoints = [
        {"id": "embedding_gr_input", "location": "after token embedding replication", "fields": ["input_ids", "position_ids", "four_branches"]},
        {"id": "layer_00_gdn", "location": "after first GDN and GR write", "fields": ["gdn_output", "recurrent_state", "conv_state", "branches"]},
        {"id": "layer_01_ple", "location": "PLE address/retrieval/injection before layer-1 GDN read", "fields": ["addresses", "ple_direct", "ple_convolved", "branches_after_injection"]},
        {"id": "layer_03_qsa", "location": "first QSA layer", "fields": ["index_scores", "selected_blocks", "selected_positions", "raw_index_keys", "k_cache", "v_cache", "attention_output"]},
        {"id": "routers", "location": "layers 0, 1, 3, 23, 47", "fields": ["router_logits", "top10_ids", "normalized_weights", "shared_gate"]},
        {"id": "middle", "location": "after layers 23 and 24", "fields": ["four_branches", "persistent_state_digests"]},
        {"id": "late", "location": "after layer 47 and final GR read", "fields": ["four_branches", "final_hidden"]},
        {"id": "final", "location": "last prompt position and each generated step", "fields": ["full_logits", "top64_ids", "top64_logits", "greedy_id", "generated_ids"]},
    ]
    return {
        "asset_id": "KQ-GOLD-C-FULL-PLAN-001",
        "capable_environment": {
            "accelerators": "8 x NVIDIA H100 80 GiB or equivalent aggregate BF16 capacity with safety headroom",
            "host_ram_min_gib": 768,
            "os": "Ubuntu 24.04 x86_64",
            "python": "3.11.9",
            "storage_free_min_gib": 750,
            "storage_requirement": "local NVMe; exact pinned 131-shard checkpoint; network disabled during execution",
        },
        "checkpoints": checkpoints,
        "class": "C",
        "dependencies": {
            "accelerate": "1.12.0",
            "huggingface_hub": "1.29.0",
            "jinja2": "3.1.6",
            "numpy": "2.3.5",
            "safetensors": "0.8.0",
            "tokenizers": "0.23.1",
            "torch": "2.9.1+cu128",
            "transformers_revision": TRANSFORMERS_REVISION,
        },
        "execution_contract": {
            "batch_size": 1,
            "canonical_dtype": "BF16",
            "determinism": [
                "torch.use_deterministic_algorithms(True)",
                "torch.backends.cuda.matmul.allow_tf32=False",
                "torch.backends.cudnn.allow_tf32=False",
                "torch.backends.cudnn.benchmark=False",
                "fixed device map recorded before capture",
            ],
            "do_sample": False,
            "max_new_tokens": 8,
            "mtp": False,
            "prompt_ids": ["KQ-PROMPT-001", "KQ-PROMPT-006", "KQ-PROMPT-008"],
            "speculation": False,
            "text_only": True,
            "vision": False,
        },
        "future_acceptance_gate": "Before canonical forward-pass correctness is claimed, execute this plan, record environment/device mapping and capture hashes, then calibrate floating tolerances from repeated independent runs without hiding mismatches.",
        "model_artifact_license": "Qwen Community License 1.0",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "oracle": {
            "implementation": "huggingface/transformers",
            "license": "Apache-2.0",
            "revision": TRANSFORMERS_REVISION,
        },
        "resource_decision": "Canonical checkpoint is 359999963128 payload bytes; generation is prohibited on KQ-01.",
        "status": "DEFERRED_CAPABLE_REFERENCE_ENV",
        "tolerance_policy": {
            "discrete": "EXACT",
            "floating": numeric_contract("CHECKPOINT_REPORTED", ["CHECKPOINT_SPECIFIC"]),
        },
    }


def quantized_plan() -> dict[str, Any]:
    return {
        "artifact": {
            "filename": GGUF_FILENAME,
            "model_artifact_license": "Qwen Community License 1.0",
            "sha256": GGUF_SHA256,
            "size_bytes": GGUF_SIZE,
            "unsloth_revision": UNSLOTH_REVISION,
        },
        "asset_id": "KQ-GOLD-Q-FULL-PLAN-001",
        "build_verification": {
            "command": "cmake -S $LLAMA_CPP_SOURCE -B $LLAMA_CPP_BUILD -G 'Visual Studio 17 2022' -A x64 -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF && cmake --build $LLAMA_CPP_BUILD --config Release --target llama-cli",
            "observed_binary_version": "0.3.0-dev (build 1, commit 90c26fc), MSVC 19.44.35228.0 x64",
            "result": "PASS_NO_MODEL_LOAD",
            "source_hashes": LLAMA_CPP_FILE_HASHES,
        },
        "capable_environment": {
            "host_ram_min_gib": 256,
            "os": "Windows 11 x64 or Ubuntu 24.04 x86_64; record exact selection",
            "storage": "local NVMe with at least 160 GiB free",
        },
        "class": "Q",
        "execution_contract": {
            "batch_size": 1,
            "command_template": "llama-cli --model $KQ_GGUF_PATH --ctx-size 4096 --batch-size 512 --ubatch-size 512 --n-gpu-layers 0 --load-mode mmap --spec-type none --temp 0 --seed 0 --no-warmup --simple-io --no-display-prompt --n-predict 8 --prompt-file $KQ_RENDERED_PROMPT_FILE",
            "context_size": 4096,
            "device_backend": "CPU_ONLY_REFERENCE",
            "greedy": True,
            "mmap": "ENABLED_EXPLICIT",
            "offload": "n_gpu_layers=0",
            "prompt_ids": ["KQ-PROMPT-001", "KQ-PROMPT-006", "KQ-PROMPT-008"],
            "speculation": "DISABLED_EXPLICIT_--spec-type_none",
            "text_only": True,
        },
        "future_acceptance_gate": "Verify the exact GGUF SHA immediately before execution, capture binary/source/build identity and stdout bytes, and require repeated greedy token IDs to match exactly before Class-Q correctness is claimed.",
        "oracle": {
            "implementation": "ggml-org/llama.cpp llama-cli",
            "license": "MIT",
            "revision": LLAMA_CPP_REVISION,
        },
        "resource_decision": "DEFER: the 103.688 GiB file exceeds KQ-01's 32 GiB RAM and 10 GiB VRAM; its SATA backing makes a controlled no-thrash run infeasible.",
        "runtime_capability_evidence": {
            "architecture": "qwen4exp registered and has a complete graph implementation",
            "ordinary_generation": "llama-cli built and exposes model/prompt/greedy controls",
            "speculation_default_and_override": "none; explicit --spec-type none required",
            "tensor_types": ["F32", "Q5_1", "Q8_0", "Q4_K", "Q5_K", "IQ4_NL", "BF16"],
        },
        "status": "DEFERRED_CAPABLE_REFERENCE_ENV",
        "tolerance_policy": {
            "greedy_token_ids_and_stdout_utf8": "EXACT",
            "floating_logits_if_future_tool_exposes_them": numeric_contract("REFERENCE_REPORTED", [1, 1, 248320]),
        },
    }


def asset_record(
    asset_id: str,
    golden_class: str,
    status: str,
    path: str,
    output_root: Path,
    comparison: str,
    dependency: list[str],
) -> dict[str, Any]:
    absolute = output_root / path
    return {
        "asset_id": asset_id,
        "class": golden_class,
        "comparison_mode": comparison,
        "dependencies": dependency,
        "generation_command": "python tools/generate-reference-goldens.py --model-dir $KQ_MODEL_METADATA_DIR --transformers-source $KQ_TRANSFORMERS_SOURCE --llama-source $KQ_LLAMA_SOURCE --output-dir research/goldens/Qwen3.8-Flash-Next",
        "generation_tool": "tools/generate-reference-goldens.py",
        "path": f"research/goldens/Qwen3.8-Flash-Next/{path}",
        "sha256": sha256_file(absolute),
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--transformers-source", type=Path, required=True)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    transformers_source = args.transformers_source.resolve()
    llama_source = args.llama_source.resolve()
    output_dir = args.output_dir.resolve()
    validate_model_metadata(model_dir)
    validate_source_tree(transformers_source, TRANSFORMERS_REVISION, TRANSFORMERS_FILE_HASHES, "Transformers")
    validate_source_tree(llama_source, LLAMA_CPP_REVISION, LLAMA_CPP_FILE_HASHES, "llama.cpp")

    try:
        import jinja2
        import tokenizers
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("pinned tokenizer-generation dependencies are unavailable") from exc

    versions = {
        "jinja2": jinja2.__version__,
        "python": platform.python_version(),
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
    }
    expected_versions = {
        "jinja2": "3.1.6",
        "python": "3.13.12",
        "tokenizers": "0.23.1",
        "transformers": "5.16.0.dev0",
    }
    if versions != expected_versions:
        raise RuntimeError(f"generation dependency mismatch: {versions} != {expected_versions}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    if type(tokenizer).__name__ != "Qwen2Tokenizer":
        raise RuntimeError(f"unexpected tokenizer class: {type(tokenizer).__name__}")

    suite = prompt_suite()
    write_json(output_dir / "prompt-suite.json", suite)
    write_json(output_dir / "canonical/tokenizer-vectors.json", tokenizer_vectors(tokenizer, suite))
    write_json(output_dir / "canonical/chat-template-vectors.json", chat_vectors(tokenizer, suite))
    write_json(output_dir / "canonical/ple-address-vectors.json", ple_vectors())
    write_json(output_dir / "canonical/operator-vector-plan.json", operator_plan())
    write_json(output_dir / "canonical/full-model-vector-plan.json", canonical_full_model_plan())
    write_json(output_dir / "quantized/gguf-vector-plan.json", quantized_plan())

    assets = [
        asset_record("KQ-GOLD-PROMPT-C-001", "C", "GENERATED_VERIFIED", "prompt-suite.json", output_dir, "EXACT", []),
        asset_record("KQ-GOLD-PROMPT-Q-001", "Q", "GENERATED_VERIFIED", "prompt-suite.json", output_dir, "EXACT", []),
        asset_record("KQ-GOLD-TOK-001", "C", "GENERATED_VERIFIED", "canonical/tokenizer-vectors.json", output_dir, "EXACT", ["prompt-suite.json", "official tokenizer metadata"]),
        asset_record("KQ-GOLD-CHAT-001", "C", "GENERATED_VERIFIED", "canonical/chat-template-vectors.json", output_dir, "EXACT", ["prompt-suite.json", "official chat_template.jinja"]),
        asset_record("KQ-GOLD-PLE-ROOT-001", "C", "GENERATED_VERIFIED", "canonical/ple-address-vectors.json", output_dir, "EXACT", ["official config", "pinned Transformers PLE algorithm"]),
        asset_record("KQ-GOLD-OPERATOR-PLAN-001", "C", "PLANNED_REFERENCE_REQUIRED", "canonical/operator-vector-plan.json", output_dir, "NUMERIC_TOLERANCE", ["pinned Transformers full operator execution"]),
        asset_record("KQ-GOLD-C-FULL-PLAN-001", "C", "DEFERRED_CAPABLE_REFERENCE_ENV", "canonical/full-model-vector-plan.json", output_dir, "EXACT_AND_NUMERIC_TOLERANCE", ["canonical BF16 checkpoint", "capable reference environment"]),
        asset_record("KQ-GOLD-Q-FULL-PLAN-001", "Q", "DEFERRED_CAPABLE_REFERENCE_ENV", "quantized/gguf-vector-plan.json", output_dir, "EXACT_AND_NUMERIC_TOLERANCE", ["exact registered GGUF", "capable reference environment"]),
    ]
    prompt_hash = sha256_file(output_dir / "prompt-suite.json")
    manifest = {
        "assets": assets,
        "canonical_model": {
            "artifact_license": "Qwen Community License 1.0",
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "research_revision": RESEARCH_REVISION,
        },
        "generated_safe_vector_count": 3,
        "generation_dependencies": versions,
        "gguf": {
            "artifact_id": "KQ-MODEL-ARTIFACT-001",
            "artifact_license": "Qwen Community License 1.0",
            "filename": GGUF_FILENAME,
            "sha256": GGUF_SHA256,
            "size_bytes": GGUF_SIZE,
            "unsloth_revision": UNSLOTH_REVISION,
        },
        "manifest_id": "KQ-GOLDEN-MANIFEST-001",
        "oracles": [
            {
                "class": "C",
                "implementation": "huggingface/transformers",
                "license": "Apache-2.0",
                "revision": TRANSFORMERS_REVISION,
                "status": "PINNED_TOKENIZER_EXECUTED_MODEL_SOURCE_VERIFIED",
                "source_hashes": TRANSFORMERS_FILE_HASHES,
            },
            {
                "class": "Q",
                "implementation": "ggml-org/llama.cpp llama-cli",
                "license": "MIT",
                "revision": LLAMA_CPP_REVISION,
                "status": "PINNED_BUILD_VERIFIED_NO_MODEL_RUN",
                "source_hashes": LLAMA_CPP_FILE_HASHES,
            },
        ],
        "prompt_suite_sha256": prompt_hash,
        "schema_version": 1,
        "weight_dependent_vector_status": "DEFERRED_CAPABLE_REFERENCE_ENV",
        "weight_payload_downloaded_or_executed": False,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output_dir / "manifest.json"), "prompt_suite_sha256": prompt_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
