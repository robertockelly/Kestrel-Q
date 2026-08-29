#!/usr/bin/env python3
"""Generate Task 2.3 canonical tokenizer divergence evidence.

Expected values are produced only by the pinned Transformers/tokenizers oracle.
The GGUF is not read and Kestrel-Q output is never used as an expectation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_REVISION = "805a9e939fa8c1bff8d8ffdf041c051b71a914aa"
TOKENIZERS_REVISION = "7f1623b90b5adfb9bc327d4c3468d2f70bbce262"
MODEL_HASHES = {
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
}


class EvidenceError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def require_assets(model_dir: Path) -> None:
    for name, expected in MODEL_HASHES.items():
        path = model_dir / name
        if not path.is_file():
            raise EvidenceError(f"missing pinned tokenizer asset: {name}")
        observed = sha256_bytes(path.read_bytes())
        if observed != expected:
            raise EvidenceError(f"pinned tokenizer asset hash mismatch: {name}")


def hash_string_array(label: bytes, values: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(label)
    digest.update(struct.pack("<Q", len(values)))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def hash_i32_array(label: bytes, values: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(label)
    digest.update(struct.pack("<Q", len(values)))
    for value in values:
        digest.update(struct.pack("<i", value))
    return digest.hexdigest()


def substrate_hashes(model_dir: Path) -> dict[str, object]:
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    base = [None] * 248_044
    for token, token_id in vocab.items():
        if not isinstance(token_id, int) or token_id < 0 or token_id >= len(base):
            raise EvidenceError("official vocabulary ID is outside the canonical base range")
        if base[token_id] is not None:
            raise EvidenceError("duplicate official vocabulary ID")
        base[token_id] = token
    if any(token is None for token in base):
        raise EvidenceError("official base vocabulary is not contiguous")

    added_decoder = config.get("added_tokens_decoder")
    if not isinstance(added_decoder, dict) or len(added_decoder) != 33:
        raise EvidenceError("official added-token inventory mismatch")
    added: list[str] = []
    for token_id in range(248_044, 248_077):
        entry = added_decoder.get(str(token_id))
        if not isinstance(entry, dict) or not isinstance(entry.get("content"), str):
            raise EvidenceError(f"missing official added token {token_id}")
        added.append(entry["content"])
    tokens = [str(token) for token in base] + added
    tokens.extend(f"[PAD{token_id}]" for token_id in range(248_077, 248_320))

    merge_lines = (model_dir / "merges.txt").read_text(encoding="utf-8").splitlines()
    # This pinned file has no version banner; a leading '#' can be token data.
    # Ignore only the exact optional GPT-2 banner rather than every hash prefix.
    merges = [line for line in merge_lines if line and not line.startswith("#version:")]
    if len(merges) != 247_587:
        raise EvidenceError("official merge count mismatch")

    token_types = [1] * 248_044
    token_types += [3] * 14
    token_types += [4] * 2
    token_types += [3] * 6
    token_types += [4] * 4
    token_types += [3] * 7
    token_types += [5] * 243
    if len(token_types) != 248_320:
        raise EvidenceError("internal token-type contract count mismatch")

    return {
        "token_count": len(tokens),
        "merge_count": len(merges),
        "token_type_count": len(token_types),
        "token_strings_binary_sha256": hash_string_array(b"KQ-TOKEN-STRINGS-v1\0", tokens),
        "merges_binary_sha256": hash_string_array(b"KQ-MERGES-v1\0", merges),
        "token_types_binary_sha256": hash_i32_array(b"KQ-TOKEN-TYPES-v1\0", token_types),
    }


ENCODE_CASES = [
    ("KQ-TOK-DIFF-001", "e\u0301"),
    ("KQ-TOK-DIFF-002", "A\u030a"),
    ("KQ-TOK-DIFF-003", "n\u0303 man\u0303ana"),
    ("KQ-TOK-DIFF-004", "\u212b and A\u030a"),
    ("KQ-TOK-DIFF-005", "\u1100\u1161\u11a8"),
    ("KQ-TOK-DIFF-006", "a\u0323\u0301"),
    ("KQ-TOK-DIFF-007", " \u0301a"),
    ("KQ-TOK-DIFF-008", "a \u0301 b"),
    ("KQ-TOK-DIFF-009", "क\u093f क"),
    ("KQ-TOK-DIFF-010", "'s 'T 're 'VE 'm 'LL 'd"),
    ("KQ-TOK-DIFF-011", "123 ٤٥ १२"),
    ("KQ-TOK-DIFF-012", "  !?\r\n\tword"),
    ("KQ-TOK-DIFF-013", "line1\r\n\r\n line2\t  "),
    ("KQ-TOK-DIFF-014", "emoji🙂+symbols—ok"),
    ("KQ-TOK-DIFF-015", "Ελληνικά 日本語 한국어"),
    ("KQ-TOK-DIFF-016", "<|fim_prefix|>left<|fim_middle|>right<|fim_suffix|>"),
    ("KQ-TOK-DIFF-017", "<|repo_name|>Kestrel-Q<|file_sep|>src/main.c"),
    ("KQ-TOK-DIFF-018", "<|im_end|>"),
    ("KQ-TOK-DIFF-019", ""),
    ("KQ-TOK-DIFF-020", "\u00a0word\u2003next\u2028last"),
    # Unicode 16 assigns/composes U+105D2 U+0307 as U+105C9. The pinned
    # tokenizers NFC table is Unicode 9 and must leave this sequence decomposed,
    # while its Oniguruma regex properties are Unicode 16.
    ("KQ-TOK-DIFF-021", "\U000105d2\u0307"),
    ("KQ-TOK-DIFF-022", "'\u017fx"),
]


DECODE_CASES = [
    ("KQ-TOK-DECODE-001", [248060, 248061, 248062, 248063, 248064, 248065], False),
    ("KQ-TOK-DECODE-002", [248060, 248061, 248062, 248063, 248064, 248065], True),
    ("KQ-TOK-DECODE-003", [248044, 248045, 74455, 198, 248046], False),
    ("KQ-TOK-DECODE-004", [248044, 248045, 74455, 198, 248046], True),
    ("KQ-TOK-DECODE-005", [127], False),
]


CHAT_CASES = [
    {
        "id": "KQ-CHAT-DIFF-001",
        "messages": [{"role": "user", "content": "Hello."}],
        "add_generation_prompt": False,
    },
    {
        "id": "KQ-CHAT-DIFF-002",
        "messages": [{"role": "user", "content": "  trimmed text  "}],
        "add_generation_prompt": True,
    },
]


CHAT_REJECTION_CASES = [
    {
        "id": "KQ-CHAT-DIFF-REJECT-001",
        "reason": "developer role is outside the official initial subset",
        "messages": [{"role": "developer", "content": "merged by the derived template"}, {"role": "user", "content": "Hi"}],
    },
    {
        "id": "KQ-CHAT-DIFF-REJECT-002",
        "reason": "a system message after user content is invalid",
        "messages": [{"role": "user", "content": "Hi"}, {"role": "system", "content": "late"}],
    },
    {
        "id": "KQ-CHAT-DIFF-REJECT-003",
        "reason": "empty chat is invalid",
        "messages": [],
    },
]


def generate(model_dir: Path) -> dict[str, object]:
    from transformers import AutoTokenizer
    import tokenizers
    import transformers
    import unicodedata

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    encode = []
    for case_id, text in ENCODE_CASES:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        encode.append(
            {
                "case_id": case_id,
                "text": text,
                "utf8_sha256": sha256_bytes(text.encode("utf-8")),
                "token_ids": token_ids,
                "decoded_keep_special": tokenizer.decode(
                    token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                ),
                "decoded_skip_special": tokenizer.decode(
                    token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ),
            }
        )

    decode = []
    for case_id, token_ids, skip_special in DECODE_CASES:
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special,
            clean_up_tokenization_spaces=False,
        )
        decode.append(
            {
                "case_id": case_id,
                "token_ids": token_ids,
                "skip_special_tokens": skip_special,
                "decoded": decoded,
                "decoded_utf8_sha256": sha256_bytes(decoded.encode("utf-8")),
            }
        )

    chats = []
    for case in CHAT_CASES:
        rendered = tokenizer.apply_chat_template(
            case["messages"],
            tokenize=False,
            add_generation_prompt=case["add_generation_prompt"],
            add_vision_id=False,
            enable_thinking=False,
            preserve_thinking=False,
            tools=None,
        )
        tokenized = tokenizer.apply_chat_template(
            case["messages"],
            tokenize=True,
            add_generation_prompt=case["add_generation_prompt"],
            add_vision_id=False,
            enable_thinking=False,
            preserve_thinking=False,
            tools=None,
        )
        token_ids = tokenized.input_ids if hasattr(tokenized, "input_ids") else tokenized
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise EvidenceError("unexpected batched chat-template output")
            token_ids = token_ids[0]
        token_ids = list(token_ids)
        chats.append(
            {
                **case,
                "rendered": rendered,
                "rendered_utf8_sha256": sha256_bytes(rendered.encode("utf-8")),
                "token_ids": token_ids,
            }
        )

    rejections = []
    for case in CHAT_REJECTION_CASES:
        try:
            tokenizer.apply_chat_template(
                case["messages"],
                tokenize=False,
                add_generation_prompt=False,
                add_vision_id=False,
                enable_thinking=False,
                preserve_thinking=False,
                tools=None,
            )
        except Exception as error:  # exact oracle rejection is the evidence
            rejections.append(
                {
                    **case,
                    "oracle_status": "REJECTED",
                    "oracle_error_type": type(error).__name__,
                    "oracle_error": str(error),
                }
            )
        else:
            raise EvidenceError(f"canonical oracle unexpectedly accepted {case['id']}")

    return {
        "schema_version": 1,
        "asset_id": "KQ-TOK-DIFFERENTIAL-001",
        "comparison_mode": "EXACT",
        "generated_by": "tools/generate-tokenizer-differential.py",
        "model_revision": MODEL_REVISION,
        "oracle": {
            "implementation": "huggingface/transformers",
            "revision": TRANSFORMERS_REVISION,
            "tokenizers_source_revision": TOKENIZERS_REVISION,
            "license": "Apache-2.0",
            "transformers_version": transformers.__version__,
            "tokenizers_version": tokenizers.__version__,
            "python_unicodedata_version": unicodedata.unidata_version,
            "nfc_unicode_version": "9.0.0",
            "regex_unicode_version": "16.0.0",
        },
        "substrate_contract": substrate_hashes(model_dir),
        "encode_cases": encode,
        "decode_cases": decode,
        "chat_cases": chats,
        "chat_rejection_cases": rejections,
        "bos_policy": "ABSENT_NO_AUTO_ADD",
        "eos_policy": "ID_248046_NO_AUTO_ADD",
        "self_oracle": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_assets(args.model_dir)
    evidence = generate(args.model_dir)
    encoded = canonical_json(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_bytes(encoded),
                "encode_cases": len(evidence["encode_cases"]),
                "decode_cases": len(evidence["decode_cases"]),
                "chat_cases": len(evidence["chat_cases"]),
                "chat_rejections": len(evidence["chat_rejection_cases"]),
                "substrate_contract": evidence["substrate_contract"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}")
        raise SystemExit(1)
