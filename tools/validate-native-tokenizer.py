#!/usr/bin/env python3
"""Compare the native Task 2.3 tokenizer/chat APIs to independent evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_HASHES = {
    "prompt": "ffee472cac6e57f85df5f50104535b6e4e2d801c4ae6cac1f775840a29b7ed15",
    "tokenizer": "cbe1290d84a7a61113cf201edaf5034893eb1e70ba5a3c4bc9f4ea50bdcaf153",
    "chat": "72858a105a0f009b94eaea5dec24ae9f45ec0a400b9ce7325efc9341f0fbd1d6",
    "manifest": "aa572756672f288957d429a60d7180650ffb2d603a792b21cd72def0a14ec0c4",
    "differential": "f383e213ccc4cde06b47a9855ca1eabb54d0b8911acbd43b2078be5fc546b463",
}


class ValidationError(RuntimeError):
    pass


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checked(path: Path, expected: str) -> object:
    observed = file_hash(path)
    if observed != expected:
        raise ValidationError(f"evidence hash mismatch: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_probe(probe: Path, model: Path, arguments: list[str], expect_error: bool = False) -> str:
    completed = subprocess.run(
        [str(probe), str(model), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    output = completed.stdout.strip()
    if expect_error:
        if completed.returncode != 2 or not output.startswith("ERROR:"):
            raise ValidationError(f"native probe unexpectedly accepted: {' '.join(arguments[:2])}")
    elif completed.returncode != 0:
        raise ValidationError(
            f"native probe failed ({completed.returncode}): {output or completed.stderr.strip()}"
        )
    return output


def parse_ids(text: str) -> list[int]:
    return [] if text == "" else [int(item) for item in text.split(",")]


def encode(probe: Path, model: Path, text: str) -> list[int]:
    return parse_ids(run_probe(probe, model, ["encode", text.encode("utf-8").hex()]))


def decode(probe: Path, model: Path, ids: list[int], skip: bool) -> str:
    encoded_ids = ",".join(str(value) for value in ids)
    output = run_probe(probe, model, ["decode", "skip" if skip else "keep", encoded_ids])
    return bytes.fromhex(output).decode("utf-8")


def chat_args(operation: str, case: dict[str, object]) -> list[str]:
    args = [operation, "1" if case.get("add_generation_prompt") else "0"]
    for message in case["messages"]:
        args.extend([message["role"], message["content"].encode("utf-8").hex()])
    return args


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--differential", type=Path, required=True)
    args = parser.parse_args()

    raw_model = os.environ.get("KQ_GGUF_PATH")
    if not raw_model:
        print("KQ_GGUF_PATH unavailable; native tokenizer oracle skipped")
        return 77
    model = Path(raw_model)
    if not model.is_file():
        raise ValidationError("KQ_GGUF_PATH does not identify a file")
    if not args.probe.is_file():
        raise ValidationError("native tokenizer probe is unavailable")

    prompt_path = args.golden_dir / "prompt-suite.json"
    tokenizer_path = args.golden_dir / "canonical" / "tokenizer-vectors.json"
    chat_path = args.golden_dir / "canonical" / "chat-template-vectors.json"
    manifest_path = args.golden_dir / "manifest.json"
    load_checked(prompt_path, EXPECTED_HASHES["prompt"])
    tokenizer_vectors = load_checked(tokenizer_path, EXPECTED_HASHES["tokenizer"])
    chat_vectors = load_checked(chat_path, EXPECTED_HASHES["chat"])
    load_checked(manifest_path, EXPECTED_HASHES["manifest"])
    differential = load_checked(args.differential, EXPECTED_HASHES["differential"])

    prompt_count = len(tokenizer_vectors["vectors"])
    segment_count = 0
    for prompt in tokenizer_vectors["vectors"]:
        for segment in prompt["segments"]:
            segment_count += 1
            expected_ids = segment["token_ids_add_special_tokens_false"]
            actual_ids = encode(args.probe, model, segment["decoded"])
            if actual_ids != expected_ids:
                raise ValidationError(f"tokenizer mismatch: {segment['segment_id']}")
            actual_decoded = decode(args.probe, model, expected_ids, False)
            if actual_decoded != segment["decoded"]:
                raise ValidationError(f"decode mismatch: {segment['segment_id']}")
    if prompt_count != 10 or segment_count != 14:
        raise ValidationError("original tokenizer coverage mismatch")

    for case in chat_vectors["vectors"]:
        rendered_hex = run_probe(args.probe, model, chat_args("chat-render", case))
        if bytes.fromhex(rendered_hex).decode("utf-8") != case["rendered"]:
            raise ValidationError(f"chat rendering mismatch: {case['prompt_id']}")
        actual_ids = parse_ids(run_probe(args.probe, model, chat_args("chat-token", case)))
        if actual_ids != case["token_ids"]:
            raise ValidationError(f"chat token mismatch: {case['prompt_id']}")
    if len(chat_vectors["vectors"]) != 4:
        raise ValidationError("original chat coverage mismatch")

    for case in differential["encode_cases"]:
        actual_ids = encode(args.probe, model, case["text"])
        if actual_ids != case["token_ids"]:
            raise ValidationError(f"differential encode mismatch: {case['case_id']}")
        if decode(args.probe, model, actual_ids, False) != case["decoded_keep_special"]:
            raise ValidationError(f"differential round-trip mismatch: {case['case_id']}")
    for case in differential["decode_cases"]:
        actual = decode(
            args.probe,
            model,
            case["token_ids"],
            case["skip_special_tokens"],
        )
        if actual != case["decoded"]:
            raise ValidationError(f"differential decode mismatch: {case['case_id']}")
    for case in differential["chat_cases"]:
        rendered_hex = run_probe(args.probe, model, chat_args("chat-render", case))
        if bytes.fromhex(rendered_hex).decode("utf-8") != case["rendered"]:
            raise ValidationError(f"differential chat mismatch: {case['id']}")
        actual_ids = parse_ids(run_probe(args.probe, model, chat_args("chat-token", case)))
        if actual_ids != case["token_ids"]:
            raise ValidationError(f"differential chat tokens mismatch: {case['id']}")
    for case in differential["chat_rejection_cases"]:
        if case["messages"]:
            run_probe(args.probe, model, chat_args("chat-render", case), expect_error=True)

    print(
        json.dumps(
            {
                "status": "PASS",
                "original_prompts": prompt_count,
                "original_segments": segment_count,
                "original_chat_vectors": len(chat_vectors["vectors"]),
                "differential_encode": len(differential["encode_cases"]),
                "differential_decode": len(differential["decode_cases"]),
                "differential_chat": len(differential["chat_cases"]),
                "differential_chat_rejections": len(differential["chat_rejection_cases"]),
                "model_tensor_payload_bytes_touched": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
