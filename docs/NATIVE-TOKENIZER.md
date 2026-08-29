# Native Qwen3.8 tokenizer

Status: **IMPLEMENTED / VERIFIED — Task 2.3**

Kestrel-Q implements the pinned Qwen3.8-Flash-Next tokenizer in C17. The
verified GGUF supplies only the physical vocabulary strings, token IDs and
ordered BPE merges. A compiled model-specific adapter supplies the canonical
semantics that the GGUF cannot express faithfully.

## Construction and compatibility

`kq_tokenizer_open_from_gguf` requires the model/GGUF objects to have already
passed native parsing and validates:

- `qwen4exp`, `gpt2`, `qwen35`, EOS 248046, padding 248044, recorded BOS
  248044 with `add_bos_token=false`;
- 248,320 token strings, 248,320 token types and 247,587 ordered merges;
- exact ordered binary stream SHA-256 values for token strings/IDs
  (`90536bd...b340`), merges/ranks (`dc39bdbc...64b4`) and token types
  (`2cc07d08...8c84`);
- all 33 canonical added-token strings at IDs 248044–248076; and
- 243 unique `[PAD<id>]` entries at 248077–248319 marked unused.

Any near-match fails with `KQ_STATUS_INCOMPATIBLE_TOKENIZER`. The GGUF is not
mutated and no tokenizer sidecar is loaded.

## Canonical adapter

The adapter implements the contract in `TOKENIZER-CONTRACT.md`:

1. validate UTF-8 and normalize with the pinned oracle's Unicode 9.0.0 NFC
   semantics, using a generated assigned-range gate plus vendored utf8proc
   2.10.0 composition;
2. scan the exact marks-excluding pinned `Qwen2Tokenizer` pre-tokenization
   branches without using the C/Windows locale;
3. apply the reversible GPT-2 byte alphabet and ordered byte-level BPE;
4. recognize or explicitly reject the 33 canonical special-token literals;
5. never add BOS or EOS automatically; and
6. decode only IDs 0–248076, with canonical keep/skip-special policy and
   U+FFFD replacement for an incomplete final UTF-8 sequence.

IDs 248077–248319 are model capacity padding, not tokenizer output, and decode
rejects them. IDs 248060–248065 are intentionally retained when canonical
skip-special decoding is requested, despite conflicting GGUF token types.

## Ownership and payload boundary

The tokenizer is immutable after construction and owns only lookup tables. Its
token byte views borrow GGUF metadata, so the tokenizer must be closed before
the GGUF/model/file objects. There is no global mutable state.

Tokenizer construction and execution read metadata only and never request a
tensor view. Real integration retained `model_tensor_payload_bytes_touched=0`.

## Validation

The native result exactly matches the independent pinned oracle for all 10
prompts, 14 segments, decode/round-trip cases and the expanded 22-encode/
5-decode divergence corpus. The evidence file is
`research/tokenizer/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/canonical-differential.json`
(SHA-256 `f383e213ccc4cde06b47a9855ca1eabb54d0b8911acbd43b2078be5fc546b463`).

One observed Release construction run on KQ-01 took 114,608,200 ns. The
tokenizer reported 24,945,728 owned heap bytes: 12,361,728 for token views and
lookup plus 12,582,912 for merge lookup. Encode temporary allocation is input
dependent and is not retained in the immutable object. These are
characterization observations, not a stable performance baseline.

## Limits

- Maximum public encode input is 16 MiB per call.
- Only the pinned artifact contract is accepted; this is not a generic
  Qwen/GPT-2 tokenizer loader.
- No inference, tensor math or model payload access is introduced.
