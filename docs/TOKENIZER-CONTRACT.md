# Qwen3.8-Flash-Next tokenizer and chat contract

Status: **CHARACTERIZED / CANONICAL OVERRIDE VERIFIED — 2026-08-29**

The initial Task 2.3 run stopped at its mandatory characterization gate because
the registered GGUF metadata is not sufficient by itself. That blocker evidence
is retained below. The maintainer then selected the governed model-specific
canonical override adapter documented in
`TOKENIZER-PRODUCTION-SOURCE-DECISION.md`; the completed production tokenizer
uses the verified GGUF only as its physical substrate.

## Authority and inspected revisions

Canonical behavior is defined by:

- `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`;
- Qwen research revision `69885871a64393807d988b27b1b5e380e8f28526`;
- the executed Apache-2.0 Transformers oracle at
  `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`;
- `tokenizers==0.23.1`, `transformers==5.16.0.dev0`, Python 3.13.12 and
  Jinja2 3.1.6, as pinned by Task 1.4; and
- the exact Class-C golden hashes already recorded in `docs/GOLDEN-VECTORS.md`.

The inspected derived source is
`unsloth/Qwen3.8-Flash-Next-GGUF@c8b5954a88c2775c546b92593eda40ea041d3176`
and the registered local artifact SHA-256 is
`8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2`.
Only its GGUF header/metadata were read during this characterization.

Pinned official tokenizer asset hashes are:

| Asset | SHA-256 |
|---|---|
| `tokenizer.json` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `tokenizer_config.json` | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` |
| `vocab.json` | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |
| `merges.txt` | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `chat_template.jinja` | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |

## Executed canonical tokenizer

Task 1.4 instantiated `Qwen2Tokenizer` through the pinned Transformers source,
not an older-Qwen assumption. Its active backend contract is:

- model: byte-level BPE;
- base vocabulary: **248,044** tokens with contiguous IDs `0..248043`;
- ordered merges: **247,587**;
- added tokens: **33** at IDs `248044..248076`;
- tokenizer length: **248,077**;
- model/logit vocabulary: **248,320**, leaving IDs `248077..248319` outside
  the canonical tokenizer vocabulary;
- BPE dropout: none;
- unknown token: none;
- byte fallback: false;
- continuing-subword prefix and end-of-word suffix: empty;
- automatic BOS/EOS insertion: none;
- normalization: NFC before pre-tokenization;
- invalid UTF-8 is not an accepted Kestrel-Q public input contract;
- decode uses the reversible GPT-2 byte alphabet, concatenates decoded bytes,
  and the pinned backend replaces incomplete/invalid final UTF-8 with U+FFFD;
- `clean_up_tokenization_spaces=false` for the golden path.

The active pre-tokenization expression is exactly:

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

It is followed by byte-level transformation with `add_prefix_space=false` and
no second byte-level regex pass. The expression deliberately makes one numeric
code point one pre-token, preserves the exact contraction alternatives, treats
CR/LF separately, and retains the GPT-style trailing/inter-word whitespace
rules. Unicode property and normalization behavior is pinned to the executed
`tokenizers==0.23.1` oracle; a native implementation may not delegate it to the
current C locale or silently substitute a different Unicode table version.
Neither the GGUF nor the pinned Python package metadata exposes a standalone
Unicode-data revision suitable for a native implementation; an accepted
production-source policy must also pin the tables that reproduce the executed
oracle.

The standalone official `tokenizer.json` is not identical to that executed
path: its regex includes `\p{M}` in the letter and non-letter alternatives.
The committed Task 1.4 goldens were produced by the active pinned
`Qwen2Tokenizer` path above, so that executed path remains the discrete oracle.

## Added-token inventory

| IDs | Canonical strings | Canonical skip-special behavior |
|---|---|---|
| 248044 | `<|endoftext|>` | skip |
| 248045 | `<|im_start|>` | skip |
| 248046 | `<|im_end|>` | skip |
| 248047–248052 | `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, `<|box_end|>`, `<|quad_start|>`, `<|quad_end|>` | skip |
| 248053–248057 | `<|vision_start|>`, `<|vision_end|>`, `<|vision_pad|>`, `<|image_pad|>`, `<|video_pad|>` | skip |
| 248058–248059 | `<tool_call>`, `</tool_call>` | preserve |
| 248060–248065 | `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|fim_pad|>`, `<|repo_name|>`, `<|file_sep|>` | preserve |
| 248066–248069 | `<tool_response>`, `</tool_response>`, `<think>`, `</think>` | preserve |
| 248070–248076 | `<|audio_start|>`, `<|audio_end|>`, `<tts_pad>`, `<tts_text_bos>`, `<tts_text_eod>`, `<tts_text_bos_single>`, `<|audio_pad|>` | skip |

Canonical configuration has no BOS token. EOS is `<|im_end|>` / 248046 and
padding is `<|endoftext|>` / 248044, but neither BOS nor EOS is automatically
inserted. The golden literal
`<|endoftext|><|im_start|>assistant\n<|im_end|>` encodes as
`[248044, 248045, 74455, 198, 248046]`; preserving special tokens decodes the
literal exactly, while skip-special decode returns `assistant\n`.

## GGUF tokenizer evidence

The registered GGUF exposes:

| Key | Observed value |
|---|---|
| `tokenizer.ggml.model` | `gpt2` |
| `tokenizer.ggml.pre` | `qwen35` |
| `tokenizer.ggml.tokens` | 248,320 unique strings |
| `tokenizer.ggml.token_type` | 248,320 int32 entries |
| `tokenizer.ggml.merges` | 247,587 strings |
| `tokenizer.ggml.eos_token_id` | 248046 |
| `tokenizer.ggml.padding_token_id` | 248044 |
| `tokenizer.ggml.bos_token_id` | 248044 |
| `tokenizer.ggml.add_bos_token` | false |
| `tokenizer.chat_template` | 9,993-byte Unsloth-modified template |

The data arrays reconcile strongly:

- IDs `0..248043` match the official base vocabulary exactly;
- IDs `248044..248076` match all 33 official added-token strings exactly;
- IDs `248077..248319` are 243 unique `[PAD<id>]` entries marked unused;
- all 247,587 ordered merges match exactly;
- token-array canonical-JSON SHA-256 is
  `71c95a5e4e975de80793f13f2c4af8dbfbff055c9c1fdd92a9960439e591dd7f`;
- merge-array canonical-JSON SHA-256 is
  `32a438cb57f9abf2b1d2b6f55af4bc188b404ba239acdffd31c9b1983171d53f`;
- token-type-array canonical-JSON SHA-256 is
  `7da085fec5b7e4c9d71d7385c1fa63c35eca2402f2112db55512712aec274324`.

GGUF token-type distribution is 248,044 normal, 27 control, six user-defined
and 243 unused tokens.

## Production-source sufficiency result

**Result: INSUFFICIENT — FAIL CLOSED.** The vocabulary and merge payload is
complete, but the metadata cannot independently reproduce all canonical
behavior:

1. The canonical NFC normalizer is not represented by a GGUF normalizer key or
   precompiled character map. Removing NFC changes `e` + U+0301 from canonical
   token `[933]` to `[68, 52033]`.
2. GGUF `qwen35` selects a regex containing `\p{M}`; the executed Class-C
   `Qwen2Tokenizer` regex above does not. Current goldens do not establish that
   the two expressions are equivalent for every valid UTF-8 input.
3. GGUF marks IDs `248060..248065` as control tokens. The canonical oracle
   preserves all six during `skip_special_tokens=true`, so trusting GGUF token
   types would silently change decode behavior.
4. GGUF records BOS ID 248044 while the canonical tokenizer has no BOS token.
   `add_bos_token=false` preserves the default golden behavior, but explicit BOS
   semantics are not equivalent.
5. The GGUF has no explicit canonical `add_eos_token=false` field.
6. The embedded chat template SHA-256 is
   `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`,
   not the official template hash. It adds developer-role/merged-system
   behavior, a `high` reasoning alias and changed tool validation/control flow.

The modified GGUF template was executed through the pinned independent oracle
for the exact Task 1.4 subset. All four combinations (two chats, generation
prompt off/on) rendered and tokenized identically to the committed goldens.
That bounded equivalence does not make the modified template canonical for its
other branches.

Treating the exact GGUF arrays as production data while silently hard-coding
official normalizer, regex, token-type and template overrides would have been a
new artifact-adapter policy. The original run correctly stopped until the
maintainer selected one of these separately reviewed resolutions:

- register a corrected derived artifact whose tokenizer metadata expresses the
  canonical contract;
- accept an ADR-governed, model-specific canonical override table in production
  code, with exact source hashes and mismatch validation; or
- authorize a governed sidecar/container extension under a separate artifact
  and provenance decision.

## Canonical chat-template characterization

The official template supports:

- string, null and iterable multimodal content, with explicit invalid-content
  errors and system-message image/video rejection;
- optional reasoning instructions with `xhigh` default plus `medium` and `low`;
- optional tool declarations and tool-call/tool-response rendering;
- an optional first system message and rejection of later system messages;
- user, assistant and tool roles, with rejection of unknown roles;
- optional assistant `reasoning_content`, `preserve_thinking`, tool calls and
  multi-step tool-response handling;
- `add_generation_prompt`; with `enable_thinking=false` its exact suffix is
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`, otherwise it opens
  `<think>\n`.

The intended Task 2.3 initial subset is text-only string content, optional
first system message, user/assistant messages, `tools=null`,
`add_vision_id=false`, `enable_thinking=false`, `preserve_thinking=false`, and
generation prompt off/on. Multimodal content, developer/tool roles, tool calls,
reasoning modes and preserved reasoning remain explicit deferred capabilities.
No generic Jinja implementation is proposed.

## Accepted production-source strategy

The selected second resolution is now implemented. The native construction
gate validates exact ordered binary streams, not a loose metadata resemblance:

| Substrate stream | SHA-256 |
|---|---|
| 248,320 token strings with IDs | `90536bd926c0d6b3230b68cd9fd9006a59ea90950c42827aa5f1795b6e84b340` |
| 247,587 ordered merges/ranks | `dc39bdbcf9d182b14e39b1aa9a54fbc8757342cfdd2c5f55367f4615cc9a64b4` |
| 248,320 GGUF token types | `2cc07d0807489ce9c9e3a04d53fc055a8de48176e61dbf113005b5597f808c84` |

The compiled adapter supplies NFC, the marks-excluding pre-tokenizer,
canonical special classification, BOS absence, EOS identity/no-auto-add and
the official chat subset. Source inspection of
`tokenizers@7f1623b90b5adfb9bc327d4c3468d2f70bbce262` proved that the pinned wheel
uses `unicode-normalization-alignments==0.1.12` with Unicode 9.0.0 tables and
`onig_sys==69.9.3` with Unicode 16.0.0 properties. Kestrel-Q uses utf8proc
2.10.0/Unicode 16.0.0 for validation/categories and a generated assigned-range
gate from Unicode 9 `DerivedAge.txt` (SHA-256
`5cb15b04693c43df16e0d304deca049e93b001445d163184e0ff1b7c8c852146`)
to preserve Unicode-9 NFC semantics. Normalization stability permits current
composition for Unicode-9-assigned runs; later code points remain opaque
boundaries. Provenance/licenses are recorded under `third_party/utf8proc/`,
`src/kq_unicode9_assigned.inc` and `NOTICE`. No GGUF bytes are changed and no
sidecar is loaded.

The expanded independent differential evidence contains 22 encode, five
decode, two supported-chat and three rejected-chat cases. Its SHA-256 is
`f383e213ccc4cde06b47a9855ca1eabb54d0b8911acbd43b2078be5fc546b463`.
It includes the Unicode-16 Todhri sequence U+105D2 U+0307, which the pinned
Unicode-9 NFC must leave decomposed while the Unicode-16 regex still classifies
U+105D2 as a letter.

## Golden boundary

The existing prompt, tokenizer and chat assets remain unchanged:

| Asset | Coverage | SHA-256 |
|---|---|---|
| prompt suite | 10 prompts | `ffee472cac6e57f85df5f50104535b6e4e2d801c4ae6cac1f775840a29b7ed15` |
| tokenizer vectors | 10 prompts / 14 segments | `cbe1290d84a7a61113cf201edaf5034893eb1e70ba5a3c4bc9f4ea50bdcaf153` |
| chat vectors | 2 chats × generation prompt off/on | `72858a105a0f009b94eaea5dec24ae9f45ec0a400b9ce7325efc9341f0fbd1d6` |

They remain independent test oracles and are never production inputs. Native
Kestrel-Q output was compared to these assets and did not define or replace
them.

The pinned offline generator was rerun after characterization. Its eight output
files were byte-identical to the committed golden tree, and the independent
validator reported all eight manifest assets valid. The manifest SHA-256
remains `aa572756672f288957d429a60d7180650ffb2d603a792b21cd72def0a14ec0c4`.

## Payload and implementation boundary

The original characterization stopped after GGUF metadata byte 10,946,618,
before the tensor directory and data section. The continuation reads only the
same parsed tokenizer metadata and does not request tensor views. Model tensor
payload bytes touched remain zero. No dequantizer, inference path, PLE
executor, cache, prefetcher, scheduler or model CUDA kernel was added.
