# ADR 0011 — Native tokenizer and chat formatting

Status: **Accepted — 2026-08-29**

## Context

Epic 1 pinned exact tokenizer and chat-template behavior through independent
canonical golden vectors.

Task 2.0-2.2 established the native GGUF, semantic and bounded physical-weight
layers. Kestrel-Q now needs an exact native text-to-token boundary before model
execution.

## Initial requirements

1. Kestrel-Q implements a native C17 Qwen3.8-Flash-Next tokenizer.
2. Behavior is pinned to the canonical model revision and Epic 1 goldens.
3. Production tokenizer data comes from verified GGUF metadata when that
   metadata is proven sufficient for canonical equivalence.
4. Missing tokenizer semantics fail closed; no undocumented sidecar is
   introduced automatically.
5. Python/Transformers/tokenizers are oracle/research tools only, not runtime
   dependencies.
6. Chat formatting is a separate model-specific module above the tokenizer.
7. Kestrel-Q implements supported pinned template semantics, not a general
   Jinja interpreter.
8. Unsupported roles/content/template options fail explicitly.
9. Golden JSON files are test inputs only.
10. Kestrel-Q output never defines its own expected output.
11. Tokenizer/chat construction and execution do not access tensor payload.

## Evidence gate

Task 2.3 must prove:
- exact 10-prompt / 14-segment tokenizer golden match;
- exact 2-chat x generation-prompt off/on match;
- unchanged independent golden hashes;
- production tokenizer source sufficiency;
- fail-closed incompatible metadata/input;
- zero model-weight payload access.

## Original blocked decision evidence

The initial Task 2.3 run reached the mandatory source-sufficiency gate and found
that the registered GGUF is not a self-sufficient canonical tokenizer source:

- vocabulary strings and ordered merges reconcile exactly;
- canonical NFC normalization is absent from GGUF metadata;
- `tokenizer.ggml.pre=qwen35` selects a marks-inclusive expression rather than
  the executed Class-C `Qwen2Tokenizer` expression;
- GGUF marks IDs 248060–248065 as control tokens although the canonical oracle
  preserves them during skip-special decode; and
- the embedded Unsloth chat template is not byte-identical to the official
  pinned template, despite matching the four committed initial-subset cases.

Silently compiling canonical overrides or inventing a sidecar would have
created an ungoverned source substitution. The implementation therefore
remained blocked until explicit maintainer direction was recorded.

## Accepted decision

The maintainer selected a governed model-specific canonical tokenizer override
adapter:

1. The verified immutable GGUF supplies vocabulary strings, IDs and ordered
   merge/rank data only after exact full-substrate digest validation.
2. The compiled Qwen3.8 adapter supplies canonical NFC, the pinned
   marks-excluding pre-tokenizer, byte-level BPE behavior, canonical 33-token
   classification, BOS absence, EOS/no-auto-add policy and the supported
   official chat-template semantics.
3. The pinned oracle's split Unicode contract is preserved: Unicode 9.0.0 NFC
   from `unicode-normalization-alignments==0.1.12` and Unicode 16.0.0 regex
   properties from `onig_sys==69.9.3`. A generated Unicode-9 assigned-range
   gate plus utf8proc 2.10.0 at commit
   `a1b99daa2a3393884220264c927a48ba1251a9c6` provides the governed,
   locale-independent implementation under the MIT/Unicode-data boundary.
4. The 111 GB GGUF remains unchanged. No tokenizer sidecar is introduced.
5. Any missing or changed token string/ID, merge/rank, token type, special
   mapping, padded ID or required tokenizer metadata fails closed.
6. The embedded Unsloth chat template is not production semantic authority.
7. Python/Transformers/tokenizers remain independent test/research oracles only.
8. Tokenizer/chat code remains CPU-only and does not touch tensor payload.

This is deliberately model-specific. It does not establish a generic override
framework or authorize Task 2.4/model execution.

## Consequences

- Exact source pins and compatibility digests make the otherwise compiled
  override reconstructible and auditable.
- A repacked GGUF and a distributed sidecar are avoided for the initial
  runtime.
- Supporting another artifact revision or broader chat branch requires new
  evidence and an explicit contract update rather than fallback matching.
- Kestrel-Q now carries the small utf8proc dependency and its attribution.
