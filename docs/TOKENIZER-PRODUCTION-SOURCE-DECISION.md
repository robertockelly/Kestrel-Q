# TOKENIZER-PRODUCTION-SOURCE-DECISION.md

Status: **IMPLEMENTED / VERIFIED — 2026-08-29**

## Decision

Kestrel-Q will continue Task 2.3 with a **model-specific canonical tokenizer override adapter**.

The verified GGUF remains immutable and supplies tokenizer vocabulary strings,
token IDs and ordered merges only after exact compatibility validation.

Canonical semantics override the GGUF only where Task 2.3 proved divergence.

## Rejected alternatives for the initial runtime

### Corrected derived GGUF
Not required now. Repacking the 111 GB model solely for tokenizer metadata
would create another large artifact identity and couple semantic correctness to
physical repackaging.

### Separate tokenizer sidecar
Not required now. It would introduce another distributed/versioned artifact and
new version-skew risk before evidence shows one is necessary.

## Proven GGUF substrate retained

Task 2.3 proved exact reconciliation for:
- vocabulary strings;
- vocabulary IDs;
- ordered merges.

Construction must fail closed if these no longer match the supported canonical
artifact contract.

## Canonical overrides

Implement exactly from `docs/TOKENIZER-CONTRACT.md`, including the proven
divergences:
- NFC normalization;
- pinned canonical Qwen2Tokenizer pre-tokenization semantics;
- canonical special-token classification/skip-special behavior;
- canonical BOS absence;
- canonical EOS identity and `add_eos_token=false`;
- pinned official canonical chat-template semantics.

Known conflicting GGUF metadata must not be trusted as semantic authority,
including IDs 248060-248065 and the embedded Unsloth template.

## Provenance pins

- Qwen model: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Qwen research: `69885871a64393807d988b27b1b5e380e8f28526`
- Transformers oracle: `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`
- Unsloth GGUF revision: `c8b5954a88c2775c546b92593eda40ea041d3176`
- GGUF SHA-256: `8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2`

## Scope

This unblocks Task 2.3 only. It is not a generic override framework for
arbitrary models or GGUF files.

## Verification result

The implementation validates exact token, merge and token-type substrate
digests before constructing an immutable tokenizer, then applies the compiled
canonical adapter. The original Task 1.4 goldens and the independent divergence
corpus pass exactly; incompatible near-match mutations fail closed. The GGUF
remains unchanged, no sidecar is required and model tensor payload touched is
zero.
