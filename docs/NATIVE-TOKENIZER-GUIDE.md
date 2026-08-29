# NATIVE-TOKENIZER-GUIDE.md

Status: **TASK 2.3 COMPLETE / PASS**

The exact characterization, preserved fail-closed source finding and governed
canonical-override resolution are recorded in `docs/TOKENIZER-CONTRACT.md`.
The production API and implementation rules are documented in
`docs/NATIVE-TOKENIZER.md` and `docs/KQ-TOKENIZER-API.md`.

## Characterize before implementing

Do not identify the tokenizer by resemblance to older Qwen releases.

Use pinned canonical tokenizer artifacts, actual GGUF metadata and the pinned
reference implementation to establish the exact contract first.

## Runtime source

Use only the exactly validated GGUF token/ID and merge/rank substrate. Apply
the governed model-specific canonical semantics; never fall back to divergent
GGUF token-type/pre-tokenizer/chat-template behavior. Any substrate mismatch
fails closed. No sidecar is used.

## Exactness

Tokenizer correctness is discrete.

Expected token IDs use exact equality.
Rendered chat prompts use exact UTF-8 byte equality.

Do not use approximate, normalized or human-equivalent comparisons where exact
evidence exists.

## Separation

```text
kq_tokenizer
  UTF-8 <-> token IDs

kq_chat
  structured messages -> rendered UTF-8
```

The chat formatter calls the tokenizer; tokenizer core does not know chat
roles.

## No generic template engine by default

The runtime needs the pinned Qwen3.8 template semantics, not arbitrary Jinja.

Unsupported branches/options fail explicitly.

## Golden source boundary

`research/goldens/...` files are tests/oracles only.

Production C must not read them.

## No tensor payload access

Tokenizer state is metadata/configuration.

Task 2.3 must not open model-weight payload views.
