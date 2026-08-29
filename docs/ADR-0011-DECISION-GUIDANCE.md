# ADR-0011-DECISION-GUIDANCE.md

Status: **APPLIED**

This guidance was applied to ADR 0011 after implementation evidence passed.

## Final decision

The verified Unsloth GGUF is not, by itself, the semantic source of truth for
tokenizer behavior.

Kestrel-Q shall:

1. retain GGUF vocabulary strings, token IDs and ordered merges only after
   exact compatibility validation against the pinned canonical tokenizer;
2. supply Qwen3.8-Flash-Next tokenizer semantic behavior from a compiled,
   model-specific canonical adapter pinned to the official model/oracle
   revisions;
3. override only evidence-proven divergent semantics, including normalization,
   pre-tokenization, special-token behavior, BOS/EOS policy and chat-template
   behavior;
4. fail closed if the physical tokenizer substrate no longer matches the
   supported canonical artifact contract;
5. leave the source GGUF immutable;
6. avoid an external tokenizer sidecar for the initial runtime;
7. avoid a generic override framework for arbitrary models.

## Rationale

This preserves one verified large weight artifact, canonical behavioral
correctness, container/semantic separation, and no runtime Python dependency,
while avoiding a new sidecar versioning surface.
