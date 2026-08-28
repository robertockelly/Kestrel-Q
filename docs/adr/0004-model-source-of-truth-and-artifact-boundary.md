# ADR 0004 — Canonical model source and artifact boundary

Status: **Accepted — 2026-08-28**

## Context

Kestrel-Q targets Qwen3.8-Flash-Next while the development workflow may use multiple representations of that model:

- official Hugging Face Safetensors artifacts;
- official configuration/tokenizer/processor metadata;
- official Qwen architecture documentation;
- locally downloaded GGUF quantizations;
- reference-runtime internal representations.

Converted/quantized artifacts can rename, fuse, omit, repack or change the precision/layout of tensors.

Kestrel-Q therefore needs an explicit source-of-truth hierarchy.

## Decision

1. Official Hugging Face revision
   `de4b8e4d43b917e7706784d8bb445c9af86a3540` of
   `Qwen/Qwen3.8-Flash-Next` is the canonical released artifact source for this
   baseline.
2. The official Safetensors tensor inventory is canonical for released tensor identity and checkpoint composition.
3. Official Qwen technical documentation pinned at GitHub commit
   `69885871a64393807d988b27b1b5e380e8f28526` is canonical for architecture
   semantics not completely encoded in artifact metadata.
4. GGUF files are derived runtime artifacts.
5. Kestrel-Q must not infer canonical architecture solely from GGUF naming/layout.
6. Model weights remain external and must not be committed to the Kestrel-Q source repository.
7. Kestrel-Q's Apache-2.0 source-code license does not relicense upstream Qwen model artifacts.

## Consequences

Positive:

- implementation remains tied to model semantics rather than one converter;
- future quantization formats can be mapped to one canonical tensor model;
- provenance and licensing boundaries remain explicit;
- correctness tests can distinguish canonical reference artifacts from runtime representations.

Negative:

- supporting GGUF requires a mapping layer;
- some research requires parsing large upstream indexes;
- canonical reference validation may eventually require selected official weight shards.

## Validation

- exact official artifact and research revisions are recorded;
- the pinned repository identifies model artifacts under Qwen Community License
  1.0, separately from Kestrel-Q's Apache-2.0 source code;
- the official checkpoint contains 131 shards and its index references 1,658
  tensor names across all 131 shard names;
- the metadata manifest is reproducible and no weight shard was downloaded;
- local GGUF `KQ-MODEL-ARTIFACT-001` is registered with exact size, SHA-256,
  GGUF metadata and pinned Unsloth distribution provenance.

The registered GGUF remains a derived representation and does not alter the
canonical source-of-truth hierarchy.
