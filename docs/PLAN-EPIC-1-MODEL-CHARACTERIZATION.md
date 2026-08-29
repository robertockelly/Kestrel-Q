# PLAN-EPIC-1-MODEL-CHARACTERIZATION.md

Status: **COMPLETE / PASS — TASKS 1.0–1.4 COMPLETE / PASS**

Task 1.0's official-source baseline and local derived-GGUF registration, Task
1.1's architecture characterization, Task 1.2's canonical tensor/footprint
baseline, Task 1.3's canonical-to-GGUF mapping and Task 1.4's two-oracle
golden-vector strategy are complete/pass. Weight-dependent full-model vectors
remain explicit future correctness gates on a capable reference environment;
they were not fabricated to close this characterization epic.

## Epic

**Epic 1 — Qwen3.8-Flash-Next Model Characterization**

## Purpose

Establish a precise, evidence-backed understanding of Qwen3.8-Flash-Next before Kestrel-Q implements a model loader or inference path.

Kestrel-Q is a model-specific runtime. The architecture must therefore be derived from authoritative model artifacts and specifications rather than inferred from a converted GGUF or from another runtime's abstractions.

## Authoritative upstream sources

Primary upstream sources for this epic:

1. Qwen official model repository:
   - https://huggingface.co/Qwen/Qwen3.8-Flash-Next
2. Qwen official architecture/research repository:
   - https://github.com/QwenLM/Qwen3.8-Flash-Next
3. Qwen official release article:
   - https://qwen.ai/blog?id=qwen3.8-flash-next
4. Qwen model license:
   - https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE

Reference implementations may be studied, but they are secondary evidence unless an upstream source explicitly designates them.

## Starting facts to verify, not blindly assume

Current upstream materials indicate:

- model type: `qwen4_exp`
- architecture: `Qwen4ExpForConditionalGeneration`
- multimodal model (`language_model_only: false`)
- 48 text layers
- repeating 3× linear-attention / 1× full-attention pattern
- hidden size 2560
- 512 routed experts
- 10 routed experts selected per token
- 1 shared expert
- 262,144-token configured maximum position
- Gated DeltaNet + Qwen Sparse Attention
- Gated Residual with 4 residual branches
- N-gram embedding / PLE
- approximately 125B main parameters
- approximately 51B N-gram embedding parameters
- approximately 6B parameters activated per token
- MTP component present in configuration

Every value used by implementation must be re-derived from pinned upstream evidence.

## Scope sequence

### Task 1.0 — Model source & artifact baseline

Establish:

- pinned upstream revision;
- metadata inventory;
- metadata hashes;
- licensing boundary;
- canonical-vs-derived artifact roles;
- local GGUF artifact registration;
- no-weight-download guardrails.

### Task 1.1 — Architecture characterization

Derive:

- text model topology;
- exact layer pattern;
- attention/GDN/QSA equations and state;
- gated residual path;
- MoE routing;
- shared expert;
- N-gram/PLE layout and lookup;
- MTP role;
- vision boundary;
- tokenizer/prompt semantics.

### Task 1.2 — Tensor inventory & footprint

Derive from the official Safetensors index:

- tensor naming families;
- tensor shapes;
- dtypes;
- shard mapping;
- byte footprint by subsystem;
- text-only versus vision/MTP footprint;
- candidate VRAM/RAM/storage placement.

### Task 1.3 — GGUF mapping

Compare the user's quantized GGUF against the canonical tensor model:

- GGUF architecture metadata;
- quantization scheme;
- tensor name transformations;
- packing/layout changes;
- omitted/fused tensors;
- PLE representation;
- MTP/vision representation;
- differences that Kestrel-Q must not mistake for canonical architecture.

### Task 1.4 — Reference behavior plan

Define how Kestrel-Q will obtain authoritative reference vectors for:

- tokenizer IDs;
- prompt rendering;
- routing;
- selected layer checkpoints;
- logits;
- greedy token sequences.

Do not implement the full CPU inference engine in Epic 1.

## Epic exit gate

Epic 1 closes only when:

1. canonical upstream sources are pinned;
2. model licensing/provenance is documented;
3. architecture is described sufficiently to implement a forward pass;
4. tensor families and footprints are quantified;
5. the GGUF mapping is understood;
6. text-only scope is accepted/rejected through an ADR based on evidence;
7. a correctness/reference-vector strategy exists.

## Non-goals

- no model inference;
- no CUDA model kernels;
- no full Safetensors checkpoint download merely for convenience;
- no generic GGUF framework;
- no copying implementation code from incompatible projects;
- no model-weight redistribution inside the Kestrel-Q repository.
