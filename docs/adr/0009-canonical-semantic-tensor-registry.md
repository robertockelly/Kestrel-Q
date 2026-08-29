# ADR 0009 — Canonical semantic tensor registry

Status: **Accepted — 2026-08-29**

## Context

Task 2.0 introduced a native GGUF physical container layer.

Epic 1 demonstrated that GGUF physical representation differs from the
canonical Qwen checkpoint through naming changes, fusions, splits,
metadata-derived semantics and scope omissions.

Allowing future GDN/QSA/MoE/PLE code to address raw GGUF names would couple
model execution to one converter/container representation.

## Decision

1. Introduce a Qwen3.8-Flash-Next canonical semantic tensor registry.
2. Stable semantic keys are independent from GGUF names.
3. A target-specific GGUF adapter maps physical descriptors to semantic
   descriptors.
4. Binding relations explicitly represent one-to-one, split, fused,
   transformed and metadata-derived forms.
5. Future execution layers consume semantic descriptors, not GGUF names.
6. Epic 1 CSV/JSON mapping evidence is test oracle only, never runtime input.
7. Unknown, ambiguous, missing or structurally incompatible mappings fail
   closed.
8. Preliminary placement hints may live on semantic descriptors but do not
   constitute memory policy.
9. Vision/MTP absence is intentional initial scope under ADR 0005.

## Consequences

Positive:
- future container changes do not rewrite model execution logic;
- mapping correctness becomes independently testable;
- placement/scheduler logic can operate on semantic roles;
- fusion/split representation is explicit.

Negative:
- Kestrel-Q owns a model-specific mapping layer;
- mapping rules must track future artifact revisions;
- one physical tensor may participate in multiple semantic descriptors.

## Evidence gate

Task 2.1 must demonstrate:

- 1224/1224 physical tensors reconciled;
- 1294/1294 initial-text semantic entries;
- zero unknown physical tensors;
- zero unbound required semantics;
- full Epic 1 mapping-oracle comparison;
- zero payload access.

## Evidence result

Task 2.1 satisfies the gate:

- the target-first adapter validates `qwen4exp`, hidden/vocabulary/context
  dimensions, exact 48-layer GDN/QSA topology, 512 experts/top-k 10, shared
  expert geometry and complete PLE configuration;
- 1,294/1,294 initial-text semantic descriptors are constructed with 847
  renamed, 256 transformed, 60 split, 128 fused and three metadata-derived
  relations;
- all 1,224 physical descriptors are covered exactly by reviewed rules, with
  zero unknown physical tensors and zero unbound required semantics;
- the deterministic native registry dump matches every initial-text row and
  every physical name in the pinned Epic 1 mapping oracle;
- synthetic coverage fails closed on near-match metadata/topology, missing or
  ambiguous tensors, incompatible rank/shape/type/expert axis, invalid split or
  fusion geometry and missing PLE metadata; and
- real-artifact construction retains `payload_bytes_accessed = 0`.

The registry is immutable after construction and has no global mutable mapping
state. It introduces no payload view, dequantization, inference, tokenizer,
PLE execution, allocation, cache or scheduler. Task 2.2 may build bounded
quantized views over these bindings under its own correctness gate.
