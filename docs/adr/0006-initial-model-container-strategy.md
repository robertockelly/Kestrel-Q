# ADR 0006 — Initial model container strategy

Status: **Accepted**

## Context
Kestrel-Q has a verified local Qwen3.8-Flash-Next `UD-Q4_K_XL` GGUF and an exact canonical Safetensors inventory. The runtime ultimately needs model-specific out-of-core handling for hot dense/shared weights, routed experts, PLE and context state.

## Options

### A. Consume GGUF directly
Potential benefits: existing verified quantized artifact, mature metadata/tensor directory, no initial conversion step, easier comparison with community runtimes.

Potential costs: generic layout may not be ideal for Kestrel-Q's prefetch/residency needs, naming/layout translation, physical locality constraints.

### B. Convert first to a Kestrel-Q-native container
Potential benefits: exact model-specific physical grouping and indexes.

Potential costs: larger implementation surface, converter correctness/provenance burden, extra disk/storage workflow.

### C. Staged strategy
Use GGUF for initial correctness/runtime work while keeping Kestrel-Q's internal tensor semantics canonical/model-specific. Introduce a native converted format only if profiling proves material value.

## Evidence gate
Task 1.3 must quantify direct-GGUF mapping complexity, packed family footprints, physical locality relevant to experts/PLE, file overhead and random-access suitability.

## Decision

Adopt **C. Staged strategy**.

The initial correctness/loader path may consume the verified GGUF directly,
but Kestrel-Q's internal identities and validation rules remain canonical and
model-specific. A native converted container is deferred until profiling of a
correct direct-GGUF path demonstrates a material benefit that justifies another
converter, artifact-validation surface and storage workflow.

## Evidence

Task 1.3 establishes:

- complete deterministic coverage of all 1,658 canonical tensors and all 1,224
  GGUF tensors, with zero unresolved or unexplained entries;
- valid GGUF v3 metadata, type/block geometry and non-overlapping random-access
  spans for 111,323,630,080 packed tensor bytes;
- only 11,024,320 bytes of container overhead;
- a 26.855 GiB PLE region and 71.729 GiB routed-expert region that require
  out-of-core policy regardless of the initial container;
- per-layer interleaving and 144 physical runs each for routed and shared expert
  tensor families, which might motivate later repacking but does not prove a
  runtime bottleneck before measurement.

## Consequences

- Direct GGUF support must fail closed on identity, metadata, tensor mapping,
  types, dimensions and spans.
- GGUF names remain physical aliases; public/internal semantic identities come
  from the pinned canonical model.
- No implicit promise is made to support arbitrary GGUF models or quantizations.
- A future native container requires a new evidence-backed decision, exact
  conversion provenance, byte/value validation and measured performance benefit.
- This ADR authorizes no loader, container, tensor kernel or inference work in
  Task 1.3.
