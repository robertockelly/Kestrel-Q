# Project Plan

This plan is intentionally high level. Detailed implementation plans should be created only when a phase is about to start.

## Phase 0 — Repository and research baseline

Goal: establish facts before implementing inference.

Deliverables:

- project governance
- agent rules
- reference hardware specification
- authoritative model architecture notes
- model artifact inventory
- Apache-2.0 licensing baseline and third-party compatibility review
- benchmark methodology
- reference implementation selection
- initial ADRs

Exit gate:

- model architecture is sufficiently understood to describe one forward pass;
- required tensor metadata is catalogued;
- reference outputs can be generated independently.

## Phase 1 — File loading and model introspection

Goal: load the target model safely without performing full inference.

Workstreams:

- model container/file-format decision
- memory mapping
- tensor metadata parser
- tensor lookup
- datatype/quantization metadata
- tokenizer assets
- diagnostics CLI

Exit gate:

- runtime enumerates and validates all expected tensors;
- unsupported/malformed artifacts fail closed;
- loading does not require full model residency in RAM.

## Phase 2 — CPU correctness runtime

Goal: implement the smallest mathematically correct forward path.

Workstreams:

- tensor primitives
- dequantization/reference kernels
- tokenizer
- embeddings
- normalization
- model blocks
- MoE router
- expert execution
- attention/GDN/QSA mechanisms
- logits
- sampling

This path is for correctness, debugging and reference comparison, not speed.

Exit gate:

- selected logits match reference vectors within documented tolerances;
- greedy generation matches reference token sequences for controlled tests.

## Phase 3 — CUDA backend

Goal: move the dominant compute path to the NVIDIA GPU.

Workstreams:

- device abstraction
- VRAM allocator
- upload/download primitives
- CUDA kernels
- stream/event synchronization
- fused operations where justified
- profiling hooks

Exit gate:

- CUDA output passes the CPU/reference correctness suite;
- performance baseline recorded on the reference PC.

## Phase 4 — Constrained-memory execution

Goal: make 10 GB VRAM / 32 GB RAM a first-class target.

Workstreams:

- tensor placement classification
- pinned host memory
- expert residency policy
- asynchronous expert prefetch
- eviction
- memory-mapped weights
- NVMe streaming
- overlap of I/O, transfer and compute
- adaptive scheduling

Exit gate:

- target quantization runs without memory exhaustion;
- placement and transfer behavior is observable;
- cold/warm benchmark matrix exists.

## Phase 5 — Quantization optimization

Goal: reduce memory bandwidth and capacity requirements while preserving useful model quality.

Workstreams:

- supported weight quantizations
- model-specific mixed quantization
- routed expert quantization strategies
- activation precision policy
- quantized CUDA kernels
- calibration/imatrix research if useful

Exit gate:

- at least one reference-machine profile is judged practically usable;
- quality and speed tradeoffs are documented.

## Phase 6 — Long-context state

Goal: make context growth predictable under constrained memory.

Workstreams:

- model-specific recurrent/attention state analysis
- cache/state memory accounting
- host offload
- disk persistence where beneficial
- session restore
- corruption/version validation

Exit gate:

- context scaling benchmark exists;
- session state can be saved/restored without ambiguity where supported.

## Phase 7 — Product surface

Goal: make the runtime pleasant to use.

Workstreams:

- production CLI
- streaming output
- local server
- OpenAI-compatible subset if appropriate
- configuration profiles
- diagnostics
- telemetry that remains local
- packaging for Windows

Exit gate:

- clean-machine installation is documented;
- local applications can consume the runtime reliably.

## Phase 8 — Native coding agent

Goal: exploit the vertical stack instead of merely exposing token generation.

Workstreams:

- model-native tool-call rendering/parsing
- file tools
- repository context
- persistent sessions
- prefix/state reuse
- bounded command execution
- agent benchmark suite

Exit gate:

- repeatable coding tasks can be completed locally and measured.

## Phase 9 — Community hardening

Goal: prepare for broader contribution.

Workstreams:

- CI matrix
- issue templates
- reproducible benchmark submissions
- contributor hardware profiles
- security policy
- release process
- compatibility policy
- public roadmap governance

## Decision gates

At the end of every major phase:

1. Is correctness demonstrated?
2. Did the architecture remain narrow?
3. Is the current bottleneck known?
4. Does the next phase still make sense on the reference hardware?
5. Are we learning something that generic runtimes do not already solve sufficiently?
