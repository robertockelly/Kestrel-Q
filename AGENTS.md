# AGENTS.md

This file defines the operating contract for AI coding agents working in Kestrel-Q.

It is normative. If an agent instruction conflicts with an informal comment, this file wins unless a human maintainer explicitly overrides it.

## 1. Mission

Build a small, understandable, measurable and model-specific native inference runtime for Qwen3.8-Flash-Next on Windows consumer hardware.

Optimize for the reference constraint:

- approximately 10 GB NVIDIA VRAM
- 32 GB RAM
- NVMe SSD
- Windows 11

Do not broaden the project into a generic inference framework without an explicit architectural decision record (ADR).

## 2. Core engineering principles

1. Correctness before speed.
2. Measure before optimizing.
3. One model first.
4. Keep the C core understandable.
5. GPU-specific code stays behind a narrow backend interface.
6. Fail closed on unsupported model metadata, tensor layouts or quantization types.
7. Never silently reinterpret model data.
8. Preserve reproducibility.
9. Avoid hidden global state.
10. Prefer explicit ownership and lifetime rules.
11. Avoid dependencies unless their value clearly exceeds their maintenance cost.
12. Do not copy implementation code from projects whose licensing is incompatible with this repository.

## 3. Source-language policy

Primary runtime:

- C17 for platform-neutral/runtime code.
- CUDA `.cu` files are allowed where required by the NVIDIA toolchain.
- CUDA implementation details must be exposed to the C runtime through a C ABI.
- C++ must not leak into the public core API unless an ADR explicitly authorizes it.

Python may be used only for offline developer tooling, reference generation, conversion or analysis unless explicitly approved otherwise.

## 4. Required workflow for every implementation task

Before editing:

1. Read `README.md`.
2. Read `docs/VISION.md`.
3. Read `docs/ARCHITECTURE.md`.
4. Read `docs/PROJECT-PLAN.md`.
5. Read `docs/TASKS.md`.
6. Read relevant ADRs.
7. Inspect current code and tests before proposing changes.

During implementation:

1. Establish the current baseline.
2. Add or identify a correctness test.
3. Implement the smallest coherent change.
4. Build with warnings enabled.
5. Run relevant tests.
6. Run benchmarks only when performance may have changed.
7. Compare results to the previous baseline.
8. Update documentation when behavior or architecture changed.
9. Update `CHANGELOG.md` for material changes.

Before declaring completion:

- working tree changes must be understood;
- tests must pass;
- no known correctness regression may be hidden;
- performance claims must include benchmark evidence;
- unsupported cases must fail explicitly;
- no secrets, model weights or large generated artifacts may be committed.

## 5. Benchmark discipline

Never report a speed improvement from a single unqualified number.

Record at minimum:

- commit
- model artifact identifier
- quantization
- prompt length
- generated token count
- context length
- GPU
- VRAM
- RAM
- CPU
- storage
- CUDA version
- compiler
- build type
- prefill tokens/s
- decode tokens/s
- peak VRAM
- peak committed RAM
- disk read volume when streaming
- run count
- warm/cold state

For optimization work, compare against the immediately preceding accepted baseline.

## 6. Correctness gates

The project must eventually maintain reference vectors obtained from an authoritative implementation.

For each supported execution path, validate where technically possible:

- tokenizer IDs
- selected tensor values after load/dequantization
- layer checkpoints
- routing decisions
- selected logits
- final logits
- greedy token sequence

Tolerance must be documented per datatype/kernel. Never widen tolerance merely to make a failing test pass without understanding the cause.

## 7. Memory-management rules

This project is specifically about constrained-memory inference. Treat memory placement as architecture, not implementation detail.

Every major tensor class must eventually have an explicit placement policy:

- persistent VRAM
- transient VRAM
- pinned host RAM
- pageable RAM
- memory-mapped storage
- streamed storage

Any new cache must define:

- ownership
- maximum size
- eviction rule
- synchronization rule
- persistence behavior
- corruption handling

## 8. Performance rules

Do not optimize based on intuition alone.

Preferred progression:

1. correctness
2. instrumentation
3. profile
4. identify bottleneck
5. hypothesis
6. isolated experiment
7. benchmark
8. correctness regression test
9. keep or revert

Avoid micro-optimizations while architecture-level transfers dominate runtime.

## 9. Dependencies and licensing

The repository is licensed under **Apache License 2.0**.

Before adding a dependency or incorporating third-party source, document:

- purpose
- license and compatibility with Apache-2.0 distribution
- attribution/NOTICE obligations
- binary/runtime cost
- alternative considered
- removal difficulty

Do not copy implementation code from GPL, AGPL, proprietary, source-available or otherwise incompatible sources into this repository.

Studying public algorithms, papers, model specifications and behavior is allowed, but implementations must have clean provenance unless reuse is explicitly license-compatible.

If licensing compatibility is uncertain, stop and require maintainer review before incorporation.

A dependency that duplicates a small amount of straightforward C code should receive additional scrutiny.

## 10. Commit hygiene

Keep commits conceptually coherent.

Recommended prefixes:

- `feat:`
- `fix:`
- `perf:`
- `test:`
- `bench:`
- `docs:`
- `build:`
- `refactor:`

Do not mix formatting-only churn with functional optimization.

## 11. Documentation obligations

Update documentation in the same change when modifying:

- model assumptions
- file format expectations
- memory placement
- scheduling
- backend interfaces
- supported quantization
- CLI/API behavior
- benchmark methodology

Architectural decisions belong under `docs/adr/`.

### Canonical change record

`CHANGELOG.md` is the canonical chronological record of project changes.

Every mandate or iteration that modifies source code, build/runtime
configuration, benchmarks, architecture or model assumptions, governed
documentation, supported behavior or interfaces, model/runtime capabilities,
bug fixes or their root causes, or project governance **must** update
`CHANGELOG.md` in that same iteration before completion and before commit.

Project records have distinct responsibilities:

- `CHANGELOG.md` records chronological changes;
- `docs/TASKS.md` records current Epic/Task status;
- task-specific `TASKS-*.md` files record execution checklists and evidence;
- `docs/adr/` records durable architectural decisions;
- Git history records exact implementation history.

Do not create a parallel worklog unless an ADR explicitly introduces one.

## 12. Safety and destructive actions

An agent must not:

- delete model files unless explicitly requested;
- rewrite benchmark history;
- force-push;
- change licensing;
- publish releases;
- upload private artifacts;
- change public APIs casually;
- disable failing tests to obtain green status.

## 13. When blocked

Do not invent model behavior.

When a specification detail is unknown:

1. identify the exact uncertainty;
2. locate the authoritative model/config/code reference;
3. document the finding;
4. add a test vector if possible;
5. only then implement.

## 14. Definition of done

A task is done only when its acceptance criteria are met and evidence is recorded.

"Compiles on my machine" is not a sufficient completion criterion.
