# TASKS-EPIC-3-CPU-CORRECTNESS-ENGINE.md

Status: **COMPLETE / PASS — TASKS 3.0-3.1 COMPLETE / PASS**

## Inception baseline

- [x] Published baseline is
  `b4a2565bd49684fd6b9bd81724d667a7bf7cc11c`.
- [x] Epic 0, Epic 1 and Epic 2 are COMPLETE/PASS.
- [x] Tasks 2.0-2.13 and M1 are COMPLETE/PASS.
- [x] Existing greedy multi-token sequence and rollback evidence preserved.
- [x] Phase 2/R2 scope reconstructed from committed governance.
- [x] Sampling identified as the only explicitly unfinished Epic 3/R2
  workstream.
- [x] No runtime, test or research evidence changed by inception.

## Task sequence

### Task 3.0 — Native Sampling Policy & Deterministic Selection Primitives

Status: **COMPLETE / PASS**

- [x] Characterize pinned official generation configuration and Transformers
  sampling path before code.
- [x] Pin exact supported processor order, token eligibility and special/padded
  ID semantics.
- [x] Select and document portable deterministic RNG algorithm/state semantics.
- [x] Implement separate immutable sampler config and explicit mutable RNG
  state in C17.
- [x] Validate exact masks/order, RNG vectors and fixed-input selections against
  an independent oracle.
- [x] Calibrate floating transforms and pass a disjoint holdout.
- [x] Define and pass a predeclared statistical acceptance corpus where exact
  token equality is invalid.
- [x] Preserve greedy argmax and EOG behavior.
- [x] Complete fail-closed suite.
- [x] Finalize ADR 0022 from evidence.
- [x] Clean CPU/CUDA regressions and repository-safety gates PASS.

### Task 3.1 — Sampled Incremental Generation & Oracle Validation

Status: **COMPLETE / PASS**

- [x] Create its task-specific governance package only after Task 3.0 review
  and checkpoint.
- [x] Integrate Task 3.0 selection after full-logit production.
- [x] Keep prompt prefill exactly once and generated-ID decode direct.
- [x] Include sampler RNG in the per-token transaction boundary.
- [x] Prove failed post-selection rollback and deterministic retry.
- [x] Preserve Task 2.12/2.13 greedy exact sequences.
- [x] Validate the official profile over a bounded prompt/seed matrix without a
  self-oracle.
- [x] Validate EOG, max-token and context stop behavior.
- [x] Record bounded state/payload/memory characterization without performance
  claims.
- [x] Clean CPU/CUDA regressions and repository-safety gates PASS.

## Epic 3 closure gates

- [x] Task 3.0 COMPLETE/PASS.
- [x] Task 3.1 COMPLETE/PASS.
- [x] Official Qwen3.8 default sampling profile supported.
- [x] Exact processor/RNG/fixed-input decision evidence PASS.
- [x] Calibrated floating and predeclared statistical holdout PASS.
- [x] Sampled incremental state/stop/rollback evidence PASS.
- [x] Greedy M1 and Task 2.13 evidence unchanged and PASS.
- [x] No production oracle dependency or hidden global RNG state.
- [x] No performance/scheduler/CUDA/long-context scope drift.
- [x] Epic 3 completion recorded by the final governed Task 3.1 gate.

## Deferred and future boundaries

- [x] `KQ-BACKLOG-BENCH-001` remains DEFERRED until production WDDM VRAM
  capacity policy.
- [x] `KQ-BACKLOG-BENCH-002` remains DEFERRED / REQUIRED BEFORE FINAL
  SCHEDULER DESIGN.
- [x] No CUDA optimized model kernel, SIMD, cache, prefetch, scheduler, memory
  tiering, long-context persistence, batching, MTP or vision work is included.
