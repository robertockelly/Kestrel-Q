# TASKS-TASK-3.0-NATIVE-SAMPLING-POLICY.md

Status: **COMPLETE / PASS**

## Baseline

- [x] Establish reviewed Epic 3 inception checkpoint and clean worktree.
- [x] Confirm Epic 2 COMPLETE/PASS and Task 3.0 NOT STARTED.
- [x] Confirm Task 3.1 NOT STARTED.
- [x] Confirm both backlog benchmarks remain deferred.

## Characterization

- [x] Create `SAMPLING-CONTRACT.md` before production algorithm code.
- [x] Verify official generation-config hash and exact values.
- [x] Pin executed Transformers files/functions and hashes.
- [x] Pin processor construction/execution order.
- [x] Pin score/softmax/categorical arithmetic and FP environment.
- [x] Pin top-k/top-p ties, cutoffs and minimum-token rules.
- [x] Pin canonical/padded/special/EOG token policy.
- [x] Record every unsupported processor/option.

## RNG decision

- [x] Select portable deterministic algorithm with compatible provenance.
- [x] Pin state width/version and integer wrap semantics.
- [x] Pin seed expansion, uniform mapping and draw consumption.
- [x] Pin reset/snapshot/import and failure behavior.
- [x] Generate independent exact RNG vectors before native validation.

## Oracle evidence

- [x] Generate Class-C sampling contract.
- [x] Generate calibration corpus.
- [x] Generate disjoint holdout corpus.
- [x] Generate RNG vectors.
- [x] Predeclare statistical method/trials/thresholds.
- [x] Generate statistical evidence.
- [x] Generate comparison-only native validation.
- [x] Generate deterministic manifest and SHA-256 values.
- [x] Confirm no self-oracle.

## Native C17 sampler

- [x] Immutable validated config.
- [x] Explicit caller-owned RNG state.
- [x] Explicit logits pointer/count and output capacities.
- [x] No input-logit mutation.
- [x] Supported transforms/order exact.
- [x] Stable categorical selection.
- [x] State commits on success only.
- [x] Bounded diagnostics and scratch.
- [x] No global mutable state.
- [x] No model/GGUF owner dependency.

## Correctness

- [x] Official profile exact.
- [x] Candidate membership/order exact.
- [x] RNG words/state exact.
- [x] Fixed-input selected IDs exact.
- [x] Floating calibration contracts justified.
- [x] Disjoint holdout PASS without widening.
- [x] Statistical corpus PASS under predeclared rule.
- [x] Greedy argmax and EOG regressions unchanged.

## Fail closed

- [x] Null/invalid arguments.
- [x] Invalid vocabulary/count/domain.
- [x] Unsupported processor/policy.
- [x] Invalid temperature/top-k/top-p.
- [x] Non-finite scores.
- [x] Empty/invalid probability mass.
- [x] Invalid selected/padded ID policy.
- [x] Invalid/corrupted/version-mismatched RNG state.
- [x] Overflow and insufficient capacity.
- [x] Forbidden aliasing/input mutation.
- [x] RNG state unchanged on failure.
- [x] Incompatible floating-point environment.

## Documentation / provenance

- [x] `NATIVE-SAMPLING.md`.
- [x] `KQ-SAMPLING-API.md`.
- [x] ADR 0022 finalized from evidence.
- [x] Dependency/provenance/license records complete.
- [x] Epic 3 plan/checklist updated.
- [x] TASKS/ROADMAP updated.
- [x] CHANGELOG updated in the same iteration.

## Boundaries / validation

- [x] No executor/CLI sampling integration.
- [x] No model payload or research-model artifact.
- [x] No production Python/Transformers/llama.cpp dependency.
- [x] No batching, long-context/session, MTP or vision work.
- [x] No SIMD/CUDA model kernel.
- [x] No scheduler/cache/prefetch/memory-tiering work.
- [x] BENCH-001 and BENCH-002 remain deferred.
- [x] Clean CPU Release suite PASS.
- [x] Clean CUDA Release suite PASS.
- [x] No new Kestrel-Q `/W4` warning.
- [x] Repository safety PASS.
- [x] `git diff --check` PASS.
- [x] Task 3.1 remains NOT STARTED.
- [x] No automatic commit/push.
