# TASKS-TASK-3.0-NATIVE-SAMPLING-POLICY.md

Status: **NOT STARTED**

## Baseline

- [ ] Establish reviewed Epic 3 inception checkpoint and clean worktree.
- [ ] Confirm Epic 2 COMPLETE/PASS and Task 3.0 NOT STARTED.
- [ ] Confirm Task 3.1 NOT STARTED.
- [ ] Confirm both backlog benchmarks remain deferred.

## Characterization

- [ ] Create `SAMPLING-CONTRACT.md` before production algorithm code.
- [ ] Verify official generation-config hash and exact values.
- [ ] Pin executed Transformers files/functions and hashes.
- [ ] Pin processor construction/execution order.
- [ ] Pin score/softmax/categorical arithmetic and FP environment.
- [ ] Pin top-k/top-p ties, cutoffs and minimum-token rules.
- [ ] Pin canonical/padded/special/EOG token policy.
- [ ] Record every unsupported processor/option.

## RNG decision

- [ ] Select portable deterministic algorithm with compatible provenance.
- [ ] Pin state width/version and integer wrap semantics.
- [ ] Pin seed expansion, uniform mapping and draw consumption.
- [ ] Pin reset/snapshot/import and failure behavior.
- [ ] Generate independent exact RNG vectors before native validation.

## Oracle evidence

- [ ] Generate Class-C sampling contract.
- [ ] Generate calibration corpus.
- [ ] Generate disjoint holdout corpus.
- [ ] Generate RNG vectors.
- [ ] Predeclare statistical method/trials/thresholds.
- [ ] Generate statistical evidence.
- [ ] Generate comparison-only native validation.
- [ ] Generate deterministic manifest and SHA-256 values.
- [ ] Confirm no self-oracle.

## Native C17 sampler

- [ ] Immutable validated config.
- [ ] Explicit caller-owned RNG state.
- [ ] Explicit logits pointer/count and output capacities.
- [ ] No input-logit mutation.
- [ ] Supported transforms/order exact.
- [ ] Stable categorical selection.
- [ ] State commits on success only.
- [ ] Bounded diagnostics and scratch.
- [ ] No global mutable state.
- [ ] No model/GGUF owner dependency.

## Correctness

- [ ] Official profile exact.
- [ ] Candidate membership/order exact.
- [ ] RNG words/state exact.
- [ ] Fixed-input selected IDs exact.
- [ ] Floating calibration contracts justified.
- [ ] Disjoint holdout PASS without widening.
- [ ] Statistical corpus PASS under predeclared rule.
- [ ] Greedy argmax and EOG regressions unchanged.

## Fail closed

- [ ] Null/invalid arguments.
- [ ] Invalid vocabulary/count/domain.
- [ ] Unsupported processor/policy.
- [ ] Invalid temperature/top-k/top-p.
- [ ] Non-finite scores.
- [ ] Empty/invalid probability mass.
- [ ] Invalid selected/padded ID policy.
- [ ] Invalid/corrupted/version-mismatched RNG state.
- [ ] Overflow and insufficient capacity.
- [ ] Forbidden aliasing/input mutation.
- [ ] RNG state unchanged on failure.
- [ ] Incompatible floating-point environment.

## Documentation / provenance

- [ ] `NATIVE-SAMPLING.md`.
- [ ] `KQ-SAMPLING-API.md`.
- [ ] ADR 0022 finalized from evidence.
- [ ] Dependency/provenance/license records complete.
- [ ] Epic 3 plan/checklist updated.
- [ ] TASKS/ROADMAP updated.
- [ ] CHANGELOG updated in the same iteration.

## Boundaries / validation

- [ ] No executor/CLI sampling integration.
- [ ] No model payload or research-model artifact.
- [ ] No production Python/Transformers/llama.cpp dependency.
- [ ] No batching, long-context/session, MTP or vision work.
- [ ] No SIMD/CUDA model kernel.
- [ ] No scheduler/cache/prefetch/memory-tiering work.
- [ ] BENCH-001 and BENCH-002 remain deferred.
- [ ] Clean CPU Release suite PASS.
- [ ] Clean CUDA Release suite PASS.
- [ ] No new Kestrel-Q `/W4` warning.
- [ ] Repository safety PASS.
- [ ] `git diff --check` PASS.
- [ ] Task 3.1 remains NOT STARTED.
- [ ] No automatic commit/push.
