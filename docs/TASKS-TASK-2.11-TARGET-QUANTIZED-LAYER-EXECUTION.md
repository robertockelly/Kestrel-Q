# TASKS-TASK-2.11-TARGET-QUANTIZED-LAYER-EXECUTION.md

## Baseline
- [x] Record HEAD/worktree after Task 2.10 checkpoint.
- [x] Confirm Task 2.10 COMPLETE/PASS.
- [x] Confirm Task 2.11 NOT STARTED.
- [x] Keep KQ-BACKLOG-BENCH-002 DEFERRED.

## Provider contract
- [x] TARGET-WEIGHT-PROVIDER-CONTRACT.md.
- [x] Semantic linear interface.
- [x] Expert-member interface.
- [x] PLE row provider.
- [x] Split bindings.
- [x] Transformed layouts.
- [x] Capacity/lifetime/error policy.
- [x] Payload accounting.

## Production provider
- [x] Task 2.2 bounded views.
- [x] Task 2.5 dequant/row-dot only.
- [x] No duplicate dequant.
- [x] No whole F32 matrix materialization.
- [x] Bounded row/chunk scratch.
- [x] Transactional output behavior.

## Operator adapters
- [x] GDN provider-backed path.
- [x] QSA provider-backed path.
- [x] GR/layer-owned weights.
- [x] MoE router/shared path.
- [x] Selected expert provider path.
- [x] PLE dense/provider path.
- [x] Synthetic F32 paths remain PASS.

## Independent target oracle
- [x] llama Class-Q decode helper.
- [x] Separate Python/NumPy canonical layer equations.
- [x] Deterministic hidden inputs/states.
- [x] Calibration cases.
- [x] Disjoint holdout.
- [x] Exact discrete selections.
- [x] Manifest/SHA.
- [x] No self-oracle.

## Representative execution
- [x] Select ordinary GDN layer.
- [x] Select QSA layer.
- [x] PLE-GDN layer 1.
- [x] One-token/reset cases.
- [x] Short prefill where feasible.
- [x] Prefill + decode.
- [x] Final output/state oracle match.

## Access invariants
- [x] Only selected routed experts requested.
- [x] Shared expert separate.
- [x] Only exact PLE rows requested.
- [x] No speculative provider access.
- [x] No hard-coded offsets.

## Payload budget
- [x] Logical bytes touched recorded.
- [x] Blocks touched recorded.
- [x] Unique semantic tensors touched recorded.
- [x] <= 768 MiB.
- [x] No raw model bytes committed.
- [x] No throughput/residency claims.

## Full-model preflight
- [x] 48/48 provider-compatible layers.
- [x] All semantic roles supported.
- [x] All transformed layouts supported or fail explained.
- [x] Expert selected-member path valid all layers.
- [x] PLE provider valid layer 1.

## Fail closed
- [x] Bad semantic role.
- [x] Unsupported type/layout.
- [x] Row geometry mismatch.
- [x] Dimension mismatch.
- [x] Invalid expert ID.
- [x] Arithmetic overflow.
- [x] Aliasing/capacity.
- [x] View/provider failure.
- [x] Layer-state rollback on provider failure.

## Metrics
- [x] Provider bytes.
- [x] Row/dequant scratch.
- [x] Max simultaneous F32 weight bytes.
- [x] Representative row-dot timings.
- [x] Representative layer timings.
- [x] No optimization claim.

## Readiness
- [x] FIRST-TOKEN-READINESS.md.
- [x] Remaining embedding requirement.
- [x] Remaining 48-layer loop/state orchestration.
- [x] Remaining final norm/LM head/logits/argmax/decode.
- [x] No unexplained blockers.

## Regression/governance
- [x] CPU clean PASS.
- [x] CUDA clean PASS.
- [x] Task 2.0-2.10 PASS.
- [x] No new warnings.
- [x] ADR 0019 ACCEPTED.
- [x] TASKS/ROADMAP/Epic 2/CHANGELOG updated.
- [x] Next first-token task NOT STARTED.
- [x] git diff --check PASS.

## Safety
- [x] no full 48-layer execution.
- [x] no embedding/final norm/LM head/logits.
- [x] no scheduler/cache optimization.
- [x] no SIMD/CUDA model kernels.
- [x] no production Python/llama dependency.
- [x] no tracked model weights/secrets/local paths.
