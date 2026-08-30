# TASKS-TASK-2.9-PLE-VALUE-REFERENCE-OPERATOR.md

Status: **COMPLETE / PASS**

Entry HEAD: `e5b4d8c6fc2f9dd6cd3c4ba3483162013c43014a` (Task 2.8
checkpoint on `origin/main`). The four pre-authored Task 2.9 governance files
were preserved and completed in place.

## Baseline
- [x] Record actual HEAD/worktree after Task 2.8 checkpoint: `e5b4d8c6fc2f9dd6cd3c4ba3483162013c43014a`.
- [x] Confirm Task 2.8 COMPLETE/PASS.
- [x] Confirm Task 2.9 started only after the Task 2.8 checkpoint was pushed.
- [x] Keep KQ-BACKLOG-BENCH-002 DEFERRED.

## Characterization
- [x] Create PLE-VALUE-OPERATOR-CONTRACT.md.
- [x] Pin canonical value module/functions.
- [x] Pin layer placement.
- [x] Pin lookup/member ordering.
- [x] Pin row/value width.
- [x] Pin six dense tensor roles/shapes.
- [x] Pin value-state/dilation semantics.
- [x] Pin activation/gating/combine.
- [x] Pin prefill/decode operation order.
- [x] Separate address state from value state.

## Oracle
- [x] Reduced synthetic value vectors.
- [x] Task 2.4 address-integration vectors.
- [x] Calibration corpus.
- [x] Disjoint holdout.
- [x] State vectors.
- [x] Manifest/SHA.
- [x] No self-oracle.

## Native config/state
- [x] Immutable PLE value config.
- [x] Validate 128 members.
- [x] Validate fused IQ4_NL binding.
- [x] Validate six dense semantics.
- [x] Validate address config compatibility.
- [x] Explicit value state.
- [x] Reset/snapshot.
- [x] Transactional failure policy.

## Lookup provider
- [x] Logical member+row interface.
- [x] Synthetic provider.
- [x] Real bounded GGUF provider.
- [x] No hard-coded physical offsets.
- [x] No persistent cache.

## Prefill/decode
- [x] Exact Task 2.4 intent consumption.
- [x] Scalar prefill.
- [x] Scalar one-token decode.
- [x] Exact history/dilation updates.
- [x] Prefill/decode continuation.
- [x] Reset/replay.

## Floating contracts
- [x] Lookup checkpoint calibration.
- [x] Dense checkpoint calibration.
- [x] History/convolution calibration.
- [x] Final output calibration.
- [x] Disjoint holdout PASS.
- [x] No generic epsilon.

## Target reconciliation
- [x] Row width derived.
- [x] Rows/member reconciled.
- [x] Member packed span reconciled.
- [x] 128-member aggregate reconciled.
- [x] Aggregate PLE footprint reconciled.
- [x] Real sample <= 8 MiB: 1,440 logical packed bytes.
- [x] No raw model bytes committed.

## Fail closed
- [x] Wrong layer/member count.
- [x] Missing fused/dense binding.
- [x] Wrong dense shape.
- [x] Bad IQ4_NL/member geometry.
- [x] Invalid member/row.
- [x] Wrong address count/order.
- [x] Bad value state.
- [x] Provider failure.
- [x] Capacity/overflow.
- [x] Aliasing.
- [x] No partial state mutation on failure.

## Metrics
- [x] Config bytes.
- [x] Value-state bytes.
- [x] Lookup row buffer.
- [x] Dequant scratch.
- [x] Dense/history workspace.
- [x] Prefill/decode latency.
- [x] No storage benchmark.

## Regression
- [x] CPU clean PASS.
- [x] CUDA clean PASS.
- [x] Task 2.0-2.8 PASS.
- [x] No new warnings.
- [x] git diff --check PASS.

## Docs/governance
- [x] PLE-VALUE-OPERATOR-CONTRACT.md.
- [x] NATIVE-PLE-VALUE-REFERENCE.md.
- [x] KQ-PLE-VALUE-API.md.
- [x] ADR 0017.
- [x] Architecture/runtime/footprint/goldens updated.
- [x] TASKS/ROADMAP/Epic 2 updated.
- [x] CHANGELOG updated.
- [x] Next integration task NOT STARTED.

## Safety
- [x] no final PLE cache/prefetch.
- [x] no full layer/full forward.
- [x] no SIMD/CUDA model kernel.
- [x] no production Class-C/Python dependency.
- [x] no tracked model weights/secrets/local paths.
