# TASKS-TASK-2.7-QSA-REFERENCE-OPERATOR.md

Status: **COMPLETE / PASS**

## Baseline
- [x] Record actual HEAD/worktree after Task 2.6 checkpoint.
- [x] Confirm Task 2.6 COMPLETE/PASS.
- [x] Confirm Task 2.7 NOT STARTED before edits.
- [x] Keep KQ-BACKLOG-BENCH-002 DEFERRED.

## Characterization
- [x] Create `docs/QSA-OPERATOR-CONTRACT.md`.
- [x] Pin canonical module/functions.
- [x] Pin 12 QSA layer IDs.
- [x] Pin Q/K/V/output shapes.
- [x] Pin head counts/dimensions.
- [x] Pin norm/RoPE/scale/mask.
- [x] Pin K/V cache semantics.
- [x] Pin indexer projections/state.
- [x] Pin block construction.
- [x] Pin incomplete-tail behavior.
- [x] Pin score/selection/tie order.
- [x] Pin prefill/decode transitions.

## Independent oracle
- [x] Use pinned Class-C source.
- [x] No full BF16 checkpoint download.
- [x] Reduced canonical cases.
- [x] Threshold-crossing sparse cases.
- [x] Calibration corpus.
- [x] Disjoint holdout.
- [x] Exact selection vectors.
- [x] State vectors.
- [x] Manifest/SHA.
- [x] No self-oracle.

## Native config/state
- [x] Immutable QSA config.
- [x] Reject GDN layer IDs.
- [x] Resolve all semantic bindings.
- [x] Validate split index_qk.
- [x] Validate state geometry.
- [x] Explicit bounded state capacity.
- [x] Reset/snapshot.
- [x] Transactional failure policy.

## Prefill/decode
- [x] Scalar prefill.
- [x] Scalar one-token decode.
- [x] Exact causal behavior.
- [x] Exact block formation.
- [x] Exact sparse selection.
- [x] Exact gather order.
- [x] Exact state append.
- [x] Prefill/decode continuation checks.

## Exact selection
- [x] Candidate IDs exact.
- [x] Selected block IDs exact.
- [x] Selected token IDs/positions exact.
- [x] Tie order exact.
- [x] Counts exact.
- [x] Causal exclusions exact.

## Threshold tests
- [x] No complete block.
- [x] One complete block.
- [x] Below limit.
- [x] Equal to limit.
- [x] Above limit.
- [x] Incomplete tail.
- [x] Block-boundary decode.
- [x] Partial-block decode.
- [x] Ties/near-ties.
- [x] Repeated inputs.

## Target integration
- [x] 12/12 QSA layers valid.
- [x] 36/36 GDN layer IDs reject.
- [x] Real semantic bindings resolve.
- [x] Target shapes/types/layouts valid.
- [x] State shapes valid.
- [x] State-growth reconciliation with Epic 1.
- [x] Optional real payload <= 8 MiB.
- [x] No hard-coded offsets/raw weights.

## Fail closed
- [x] Missing semantic tensor/split part.
- [x] Wrong projection/head geometry.
- [x] Wrong cache/index state.
- [x] Capacity/context overflow.
- [x] Invalid position/block geometry.
- [x] Selection range errors.
- [x] Non-finite data where forbidden.
- [x] Split/transformed misuse.
- [x] Aliasing violations.
- [x] No partial visible mutation on failure.

## Metrics
- [x] Config bytes.
- [x] Fixed state bytes.
- [x] Bytes/token state growth.
- [x] Scratch/token.
- [x] Selection workspace.
- [x] Reduced prefill/decode latency.
- [x] No optimization.

## Regression
- [x] CPU clean PASS.
- [x] CUDA clean PASS.
- [x] Task 2.0-2.6 PASS.
- [x] No new warnings.
- [x] `git diff --check` PASS.

## Docs/governance
- [x] QSA-OPERATOR-CONTRACT.md.
- [x] NATIVE-QSA-REFERENCE.md.
- [x] KQ-QSA-API.md.
- [x] ADR 0015.
- [x] ARCHITECTURE/runtime-state/goldens updated.
- [x] TASKS/ROADMAP/Epic 2 updated.
- [x] CHANGELOG updated.
- [x] Next operator task NOT STARTED.

## Safety
- [x] no MoE.
- [x] no PLE value lookup.
- [x] no full layer/full forward.
- [x] no final KV scheduler/cache policy.
- [x] no SIMD/CUDA model kernel.
- [x] no production Class-C/Python dependency.
- [x] no tracked model weights/secrets/local paths.
