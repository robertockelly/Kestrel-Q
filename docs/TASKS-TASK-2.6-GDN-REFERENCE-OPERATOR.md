# TASKS-TASK-2.6-GDN-REFERENCE-OPERATOR.md

## Baseline
- [x] Record actual HEAD/worktree.
- [x] Confirm Task 2.5 COMPLETE/PASS.
- [x] Confirm Task 2.6 NOT STARTED before edits.
- [x] Keep KQ-BACKLOG-BENCH-002 DEFERRED.

## Characterization
- [x] Create `docs/GDN-OPERATOR-CONTRACT.md`.
- [x] Pin canonical module/function names.
- [x] Pin 36 GDN layer IDs.
- [x] Pin input/output shapes.
- [x] Pin norm/epsilon.
- [x] Pin every weight/parameter role.
- [x] Pin conv geometry/state.
- [x] Pin recurrent state geometry/dtype.
- [x] Pin activations/gates/decays.
- [x] Pin exact operation order.
- [x] Pin prefill semantics.
- [x] Pin decode semantics.
- [x] Pin state initialization/mutation.
- [x] Separate GDN-owned GR semantics from future work.

## Independent oracle
- [x] Use pinned Class-C canonical source.
- [x] No full BF16 checkpoint download.
- [x] Build deterministic reduced-shape canonical cases.
- [x] Record deterministic seeds/config.
- [x] Generate calibration corpus.
- [x] Generate disjoint holdout corpus.
- [x] Generate state-transition vectors.
- [x] Generate manifest/SHA.
- [x] Kestrel-Q is not oracle.

## Native config/state
- [x] Immutable GDN layer config.
- [x] Reject QSA layer IDs.
- [x] Resolve all required semantic tensors.
- [x] Validate shape/type/layout.
- [x] Explicit bounded stream state.
- [x] Reset/init.
- [x] State-size query.
- [x] No raw-struct-padding equality tests.

## Native prefill
- [x] Scalar implementation.
- [x] Reuse Task 2.5 numerics.
- [x] Bounded scratch.
- [x] Exact canonical order.
- [x] Error-state policy documented.

## Native decode
- [x] One-token decode.
- [x] Exact state update.
- [x] Multi-step decode.
- [x] Prefill continuation == decode continuation.

## Checkpoints
- [x] Normalized input if canonical (none inside GDN; masked input captured).
- [x] Projection checkpoints.
- [x] Convolution checkpoints.
- [x] Gate/decay checkpoints.
- [x] Recurrent update checkpoints.
- [x] Recurrent state.
- [x] Convolution state.
- [x] Final output.

## Floating contracts
- [x] Per-checkpoint calibration.
- [x] ULP/abs/rel analysis.
- [x] Disjoint holdout validation.
- [x] Exact discrete fields exact.
- [x] No generic epsilon.

## Target integration
- [x] All 36 target GDN layers structurally validate.
- [x] QSA layers reject.
- [x] Required real semantic bindings resolve.
- [x] Target state shapes validate.
- [x] Physical quant types/layouts validate.
- [x] Optional real payload <= 8 MiB (not used; zero bytes).
- [x] No hard-coded GGUF offsets.
- [x] No raw model payload committed.

## Fail closed
- [x] Missing semantic tensor.
- [x] Wrong projection shape.
- [x] Wrong recurrent state.
- [x] Wrong conv state.
- [x] Wrong dtype.
- [x] Invalid sequence/count.
- [x] Non-finite input where forbidden.
- [x] Split/transformed misuse.
- [x] Forbidden aliasing.
- [x] View/range failure (invalid binding/type rejected before any view).
- [x] Partial-failure state policy.

## Metrics
- [x] Config bytes.
- [x] Batch-1 state bytes.
- [x] Scratch/token.
- [x] Prefill scratch.
- [x] Temporary dequant scratch.
- [x] Synthetic prefill latency.
- [x] Synthetic decode latency.
- [x] No optimization.

## Regression
- [x] Clean CPU Release PASS (20/20 CTest).
- [x] Clean CUDA Release PASS (22/22 CTest).
- [x] Task 2.0-2.5 PASS in focused regression.
- [x] No new Kestrel-Q warnings (the existing external NVCC-generated C4211 remains documented).
- [x] `git diff --check` PASS.

## Docs/governance
- [x] `docs/GDN-OPERATOR-CONTRACT.md`.
- [x] `docs/NATIVE-GDN-REFERENCE.md`.
- [x] `docs/KQ-GDN-API.md`.
- [x] ADR 0014 finalized.
- [x] ARCHITECTURE updated.
- [x] MODEL-RUNTIME-STATE updated.
- [x] GOLDEN-VECTORS updated.
- [x] TASKS/ROADMAP/Epic 2 updated.
- [x] CHANGELOG updated.
- [x] Next operator task NOT STARTED.

## Safety
- [x] no QSA.
- [x] no MoE execution.
- [x] no PLE value lookup.
- [x] no full transformer layer.
- [x] no full model forward.
- [x] no sampling.
- [x] no SIMD/CUDA model kernels.
- [x] no production Class-C/Python dependency.
- [x] no tracked model weights/secrets/local paths.
