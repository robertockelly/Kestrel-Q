# TASKS-TASK-2.10-CANONICAL-LAYER-COMPOSITION.md

Status: **COMPLETE / PASS**

## Baseline
- [x] Record actual HEAD/worktree after Task 2.9 checkpoint.
- [x] Confirm Task 2.9 COMPLETE/PASS.
- [x] Confirm Task 2.10 NOT STARTED.
- [x] Keep KQ-BACKLOG-BENCH-002 DEFERRED.

## Characterization
- [x] Create TRANSFORMER-LAYER-CONTRACT.md.
- [x] Pin decoder-layer functions.
- [x] Pin family classification.
- [x] Pin norms/epsilons.
- [x] Pin GR module/parameters/equations.
- [x] Pin branch count/rank/order.
- [x] Pin PLE exact placement/combine.
- [x] Pin GDN/QSA call order.
- [x] Pin MoE call order.
- [x] Pin final branch/output composition.
- [x] Separate forward-only GR activations from persistent state.

## Oracle
- [x] Ordinary GDN-layer vectors.
- [x] QSA-layer vectors.
- [x] PLE-enabled layer vectors.
- [x] Calibration corpus.
- [x] Disjoint holdout.
- [x] State vectors.
- [x] Family vectors.
- [x] Manifest/SHA.
- [x] No self-oracle.

## Native config
- [x] Immutable layer config.
- [x] Exact layer family.
- [x] Resolve layer-owned norm/GR bindings.
- [x] Resolve correct mixer config.
- [x] Resolve MoE config.
- [x] Resolve PLE only where canonical.
- [x] 48/48 real configs validate.

## GR/residual
- [x] Scalar C17 implementation.
- [x] Exact read gates.
- [x] Exact write gates/scalars.
- [x] Exact branch order.
- [x] No unintended branch mixing.
- [x] No persistent GR cache if forward-only.

## Prefill/decode
- [x] Ordinary GDN family.
- [x] QSA family.
- [x] PLE-enabled family.
- [x] Multi-step decode.
- [x] Reset/replay.
- [x] Persistent state transitions.
- [x] Transactional failure behavior.

## Composition checkpoints
- [x] Layer input.
- [x] GR/read branch checkpoints.
- [x] PLE-enhanced hidden where applicable.
- [x] Mixer input/output.
- [x] GR write/update.
- [x] MoE input/output.
- [x] Final layer output.
- [x] Only canonical checkpoints retained.

## Floating contracts
- [x] Layer-specific calibration.
- [x] Disjoint holdout.
- [x] Per-checkpoint tolerance.
- [x] Existing suboperator state contracts remain PASS.
- [x] No generic epsilon.

## Target integration
- [x] 48/48 layers classify correctly.
- [x] GDN/QSA family counts reconciled.
- [x] Exactly one PLE-enabled layer at canonical ID.
- [x] Norm/GR bindings valid.
- [x] Suboperator configs valid.
- [x] No missing/extra semantic bindings.
- [x] Optional payload <= 16 MiB.
- [x] No hard-coded offsets/raw weights.

## State footprint
- [x] GDN-family persistent state derived.
- [x] QSA-family persistent state derived.
- [x] PLE-enabled family state derived.
- [x] GR forward activations excluded if non-persistent.
- [x] Reconcile MODEL-RUNTIME-STATE.md.

## Fail closed
- [x] Wrong layer family/ID.
- [x] Wrong/missing PLE placement.
- [x] Missing norm/GR binding.
- [x] Wrong GR branch/rank/shape.
- [x] Invalid suboperator config/state.
- [x] QSA capacity failure.
- [x] PLE provider/address failure.
- [x] MoE failure.
- [x] Non-finite input where forbidden.
- [x] Count/context overflow.
- [x] Aliasing/capacity failures.
- [x] Transaction rollback.

## Metrics
- [x] Config bytes/family.
- [x] State bytes/family.
- [x] Layer scratch.
- [x] GR workspace.
- [x] Transaction staging overhead.
- [x] Reduced prefill/decode latency.
- [x] No optimization.

## Regression
- [x] CPU clean PASS.
- [x] CUDA clean PASS.
- [x] Task 2.0-2.9 PASS.
- [x] No new warnings.
- [x] git diff --check PASS.

## Docs/governance
- [x] TRANSFORMER-LAYER-CONTRACT.md.
- [x] NATIVE-LAYER-REFERENCE.md.
- [x] KQ-LAYER-API.md.
- [x] ADR 0018.
- [x] Architecture/model/runtime/goldens updated.
- [x] TASKS/ROADMAP/Epic 2 updated.
- [x] CHANGELOG updated.
- [x] Next model-executor task NOT STARTED.

## Safety
- [x] no embedding.
- [x] no 48-layer executor.
- [x] no final norm/LM head/logits.
- [x] no sampling.
- [x] no final scheduler/cache policy.
- [x] no SIMD/CUDA model kernels.
- [x] no production Class-C/Python dependency.
- [x] no tracked model weights/secrets/local paths.
