# TASKS-TASK-2.8-MOE-REFERENCE-OPERATOR.md

Status: **COMPLETE / PASS**

Entry HEAD: `21189a45e1bcc277ba35a5ad5ba483557da73daa` (Task 2.7
checkpoint on `origin/main`). The four pre-authored Task 2.8 governance files
were preserved and completed in place.

## Baseline
- [x] Record actual HEAD/worktree after Task 2.7 checkpoint.
- [x] Confirm Task 2.7 COMPLETE/PASS.
- [x] Confirm Task 2.8 NOT STARTED.
- [x] Keep KQ-BACKLOG-BENCH-002 DEFERRED.

## Characterization
- [x] Create MOE-OPERATOR-CONTRACT.md.
- [x] Pin canonical module/functions.
- [x] Pin router shape/dtype.
- [x] Pin expert count/top-k.
- [x] Pin top-k membership/order/ties.
- [x] Pin selected-weight normalization.
- [x] Pin routed expert equations.
- [x] Pin shared expert equations.
- [x] Pin shared gate equation.
- [x] Pin final combine.
- [x] Exclude training-only branches.

## Oracle
- [x] Reduced expert-count vectors.
- [x] Canonical 512/top-10 routing vectors.
- [x] Equal-logit ties.
- [x] kth-boundary ties.
- [x] Near-ties.
- [x] Expert 0/511 boundaries.
- [x] Calibration corpus.
- [x] Disjoint holdout.
- [x] Routing vectors.
- [x] Manifest/SHA.
- [x] No self-oracle.

## Native config
- [x] Immutable MoE config.
- [x] Resolve router.
- [x] Resolve routed gate/up/down.
- [x] Resolve shared expert.
- [x] Resolve shared gate.
- [x] Validate expert axis=512.
- [x] Validate hidden/intermediate dims.
- [x] Validate split gate/up.

## Router/top-k
- [x] FP32 canonical router path.
- [x] Exact expert membership.
- [x] Exact ordering.
- [x] Exact tie behavior.
- [x] Exact selected-weight normalization.
- [x] No approximate routing pass criteria.

## Routed experts
- [x] Access selected expert members only.
- [x] Gate/up/down reference path.
- [x] Exact activation/order.
- [x] Weighted accumulation order.
- [x] No all-expert dequant/materialization.

## Shared expert
- [x] Shared gate/up/down.
- [x] Shared activation.
- [x] Shared sigmoid gate.
- [x] Gated shared output.
- [x] Final routed+shared combine.

## Floating contracts
- [x] Router logits/probabilities calibrated.
- [x] Selected weights calibrated.
- [x] Expert outputs calibrated.
- [x] Routed sum calibrated.
- [x] Shared path calibrated.
- [x] Final output calibrated.
- [x] Disjoint holdout PASS.
- [x] No generic epsilon.

## Target integration
- [x] All target MoE configs validate.
- [x] Every routed expert axis = 512.
- [x] Shared expert/gate bindings valid.
- [x] Split gate/up bindings valid.
- [x] Physical types/layouts valid.
- [x] Per-expert packed footprint reconciled.
- [x] Optional real payload <= 16 MiB.
- [x] No hard-coded offsets/raw weights.

## Fail closed
- [x] Wrong expert count/top-k.
- [x] Missing router/stack/shared/gate.
- [x] Wrong expert axis.
- [x] Wrong shapes.
- [x] Broken split relation.
- [x] Invalid expert ID.
- [x] Non-finite input where forbidden.
- [x] Capacity/overflow.
- [x] Aliasing/view failures.

## Metrics
- [x] Config bytes.
- [x] Router workspace.
- [x] Top-k workspace.
- [x] Expert workspace.
- [x] Routed accumulation workspace.
- [x] Shared workspace.
- [x] Reduced latencies.
- [x] No optimization.

## Regression
- [x] CPU clean PASS.
- [x] CUDA clean PASS.
- [x] Task 2.0-2.7 PASS.
- [x] No new warnings.
- [x] git diff --check PASS.

## Docs/governance
- [x] MOE-OPERATOR-CONTRACT.md.
- [x] NATIVE-MOE-REFERENCE.md.
- [x] KQ-MOE-API.md.
- [x] ADR 0016.
- [x] Architecture/goldens/footprint updated where justified.
- [x] TASKS/ROADMAP/Epic 2 updated.
- [x] CHANGELOG updated.
- [x] Next operator task NOT STARTED.

## Safety
- [x] no PLE value lookup.
- [x] no full layer/full forward.
- [x] no expert cache/prefetch.
- [x] no scheduler.
- [x] no SIMD/CUDA model kernel.
- [x] no production Class-C/Python dependency.
- [x] no tracked model weights/secrets/local paths.
