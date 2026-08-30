# TASKS-TASK-2.5-CPU-REFERENCE-NUMERICS.md

Status: **COMPLETE / PASS — 2026-08-29**

## Baseline
- [x] Record actual HEAD/worktree.
- [x] Confirm Task 2.4 COMPLETE/PASS.
- [x] Confirm Task 2.5 NOT STARTED before edits.
- [x] Keep `KQ-BACKLOG-BENCH-002` DEFERRED.

## Characterization
- [x] Create `docs/CPU-NUMERIC-CONTRACT.md` before production code.
- [x] Pin all seven block layouts/formulas.
- [x] Pin BF16/F32 semantics.
- [x] Pin accumulation dtype per primitive.
- [x] Pin rounding/FMA/floating compiler mode.
- [x] Pin NaN/Inf/subnormal policy.
- [x] Pin stable top-k tie behavior.
- [x] Separate generic primitives from later model operators.

## Dequantization
- [x] F32.
- [x] BF16.
- [x] Q5_1.
- [x] Q8_0.
- [x] Q4_K.
- [x] Q5_K.
- [x] IQ4_NL.
- [x] Checked capacities/counts.
- [x] No whole-tensor requirement.

## Row-dot / matvec reference
- [x] Quant-row dot F32 activation.
- [x] Block-by-block bounded scratch.
- [x] No SIMD/intrinsics.
- [x] No fast-math/FMA ambiguity.
- [x] Optional row-matvec wrapper deliberately not added; it is not required by
  the accepted minimal boundary.

## Scalar primitive set
- [x] Implement only characterized required primitives.
- [x] Vector add/mul/scale.
- [x] F32 dot.
- [x] sigmoid/SiLU/SwiGLU.
- [x] RMSNorm.
- [x] softmax.
- [x] stable top-k and selected-weight renormalization.
- [x] No model-specific layer/operator graph.

## Independent oracles
- [x] Pinned llama.cpp Class-Q helper for dequant only.
- [x] Independent NumPy primitive reference generator.
- [x] Production links to neither oracle.
- [x] Exact revisions/versions/hashes recorded.

## Calibration / holdout
- [x] Deterministic 31-case calibration corpus.
- [x] Measure abs/rel/ULP differences.
- [x] Per-primitive tolerance contract.
- [x] Separate deterministic 21-case holdout corpus.
- [x] Holdout PASS.
- [x] EXACT_BITS / EXACT_DISCRETE / CALIBRATED_FLOAT classifications.

## Synthetic quant vectors
- [x] All-zero/IEEE edge patterns.
- [x] Min/max quant codes.
- [x] Alternating patterns.
- [x] Deterministic random patterns.
- [x] K-quant metadata edge cases.
- [x] IQ4_NL complete codebook coverage.
- [x] Class-Q comparison PASS (39 decode + seven row-dot cases).

## Real GGUF bounded samples
- [x] Sample budget guard <= 1 MiB logical bytes.
- [x] Resolve only through semantic + Task 2.2 views.
- [x] F32 sample.
- [x] BF16 sample.
- [x] Q5_1 sample.
- [x] Q8_0 sample.
- [x] Q4_K sample.
- [x] Q5_K sample.
- [x] IQ4_NL sample.
- [x] Routed-expert sample.
- [x] Layer-2 Q8_0/Q5_K special mix.
- [x] PLE/IQ4_NL sample.
- [x] No raw model blocks committed.
- [x] 612 logical packed bytes / nine blocks reported.

## Fail closed
- [x] Unsupported type.
- [x] Bad block byte count.
- [x] Output capacity.
- [x] Invalid arguments.
- [x] Arithmetic overflow.
- [x] Dead/malformed view validation where detectable; borrowed lifetime is
  documented.
- [x] Out-of-range span.
- [x] Non-contiguous member misuse rejected by Task 2.2 before numeric access.
- [x] Transformed-layout canonical-order misuse.
- [x] Invalid top-k.
- [x] Shape/dimension mismatch.
- [x] Forbidden aliasing.
- [x] NaN/Inf and non-default rounding policy.

## Evidence
- [x] Separate `research/numerics/...` namespace.
- [x] Deterministic manifest/SHA.
- [x] No Task 1.4 golden mutation.
- [x] No raw real-model payload committed.

## Metrics
- [x] Scratch bytes: 1,024.
- [x] Scalar timing/throughput intentionally not promoted to a performance
  baseline.
- [x] No SIMD/CUDA performance work.

## Regression
- [x] Clean CPU Release PASS (17/17 CTest).
- [x] Clean CUDA Release PASS (19/19 CTest).
- [x] Task 2.0-2.4 PASS.
- [x] No new Kestrel-Q warnings; the previously documented external NVCC-generated C4211 remains.
- [x] `git diff --check` PASS.

## Docs/governance
- [x] `docs/CPU-REFERENCE-NUMERICS.md`.
- [x] `docs/KQ-NUMERIC-API.md`.
- [x] ADR 0013 finalized.
- [x] ARCHITECTURE/TASKS/ROADMAP/Epic 2 updated.
- [x] CHANGELOG updated.
- [x] Next model-operator task NOT STARTED.

## Safety
- [x] No full forward pass.
- [x] No GDN/QSA/GR/MoE/PLE model operator implementation.
- [x] No sampler/generation.
- [x] No production llama.cpp/Python dependency.
- [x] No tracked model weights/secrets/local paths.
- [x] Build trees untracked.
