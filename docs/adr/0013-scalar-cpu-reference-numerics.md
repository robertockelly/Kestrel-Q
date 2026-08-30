# ADR 0013 — Scalar CPU reference numerics

Status: **Accepted — 2026-08-29**

## Context

Tasks 2.0-2.4 established container access, semantic identity, bounded quantized views, tokenizer/chat, and PLE address generation. Kestrel-Q now needs a trusted numerical path before implementing model operators or optimized kernels.

## Decision

1. Maintain a scalar C17 CPU correctness/reference numerical path.
2. Decode all seven physical storage types used by the verified GGUF.
3. Provide a bounded block-by-block quantized-row dot reference path rather than requiring full tensor dequantization.
4. Keep the reference path free of SIMD, fast-math and model CUDA kernels.
5. Define FP accumulation, rounding and contraction semantics explicitly.
6. Calibrate floating tolerances per primitive using an independent calibration corpus and validate a separate holdout corpus.
7. Use exact equality for bit-exact and discrete outputs.
8. Use pinned llama.cpp only as a test/research Class-Q oracle for quantized decode; production never links it.
9. Allow bounded real-GGUF sample reads for validation without committing raw model bytes.
10. Require future optimized kernels to validate against this reference layer.

## Evidence gate (satisfied)

Task 2.5 must demonstrate independent synthetic decode validation for all seven types, calibrated/held-out primitive validation, bounded real-model sample validation, CPU/CUDA regression stability, and no model-specific forward operator implementation.

## Evidence

- All seven target storage types match the pinned llama.cpp Class-Q decoder
  bit-for-bit over 39 deterministic synthetic edge/pattern cases.
- Seven quantized row-dot cases match an independently ordered F32 oracle with
  fixed 1,024-byte scratch.
- Thirty-one calibration and 21 disjoint holdout cases establish exact or
  per-primitive ULP contracts; no blanket tolerance is used.
- Nine real semantic/view-resolved blocks cover all seven types and match the
  independent decoder exactly while touching 612 packed bytes, well below the
  1 MiB guard. No raw model bytes are committed.
- Production is C17, scalar and `/fp:strict`; it links neither llama.cpp nor
  Python and adds no full tensor materialization, model operator or forward
  path.
- Clean CPU and CUDA Release regression suites pass with no new Kestrel-Q
  warning.

Future optimized CPU/CUDA implementations must compare against this reference
layer under its exact/discrete/calibrated contracts. This ADR does not accept a
model operator, performance kernel, cache, prefetch or scheduler policy.
