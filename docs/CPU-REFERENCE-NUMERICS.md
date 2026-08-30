# Scalar CPU reference numerics

Status: **TASK 2.5 COMPLETE / PASS**

Task 2.5 adds Kestrel-Q's correctness-first scalar numerical boundary. It
decodes the seven storage formats present in the registered GGUF, supplies a
bounded quantized-row dot path and implements only the generic F32 primitives
characterized in `CPU-NUMERIC-CONTRACT.md`.

## Production scope

The C17 implementation is `src/kq_numeric.c`, exposed by
`include/kq_numeric.h`. Its numerical translation unit is compiled with
MSVC `/fp:strict` (or the documented non-MSVC no-fast-math/no-contraction
fallback). It uses no SIMD, intrinsic, CUDA, Python, llama.cpp or GGML runtime
dependency.

Supported decode types are F32, BF16, Q5_1, Q8_0, Q4_K, Q5_K and IQ4_NL.
Decode accepts only complete blocks, returns F32, checks all count/capacity
arithmetic and never requires full-tensor materialization. F32/BF16 preserve
raw special-value bits; malformed non-finite quantized scales fail closed.

The selected generic primitive set is:

- F32 add, multiply, scale and left-to-right dot;
- sigmoid, SiLU and SwiGLU combine;
- Qwen-weight-delta-form RMSNorm;
- F32 softmax;
- stable top-k with lower-index tie order; and
- selected-weight renormalization.

This is not a model graph. GDN, convolution, QSA, RoPE, Gated Residual, MoE
routing/dispatch, PLE value lookup, LM-head execution, sampling and generation
remain unimplemented.

## Independent evidence

Evidence root:

`research/numerics/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/`

| Asset | Coverage | SHA-256 |
|---|---|---|
| `dequant-vectors.json` | 39 synthetic blocks plus seven row-dot cases | `e95da6399fccd1940b01c15f9dc039f2062f39fbb31a946f86bb354b2e87f8b6` |
| `primitive-calibration.json` | 31 cases over 11 generic operations | `c63c25b09afde94a51cbb9f63bb2fce1b4743771c9dee328d88fbfa075ff91d4` |
| `primitive-holdout.json` | 21 disjoint cases over the same operations | `ca8feec63f84028b9b35c9f3bc1afaf6a44f18817d1dda82466593edb593e6bb` |
| `real-gguf-samples.json` | nine bounded real blocks, all seven formats | `5a4846d74b029389dc80cef24172b37623c2e815f9bec36b59cc079e3e6e3a79` |
| `manifest.json` | revisions, licenses, tool and asset hashes | `28ef7a19142bd922344e598ab46aaff60cc94724ff6ad4f177184dfd12000e06` |

Synthetic storage expected values come from the pinned MIT llama.cpp revision
`90c26fcd4b2114b4aa39d09d69318cb8f438d27a`. The helper is built in ignored
research cache and uses the pinned GGML `to_float` traits; production never
links it. Primitive expected values come from Python 3.13.12/NumPy 2.5.2
(BSD-3-Clause) with explicit F32 operations and reduction order. Native output
is recorded only as an observation.

Add, multiply, scale, dot, RMSNorm and renormalization were bit-exact on the
calibration and holdout corpora. Top-k indices and values were exact. The
libm-sensitive contracts are separately calibrated:

| Primitive | Calibration max ULP | Accepted ULP | Holdout max ULP |
|---|---:|---:|---:|
| sigmoid | 2 | 3 | 2 |
| SiLU | 2 | 3 | 2 |
| softmax | 1 | 2 | 2 |
| SwiGLU | 3 | 4 | 2 |

Each limit is its calibration maximum plus one adjacent representable F32
step; there is no blanket absolute/relative tolerance. The evidence also
records measured maximum absolute and relative differences and forbids NaN/
Inf.

## Bounded real-artifact samples

The real test resolves every sample by stable semantic ID and Task 2.2 view;
it contains no physical file offset. Exactly one block per sample is decoded:

| Sample | Semantic | Type | Packed bytes |
|---|---|---:|---:|
| router | `layer.00.moe.router` | F32 | 4 |
| QSA query split | `layer.11.qsa.indexer.qk` | BF16 | 2 |
| routed expert | `layer.00.moe.routed.down`, expert 7 | Q5_1 | 24 |
| routed expert gate | `layer.00.moe.routed.gate_up`, expert 7 | Q4_K | 144 |
| shared expert down | `layer.00.moe.shared.down` | Q8_0 | 34 |
| layer-2 expert down | `layer.02.moe.routed.down`, expert 7 | Q8_0 | 34 |
| layer-2 expert gate | `layer.02.moe.routed.gate_up`, expert 7 | Q5_K | 176 |
| layer-2 expert up | `layer.02.moe.routed.gate_up`, expert 7 | Q5_K | 176 |
| PLE member 64 | `layer.01.ple.table.064` | IQ4_NL | 18 |

Total real model payload touched is **612 logical packed bytes in 9 blocks**,
below the predeclared 1 MiB hard limit. Each decoded-output SHA-256 matches the
independent llama.cpp result bit-for-bit. Only semantic ID, type, logical block
index, byte count, decoded hash/statistics and comparison result are committed;
raw real-model bytes are not.

These reads do not measure disk throughput, paging, residency or cache
behavior. `KQ-BACKLOG-BENCH-002` remains deferred.

## Failure and memory behavior

Tests reject unsupported types, partial blocks, insufficient capacity,
overflow, dimension mismatch, forbidden aliasing, non-finite primitive input,
non-finite quant scales, invalid top-k, changed FP rounding mode, out-of-view
ranges and transformed-layout canonical-order misuse. A physical-order API is
explicitly named and required for converter-transformed samples.

The implementation allocates no heap memory. Quantized row-dot uses exactly
256 F32 scratch elements (**1,024 bytes**) and decodes one block at a time.
No scalar throughput target or storage-performance claim is made in this task.

## Findings corrected during implementation

- The first dot test transcribed the expected sum as `1` instead of `-5`;
  recomputation identified the test error and the corrected case now guards
  left-to-right accumulation.
- The initial research helper assumed every GGML trait exposes `to_float`;
  F32 deliberately does not. A helper-only bit-copy branch fixed the oracle
  harness without changing production semantics.
- The first real-evidence parser expected 12 TSV fields instead of the emitted
  11 and stopped before writing evidence. The exact record-shape check was
  corrected; no raw block was persisted.
- A context patch initially placed the scalar dot-alias guard in the binary
  vector helper. The new alias regression failed, the guard was moved to dot,
  and the clean suite passed.

## Limitations

The path is scalar, contiguous and deliberately slow. Its libm contract is
currently calibrated for the accepted MSVC/KQ-01 toolchain. It does not encode
or quantize, implement strided layouts, concatenate split tensors, invert
converter transforms, expose an in-place primitive API or establish any model
operator/full-forward correctness claim.
