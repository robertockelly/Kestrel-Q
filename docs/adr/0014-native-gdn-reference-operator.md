# ADR 0014 — Native GDN reference operator

Status: **Accepted**

## Context

Task 2.0-2.5 established container access, canonical semantics, bounded tensor
views, tokenizer/PLE addressing and an independently validated scalar numeric
layer.

Kestrel-Q now needs its first model-specific execution operator.

## Decision

1. Implement Qwen3.8-Flash-Next GDN as the first native model operator.
2. Maintain a scalar CPU reference implementation before optimized kernels.
3. Define correctness against pinned independent Class-C operator vectors.
4. Treat recurrent and convolution state transitions as first-class outputs.
5. Use reduced-shape canonical vectors to validate algorithm semantics when the
   pinned canonical module supports them.
6. Use target-shape semantic/view integration to validate the real artifact
   independently from synthetic algorithm vectors.
7. Do not download the full BF16 checkpoint merely for Task 2.6.
8. Introduce no SIMD/CUDA GDN kernel until the reference path is stable.
9. Future optimized GDN paths must validate against this reference.
10. Keep GDN separate from QSA, MoE and PLE value execution.

## Evidence gate

Require:

- complete canonical GDN contract;
- independent calibration and holdout vectors;
- native prefill PASS;
- native incremental decode PASS;
- state-transition/checkpoint PASS;
- all 36 real GDN layers structurally valid;
- QSA layers fail closed;
- CPU/CUDA regressions PASS.

The evidence gate passed in Task 2.6. The pinned module supports the selected
reduced dimensions without an algorithm change; calibration, disjoint holdout,
state transitions and 21 intermediate native checkpoint classes pass their
independently generated contracts. Native split prefill/decode and reset/replay
are bit-identical internally. All 36 target GDN descriptors validate and all 12
QSA IDs reject with zero real payload bytes touched.

The accepted production boundary distinguishes the released BF16 target-state
descriptor from the scalar F32 correctness descriptor. Both validate the same
canonical semantic registry. Real GGUF transformed bytes are never presented
as canonical-contiguous F32 weights.

No claim is made that reduced F32 vectors replace a future full-weight target
operator oracle. QSA, GR composition, MoE, PLE values, complete-layer/full-model
execution and optimized kernels require separate tasks and evidence.
