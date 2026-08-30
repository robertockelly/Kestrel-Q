# PLAN-TASK-2.5-CPU-REFERENCE-NUMERICS.md

Status: **COMPLETE / PASS — 2026-08-29**

## Objective

Implement Kestrel-Q's first correctness-first CPU numerical layer:

1. scalar storage decoding/dequantization for every physical type present in the verified Qwen3.8-Flash-Next GGUF;
2. a minimal reusable scalar C17 reference primitive set for later model-operator work;
3. independent numeric evidence with per-primitive floating contracts calibrated from reference runs rather than guessed tolerances.

Task 2.5 is the first task allowed to dereference **bounded samples** of the real model-weight payload.

It does **not** implement a full forward pass, model layers, SIMD optimization, CUDA model kernels, PLE storage policy, scheduler, sampling, or generation.

## Baseline

Reviewed checkpoint:

`1cf27c9fd0ef39627b38c1225c9ff9c2293c0388`

Status:

- Epic 1 COMPLETE/PASS
- Epic 2 IN PROGRESS
- Task 2.0-2.4 COMPLETE/PASS
- Task 2.5 NOT STARTED
- ADR 0008-0012 ACCEPTED
- `KQ-BACKLOG-BENCH-002` DEFERRED

Task 2.2 storage geometry:

| Type | Elements/block | Bytes/block |
|---|---:|---:|
| F32 | 1 | 4 |
| BF16 | 1 | 2 |
| Q5_1 | 32 | 24 |
| Q8_0 | 32 | 34 |
| Q4_K | 256 | 144 |
| Q5_K | 256 | 176 |
| IQ4_NL | 32 | 18 |

The frozen Task 1.4 `operator-vector-plan.json` remains unchanged. Task 2.5 creates a separate numeric-evidence namespace.

## 1. Mandatory numeric characterization gate

Before production implementation create `docs/CPU-NUMERIC-CONTRACT.md`.

Record and pin:

- exact block layouts and bit/nibble packing;
- scale/min fields and decode formulas;
- BF16 conversion semantics;
- decode output dtype;
- accumulation dtype for each reference primitive;
- floating-point rounding and contraction/FMA policy;
- compiler FP mode actually used by the reference target;
- NaN/Inf/subnormal policy;
- stable top-k tie ordering;
- shape/stride/aliasing assumptions;
- which operations are generic numeric primitives versus future Qwen-specific operators.

Use the pinned quant-format provenance:

`ggml-org/llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a` (MIT).

Production must implement independently and must not link to llama.cpp/GGML.

For model arithmetic semantics, inspect the pinned canonical Qwen/Transformers source rather than assuming accumulation behavior.

## 2. Storage decode / dequantization

Implement scalar decode to F32 for:

- F32
- BF16
- Q5_1
- Q8_0
- Q4_K
- Q5_K
- IQ4_NL

Requirements:

- checked block count and output capacity;
- no OOB reads;
- no unnecessary alignment assumptions;
- no whole-tensor materialization requirement;
- deterministic output;
- explicit unsupported-type failure;
- no quantization/encoding API unless separately justified.

Preferred conceptual API:

```c
kq_status kq_dequantize_blocks_f32(
    kq_gguf_type type,
    const void *packed,
    size_t packed_bytes,
    float *out,
    size_t out_capacity,
    size_t *out_count);
```

Exact API may differ.

## 3. BF16 / F32 exactness

F32 storage decode is bit-preserving.

BF16 -> F32 uses explicit bit-level semantics. Test +0, -0, normal values, subnormals where applicable, +/-Inf and NaN behavior defined by the contract.

## 4. Quantized row-dot reference path

Provide a correctness-first scalar operation equivalent to:

```text
quantized weight row dot F32 activation -> scalar
```

The implementation may decode block-by-block internally.

Requirements:

- bounded scratch;
- no full weight-tensor dequantization;
- validated row/block geometry;
- accumulation dtype pinned by the numeric contract;
- no SIMD/intrinsics;
- no `/fp:fast`;
- no hidden FMA assumption.

A small row-wise matvec helper is allowed only if it is a checked loop over the reference row-dot primitive.

Do not implement a high-performance GEMM in this task.

## 5. Minimal scalar reference primitives

Implement only the minimal reusable primitive set justified for later Qwen operator work. Candidate set, subject to characterization:

- F32 vector add;
- elementwise multiply;
- scale;
- F32 dot;
- sigmoid;
- SiLU;
- SwiGLU elementwise combine;
- RMSNorm;
- softmax;
- stable top-k;
- top-k probability renormalization if evidence-backed and generic.

Do not implement GDN recurrence, causal convolution, QSA sparse selection/attention, RoPE unless separately justified, GR model semantics, MoE routing model logic, PLE value lookup/convolution, full LM head, sampler, or complete layer code.

If a candidate's canonical arithmetic contract is unclear, defer it rather than guess.

## 6. Reference FP mode

The scalar CPU path becomes the correctness reference for future optimized kernels.

Requirements:

- no fast-math;
- no reassociation;
- no silent reduction-order changes;
- document actual MSVC FP flags;
- establish rounding mode;
- document contraction/FMA policy;
- avoid locale-dependent behavior.

Prefer target-specific compiler options rather than globally changing CUDA/build semantics.

## 7. Independent Class-Q dequant oracle

Use the already-pinned Class-Q oracle `llama.cpp@90c26fcd...` only in test/research tooling.

Preferred workflow:

1. build a small ignored reference helper against the pinned source;
2. feed deterministic raw quant blocks;
3. obtain reference F32 outputs;
4. compare native Kestrel-Q output;
5. never copy the implementation into production;
6. never link production Kestrel-Q to llama.cpp.

The helper must not require loading the 111 GB model.

## 8. Independent primitive oracle

For non-quantized primitives use deterministic test/research tooling with explicit operation order and dtype semantics. Python/NumPy is acceptable only outside production, with versions recorded.

Kestrel-Q output must never define expected values.

## 9. Calibration + holdout, no guessed tolerances

Do not apply one generic `1e-5` tolerance.

For each floating primitive:

1. generate a deterministic calibration corpus;
2. compare independent oracle vs native implementation;
3. measure absolute/relative/ULP differences as applicable;
4. set a documented contract with justified margin;
5. validate a separate deterministic holdout corpus.

Classify evidence as:

```text
EXACT_BITS
EXACT_DISCRETE
CALIBRATED_FLOAT
```

Discrete outputs such as top-k indices remain exact.

Operations proven bit-exact use exact F32-bit equality.

## 10. Numeric evidence namespace

Create a separate evidence root, for example:

```text
research/numerics/Qwen3.8-Flash-Next/<canonical-revision>/
  manifest.json
  dequant-vectors.json
  primitive-calibration.json
  primitive-holdout.json
  real-gguf-samples.json
```

Requirements:

- deterministic regeneration;
- SHA-256 manifest;
- oracle/source revisions recorded;
- no raw model-weight bytes committed;
- real samples identified by semantic ID/type/block index, not local path or physical offset as the durable identity.

Do not change the original Task 1.4 golden files/hashes.

## 11. Synthetic quant blocks

For every storage type generate deterministic synthetic blocks covering representative and edge patterns, including where meaningful:

- zero values;
- min/max quant codes;
- alternating bit/nibble patterns;
- positive/negative scales;
- deterministic pseudo-random data;
- K-quant scale/min metadata extremes;
- IQ4_NL code coverage.

Compare against the independent Class-Q oracle.

Synthetic blocks are not model weights and may be committed when reproducibly generated.

## 12. Bounded real-GGUF sample validation

Task 2.5 may read a small deterministic subset of the real GGUF payload.

Predeclare a test-only guard:

```text
KQ_TASK25_REAL_PAYLOAD_SAMPLE_BUDGET <= 1 MiB logical payload bytes
```

Prefer much less.

Resolve samples through semantic descriptors and Task 2.2 bounded views, never hard-coded file offsets.

Representative coverage must include:

- all seven storage families;
- at least one routed-expert member;
- layer-2 expert special quant mix;
- one IQ4_NL/PLE block if that is the verified family.

Record only safe derived evidence:

- semantic ID;
- type;
- logical block index;
- packed bytes touched;
- decoded-output hash/statistics as appropriate;
- independent oracle result.

Do **not** commit the raw packed model bytes.

Track:

```text
real_model_payload_logical_bytes_touched
real_model_payload_blocks_touched
```

No disk-performance conclusion may be drawn from these reads.

## 13. Tensor-view integration

All model-payload access must follow:

```text
semantic descriptor
-> Task 2.2 validated view/member view
-> bounded packed block
-> Task 2.5 numeric primitive
```

Do not reopen raw offsets independently.

Split bindings remain separate. Transformed layouts may not be treated as canonical-contiguous unless the view contract explicitly permits it.

## 14. Fail closed

Cover:

- unsupported type;
- malformed packed byte count;
- insufficient output capacity;
- invalid arguments;
- count/size overflow;
- malformed/foreign view;
- range beyond tensor/member;
- non-contiguous range when contiguous required;
- transformed-layout misuse;
- invalid top-k request;
- shape mismatch;
- forbidden aliasing;
- invalid NaN/Inf input policy where relevant.

No UB on malformed inputs.

## 15. Memory discipline

Reference code uses bounded scratch and never allocates proportional to the full model.

Row-dot decodes only required blocks/chunks.

Synthetic matrix/vector tests remain small and deterministic.

## 16. Performance scope

Task 2.5 is not an optimization task.

Optional characterization only:

- scalar dequant throughput on small synthetic buffers;
- row-dot latency;
- small primitive latency;
- scratch bytes.

No SIMD/CUDA comparison and no performance target.

## 17. Documentation

Create:

- `docs/CPU-NUMERIC-CONTRACT.md`
- `docs/CPU-REFERENCE-NUMERICS.md`
- `docs/KQ-NUMERIC-API.md`

Create/finalize:

- `docs/adr/0013-scalar-cpu-reference-numerics.md`

Update ARCHITECTURE, TASKS, ROADMAP, Epic 2 status, Task 2.5 checklist, relevant tensor-view docs, tools/provenance docs, GOLDEN-VECTORS cross-reference if useful, and CHANGELOG.

## 18. ADR 0013 decision

ADR 0013 should capture:

1. scalar CPU correctness/reference numerical path;
2. all seven target storage decoders;
3. bounded block-by-block quantized row-dot reference path;
4. no SIMD/fast-math/model CUDA in the reference path;
5. explicit accumulation/FP semantics;
6. calibration + independent holdout per floating primitive;
7. exact equality for bit-exact/discrete outputs;
8. no production llama.cpp dependency;
9. bounded real-model sample validation without committing raw model bytes;
10. future optimized kernels must validate against this reference layer.

Accept only after independent validation.

## 19. Build/regression

Run clean CPU Release and CUDA Release builds/CTest. Task 2.5 itself has no CUDA dependency.

All Task 2.0-2.4 tests remain PASS. No new Kestrel-Q `/W4` warnings. Existing external NVCC C4211 may remain documented. `git diff --check` PASS.

## Definition of done

Task 2.5 is COMPLETE/PASS when Kestrel-Q has a deterministic scalar CPU numeric reference layer that decodes all seven target storage types, performs the selected low-level primitives under evidence-backed floating contracts, and validates bounded real-model samples without full tensor materialization or model-specific forward execution.

Do not commit or push automatically.
