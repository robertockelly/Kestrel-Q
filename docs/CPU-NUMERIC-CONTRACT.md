# CPU numeric contract

Status: **PINNED FOR TASK 2.5 IMPLEMENTATION**

This document fixes the scalar storage and arithmetic contract before the
Task 2.5 production implementation. It is deliberately narrower than a model
operator contract: it defines packed-block decoding and reusable F32
primitives, not Qwen layer execution.

## Provenance boundary

The Class-Q storage oracle is `ggml-org/llama.cpp` revision
`90c26fcd4b2114b4aa39d09d69318cb8f438d27a` under MIT. The inspected source
files at that revision are:

| Source | SHA-256 |
|---|---|
| `ggml/src/ggml-common.h` | `c7a75460f797ac4406f8af4a9e318c2f0674b7b660a1c81ce5b4c019dc29a89a` |
| `ggml/src/ggml-quants.c` | `c5829ed1b4ced3970964464eaf2c985af38008fa76c91af15f4f63ef4447ab1f` |
| `ggml/include/ggml.h` | `f81d69f7711740e3b2a2f1c47513a187ae985c4966a2a7788eafdbdd51941988` |

The model-arithmetic source is the Apache-2.0 Transformers revision
`805a9e939fa8c1bff8d8ffdf041c051b71a914aa`. Its pinned
`modeling_qwen4_exp.py` SHA-256 is
`91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f`;
`activations.py` is
`5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665`.
It establishes F32 RMSNorm reduction, SiLU-gated MLPs, F32 router/attention
softmax and top-k probability renormalization. It does not define a portable
tie order for `torch.topk`; Kestrel-Q therefore defines a deterministic generic
tie rule below, while future MoE operator parity remains a separate gate.

Kestrel-Q's production implementation is independent C17 and links neither
oracle. The llama.cpp helper and Python 3.13.12/NumPy 2.5.2 primitive oracle
are ignored/test-research tools only.

## Common representation rules

- Packed data is little-endian and may be unaligned. Scalar fields are loaded
  through byte copies, never through unaligned typed pointer casts.
- Decode output is IEEE-754 binary32 (`float`). A block is the smallest decode
  unit and only whole packed blocks are accepted.
- The caller supplies packed byte count and F32 capacity. Counts, products and
  block spans use checked 64-bit arithmetic before conversion to `size_t`.
- Packed input and output must not overlap. Primitive input/output ranges must
  obey each API's explicit no-alias rule; partial overlap is never accepted.
- Storage decoding has no tensor-shape meaning. Canonical-versus-physical
  ordering is supplied by the Task 2.2 view API.

## Exact storage layouts

`h16(x)` below means exact IEEE binary16-to-binary32 expansion. `b16(x)` means
placing the unsigned BF16 payload in the high 16 bits of a binary32 word. All
integer indexing is unsigned and range checked.

### F32 and BF16

- F32: one element in four bytes, copied bit-for-bit to the F32 output.
- BF16: one element in two bytes, decoded as `bits32 = bits16 << 16`.

Both preserve signed zero, infinities, subnormals and NaN sign/payload bits.

### Q5_1

One 24-byte block contains 32 values:

```text
bytes 0..1   binary16 d
bytes 2..3   binary16 m
bytes 4..7   32 high bits qh, little-endian
bytes 8..23  16 packed low/high nibbles qs
```

For `j=0..15`, low-half element `j` uses the low nibble of `qs[j]` and
high bit `j` of `qh`; high-half element `j+16` uses the high nibble and high
bit `j+16`. The exact result is `F32(code * h16(d) + h16(m))` with code 0..31.

### Q8_0

One 34-byte block contains binary16 `d` followed by 32 signed two's-complement
bytes. Element `j` is `F32(int8(qs[j]) * h16(d))`.

### Q4_K

One 144-byte super-block contains 256 values:

```text
bytes 0..1     binary16 d
bytes 2..3     binary16 dmin
bytes 4..15    eight packed 6-bit scales and eight packed 6-bit minima
bytes 16..143  four 64-element groups, each 32 low nibbles then 32 high nibbles
```

For sub-block `j=0..7`, `(scale_j,min_j)` is unpacked as:

```text
j < 4: scale = scales[j] & 63; min = scales[j+4] & 63
j >=4: scale = (scales[j+4]&15) | ((scales[j-4]>>6)<<4)
       min   = (scales[j+4]>>4) | ((scales[j]>>6)<<4)
```

Each 32-element sub-block decodes as
`F32((h16(d) * scale_j) * code - (h16(dmin) * min_j))`, code 0..15.

### Q5_K

One 176-byte super-block adds 32 high-bit bytes between the 12-byte scale/min
stream and the 128 low-nibble bytes. Scale/min decoding is identical to Q4_K.
For each successive 64-element group, high-bit masks are `(1,2)`, `(4,8)`,
`(16,32)`, `(64,128)` for the low/high nibble halves. Each sub-block decodes
with the Q4_K formula and a code in 0..31.

### IQ4_NL

One 18-byte block contains binary16 `d` and 16 packed nibbles. Low nibbles
produce elements 0..15; high nibbles produce elements 16..31. The pinned
non-linear codebook is:

```text
-127, -104, -83, -65, -49, -35, -22, -10,
   1,   13,  25,  38,  53,  69,  89, 113
```

Each element is `F32(h16(d) * codebook[code])`.

Quantized scales must decode to finite F32. A non-finite scale/min is malformed
numeric data and fails closed. F32/BF16 raw storage remains bit-preserving,
including special values.

## Floating-point execution contract

The numerical translation unit is compiled by MSVC 19.44 with `/fp:strict`;
non-MSVC fallback flags are `-fno-fast-math -ffp-contract=off`. No SIMD or
architecture intrinsics are used. The required process rounding mode is
round-to-nearest/ties-to-even. APIs reject another active C rounding mode and
do not mutate caller floating-point state.

F32 reductions accumulate left-to-right in a `float` temporary and store after
every defined source-level step. There is no reassociation or contraction/FMA.
The reference environment preserves subnormals; enabling FTZ/DAZ is outside
this accepted contract and is caught by the numeric regression probes.

All math primitives reject NaN or infinity inputs and fail if they produce a
non-finite output. Signed finite zeros are accepted. This policy makes numeric
corruption explicit. Raw F32/BF16 decode is the sole bit-preserving exception.

## Primitive contracts

- `add`, `multiply`, `scale`: element order 0..N-1, one F32 operation per
  output; no overlapping arrays.
- `dot`: F32 multiply followed by F32 left-to-right addition, starting at +0.
- quantized row-dot: decode one physical block into bounded scratch, then use
  the same F32 multiply/add order; scratch is at most 256 F32 values.
- `sigmoid`: `1 / (1 + expf(-x))`, with a sign-stable branch that avoids
  positive exponential overflow.
- `SiLU`: the contracted semantic expression is `x * sigmoid(x)` but is
  evaluated as two non-contracted F32 steps.
- `SwiGLU combine`: `SiLU(gate[i]) * up[i]`, again as separate F32 steps.
- `RMSNorm`: input and weight are F32; sum of squares and mean are F32 in
  left-to-right order, inverse root is `1/sqrtf(mean+epsilon)`, and output is
  `x[i] * inverse_root * (1 + weight[i])` with explicit F32 intermediates.
  `epsilon` must be positive finite. Grouped/gated RMSNorm remains a later
  operator concern.
- `softmax`: finite input, F32 maximum reduction, `expf(x-max)`, F32
  left-to-right sum, then F32 division. Output order is unchanged.
- stable top-k: descending F32 score; exact ties choose the lower original
  index. Returned indices are exact discrete values. This generic deterministic
  rule is not yet a claim about PyTorch tie behavior in the Qwen MoE operator.
- top-k renormalization: selected non-negative finite weights are summed and
  divided in selected order. A zero/non-finite sum fails closed.

All vector lengths are non-zero. Row dimensions must be a whole number of the
selected type's blocks. The reference API is contiguous/stride-1 only.

## Evidence classes and acceptance

- `EXACT_BITS`: F32/BF16 decode and operations whose independent reference
  matches every F32 bit.
- `EXACT_DISCRETE`: top-k indices and status/count results.
- `CALIBRATED_FLOAT`: libm/reduction results with a per-primitive maximum
  absolute, relative and ULP observation from a deterministic calibration
  corpus, a documented acceptance margin, and a disjoint holdout corpus.

No blanket tolerance exists. Task 1.4's frozen model-operator plan remains
unchanged; Task 2.5 evidence covers only this low-level contract.

## Deliberate exclusions

This contract does not define GDN, causal convolution, QSA selection or
attention, RoPE, Gated Residual, MoE routing/dispatch semantics, PLE value
lookup, LM-head execution, sampling, generation, a high-performance GEMM,
SIMD, CUDA numerics, cache, prefetch or scheduling.
