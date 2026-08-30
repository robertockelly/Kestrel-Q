# PLE value operator contract

Status: **TASK 2.9 CHARACTERIZATION COMPLETE — PINNED CONTRACT**

This document fixes the Qwen3.8-Flash-Next PLE value-path semantics before the
native implementation. Task 2.4 remains the authority for address generation;
this operator consumes its ordered logical member/row intents without deriving,
reordering, or replacing them.

## Provenance

Canonical artifacts are pinned to
`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`.
Executable Class-C behavior is pinned to the Apache-2.0 Transformers revision
`805a9e939fa8c1bff8d8ffdf041c051b71a914aa`.

| Source | SHA-256 |
|---|---|
| `modeling_qwen4_exp.py` | `91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f` |
| `configuration_qwen4_exp.py` | `26b47995740e3bc596b44b2011ee6c3d971d46136438b00dd5fad9557bec4254` |
| official `config.json` | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` |

The canonical modules are `Qwen4ExpTextNGramEmbedding` and
`Qwen4ExpTextPLELayer`; the value execution is
`Qwen4ExpTextPLELayer.forward` plus `_short_conv`. The surrounding placement is
`Qwen4ExpTextDecoderLayer.forward`: the returned PLE tensor is added to the four
residual streams immediately before layer 1 zero-based's attention/GDN gated-
residual read. PLE does not own that later Gated Residual or GDN execution.

## Target geometry and semantic bindings

The only PLE is decoder layer 1 zero-based (one-based layer 2), a GDN layer.
Orders 2 and 3 have eight heads each. Task 2.4 emits the canonical order:
bigram heads 0–7 followed by trigram heads 0–7. Each intent selects one
160-wide row. The 16 rows concatenate in that order to a 2,560-wide embedding.

The 128 canonical logical members each have 2,500,012 rows. They are fused in
the registered GGUF as one physical `[160,320001536]` IQ4_NL tensor. The 16
active head ranges address 320,001,446 rows; the final 90 padded rows are not
canonical addresses.

The six dense semantic bindings are:

| Stable ID | Canonical role and shape |
|---|---|
| `layer.01.ple.key` | bias-free key projection `[10240,2560]` |
| `layer.01.ple.value` | bias-free value projection `[2560,2560]` |
| `layer.01.ple.norm_key` | zero-centered Group-RMSNorm delta `[10240]` |
| `layer.01.ple.norm_query` | zero-centered Group-RMSNorm delta `[10240]` |
| `layer.01.ple.norm_conv` | zero-centered Group-RMSNorm delta `[10240]` |
| `layer.01.ple.conv` | depthwise convolution `[10240,1,4]` |

The three norm and convolution tensors use explicit transformed-layout
bindings in the GGUF. Their physical F32 representation must not be silently
treated as the canonical BF16 tensor layout. Key/value are Q8_0 physical
bindings. The table is IQ4_NL. Structure validation and value execution remain
separate: scalar reference tests receive canonical F32 weights, while bounded
real-row plumbing decodes only IQ4_NL table rows.

## Exact value equations

Let `E_t` be the 2,560-wide concatenation of the 16 lookup rows and let
`H_t[b]`, `b=0..3`, be the four 2,560-wide residual branches. All linear
layers are bias-free.

```text
K_t = linear(E_t, key_proj)       # 4 x 2560
V_t = linear(E_t, value_proj)     # 2560
Kn_t[b] = group_rmsnorm(K_t[b], norm_key[b], epsilon=1e-6)
Qn_t[b] = group_rmsnorm(H_t[b], norm_query[b], epsilon=1e-6)

g_t[b] = dot(Kn_t[b], Qn_t[b]) / sqrt(2560)
g_t[b] = sign(g_t[b]) * sqrt(max(abs(g_t[b]), 1e-6))
G_t[b] = sigmoid(g_t[b]) * V_t
C_t    = group_rmsnorm(flatten(G_t), norm_conv, epsilon=1e-6)
```

With kernel coefficients `W[c,0..3]`, kernel width 4 and dilation 3, the
causal depthwise convolution is cross-correlation in canonical PyTorch order:

```text
D_t[c] = W[c,0]*C_(t-9)[c] + W[c,1]*C_(t-6)[c]
       + W[c,2]*C_(t-3)[c] + W[c,3]*C_t[c]
Y_t[c] = G_t[c] + silu(D_t[c])
```

Missing history is zero. The return shape is `[B,T,4,2560]` when viewed as
residual branches (the canonical module returns the flattened final axis).
There is no PLE-owned residual add: the decoder layer adds `Y` to its existing
four branches.

The released path uses the activation/checkpoint dtype for lookup, linear,
norm, convolution and output tensors. The executable scalar oracle and native
reference use F32 with source-order reductions and no fast math. This does not
claim that the quantized artifact is numerically identical to canonical BF16.

## Value state, prefill and decode

The value state is the most recent nine positions of **normalized gated value
`C`**, shape `[B,10240,9]`. It is cache `conv_states[1]`. It is separate from
Task 2.4's 32-byte token/address state (`conv_states[2]` in the pinned cache).
For batch 1 the semantic released-state footprint is 184,320 BF16 bytes; the
F32 scalar-reference state is 368,640 bytes.

Prefill processes tokens in sequence order. Before convolution, each current
`C_t` is appended; the retained state after a call is the last nine values,
left-zero padded when fewer exist. One-token decode uses the same transition.
The output for the current token reads retained positions corresponding to
`t-9`, `t-6`, `t-3`, then current `t`. Reset zeros all nine positions.
Failure in Kestrel-Q is transactional: externally visible value state is not
changed unless every lookup and numeric step succeeds.

The optional canonical `conv_mask` zeros both `G` and `C` at padding
positions before convolution. Task 2.9's initial public scalar API is batch 1
and accepts only unpadded valid tokens; unsupported padding/masking fails
closed rather than being ignored. Batched execution is deferred.

## Address-provider boundary

For every token the operator requires exactly 16 intents with matching
position, global-head sequence 0..15, order/head relation, member range and row
range. Lookups occur in emitted order. The provider receives only
`logical_member + member_row` and returns exactly 160 F32 values. It owns
storage policy; the scalar operator does not cache or prefetch.

The synchronous real provider opens the corresponding Task 2.2 fused-member
view, decodes five IQ4_NL blocks (90 packed bytes) through Task 2.5, then closes
the view. It never hard-codes a file offset and never maps/dequantizes the full
logical table into an owned buffer.

## Comparison and scope

Intent/member/row/count/order fields are `EXACT_DISCRETE`. Floating lookup,
projection, norm, gate, convolution, output and state checkpoints have separate
Task 2.9 calibration-only contracts applied unchanged to disjoint holdout.
Expected values are generated from the pinned module before native comparison.

This contract does not authorize a PLE disk cache, async I/O, prefetch,
scheduler/residency policy, complete transformer layer, multi-layer/full-model
execution, LM head, sampling, SIMD, or CUDA model kernels.
`KQ-BACKLOG-BENCH-002` remains deferred.
