# GDN operator contract

Status: **TASK 2.6 COMPLETE / PASS — PINNED CONTRACT**

This document fixes the Qwen3.8-Flash-Next Gated DeltaNet (GDN) contract before
the native operator is implemented. It is model-specific. Older DeltaNet
papers, other Qwen generations and GGML execution conventions are not semantic
authorities for this contract.

## Provenance

Canonical model artifacts are pinned to
`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`.
Executable Class-C behavior is pinned to the Apache-2.0 Transformers revision
`805a9e939fa8c1bff8d8ffdf041c051b71a914aa`.

| Pinned source | SHA-256 |
|---|---|
| `modeling_qwen4_exp.py` | `91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f` |
| `configuration_qwen4_exp.py` | `26b47995740e3bc596b44b2011ee6c3d971d46136438b00dd5fad9557bec4254` |
| `cache_utils.py` | `4b284431cb3a881b6e6f8b8c6430df6f2efdcb3366a2484c7984ae88c612c61a` |
| `masking_utils.py` | `c159cd91c2a7fcafce04a8b6cbca55c320ce904b8ebf634383c97da5d9313ce3` |
| official pinned `config.json` | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` |

The canonical implementation names are
`Qwen4ExpTextGatedDeltaNet`, `Qwen4ExpTextRMSNormGated`,
`apply_mask_to_padding_states`, `causal_conv1d_fn`,
`causal_conv1d_update`, `l2norm`, `torch_chunk_gated_delta_rule` and
`torch_recurrent_gated_delta_rule`. The cache implementation is
`LinearAttentionLayer` inside `DynamicCache`.

The independent operator oracle instantiates the pinned module directly. Its
configuration accepts reduced positive dimensions while executing the same
functions and equations. Task 2.6 therefore uses a reduced F32 configuration
for bounded algorithm vectors and the real target only for structural binding
validation. No canonical BF16 checkpoint is downloaded.

## Target topology

Layer IDs are zero-based. GDN appears at every layer except `3, 7, 11, 15,
19, 23, 27, 31, 35, 39, 43, 47`, giving the exact GDN set:

```text
0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22,
24,25,26, 28,29,30, 32,33,34, 36,37,38, 40,41,42, 44,45,46
```

The remaining twelve layers are QSA and must be rejected by GDN construction.
The accepted production target has:

| Property | Target value |
|---|---:|
| batch supported by the scalar baseline | 1 |
| input/output hidden size | 2,560 |
| key heads / key head width | 16 / 128 |
| value heads / value head width | 48 / 128 |
| key projection width | 2,048 |
| value projection width | 6,144 |
| fused QKV/conv channels | 10,240 |
| key-to-value head repeat | 3 |
| convolution kernel width | 4 |
| recurrent state | `[B,48,128,128]` |
| convolution state | `[B,10240,4]` |
| RMS epsilon | `1e-6` |
| convolution activation | SiLU |
| output-gate activation | sigmoid |
| recurrent storage/accumulation | F32 |
| checkpoint activation/storage | BF16 in the released model |

The native reduced-shape reference configuration uses F32 activations. It is
accepted only for oracle cases and does not change ordering, equations, head
repeat, convolution width or state semantics.

## Required canonical tensors

For each GDN layer, all nine semantic bindings are mandatory:

| Semantic role | Canonical tensor | Target shape |
|---|---|---:|
| decay base | `linear_attn.A_log` | `[48]` |
| depthwise convolution | `linear_attn.conv1d.weight` | `[10240,1,4]` |
| time-step bias | `linear_attn.dt_bias` | `[48]` |
| alpha/time projection | `linear_attn.in_proj_a.weight` | `[48,2560]` |
| beta projection | `linear_attn.in_proj_b.weight` | `[48,2560]` |
| fused QKV projection | `linear_attn.in_proj_qkv.weight` | `[10240,2560]` |
| output gate projection | `linear_attn.in_proj_z.weight` | `[6144,2560]` |
| per-head gated RMS weight | `linear_attn.norm.weight` | `[128]` |
| output projection | `linear_attn.out_proj.weight` | `[2560,6144]` |

The released canonical tensors are BF16. Their GGUF bindings are not all
canonical-contiguous: alpha, beta, QKV, gate and output are transformed
layouts; convolution removes the singleton dimension; and `ssm_a` stores the
converter result `-exp(A_log)` rather than canonical `A_log`. The native layer
descriptor must record and validate these relations. It must never pretend
that transformed physical bytes are canonical row-major weights.

## Exact operation order

For input `x[B,T,hidden]` and optional two-dimensional padding mask `m[B,T]`:

1. If `m` is present, multiply `x` by `m[...,None]` in the activation dtype.
   The model mask factory omits this operation for an all-one mask and for a
   one-token decode. A multi-token cache continuation receives only the local
   trailing mask segment.
2. Apply four bias-free projections to the masked input:
   `mixed_qkv = W_qkv x`, `z = W_z x`, `b = W_b x`, `a = W_a x`.
3. Treat `mixed_qkv` as channel-first `[B,conv_dim,T]`. Apply causal depthwise
   convolution with one independent four-tap kernel per channel, zero history
   at reset and left zero padding. Apply SiLU after convolution.
4. Split the convolved channel order exactly as query `key_dim`, key
   `key_dim`, then value `value_dim`. Reshape to key/value heads.
5. Compute `beta = sigmoid(b)` in projection order.
6. Convert `A_log`, `a` and `dt_bias` to F32 and compute, in source order,
   `g = -exp(A_log) * softplus(a + dt_bias)`. Thus `g <= 0` for finite valid
   parameters.
7. Repeat each key/query head three consecutive times to match the 48 value
   heads.
8. Convert query, key, value, beta and `g` to F32. L2-normalize query and key
   over the 128 key features using
   `x * rsqrt(sum(x*x) + 1e-6)`. Scale query by `1/sqrt(128)`.
9. Execute the gated-delta recurrence in canonical token order.
10. Reshape each value head to 128 features. Apply gated RMS normalization:
    convert core output to F32; compute the mean square over 128 features;
    multiply by `rsqrt(mean_square + 1e-6)`; cast to the activation dtype;
    multiply by the learned norm weight; multiply by `sigmoid(z)` computed in
    F32; cast the result back to the activation dtype.
11. Concatenate 48 value heads and apply the bias-free output projection to
    produce `[B,T,2560]`.

The GDN operator owns none of the decoder residual or hyper-connection update.
`Qwen4ExpTextDecoderLayer` first obtains the attention hyper-connection input,
calls GDN, and then performs injection/residual composition outside GDN. PLE,
MoE and final residual-stream behavior are also outside this contract.

## Recurrent equation

For value head `h` at token `t`, with normalized repeated key/query vectors
`k_t`, `q_t`, value `v_t`, scalar `beta_t`, log decay `g_t` and previous F32
state `S` of shape `[key_dim,value_dim]`, the sequential semantic recurrence is:

```text
q_t = l2norm(q_t, eps=1e-6) / sqrt(key_dim)
k_t = l2norm(k_t, eps=1e-6)
S    = S * exp(g_t)
read = sum_key(S * k_t[:,None])
delta = (v_t - read) * beta_t
S    = S + k_t[:,None] * delta[None,:]
o_t  = sum_key(S * q_t[:,None])
```

All recurrence values and reductions are F32. Multiplication/addition order is
the explicit scalar order above; signed overflow does not apply. Inputs,
weights, state and intermediates must remain finite for the native reference.

Canonical multi-token prefill calls `torch_chunk_gated_delta_rule` with chunk
size 64; cached one-token decode calls
`torch_recurrent_gated_delta_rule`. The pinned source documents and implements
the same recurrence. The scalar Kestrel-Q reference deliberately executes the
sequential equation for every length, and the independent evidence verifies
both the pinned module output and sequential checkpoint representation under a
GDN-specific calibrated floating contract. Chunk implementation artifacts are
not a new model semantic.

## Convolution and cache transitions

The convolution state is the last four pre-activation projected QKV channel
values, not the convolved output.

- Reset initializes convolution and recurrent state to exact positive zero and
  marks the stream empty.
- First prefill left-pads projected values with zeros when `T < 4`, saves the
  last four projected values, and emits the first `T` causal convolution
  results.
- A multi-token continuation concatenates the previous four projected values
  with the new projected segment, saves the last four values, and emits only
  the new segment.
- Cached one-token decode shifts the four-value history, appends the new
  projected value, evaluates one depthwise dot and applies SiLU.
- The recurrence begins from the supplied F32 state or zeros, scans tokens in
  order and commits the final state only after successful execution.

Kestrel-Q's public failure policy is transactional for mutable stream state:
argument, shape, aliasing, finiteness and capacity checks occur before state
mutation; execution uses a bounded working state; failure leaves the caller's
state unchanged. Output contents are unspecified on failure.

## Padding, lengths and state boundary

The initial native baseline supports batch 1, non-empty contiguous sequences
and an optional byte mask containing only zero or one for multi-token prefill.
A one-token decode is never padding. There is no causal attention mask inside
GDN beyond causal convolution and recurrent scan order.

Sequence lengths 1, below/equal/above the four-token convolution history,
multi-token continuations and repeated reset/replay are defined. Zero length,
count overflow, invalid masks, non-finite values and incompatible state/config
pairs fail closed.

The GDN stream state contains only the convolution history, recurrent matrix,
initialization flag and compatibility identity needed by this operator. PLE
history/value state, QSA cache, GR state, residual streams and scheduler/cache
state are separate.

## Floating comparison policy

No blanket tolerance is inherited from Task 2.5. Task 2.6 generates a
deterministic calibration corpus and a disjoint holdout corpus from the pinned
module before native comparisons. Each checkpoint class records maximum
absolute, relative and ULP observations and an evidence-derived acceptance
limit. Discrete shape, layer, mask and state-status fields use
`EXACT_DISCRETE`; F32 arrays use `CALIBRATED_FLOAT` unless measured bit identity
supports `EXACT_BITS`.

Task 2.5's `/fp:strict` (or `-fno-fast-math -ffp-contract=off`) policy remains
in force for this translation unit. The native operator is scalar, performs no
SIMD/CUDA execution, uses no production Python/Transformers dependency and
does not access full model payloads.
