# QSA Operator Contract

## Status and authority

This document is the Task 2.7 executable contract for the Qwen3.8-Flash-Next
QSA operator.  It was characterized from the official model revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` and the pinned Transformers
revision `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`.  The authoritative files are
`modeling_qwen4_exp.py`, `configuration_qwen4_exp.py`, `cache_utils.py`, the
pinned model `config.json`, and the generated independent Class-C evidence.
Generic sparse-attention descriptions are not an authority for this runtime.

The reference entry points are `Qwen4ExpTextQSAIndexer.forward`,
`Qwen4ExpTextAttention.forward`, `Qwen4ExpTextRMSNorm.forward`,
`Qwen4ExpTextRotaryEmbedding.forward`, `apply_rotary_pos_emb`, `repeat_kv`,
and `eager_attention_forward`.  Dynamic decode state is implemented by
`DynamicIndexedLayer.update`, `DynamicIndexedLayer.update_indexer`, and
`DynamicCache`.

## Target topology

The text backbone has 48 zero-based layers.  QSA is used by layers
`3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47`; the remaining 36 layers are
GDN.  The native QSA constructor accepts only these registry-validated QSA
layers.

The target geometry is:

| quantity | value |
|---|---:|
| hidden width | 2560 |
| query heads | 24 |
| key/value heads | 2 |
| key/value repeat | 12 |
| head width | 256 |
| query-plus-gate projection | `[12288, 2560]` |
| key projection | `[512, 2560]` |
| value projection | `[512, 2560]` |
| output projection | `[2560, 6144]` |
| query/key norm weights | `[256]` each |
| index query heads | 4 |
| index key heads | 1 |
| index head width | 128 |
| combined canonical index projection | `[640, 2560]` |
| physical index-query part | logical `[512, 2560]` |
| physical index-key part | logical `[128, 2560]` |
| index norm weights | `[128]` each |
| maximum positions | 262144 |

All projections are bias-free.  The GGUF represents the canonical
`index_qk_proj` as two ordered physical parts: index query first, index key
second.  This split is a binding fact, not a change in canonical semantics.

## Normalization and rotary position encoding

Core Q/K and index Q/block-K use `Qwen4ExpTextRMSNorm`.  For a vector `x` and
stored delta-weight `w`, it evaluates in FP32:

`y = x * rsqrt(mean(x*x) + 1e-6) * (1 + w)`

and casts the result back to the input activation dtype.  Norm weights are
initialized as deltas, not direct multiplicative weights.

Default RoPE uses base `10,000,000` and partial rotary factor `0.25`.  For the
target head width 256 this rotates the first 64 values; the remaining 192 are
unchanged.  The same 64-value cosine/sine vector is valid for the 128-wide
index heads.  Frequencies are FP32 and positions are absolute cache
positions.  Text-only positions use equal T/H/W axes, so interleaved MRoPE
reduces to the temporal position.  `rotate_half` splits the rotary prefix into
two equal halves and returns `[-second_half, first_half]`.

## Index projection, state, and block selection

For each input token, the combined index projection produces four 128-value
query heads followed by one raw 128-value key.  Index queries are normalized
and RoPE-rotated at the current absolute position.  Raw index keys are appended
to cache before selection and are stored before normalization and RoPE.

For each query, the four-dimensional causal mask supplies the visible token
indices in ascending order.  The initial native contract is batch one,
text-only, contiguous causal positions without padding.  A prefix of
`floor(visible_count / 4) * 4` tokens forms complete consecutive blocks.  Any
remaining zero to three tokens form the incomplete tail.

For each complete block:

1. average its four raw index keys in FP32, then cast to the activation dtype;
2. apply index K RMS normalization;
3. apply partial RoPE at the absolute position of the first token in the
   block;
4. compute four index-head dot products against the current rotated index
   queries in FP32;
5. apply ReLU to each dot product, sum the four values, and divide by
   `sqrt(128)`.

The candidate block IDs are `0..complete_block_count-1`.  Select at most 512
blocks (`2048 / 4`) by descending score.  The pinned CPU oracle's deterministic
tie result is ascending block ID; native selection therefore orders by
`(-score, block_id)`.  Candidate, selected-block, and selected-token ordering
are `EXACT_DISCRETE`.  Near ties are compared by their FP32 scores before the
block-ID tiebreak.

Selected token positions are emitted block by block in selected-block order,
with the four positions inside each block in ascending order.  The incomplete
tail is always appended afterward in ascending order.  There is no additional
mandatory local or recent window.  The selected-token mask is intersected with
the causal mask; causal exclusions can never be restored by selection.

## Core sparse attention

The query projection is viewed as `[query_head, 2 * head_width]` per token,
then split on its last dimension into query and sigmoid-gate halves.  This is
an interleaved per-head query/gate row layout, not one global Q matrix followed
by one global gate matrix.  Q and K are RMS-normalized and partial-RoPE rotated.
K and V are appended to state before attention.

K/V heads are repeated 12 times to the 24 query heads.  For every selected
token and head:

`logit = dot(Q, K) / sqrt(256) = dot(Q, K) / 16`

Softmax is evaluated in FP32 over the selected causal token set and cast back
to the query dtype before the weighted V sum.  Per-head contexts are flattened
to 6144 values, multiplied elementwise by `sigmoid(gate)`, then passed through
the output projection.  Dropout is zero in reference evaluation.  QSA owns no
residual addition or surrounding hyper-connection mixing.

## Prefill, decode, and state

The semantic state for batch one is:

- K cache `[2, T, 256]` in activation dtype;
- V cache `[2, T, 256]` in activation dtype;
- raw index-key cache `[T, 128]` in activation dtype;
- explicit current length/capacity and absolute next position.

Prefill processes queries causally in sequence order.  A one-token decode
projects and appends its raw index key, K, and V, selects against the resulting
causal prefix, emits output, then commits the new length.  Prefill followed by
decode must equal full causal recomputation at the same positions.  Reset sets
length to zero and invalidates no immutable configuration.  Native operations
are transactional: argument, numeric, capacity, selection, or workspace
failure leaves externally visible state and output unchanged.

No persistent block-summary cache or partial-block buffer exists in the
canonical dynamic cache: complete block summaries are recomputed from raw
index keys.  Consequently partial-block semantic payload overhead is zero.

At target BF16, per layer and token:

- K: `2 * 256 * 2 = 1024` bytes;
- V: `2 * 256 * 2 = 1024` bytes;
- raw index key: `128 * 2 = 256` bytes;
- total: `2304` bytes.

Across 12 QSA layers the exact semantic growth is `27,648` bytes/token.  Native
object bookkeeping and allocator capacity are representation overhead and are
reported separately.  The scalar executable reference uses F32 state; its
storage is twice the BF16 semantic payload and is not a target placement
decision.

## Numeric comparison policy

Candidate IDs, selected block IDs, selected token positions, counts, ordering,
causal exclusions, state lengths, and geometry are `EXACT_DISCRETE`.
Floating checkpoints are independently calibrated against the pinned oracle;
each evidence checkpoint records its maximum absolute, relative, and ULP
difference and the resulting `CALIBRATED_FLOAT` contract.  No Task 2.5 or GDN
tolerance is inherited.  Non-finite inputs, weights, or imported state fail
closed in the native baseline.

## Deliberate boundaries

Task 2.7 does not implement padding masks, batch sizes above one, quantized
weight execution, the final KV-cache allocator/scheduler, QSA SIMD/CUDA,
MoE, PLE values, a transformer layer, or model execution.  Target construction
validates physical types and bindings without reading payload.  These limits
are explicit rather than silently approximated.
