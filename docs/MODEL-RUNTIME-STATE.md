# Qwen3.8-Flash-Next runtime-state model

Status: **PINNED / VERIFIED** for Task 1.1

Scope: canonical ordinary text-only prefill and decode, with optional-state
boundaries recorded for later work.

This inventory distinguishes model weights, per-request persistent state, and
forward-call temporaries. Symbolic shapes use:

- `B`: batch or active-sequence count;
- `T`: cached context length;
- `N`: token count in the current prefill/chunk;
- `D = 2560`: text hidden width; and
- `R = 4`: Gated Residual branch count.

Detailed byte accounting, allocator policy and placement are Task 1.2 work.

## State inventory

| State | Owner/count | Lifetime | Growth | Symbolic shape | Source-backed dtype | Text-only required |
|---|---|---|---|---|---|---|
| GDN recurrent matrix | each of 36 GDN layers | request/context | fixed | `[B,48,128,128]` per layer | FP32 in pinned Tier B; config `mamba_ssm_dtype=float32` | yes |
| GDN causal-conv tail | each of 36 GDN layers | request/context | fixed | `[B,10240,4]` per layer | projection/activation dtype, normally BF16 | yes |
| QSA K cache | each of 12 QSA layers | request/context | grows with `T` | `[B,2,T,256]` | model activation dtype, normally BF16 | yes |
| QSA V cache | each of 12 QSA layers | request/context | grows with `T` | `[B,2,T,256]` | model activation dtype, normally BF16 | yes |
| QSA raw index keys | each of 12 QSA layers | request/context | grows with `T` | `[B,T,128]` | index projection dtype, normally BF16 | yes |
| GR residual branches | current layer-stack activation | current forward call | proportional to current `N`, not prior `T` | `[B,N,4,2560]` (often flattened to `[B,N,10240]`) | normally BF16; report supports later FP8 study | yes, transient |
| PLE token history | layer 2 PLE | request/context | fixed | `[B,2]` token IDs | integer token-ID type | yes |
| PLE dilated-conv tail | layer 2 PLE | request/context | fixed | `[B,10240,9]` | PLE activation dtype, normally BF16 | yes |
| PLE table rows | layer 2 lookup | per token/chunk | transient/address-driven | 16 rows × 160 values per token before concatenation | checkpoint/table dtype | yes |
| Position count | request | request/context | scalar per sequence | logical next position | integer | yes |
| Cached QSA position IDs | hybrid cache/reference implementation | request/context | grows with `T` | text-only equivalent of `[3,B,T]` | integer | reference-specific representation; equivalent position state required |
| QSA pooled block keys/scores/top-k/mask | each QSA call | current call | transient; derived from `T` | implementation-dependent | mixed: FP32 scoring in Tier B, masks bool/float | yes, transient |
| MoE route IDs/weights | each layer/current tokens | current sublayer | transient | `[B*N,10]` each | IDs integer; score path FP32 then activation dtype | yes, transient |
| MTP draft cache/state | optional MTP module | speculative request | depends on draft steps/context | one QSA/MoE draft layer plus input-fusion state | not fixed by Task 1.1 | no |
| Vision features/M-RoPE state | optional vision wrapper | multimodal request | input-dependent | vision-grid dependent | model dtype/integer positions | no |

Task 2.6 revalidated the two GDN rows directly against the pinned
`Qwen4ExpTextGatedDeltaNet` and `LinearAttentionLayer` cache implementation.
The convolution state is exactly the last four **pre-convolution projected
QKV** values per channel, not convolved output. Reset zeroes both state classes
and clears the initialized flag. The scalar operator performs a bounded
working-state scan and commits only after successful prefill/decode, so an
operator-local failure does not expose a partially advanced GDN state.

The released-model descriptor retains BF16 convolution storage and F32
recurrent storage. The independent reduced reference configuration uses F32
for both so its arithmetic can be isolated without the full BF16 checkpoint.
Observed batch-1 per-layer target storage is 3,227,696 owned bytes including
the state object/allocation overhead; the semantic payload remains 81,920
convolution bytes plus 3,145,728 recurrent bytes.

Task 2.7 revalidated all three QSA state rows against the pinned
`Qwen4ExpTextAttention`, `Qwen4ExpTextQSAIndexer` and `DynamicIndexedLayer`
path. At batch 1 and BF16, one context token consumes exactly 1,024 K bytes,
1,024 V bytes and 256 raw-index-key bytes per QSA layer: 2,304 bytes per layer
and 27,648 bytes across all 12 layers. The reference derives complete-block
summaries on demand, so there is no persistent block-summary cache or
partial-block buffer overhead in this semantic baseline. The native scalar
reference uses F32 storage only in its reduced executable correctness config;
the real target descriptor remains BF16.

Native QSA prefill/decode has explicit capacity and current length. It stages
all new K/V/raw-index entries and outputs before committing, so an operator
failure leaves externally visible state unchanged. This transactional baseline
does not define final cache allocation, sharing, crop, eviction or scheduling
policy.

Task 2.9 revalidated the PLE dilated-convolution tail independently of Task
2.4's token/address history. The value state is exactly the most recent nine
positions of normalized gated values, shape `[B,10240,9]`; kernel width 4 and
dilation 3 read history at `t-9`, `t-6` and `t-3` plus the current value. The
released semantic state is BF16 and occupies 184,320 bytes at batch 1. The F32
scalar-reference container occupies 368,640 bytes and is explicitly labeled
as a correctness representation, not released runtime storage. Task 2.9 stages
and commits this state transactionally across prefill/decode.

Claim sources: [KQ-ARCH-GDN-004], [KQ-ARCH-QSA-004],
[KQ-ARCH-GR-003], [KQ-ARCH-PLE-004], [KQ-ARCH-MTP-004].

## Ownership and mutation rules

### GDN state

Each GDN layer owns one independent recurrent matrix and one independent causal
convolution tail per active sequence. No GDN state is shared between layers.

During prefill, the layer starts from zeros (or a validated prefix state), scans
tokens causally, and commits only the final recurrent matrix and final four
convolution positions. During single-token decode it reads those two objects,
updates both once, and writes them back. Padding/masked tokens must not mutate
semantic state. Reordering or duplicating sequences for beam-like execution
requires the same operation on every layer's state.

The fixed recurrent shape is the reason GDN does not need a context-growing KV
cache. It does not mean the state is optional: dropping it changes all later
token outputs.

### QSA state

Each QSA layer owns conventional grouped-query K and V history plus one raw
index-key vector per context token. Prefill appends `N` entries to all three;
decode appends one. The raw index-key cache is separate from core K/V because it
has one 128-wide shared head instead of two 256-wide KV heads.

For a query, complete groups of four visible raw index keys are average pooled
and positioned, scored, and top-k selected. The pinned implementation derives
these block keys every call and persists only raw keys. A later derived-block
cache is permissible only if it defines ownership, maximum size, invalidation,
prefix sharing, crop/rollback and corruption handling and proves identical
selection.

The main KV/index arrays grow with context even though selected core attention
is capped at 2,048 complete-block tokens plus up to three tail tokens.

### GR branches

The GR state is the activation stream for the current tokens. Input embeddings
are replicated into four branches, every attention/GDN and MoE block reads all
branches and adds a gated write to every branch, and the final mixer reads the
branches down to one hidden vector.

These branches must survive across all 48 layers within one forward call. They
do not survive as context history after the call: on the next decode step, the
new token starts from four copies of its own embedding while historical effects
arrive through GDN and QSA caches. Treating GR branches as a context-growing
cache would be an architecture error.

### PLE state and prefetch boundary

PLE retains the previous two token IDs, with EOS acting as the left/segment
sentinel, so it can form the next bigram and trigram. It also retains nine past
positions for the dilation-3, kernel-4 depthwise convolution. These are distinct
cache entries from the GDN convolution tail even though the pinned reference
uses a common generic cache abstraction.

For prefill, every n-gram table address is known immediately after tokenization.
For decode, addresses for processing the newly selected input token are known as
soon as that token and the two-token history are available. The lookup rows can
therefore be prefetched before layer 2. The table-row cache, transfer batching,
eviction and synchronization policy remain deliberately undefined.

### Position state

Text-only execution needs monotonically increasing causal positions and the
partial-RoPE transform used by QSA and its indexer. The pinned multimodal wrapper
represents positions in a three-axis form and the text model additionally tracks
the scalar text position; for pure text all three axes are equivalent. Kestrel-Q
does not need to copy that container shape, but it must generate numerically
equivalent text rotary positions and maintain exact alignment with KV/index
cache length.

### MoE temporaries

Router logits, top-10 IDs and normalized weights are needed only while a layer's
MoE result is computed. They are not request-persistent after the weighted
expert outputs and gated shared-expert output are combined. Training-only
load-balancing accumulators are not inference state.

## Prefill state transition

Starting from an empty request:

```text
token IDs
  -> initialize PLE history with EOS sentinels
  -> initialize 36 GDN recurrent matrices/conv tails
  -> initialize 12 empty QSA K/V/index arrays
  -> scan 48 layers over the prompt
  -> commit final GDN states and PLE histories/tail
  -> retain all QSA K/V/raw-index entries
  -> retain next causal position
  -> discard GR, router, selected-index, score and mask temporaries
```

Chunked prefill must produce the same committed state as a single pass. A chunk
boundary may not reset PLE token history, PLE convolution history, GDN state or
position numbering, and QSA selection must see the complete visible cached
prefix.

## Single-token decode state transition

For one new input token:

```text
read next position + PLE two-token history
compute PLE addresses early
for each layer:
  carry current token's four GR branches
  GDN: mutate fixed conv/recurrent state, or
  QSA: append one KV/index entry and select from the visible prefix
  route/run MoE; update only current GR branches
collapse GR -> main hidden -> LM logits
commit PLE history/conv state and advance position
discard current-token GR and routing/QSA selection temporaries
```

Transactionally, a failed step must not expose partially advanced state. A
future runtime must either validate all operations before commit or provide a
rollback/crop path for GDN, QSA, PLE and position state together.

Task 2.10 implements that rule for one layer by retaining committed and staging
suboperator slots. Ordinary GDN semantic state is 3,227,648 bytes per batch-1
layer (FP32 recurrence plus BF16 convolution tail). QSA remains 2,304 bytes per
retained token per layer. The PLE-GDN layer adds the 32-byte address history and
184,320-byte BF16 PLE value tail for 3,412,000 semantic persistent bytes. GR
branches, gates, MoE routes and layer scratch are explicitly excluded.

Task 2.11 preserves the same state objects while replacing F32 test weights
with bounded target-quantized access. Real ordinary-GDN, QSA and PLE-GDN
prefill+decode runs advance position from zero to two. Provider-budget failures
before the first payload and after the GR stage leave the active slot, output
and position unchanged; the inactive staging slot is overwritten from the
committed slot on the next call. No weight-provider accounting or diagnostic
trace is persistent model context.

## Optional-state boundaries

### MTP

MTP is not required for ordinary next-token logits. If later enabled for
speculative decoding, it needs its own one-layer QSA/MoE execution state and a
clear relationship to target-model KV/index state and QSA index reuse. Exact
draft fusion/cache semantics remain a later, separately tested scope.

### Vision

Text-only requests need no vision features, grids, projector activations or
multimodal RoPE deltas. The initial runtime must reject image/video payloads and
visual placeholder tokens before state construction. Future multimodal support
must add its state explicitly rather than making the text cache polymorphic by
accident.

## Deferred to Task 1.2 or later

- exact byte totals and peak/transient accounting;
- allocation granularity and placement across VRAM, pinned/pageable RAM or SSD;
- cache capacity, prefix sharing, eviction and rollback data structures;
- PLE row-cache and asynchronous transfer scheduler;
- precision choices below the released/reference behavior;
- MTP speculative state implementation; and
- multimodal runtime state.
