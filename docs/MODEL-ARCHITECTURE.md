# Qwen3.8-Flash-Next architecture baseline

Status: **PINNED / VERIFIED** for Task 1.1

Canonical model revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`

Canonical research revision: `69885871a64393807d988b27b1b5e380e8f28526`

This document characterizes the released Qwen3.8-Flash-Next model far enough
to guide later loader, reference-runtime and kernel work. It does not define a
Kestrel-Q tensor layout or implement inference. Canonical checkpoint artifacts
and Qwen specifications take precedence over implementation-reference naming.
The evidence behind each material statement is indexed by `KQ-ARCH-*` claim ID
in the Task 1.1 evidence file.

## Evidence boundary

Tier A evidence is the pinned official Hugging Face repository, pinned Qwen
research repository and technical report, plus the unversioned release article
as supporting context. Where Tier A does not expose exact execution mechanics,
this baseline uses Hugging Face Transformers at commit
`805a9e939fa8c1bff8d8ffdf041c051b71a914aa` as an Apache-2.0 Tier B
implementation reference. A pinned SGLang MTP implementation at commit
`9579bff86085f886cf6d1ec69349017d0caeced4` is used only as a Tier C
cross-check. No implementation code is copied into Kestrel-Q.

The released `config.json` labels every fourth layer `full_attention`.
Transformers deliberately normalizes that released-checkpoint spelling to
`qwen_sparse_attention`, and the technical report states that all backbone
full-attention positions were replaced by QSA during continued pretraining.
Therefore this document calls those 12 positions **QSA**, while retaining the
literal config label where useful. This is a terminology discrepancy, not a
different layer schedule. [KQ-ARCH-DISCREPANCY-001]

## Top-level model and text-only boundary

The released architecture is `Qwen4ExpForConditionalGeneration`, a causal
language model wrapped with a vision encoder and multimodal embedding merge.
The wrapper owns:

- a `Qwen4ExpModel` containing `visual` and `language_model` modules;
- the token embedding in the language model;
- image and video placeholder token IDs (`248056`, `248057`);
- a final untied LM head of 248,320 outputs; and
- optional MTP tensors stored separately under `mtp.*`.

For text-only input, token IDs are embedded, no image/video feature function is
called because both pixel-value arguments are absent, no embedding positions
are replaced, and the embeddings pass directly into the language model. Vision
execution occurs only under the respective non-null `pixel_values` or
`pixel_values_videos` branches. The language-model-only implementation also
explicitly ignores `model.visual.*` and `mtp.*` checkpoint keys. Kestrel-Q may
therefore implement canonical ordinary text logits without loading or executing
the vision path, provided it rejects visual inputs and visual placeholder tokens
instead of silently treating them as text. [KQ-ARCH-WRAPPER-001]
[KQ-ARCH-WRAPPER-002] [KQ-ARCH-SCOPE-001]

For supported multimodal execution, the wrapper obtains vision features,
checks that feature counts match image/video placeholder counts, and replaces
only those placeholder embedding positions. It also constructs multimodal
three-axis rotary positions. This path is outside the first Kestrel-Q inference
scope. [KQ-ARCH-WRAPPER-003]

## Text backbone

The text model has hidden width 2,560, padded vocabulary/input and output width
248,320, 48 decoder layers, SiLU activation, RMSNorm epsilon `1e-6`, BF16
checkpoint dtype, and a native maximum position setting of 262,144. Token
embeddings and the LM head are not tied. [KQ-ARCH-TEXT-001]

The model begins by repeating each token embedding into four residual branches.
Each decoder layer performs an attention/linear-attention sublayer and an MoE
sublayer; each sublayer has its own Gated Residual read and write. The PLE
augmentation is inserted immediately before the attention read in the second
one-indexed layer. After layer 48, a final Gated Residual read collapses the four
branches to one 2,560-wide hidden vector, then the LM head produces ordinary
next-token logits. [KQ-ARCH-TEXT-002] [KQ-ARCH-GR-002]

### Exact layer schedule

Layer numbers below are one-based for readers; checkpoint/config indices are
zero-based. Every fourth layer is a released `full_attention` position whose
actual released semantics are QSA. There are exactly 36 GDN and 12 QSA layers.
[KQ-ARCH-TEXT-003]

| Layer | Config index | Released config label | Execution |
|---:|---:|---|---|
| 1 | 0 | `linear_attention` | GDN |
| 2 | 1 | `linear_attention` | GDN + PLE injection |
| 3 | 2 | `linear_attention` | GDN |
| 4 | 3 | `full_attention` | QSA |
| 5 | 4 | `linear_attention` | GDN |
| 6 | 5 | `linear_attention` | GDN |
| 7 | 6 | `linear_attention` | GDN |
| 8 | 7 | `full_attention` | QSA |
| 9 | 8 | `linear_attention` | GDN |
| 10 | 9 | `linear_attention` | GDN |
| 11 | 10 | `linear_attention` | GDN |
| 12 | 11 | `full_attention` | QSA |
| 13 | 12 | `linear_attention` | GDN |
| 14 | 13 | `linear_attention` | GDN |
| 15 | 14 | `linear_attention` | GDN |
| 16 | 15 | `full_attention` | QSA |
| 17 | 16 | `linear_attention` | GDN |
| 18 | 17 | `linear_attention` | GDN |
| 19 | 18 | `linear_attention` | GDN |
| 20 | 19 | `full_attention` | QSA |
| 21 | 20 | `linear_attention` | GDN |
| 22 | 21 | `linear_attention` | GDN |
| 23 | 22 | `linear_attention` | GDN |
| 24 | 23 | `full_attention` | QSA |
| 25 | 24 | `linear_attention` | GDN |
| 26 | 25 | `linear_attention` | GDN |
| 27 | 26 | `linear_attention` | GDN |
| 28 | 27 | `full_attention` | QSA |
| 29 | 28 | `linear_attention` | GDN |
| 30 | 29 | `linear_attention` | GDN |
| 31 | 30 | `linear_attention` | GDN |
| 32 | 31 | `full_attention` | QSA |
| 33 | 32 | `linear_attention` | GDN |
| 34 | 33 | `linear_attention` | GDN |
| 35 | 34 | `linear_attention` | GDN |
| 36 | 35 | `full_attention` | QSA |
| 37 | 36 | `linear_attention` | GDN |
| 38 | 37 | `linear_attention` | GDN |
| 39 | 38 | `linear_attention` | GDN |
| 40 | 39 | `full_attention` | QSA |
| 41 | 40 | `linear_attention` | GDN |
| 42 | 41 | `linear_attention` | GDN |
| 43 | 42 | `linear_attention` | GDN |
| 44 | 43 | `full_attention` | QSA |
| 45 | 44 | `linear_attention` | GDN |
| 46 | 45 | `linear_attention` | GDN |
| 47 | 46 | `linear_attention` | GDN |
| 48 | 47 | `full_attention` | QSA |

## Gated DeltaNet

For a token position `t` and value head, let `q_t,k_t` have width 128,
`v_t` have width 128, and let recurrent state `S_t` have shape `128 x 128`.
The released model uses 16 projected Q/K heads and 48 value heads; each Q/K
head is repeated three times to align with the 48 value heads. [KQ-ARCH-GDN-001]

The canonical recurrence is:

```text
S_decay = alpha_t * S_(t-1)
error   = v_t - transpose(S_decay) * k_t
S_t     = S_decay + beta_t * k_t * transpose(error)
y_t     = transpose(S_t) * q_t
```

Equivalently, the update decays the old state, erases the component associated
with the current key according to `beta`, and writes the current value. `alpha`
and `beta` are data-dependent values in `(0,1)`. [KQ-ARCH-GDN-002]

One input projection produces Q, K and V channels. A depthwise causal
convolution of kernel width 4 and SiLU precedes the recurrence. Q and K are
L2-normalized. Independent input projections produce the beta logit and decay
parameter; the implementation evaluates the decay parameterization and the
recurrence in FP32. A separate `z` projection gates zero-centered RMS-normalized
per-head outputs with a sigmoid before the output projection. These operations
are semantics even when later fused into one kernel. [KQ-ARCH-GDN-003]

For batch `B`, each GDN layer persistently keeps:

```text
recurrent state: [B, 48, 128, 128], FP32 in the pinned implementation
conv state:      [B, 10240, 4], projection/checkpoint computation dtype
```

The recurrent dimensions are independent of context length. Prefill runs the
same causal recurrence over a sequence (the reference implementation chunks it
in groups of 64) and retains the final state and convolution tail. Cached
single-token decode updates the convolution state and recurrence once in place.
Chunking, triangular solves and fused per-step kernels are implementation
choices; the sequential recurrence above is the semantic reference.
[KQ-ARCH-GDN-004] [KQ-ARCH-GDN-005]

## Qwen Sparse Attention

Each QSA layer has 24 query heads, 2 KV heads and 256 dimensions per head. It
uses grouped-query attention with 12 query heads per KV head. Q and K are
RMS-normalized; 64 dimensions (partial rotary factor 0.25) receive RoPE. The Q
projection also produces a sigmoid output gate applied before the output
projection. [KQ-ARCH-QSA-001]

The lightweight indexer independently projects four 128-dimensional query heads
and one shared 128-dimensional key head. Its key sequence is divided into
non-overlapping blocks of four visible tokens. Complete blocks are average
pooled before normalization and partial RoPE; the block carries the starting
token position. For query `i`, each causally complete block receives the sum
over four heads of the ReLU-positive query/key dot products. The top 512 blocks
give at most 2,048 selected complete-block tokens. Tokens in the final
incomplete visible block are always appended. [KQ-ARCH-QSA-002]

The selected token set masks the ordinary causal grouped-query attention. It
does not change Q/K/V values or silently densify unavailable context. During
prefill, selection is computed independently for each query from only its
visible prefix. During decode, the new query scores all complete blocks in the
cached prefix and includes the current incomplete tail. [KQ-ARCH-QSA-003]

The persistent state for each of 12 QSA layers is context-growing:

```text
K cache:         [B, 2, T, 256]
V cache:         [B, 2, T, 256]
raw index keys:  [B, T, 128]
```

The pinned reference retains raw index keys and forms pooled block keys on
demand; it does not persist a second compressed-key cache. Selected indices,
pooled keys, scores and masks are per-call temporaries. A future optimized
runtime may cache derived block data only after proving equivalence and defining
invalidation. [KQ-ARCH-QSA-004]

The report's KL/distillation losses are training-only. They are not required to
produce inference logits. [KQ-ARCH-QSA-005]

## Gated Residual

GR widens the residual activation into four independent 2,560-wide branches.
For a sublayer read, each branch is RMS-normalized with its own gain. A low-rank
network of rank 320 consumes all normalized branches and emits an elementwise
sigmoid gate for every branch and channel. The gated branches are averaged to
produce the sublayer's 2,560-wide input. [KQ-ARCH-GR-001]

For a sublayer write, a separate projection consumes the normalized four-branch
state and produces one scalar per branch. Twice a sigmoid bounds each scalar,
and the sublayer output multiplied by that scalar is added to the corresponding
branch. There is no `H_res` branch-to-branch mixing matrix. Separate GR modules
surround the attention and MoE sublayers of every layer, and a read-only GR
module collapses the final branches before the LM head. [KQ-ARCH-GR-002]

Conceptually:

```text
read(branches):
    normalized = group_rmsnorm(branches)
    element_gate = sigmoid(up(silu(down(normalized) / 4)))
    return mean(element_gate * normalized, branch_axis)

write(branches, sublayer_output):
    branch_gate = 2 * sigmoid(write_projection(normalized) / 4)
    return branches + branch_gate * sublayer_output
```

The four branches are activations carried through the layer stack for the token
positions in the current forward call. They are not an additional
context-growing autoregressive cache. [KQ-ARCH-GR-003]

The technical report demonstrates that GR residual branches can be stored in
FP8 with almost no reported quality loss and describes fused read/write kernels.
The released checkpoint and Tier B reference use BF16 activations by default;
FP8 is therefore a source-backed optimization capability, not a canonical
precision assumption. Kestrel-Q needs an explicit later precision/parity gate
before adopting it. [KQ-ARCH-GR-004]

## Ultra-sparse MoE

Every decoder layer has the same MoE structure. A bias-free router maps each
2,560-wide token input to 512 logits. Softmax is evaluated in FP32, the top 10
experts are selected, and their selected probabilities are renormalized to sum
to one. Each routed expert is a bias-free SwiGLU MLP with intermediate width
640; its output is multiplied by the corresponding normalized routing weight,
and the ten weighted outputs are summed. [KQ-ARCH-MOE-001]

A separate shared SwiGLU expert also has intermediate width 640. A learned
scalar sigmoid gate modulates its output, which is added to the routed-expert
sum. Thus each token activates ten routed experts plus the shared expert.
[KQ-ARCH-MOE-002]

Router logits may be exposed for diagnostics/training. The auxiliary
load-balancing loss and coefficient `0.001` affect training loss only when
requested; ordinary inference uses the routing decision and weights but no
load-balancing loss. This document makes no claim about GGUF expert storage
order. [KQ-ARCH-MOE-003]

## N-gram embedding / PLE

The released model has one PLE at one-indexed layer 2 (config index 1), before
that layer's GDN. It uses n-gram size 3, eight heads per n-gram order, a
20,000,000 base head vocabulary, embedding width 2,560, and 128 published table
partitions. Bigrams and trigrams therefore supply 16 lookup heads of width 160
whose outputs concatenate to 2,560. [KQ-ARCH-PLE-001]

For each token, PLE forms bigram and trigram addresses from the current token
and up to two preceding tokens. Missing history and history across EOS segment
boundaries is replaced by the EOS token ID. For each order, token IDs are
multiplied by deterministic odd 64-bit per-position multipliers and combined
with bitwise XOR. Each of eight heads takes the result modulo its own successive
prime vocabulary size near 20 million, then adds that head's table offset.
The checkpoint's `layer_multipliers` tensor is canonical data; loaders must not
invent or silently change it. [KQ-ARCH-PLE-002]

The concatenated retrieved vector is projected to one key per residual branch
and one shared value. Normalized keys are dotted with normalized GR branches;
the signed square-root-transformed score passes through a sigmoid to gate the
shared value per branch. A group-normalized copy passes through a depthwise
SiLU convolution with kernel 4 and dilation 3. Direct and convolved values are
added, then the result is added to the four-branch residual before the layer-2
GR read. [KQ-ARCH-PLE-003]

PLE needs two token IDs of persistent history and nine positions of dilated
convolution state. All lookup addresses for a prefill prompt depend only on
already-tokenized IDs and can be calculated before layer execution. The report
explicitly places PLE at layer 2 so host-memory prefetch can overlap layer 1.
At decode, the address for the input token is known from that token and the two
saved predecessors, so the lookup is also deterministically prefetchable before
the layer-2 injection. Cache policy, batching and scheduling remain future work.
[KQ-ARCH-PLE-004] [KQ-ARCH-PLE-005]

## Multi-Token Prediction

The release records about 4B MTP parameters. Config and checkpoint inventory
show one MTP layer, a hybrid configuration whose single attention position is
QSA, no dedicated token embedding, and separate input-fusion weights/norms plus
a complete attention/GR/MoE layer. The report evaluates it as a four-step
speculative draft module and states that it reuses QSA top-k indices across
prediction steps. [KQ-ARCH-MTP-001] [KQ-ARCH-MTP-002]

The tensor names `fc_embedding`, `fc_hidden`, `pre_fc_norm_embedding` and
`pre_fc_norm_hidden`, together with the analogous pinned SGLang implementation,
show that the draft layer consumes a token embedding and a main-model hidden
state after separate normalization/fusion. Tier A does not fully specify the
exact released fusion arithmetic or the meaning of
`mtp_use_hidden_state_from_layer: null`; those details must be verified against
canonical tensors/reference outputs before MTP is implemented. This uncertainty
does not affect ordinary logits. [KQ-ARCH-MTP-003]

Ordinary autoregressive logits are produced by the main text model and LM head
without invoking MTP. The pinned Transformers causal-LM class explicitly ignores
`mtp.*` weights, while the report treats MTP as speculative decoding and reports
draft accepted length. MTP is therefore optional speculative acceleration, not
a dependency of canonical ordinary next-token logits. [KQ-ARCH-MTP-004]

## Prefill flow

For a text-only prompt:

1. Reject any visual input or image/video placeholder token.
2. Embed all token IDs and replicate embeddings into four GR branches.
3. Build causal positions/masks and initialize empty hybrid cache state.
4. At layer 1, GR-read, run GDN, GR-write, GR-read, route/run MoE, GR-write.
5. Before layer 2's GR read, compute deterministic bigram/trigram lookups,
   project/gate/convolve them and add them to the four branches.
6. Continue all 48 layers. GDN layers causally scan and retain only their final
   recurrent/conv state; QSA layers construct KV/raw-index caches and compute
   causally selected attention; every layer runs its MoE.
7. Collapse the four final branches through the read-only GR mixer.
8. Apply the LM head to required positions and return ordinary logits plus the
   persistent hybrid cache.

## Single-token decode flow

For the next known input token:

1. Embed the token and create four equal residual branches.
2. Advance position state and retain the prior two token IDs for PLE.
3. In each GDN layer, read GR, update the width-4 convolution tail, apply one
   recurrent delta update to the fixed matrix state, then GR-write.
4. At layer 2, compute/fetch the token's 16 PLE table rows, update the two-token
   history and nine-position PLE convolution state, and inject the result.
5. In each QSA layer, append one raw index key and one KV pair, score causally
   complete four-token blocks plus the trailing incomplete block, attend to the
   selected context, and GR-write.
6. Route the current token to ten experts plus the shared expert in every layer.
7. Collapse final GR branches and evaluate the LM head for the next-token
   distribution.

GDN work and state are fixed with respect to context length. QSA KV/index state
and block-scoring work grow with context (though selected core attention is
budgeted). PLE table addresses are the clearest future asynchronous host-memory
prefetch candidate. No scheduler or cache design is selected by Task 1.1.

## Implementation guardrails established by Task 1.1

- Fail closed on a layer schedule other than the exact 48-entry sequence.
- Treat released `full_attention` positions as QSA only when all QSA fields are
  present and valid; never silently fall back to dense attention.
- Preserve FP32 GDN recurrence semantics until lower-precision parity is proven.
- Load and validate checkpoint PLE multipliers and table partition metadata.
- Do not infer canonical tensor layout from the registered GGUF.
- Reject unsupported multimodal input and visual special tokens explicitly.
- Do not require or load MTP for the first ordinary text-generation path.
- Keep detailed tensor footprint/accounting in the completed Task 1.2 evidence
  and canonical/container mapping in the completed Task 1.3 evidence; neither
  document authorizes runtime implementation.
