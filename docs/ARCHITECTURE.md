# Architecture — Initial Direction

This document describes the intended shape, not a frozen implementation.

## 1. Architectural constraint

Kestrel-Q is a **model-specific runtime**.

The architecture should expose model-specific concepts directly when doing so simplifies execution or improves performance. Generic abstraction is not a goal by itself.

## 2. Layers

```text
+------------------------------------------------------+
| CLI / Server / Agent                                 |
+------------------------------------------------------+
| Session / prompt / sampling                          |
+------------------------------------------------------+
| Qwen3.8-Flash-Next model execution                   |
| routing | GDN | QSA | residual | n-gram embedding    |
+------------------------------------------------------+
| Scheduler / tensor placement / cache                 |
+----------------------+-------------------------------+
| CPU reference        | CUDA backend                  |
+----------------------+-------------------------------+
| RAM / pinned RAM     | VRAM                          |
+----------------------+-------------------------------+
| mmap / NVMe streaming                                |
+------------------------------------------------------+
```

## 3. Core modules

### Model loader

Responsibilities:

- validate model identity/version;
- parse metadata;
- map tensor files;
- expose typed tensor descriptors;
- reject unsupported layouts.

Task 2.0 establishes the first production boundary beneath the model loader: a
Windows-native read-only file/view layer and target-first GGUF v3 parser. It
exposes immutable physical metadata and tensor descriptors without payload
pointers.

Task 2.1 adds the separate Qwen3.8-Flash-Next semantic registry. The
`qwen4exp` adapter validates the complete target topology and maps all initial
text semantics to immutable stable IDs with explicit renamed, transformed,
split, fused and metadata-derived bindings. Future GDN/QSA/MoE/PLE code consumes
these IDs rather than raw GGUF names. The registry retains no payload pointers,
performs no allocation/scheduling and fails on every unknown physical tensor.

Task 2.2 adds the first bounded payload boundary. Immutable quantized tensor
views resolve semantic bindings into exact read-only physical block spans while
retaining canonical shape, physical shape, split/fused/member identity and
explicit transformed-layout metadata. Whole tensors, ordered split parts,
contiguous routed-expert members and individual fused PLE members are supported
without copies or dequantization. Mapping remains a storage primitive, not an
allocator, cache, prefetcher or scheduler policy.

Task 2.3 adds the immutable native tokenizer and separate model-specific chat
formatter. The registered GGUF supplies exactly validated vocabulary/ID and
ordered merge substrate, while a governed Qwen3.8 adapter supplies pinned NFC,
marks-excluding byte-level BPE, canonical special/BOS/EOS semantics and the
supported official chat-template subset. The original GGUF-only insufficiency
finding remains explicit; no execution layer may trust divergent fields or
substitute a sidecar. ADR 0011 accepts this model-specific boundary. The
tokenizer borrows bounded GGUF metadata, contains no global mutable state and
does not open tensor payload views.

Task 2.4 adds the immutable model-specific PLE address configuration and
explicit bounded stream state. Canonical token IDs produce 16 ordered logical
member/row intents per position using the pinned bigram/trigram arithmetic.
The 32-byte address state tracks only position and two prior raw token IDs;
future PLE value/dilation state remains separate. Intents contain no physical
offset, pointer, cache slot or scheduling policy. Construction validates the
semantic registry and its metadata-derived arrays, and address generation
opens no tensor view or storage path. ADR 0012 accepts this prefetch-consumable
semantic boundary while leaving all I/O/cache/prefetch policy deferred.

Task 2.5 adds a separate scalar CPU numeric boundary above bounded Task 2.2
views. It decodes F32, BF16, Q5_1, Q8_0, Q4_K, Q5_K and IQ4_NL blocks to F32,
supports an explicit physical-order versus canonical-order view contract and
provides a block-by-block quantized row dot with fixed 1,024-byte scratch. The
generic F32 helpers are compiled without fast math or contraction and validated
against pinned independent calibration/holdout evidence. The layer is a future
optimized-kernel oracle, not a model graph: no GDN/QSA/GR/MoE/PLE value
operator, full forward pass, scheduler or CUDA model kernel is present.

Task 2.6 adds the first model-specific execution boundary: a scalar C17
Qwen3.8 GDN reference operator. Immutable layer descriptors validate all nine
canonical GDN semantics and preserve transformed GGUF relations; explicit
per-stream convolution/recurrent state supports transactional batch-1 prefill,
one-token decode and reset. A released-model BF16 state descriptor remains
separate from the executable F32 correctness descriptor. Independent pinned
Class-C calibration/holdout/state vectors validate intermediate checkpoints,
final output and both state classes. The real artifact validates all 36 GDN
layers and rejects 12 QSA layers without opening a payload view. GR composition,
QSA, MoE, PLE value math, full layers and optimized kernels remain absent.

Task 2.7 adds the scalar C17 QSA reference boundary. Immutable descriptors
validate the nine canonical attention/indexer semantics and the ordered
physical `index_qk` split. Explicit bounded stream state owns semantic K, V and
raw index-key history; prefill and one-token decode stage output/state and
commit only after success. Sparse candidate and selected block/token IDs are
observable exact-discrete outputs, while independently calibrated floating
checkpoints cover projections, RoPE, scores, attention and final output. The
real artifact validates 12 QSA layers and rejects 36 GDN layers without reading
payload. This is not a production cache scheduler or optimized attention path.

Task 2.8 adds a stateless scalar C17 MoE reference boundary. Immutable
descriptors validate router, 512-expert routed gate/up/down stacks, ordered
physical gate/up splits, the separate shared expert and scalar shared gate on
all 48 text layers. The operator exposes exact top-10 IDs/order and synchronous
floating checkpoints, executes only selected experts, and combines the routed
sum with the independently gated shared output. Independent reduced
calibration/holdout and 512/top-10 routing vectors define correctness. Real
integration opens bounded expert-member views without dereferencing payload;
it adds no cache, residency, prefetch, scheduler or optimized MoE kernel.

Task 2.9 adds the scalar C17 PLE value reference boundary. It consumes Task
2.4's exact ordered logical member/row intents, resolves rows through a
storage-neutral provider and keeps a separate explicit nine-position value
history. The reference prefill/decode path validates projection, group RMS
normalization, gating, dilated depthwise convolution and final combination
against pinned independent Class-C calibration, holdout and state evidence.
The real GGUF provider uses Task 2.2 member views and Task 2.5 IQ4_NL decode
under a hard logical-byte budget. It is synchronous correctness plumbing, not
a final disk cache, prefetch, residency or scheduling policy.

Task 2.10 adds the first complete one-layer scalar reference boundary. It
implements the exact four-branch Gated Residual reads/writes owned by the
decoder layer and composes PLE-at-layer-1, GDN/QSA and MoE in pinned canonical
order. Persistent GDN/QSA/PLE state executes through a committed/staging pair;
the active slot changes only after the final GR write, so downstream failure
cannot partially advance a request. All 35 ordinary-GDN, 12 QSA and one
PLE-GDN target configs validate without payload access. This remains one-layer
correctness plumbing, not an embedding/model-loop/logits executor or scheduler.

Task 2.11 adds the semantic target-weight provider and executes those same
layer equations directly from the verified quantized GGUF. The provider hides
physical row orientation, GDN converter transforms, QSA/MoE split parts,
contiguous routed-expert members and fused PLE rows behind canonical
operations. Each operation uses Task 2.2 bounded views and Task 2.5 numerics,
stages output transactionally and materializes at most a small vector—not a
complete matrix—in F32. Independent llama.cpp decoding plus pinned Transformers
equations validates ordinary GDN, QSA and PLE-GDN prefill/decode, exact MoE/QSA/
PLE decisions and 48/48 provider preflight. This is synchronous correctness
plumbing; it is not a cache, prefetcher, scheduler, 48-layer executor or model
entry/logits path.

### Tensor runtime

Small set of operations required by the target model.

Do not attempt to recreate a general tensor framework.

### Model executor

Contains the explicit Qwen3.8-Flash-Next forward path.

The executor should make expert routing and model-specific state visible to the scheduler.

### Scheduler

The scheduler decides:

- what must remain in VRAM;
- what can remain in host RAM;
- what may be streamed;
- what should be prefetched next;
- when transfers occur relative to compute.

The scheduler is expected to become one of the project's defining components.

### CPU backend

Purpose:

- correctness oracle;
- diagnostics;
- low-level testing;
- fallback for operations not yet implemented on CUDA.

It is not initially expected to provide practical full-model performance.

### CUDA backend

Purpose:

- accelerated inference on NVIDIA hardware;
- async transfer/compute overlap;
- model-specific fused kernels where justified.

CUDA source may require C++, but the boundary exposed to the runtime should remain C-compatible.

### Storage layer

Responsibilities:

- memory mapping;
- aligned access;
- optional direct/unbuffered experiments;
- streaming metrics;
- persistent session/state storage later.

The initial implemented storage primitive uses `CreateFileW`,
`CreateFileMappingW` and bounded `MapViewOfFile` logical views. Windows
allocation-granularity alignment is internal to the view object; checked
64-bit logical offsets and lengths remain visible to callers. Task 2.0 maps
header/directory address space read-only and does not intentionally dereference
tensor payload during normal inspection.

## 4. Memory hierarchy

The runtime should think in tiers:

### Tier 0 — VRAM

For:

- hot activations;
- current compute tiles;
- selected experts;
- frequently reused compact state.

### Tier 1 — pinned host RAM

For:

- transfer staging;
- prefetched experts;
- tensors that benefit from predictable PCIe DMA.

### Tier 2 — pageable/mapped RAM

For:

- broader model working set;
- cold tensors;
- metadata.

### Tier 3 — NVMe

For:

- model tensors that need not remain resident;
- persistent state;
- potentially large embedding structures.

The runtime must measure traffic between tiers.

### Preliminary KQ-01 placement hypothesis — validation pending

The following is a hypothesis to test, not an accepted scheduler or residency
policy:

- PLE/N-gram has the role `PLE_DISK_BACKED_CANDIDATE`: use a disk-backed or
  memory-mapped primary store, a bounded explicit hot page/row cache in RAM,
  predictive asynchronous prefetch from the deterministically known addresses,
  and only actively materialized lookup results or required working data in
  VRAM. Full RAM residency is neither required nor desirable on KQ-01.
- Routed experts are candidates for a bounded active cache in RAM, a hot subset
  in VRAM and cold backing on disk.
- Uncontrolled Windows paging is not an implementation mechanism. Any mapped
  path must expose and measure the OS page cache separately from Kestrel-Q's
  explicit RAM cache and cold physical-storage reads.

The hypothesis follows from the registered GGUF's approximately 26.855 GiB PLE
and 71.729 GiB routed-expert footprints versus the planned approximately 20 GiB
managed host and 8 GiB VRAM budgets. KQ-01 measured approximately 0.44 GB/s SATA,
25.63 GB/s RAM and 26 GB/s large pinned PCIe transfers, while Task 1.1 established
that PLE addresses are available deterministically early enough to permit
prefetch. These facts establish the need for a measured tiering experiment;
they do not validate disk-backed PLE performance. `KQ-BACKLOG-BENCH-002` is the
required validation before final scheduler/residency policy is accepted.

## 5. Expected specialization opportunities

Potential areas to investigate:

- expert-aware prefetch based on routing;
- expert cache keyed by recent activation;
- layered VRAM residency;
- host-resident n-gram embeddings with overlapped lookup/prefetch;
- sparse-attention-specific state management;
- mixed quantization based on tensor sensitivity;
- batching transfers across consecutive operations;
- persistent session state to avoid unnecessary recomputation.

These are hypotheses, not accepted optimizations.

## 6. Threading direction

Likely execution roles:

- main inference/control thread;
- storage/prefetch worker(s);
- optional CPU compute pool;
- CUDA stream(s).

Threading must be driven by profiling. Avoid building a complex executor before observing real stalls.

## 7. Error handling

Core rules:

- explicit error codes;
- no silent fallback when it changes numerical behavior;
- model incompatibility must be diagnostic;
- corrupted persistent state must be rejected;
- all sizes/offsets from model files must be bounds checked.

## 8. Public API direction

Initial core API should remain small.

Example conceptual boundary:

```c
kq_model_open(...)
kq_context_create(...)
kq_prefill(...)
kq_decode(...)
kq_sample(...)
kq_context_save(...)
kq_context_load(...)
```

Names are placeholders. API stability is not promised before the first alpha.

## 9. Architecture documents

Major changes require an ADR under `docs/adr/`.

## 10. M1 model-execution boundary

Task 2.12 adds a model-specific C17 orchestration layer above the accepted
tokenizer, semantic provider and reference layers. `kq_model_exec_config`
validates/borrows the exact target owners and 48 immutable layer configs;
`kq_model_exec_state` owns explicit context-bounded layer state. `kq-run`
exposes only raw-text, one-token, greedy M1 execution.

The executor performs bounded embedding rows, layers `0..47`, the final
qwen4exp hyper-connection mixer, all LM-head row dots, full finite-logit
argmax and native decode. Converter-folded gamma restoration belongs to the
physical-to-canonical weight-provider boundary and occurs exactly once.
Failures do not publish outputs or position and force a complete private state
reset before retry.

Task 2.13 extends the same model-specific boundary with true incremental greedy
decode. Prefill runs once; each selected canonical token ID passes directly
through one-token layer decode, final mixing, logits and native token decode.
The model position and GDN, QSA, PLE-address and PLE-value state advance one
position per successful token. A failed later step rolls completed layers back
to their preceding transaction slots and publishes neither outputs nor model
position; retry from that state reproduces the oracle token. Provider metrics
and traces remain monotonic audit data and are not rolled back.

Task 3.0 adds `kq_sampling` as a deliberately separate CPU-only boundary. It
accepts exactly 248,320 complete finite F32 logits, an immutable validated
Qwen3.8 sampling policy, caller scratch and caller-owned versioned PCG32 state.
It applies the pinned temperature, top-k and top-p processor order and returns
one canonical token ID plus bounded diagnostics. The sampler owns no model,
GGUF, tokenizer, stop condition or executor state, and it cannot access tensor
payload. Config objects are shareable read-only; RNG state is mutable only by
its caller and commits after successful canonical-ID selection.

This separation makes processor math and randomness independently testable and
keeps the accepted greedy argmax unchanged. Task 3.1 composes the two accepted
boundaries through `kq_model_exec_sampled_prefill_f32` and
`kq_model_exec_sampled_decode_one_f32`. The executor stages a private copy of
the caller's RNG beside the existing per-token model transaction; model state,
RNG state and outputs commit only after successful selection and canonical-ID
decode. Failed model, sampler or padded-ID paths publish neither state.

The sampled CLI requires an explicit seed, offers an optional bounded stream
and never creates hidden session/RNG state. EOG is returned once and cannot be
fed to sampled decode. Temporary real full logits exist only in ignored
research capture; a separate evidence tool reprocesses them and verifies exact
survivor/order hashes, PCG transition and selected ID without becoming a
production dependency.

This is not a scheduler or product-session architecture. Its repeated scalar
logical weight touches are correctness instrumentation, not physical I/O.
Context is an explicit bounded caller choice (16 for the governed Task 2.13
and Task 3.1 cases). Greedy argmax and the accepted Task 3.0 default sampling
profile are the two integrated selection policies. Batching, additional
sampling processors, context serialization, vision, MTP, cache/prefetch, SIMD
and CUDA model kernels remain outside the boundary.
`KQ-BACKLOG-BENCH-002` is still required before final disk-backed
PLE/residency policy.
