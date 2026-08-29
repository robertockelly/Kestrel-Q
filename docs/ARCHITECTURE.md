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
