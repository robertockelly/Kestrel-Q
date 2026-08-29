# High-Level Task Backlog

Status convention:

- `[ ]` not started
- `[~]` active
- `[x]` complete
- `[!]` blocked

## Epic 0 — Foundations

- [ ] Freeze provisional project name or rename before public announcement
- [ ] Create public GitHub repository
- [x] Adopt Apache License 2.0 as the project license
- [ ] Define third-party dependency and NOTICE review policy
- [x] Task 0.1 — Record the exact KQ-01 reference hardware baseline
- [x] Task 0.2 — Complete the KQ-01 memory and bandwidth budget (0.2A–0.2E PASS)
- [x] Task 0.3 — Establish the Windows/CUDA toolchain and CUDA smoke baseline
- [ ] Configure formatting/static-analysis policy
- [ ] Add CI skeleton
- [x] Select authoritative reference runtimes for correctness vectors (Task
  1.4 two-oracle strategy; full-model captures remain future gates)

## Epic 1 — Qwen3.8-Flash-Next research

- [x] Task 1.0 — Model source & artifact baseline:
  - official source `KQ-MODEL-SOURCE-001`: **PINNED / VERIFIED**;
  - ADR 0004: **ACCEPTED**;
  - local GGUF `KQ-MODEL-ARTIFACT-001`: **REGISTERED / VERIFIED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 1.1 — Architecture characterization:
  - canonical 48-layer text execution, GDN, QSA, GR, MoE, PLE and MTP
    boundaries: **PINNED / VERIFIED**;
  - runtime-state and prefill/decode models: **DOCUMENTED**;
  - ADR 0005 initial text-only ordinary autoregressive scope: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 1.2 — Catalogue canonical tensor inventory and footprint:
  - bounded Range capture: **131 / 131 headers, zero weight payload bytes**;
  - canonical inventory: **1,658 / 1,658 tensors reconciled and classified**;
  - static, idealized-quantization and persistent-state footprint: **VERIFIED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 1.3 — Map canonical tensors to the registered derived GGUF:
  - canonical coverage: **1,658 / 1,658, UNRESOLVED = 0**;
  - GGUF coverage: **1,224 / 1,224, unexplained = 0**;
  - packed/type/family footprint and exact 434-count reconciliation: **VERIFIED**;
  - 384-byte split/merge delta: **PROVEN FORMAT OVERHEAD ONLY**;
  - ADR 0006 staged initial container strategy: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 1.4 — Establish reference behavior and golden vectors:
  - Class C: official pinned Qwen sources plus
    `transformers@805a9e939fa8c1bff8d8ffdf041c051b71a914aa`;
  - Class Q: exact registered GGUF plus
    `llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a`;
  - prompt suite and exact tokenizer/chat/PLE vectors: **GENERATED / VERIFIED**;
  - operator and full-model contracts: **PINNED**, with weight-dependent runs
    explicitly deferred to a capable reference environment;
  - ADR 0007 two-oracle strategy: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Document tokenizer/chat template (Task 1.4 exact goldens)
- [x] Document MoE topology and routing (Task 1.1)
- [x] Document Gated DeltaNet execution (Task 1.1)
- [x] Document Qwen Sparse Attention execution (Task 1.1)
- [x] Document gated residual mechanism (Task 1.1)
- [x] Document n-gram embedding mechanism (Task 1.1)
- [x] Estimate memory by tensor family and datatype (Task 1.2)
- [x] Identify preliminary placement-candidate tensor families (Task 1.2),
  including the unvalidated `PLE_DISK_BACKED_CANDIDATE`; scheduler policy
  remains deferred
- [ ] Define minimum useful quantization target

**Epic 1 status: COMPLETE / PASS.** All characterization exit gates are met;
deferred weight-dependent vectors remain mandatory before later canonical or
exact-GGUF runtime correctness claims.

## Epic 2 — Loader and introspection

**Implementation status: IN PROGRESS.** Task 2.0 is complete/pass; Task 2.1 has
not started.

- [x] Choose initial model storage/container strategy (ADR 0006 staged direct-GGUF-first path)
- [x] Task 2.0 — Native GGUF introspection and memory-mapped container layer:
  - Windows read-only file mapping and bounded logical views: **VERIFIED**;
  - target-first GGUF v3 metadata/tensor parser: **VERIFIED**;
  - seven observed storage types and packed-span validation: **VERIFIED**;
  - `kq-inspect`, deterministic malformed suite and real-artifact oracle:
    **PASS**;
  - normal inspection payload bytes accessed: **0**;
  - ADR 0008 native read-only GGUF layer: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Implement file mapping (Task 2.0 physical container layer)
- [x] Implement metadata parsing (Task 2.0 target-required subset)
- [x] Implement safe physical tensor lookup (Task 2.0)
- [x] Implement physical datatype/block registry (Task 2.0)
- [x] Implement inspection CLI (Task 2.0 summary mode)
- [x] Add malformed-artifact tests (Task 2.0)
- [ ] Task 2.1 — Canonical tensor registry and semantic descriptors

## Epic 3 — CPU correctness engine

- [ ] Tensor primitives
- [ ] Reference dequantization
- [ ] Tokenizer
- [ ] Embeddings
- [ ] Normalization
- [ ] Core block execution
- [ ] MoE routing
- [ ] Expert execution
- [ ] GDN/QSA path
- [ ] Logits
- [ ] Sampling
- [ ] Reference-vector validation

## Epic 4 — CUDA

- [ ] Backend C ABI
- [ ] Device discovery
- [ ] VRAM allocator
- [ ] Transfers
- [ ] Baseline CUDA kernels
- [ ] Correctness parity
- [ ] CUDA profiler instrumentation
- [ ] Initial performance baseline
- [ ] `KQ-BACKLOG-BENCH-001` — Measure WDDM CUDA allocation headroom before a
  production VRAM capacity policy depends on it:
  - status: **DEFERRED**;
  - trigger: **before production VRAM allocator/memory-placement capacity is set**;
  - measure bounded allocation behavior under representative WDDM load;
  - keep the result separate from Task 0.2D clean-idle capacity.

## Epic 5 — Memory tiering

- [ ] Tensor placement model
- [ ] Host pinned-memory pool
- [ ] Expert residency cache
- [ ] Async prefetch
- [ ] Eviction policy
- [ ] mmap weight access
- [ ] NVMe streaming experiments
- [ ] I/O/PCIe/compute overlap
- [ ] Adaptive scheduler
- [ ] Memory-pressure tests
- [ ] `KQ-BACKLOG-BENCH-002` — PLE Disk-Backed Access Benchmark:
  - status: **DEFERRED / REQUIRED BEFORE FINAL SCHEDULER DESIGN**;
  - purpose: validate whether KQ-01's storage hierarchy can support the
    `PLE_DISK_BACKED_CANDIDATE` efficiently;
  - measure cold and warm lookup latency; random, batched and
    predictive/prefetched lookup; mmap/page-fault behavior; physical bytes read
    per token; read amplification; RAM-cache hit ratio and working-set size;
    prefetch lead time; lookup-batching effects; SATA utilization; CPU overhead;
    and an NVMe comparison if/when a device is available;
  - distinguish OS page cache, explicit Kestrel-Q RAM cache and cold
    physical-storage reads;
  - this does not block Epic 2 loader/correctness work, but it **MUST** pass
    before final scheduler/residency policy is accepted.

## Epic 6 — Quantization

- [ ] Select first supported quantization
- [ ] Implement loader support
- [ ] Implement CPU dequant/reference path
- [ ] Implement CUDA kernels
- [ ] Mixed-precision experiments
- [ ] Quality evaluation
- [ ] Speed/memory matrix

## Epic 7 — Context and sessions

- [ ] State/KV architecture analysis
- [ ] Memory accounting
- [ ] Host offload
- [ ] Persistence format
- [ ] Save/restore validation
- [ ] Long-context benchmarks

## Epic 8 — User-facing runtime

- [ ] CLI
- [ ] Streaming generation
- [ ] Config profiles
- [ ] Local server
- [ ] API compatibility subset
- [ ] Windows packaging
- [ ] Diagnostic report command

## Epic 9 — Native coding agent

- [ ] Tool-call protocol
- [ ] File read/search tools
- [ ] Repository context handling
- [ ] Persistent agent sessions
- [ ] Safe command execution
- [ ] Agent-specific benchmarks

## Epic 10 — Community and release

- [ ] Contribution workflow
- [ ] Benchmark submission format
- [ ] Hardware compatibility matrix
- [ ] Issue/PR templates
- [ ] Security reporting
- [ ] Release checklist
- [ ] `KQ-BACKLOG-CUDA-001` — Define portable CUDA binary strategy before first public binary release:
  - status: **DEFERRED**;
  - trigger: **before first public binary release**;
  - evaluate per-machine `native` builds;
  - evaluate multi-architecture fat binaries;
  - define minimum supported NVIDIA compute capabilities;
  - define cubin/PTX fallback policy;
  - measure binary-size/startup/performance trade-offs;
  - document the selected release policy in an ADR.
- [ ] First public alpha
