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
- [ ] Select authoritative reference runtime for correctness vectors

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
- [ ] Document tokenizer/chat template
- [x] Document MoE topology and routing (Task 1.1)
- [x] Document Gated DeltaNet execution (Task 1.1)
- [x] Document Qwen Sparse Attention execution (Task 1.1)
- [x] Document gated residual mechanism (Task 1.1)
- [x] Document n-gram embedding mechanism (Task 1.1)
- [x] Estimate memory by tensor family and datatype (Task 1.2)
- [x] Identify preliminary placement-candidate tensor families (Task 1.2;
  scheduler policy remains deferred)
- [ ] Define minimum useful quantization target

## Epic 2 — Loader and introspection

- [x] Choose initial model storage/container strategy (ADR 0006 staged direct-GGUF-first path)
- [ ] Implement file mapping
- [ ] Implement metadata parsing
- [ ] Implement safe tensor lookup
- [ ] Implement datatype registry
- [ ] Implement inspection CLI
- [ ] Add malformed-artifact tests

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
