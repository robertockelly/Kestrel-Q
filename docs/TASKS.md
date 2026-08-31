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

**Implementation status: IN PROGRESS.** Tasks 2.0–2.12 are complete/pass. The
native runtime now includes independently validated scalar operators, complete
target-quantized layer composition and M1's short-context 48-layer first-token
path. The governed Epic 2 plan does not define M1 as its final closure gate;
no post-M1 task has started.

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
- [x] Task 2.1 — Canonical tensor registry and semantic descriptors:
  - target `qwen4exp` identity and 48/36/12, 512-expert/top-k-10 topology:
    **VERIFIED**;
  - initial-text semantic registry: **1,294 / 1,294**;
  - physical GGUF coverage: **1,224 / 1,224**, unknown = 0;
  - metadata-derived PLE semantics: **3**, unbound required = 0;
  - complete Epic 1 mapping-oracle comparison and deterministic native dump:
    **PASS**;
  - semantic inspection payload bytes accessed: **0**;
  - ADR 0009 canonical semantic tensor registry: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 2.2 — Quantized tensor views and block geometry:
  - all seven registered physical storage geometries and checked logical-range
    to quant-block helpers: **VERIFIED**;
  - bounded whole, ordered split, contiguous expert-member and fused PLE-member
    views: **VERIFIED**;
  - synthetic guarded payload suite and Task 1.3 all-physical geometry oracle:
    **PASS**;
  - real-artifact representative view construction with payload bytes touched
    by test = **0**;
  - ADR 0010 bounded quantized tensor views: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 2.3 — Native tokenizer and chat formatting:
  - pinned canonical `Qwen2Tokenizer`, preprocessing, vocabulary, special-token
    and template behavior: **CHARACTERIZED**;
  - original GGUF-only production-source insufficiency: **PRESERVED /
    FAIL-CLOSED**;
  - exact GGUF 248,320-token, 247,587-merge physical substrate compatibility:
    **PASS**;
  - governed Qwen3.8 canonical NFC/pre-tokenizer/BPE/special/BOS/EOS override
    and separate text-only chat formatter: **IMPLEMENTED**;
  - original 10 prompts, 14 segments and four chat vectors plus expanded
    independent divergence corpus: **EXACT PASS**;
  - model tensor payload touched: **0**;
  - ADR 0011: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 2.4 — Native PLE address engine:
  - exact layer/order/head/member/history and integer contract:
    **CHARACTERIZED**;
  - immutable model-validated config and explicit 32-byte stream state:
    **VERIFIED**;
  - transactional prefill and incremental decode with 16 ordered logical
    member/row intents per token: **VERIFIED**;
  - original 7 sequence vectors and 4 decode steps plus expanded independent
    differential corpus: **EXACT PASS**;
  - native tokenizer-to-PLE integration: **EXACT PASS**;
  - PLE payload views opened / model tensor payload touched: **0 / 0**;
  - ADR 0012: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 2.5 — CPU reference numeric primitives and quantized dequantization:
  - exact seven-format block/decode and `/fp:strict` scalar contract:
    **CHARACTERIZED / VERIFIED**;
  - bounded block-by-block quantized row-dot with 1,024-byte scratch:
    **EXACT PASS**;
  - selected generic F32 primitives with 31-case independent calibration and
    21-case holdout: **PASS**;
  - 39 synthetic Class-Q dequant cases and seven row-dot cases:
    **EXACT BITS PASS**;
  - nine semantic/view-resolved real blocks covering all seven formats:
    **612 bytes / 9 blocks / EXACT BITS PASS**;
  - no raw real-model bytes committed; no full-tensor dequantization or model
    operator implemented;
  - ADR 0013: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 2.6 — GDN reference operator and independent golden vectors:
  - exact pinned `Qwen4ExpTextGatedDeltaNet` contract and nine tensor roles:
    **CHARACTERIZED / VERIFIED**;
  - independent reduced F32 Class-C calibration, disjoint holdout and state
    vectors: **PASS**;
  - scalar batch-1 prefill, incremental decode, reset, transactional state and
    21 intermediate checkpoint classes: **PASS**;
  - native split-prefill/decode and reset/replay: **EXACT BITS PASS**;
  - all 36 real GDN configs valid, all 12 QSA IDs rejected, real payload bytes
    touched: **0**;
  - ADR 0014: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 2.7 — QSA reference operator and independent golden vectors:
  - exact pinned `Qwen4ExpTextAttention` and `Qwen4ExpTextQSAIndexer`
    contract: **CHARACTERIZED / VERIFIED**;
  - independent reduced Class-C calibration, disjoint holdout, cache-state
    and exact sparse-selection vectors: **PASS**;
  - scalar batch-1 prefill/decode, transactional state and exact selected
    block/token observation: **PASS**;
  - candidate-count boundaries at and above the 512-block limit, including
    deterministic tie cases: **EXACT DISCRETE PASS**;
  - all 12 real QSA configs valid, all 36 GDN IDs rejected, target state growth
    = **2,304 bytes/token/layer** and **27,648 bytes/token across 12 layers**;
  - real model payload bytes touched: **0**;
  - ADR 0015: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 2.8 — MoE reference operator and independent golden vectors:
  - exact pinned router, top-k, routed-expert, shared-expert/gate and final
    combination contract: **CHARACTERIZED / VERIFIED**;
  - independent Tier-A calibration/holdout and Tier-B 512-expert/top-10
    routing vectors: **PASS**;
  - expert IDs, membership, order, count and pinned tie behavior:
    **EXACT DISCRETE PASS**;
  - scalar selected routed-expert path, separate shared path and final sum:
    **PASS**;
  - all 48 target MoE configs, expert axis 512 and ordered gate/up split:
    **PASS**;
  - per-expert packed groups: **43 x 3,072,000; layer 2 x 3,993,600; layers
    4/30/46/47 x 3,584,000 bytes**;
  - real model payload bytes touched: **0**;
  - ADR 0016: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 2.9 — PLE value reference operator and independent golden vectors:
  - pinned layer-2 PLE lookup/projection/gate/value-history/convolution
    contract: **CHARACTERIZED / VERIFIED**;
  - independent reduced calibration, disjoint holdout, state and Task 2.4
    address-integration vectors: **PASS**;
  - scalar batch-1 prefill/decode with explicit transactional nine-position
    value state: **PASS**;
  - all 128 logical members, fused IQ4_NL table and six dense semantic
    bindings: **PASS**;
  - table bytes **28,800,138,240**, total PLE bytes **28,835,240,960**;
  - bounded real sample: **1,440 logical packed bytes / 80 blocks**;
  - ADR 0017: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 2.10 — Canonical transformer-layer composition and GR/residual:
  - exact pinned decoder-layer order and four-branch rank-320 GR contract:
    **CHARACTERIZED / VERIFIED**;
  - independent ordinary-GDN, QSA and PLE-GDN calibration/holdout vectors:
    **PASS**;
  - scalar prefill/decode, reset/replay and committed/staging rollback:
    **PASS**;
  - target families: **35 ordinary GDN / 12 QSA / 1 PLE-GDN**, 48/48 valid;
  - persistent semantic state reconciled and real payload bytes touched: **0**;
  - ADR 0018: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 2.11 — Target quantized layer execution:
  - semantic GGUF weight provider over Task 2.2 views and Task 2.5 numerics:
    **COMPLETE / PASS**;
  - ordinary GDN layer 0, QSA layer 3 and PLE-GDN layer 1 real quantized
    prefill+decode: **PASS**;
  - independent pinned llama.cpp decode + Transformers equation calibration
    and disjoint holdout: **PASS**;
  - top-10 route order/access, QSA local selection and 32 PLE intents:
    **EXACT DISCRETE PASS**;
  - provider preflight: **48/48 PASS**, no unsupported semantic/type/transform;
  - aggregate correctness payload: **772,826,304 bytes**, below 768 MiB;
  - transactional provider failures: **PASS**;
  - ADR 0019: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.
- [x] Task 2.12 — First Correct Native Token:
  - native tokenizer, bounded real embedding rows, canonical layers 0..47,
    final hyper-connection mixer, full LM head/logits, greedy argmax and native
    decode: **COMPLETE / PASS**;
  - independent pinned llama.cpp Class-Q oracle consumes the exact committed
    canonical IDs and selects the same token ID 271: **EXACT DISCRETE PASS**;
  - early/middle/late model-level provider faults plus control-equivalent
    recovery: **PASS**;
  - logical payload 40,208,768,960 bytes under the governed 64 GiB M1 ceiling;
  - ADR 0020: **ACCEPTED**;
  - overall status: **COMPLETE / PASS**.

## Epic 3 — CPU correctness engine

- [x] Scalar reference tensor primitives (Task 2.5 low-level boundary)
- [x] Reference dequantization (Task 2.5 seven registered formats)
- [x] Tokenizer (Task 2.3 canonical native path)
- [x] Embeddings (Task 2.12 bounded target rows)
- [x] Normalization (Task 2.5 primitives / Task 2.12 final mixer)
- [x] Core block execution (Tasks 2.6–2.12 scalar target path)
- [x] MoE routing (Task 2.8 scalar reference)
- [x] Selected routed/shared expert execution (Task 2.8 scalar reference)
- [x] GDN scalar reference path (Task 2.6)
- [x] QSA scalar reference path (Task 2.7)
- [x] PLE value scalar reference path (Task 2.9)
- [x] Complete one-layer scalar reference composition (Task 2.10)
- [x] Real target-quantized single-layer execution (Task 2.11)
- [x] Logits (Task 2.12 complete target LM head)
- [ ] Sampling
- [x] Reference-vector validation (Task 2.12 independent first-token oracle)

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
