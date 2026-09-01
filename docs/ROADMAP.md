# Roadmap

## R0 — Know the model
Architecture, tensor inventory, tokenizer, reference outputs, hardware baseline.

- Foundation Tasks 0.1–0.3 are complete: KQ-01 hardware, memory/bandwidth and
  Windows/CUDA toolchain baselines are recorded and validated.
- KQ-01 CUDA/PCIe transfer baseline captured with correctness-checked pageable
  and pinned H2D/D2H measurements; simultaneous bidirectional copy remains an
  explicit unsupported/skip case on the observed WDDM runtime.
- Task 1.0 pinned the official Hugging Face artifact revision and official Qwen
  research revision, captured a reproducible no-weight metadata manifest,
  documented the model-license boundary and accepted ADR 0004.
- `KQ-MODEL-SOURCE-001` is verified and the maintainer's merged Unsloth
  UD-Q4_K_XL GGUF is registered as `KQ-MODEL-ARTIFACT-001` with exact size,
  SHA-256, GGUF metadata, quantization evidence and pinned provenance. Task 1.0
  is complete/pass.
- Task 1.1 is complete/pass: the exact 48-layer GDN/QSA schedule, GR, MoE,
  PLE, MTP boundary, prefill/decode flow and persistent runtime state are
  evidence-backed against pinned sources. ADR 0005 is accepted for an initial
  text-only ordinary autoregressive path without vision or MTP acceleration.
- Task 1.2 is complete/pass: bounded HTTP Range capture reconciles all 131
  canonical shard headers and 1,658 tensors with zero weight-payload bytes,
  exact subsystem/static footprints, idealized quantization floors and
  batch-1 persistent-state scaling through 262,144 context tokens.
- Task 1.3 is complete/pass: all 1,658 canonical tensors and all 1,224 GGUF
  tensors reconcile with zero unexplained mappings; exact packed footprints,
  converter/quantizer provenance and the 384-byte split/merge overhead are
  evidence-backed. ADR 0006 accepts a staged direct-GGUF-first strategy while
  keeping canonical model semantics independent of the container.
- Task 1.4 is complete/pass: ADR 0007 pins separate canonical and exact-GGUF
  oracles, and the original/synthetic prompt suite plus exact tokenizer,
  chat-template and PLE-address vectors regenerate byte-identically. Canonical
  BF16 and exact-GGUF full-model runs remain explicit capable-environment gates;
  no weight-dependent output was fabricated.
- Epic 1 is complete/pass. Epic 2 is **COMPLETE / PASS**: its original
  loader/introspection and runtime-foundation objectives, Project Plan Phase 1
  exit gates and every explicitly governed Task 2.0–2.13 gate are satisfied.
  M1 First Correct Native Token and the bounded incremental continuation remain
  PASS; no Task 2.14 or additional Epic 2 closure criterion is defined.

## R1 — Read the model
Safe mapping, metadata, tensor inspection, diagnostics.

- Task 2.0 is complete/pass. ADR 0008 accepts the Kestrel-Q-owned C17 GGUF v3
  layer: read-only Win32 mapping, bounded logical views, checked physical tensor
  descriptors, `kq-inspect`, malformed-artifact coverage and exact reproduction
  of the 111 GB artifact's structural oracle with zero payload bytes accessed.
- Task 2.1 is complete/pass. ADR 0009 accepts the immutable canonical semantic
  registry: exact `qwen4exp` identity/topology validation, 1,294/1,294
  initial-text semantics, 1,224/1,224 physical coverage, explicit split/fused/
  metadata relations and zero payload bytes accessed. The complete native dump
  matches the pinned Epic 1 mapping oracle.
- Task 2.2 is complete/pass. ADR 0010 accepts immutable bounded physical views
  with checked seven-type quant geometry, explicit canonical/physical layout,
  separate split parts, contiguous expert members and bounded fused PLE
  members. Synthetic guarded payload reads pass, all 1,224 physical geometries
  match Task 1.3 and the real test dereferences zero model-payload bytes.
- Task 2.3 is complete/pass. It preserves the finding that GGUF-only tokenizer
  semantics are insufficient, then applies the governed canonical adapter over
  exactly validated GGUF vocab/ID/merge substrate. Pinned Unicode/NFC,
  marks-excluding byte-level BPE, canonical special/BOS/EOS behavior and the
  separate official text-only chat subset match all independent original and
  divergence vectors. ADR 0011 is accepted; model tensor payload touched is
  zero.
- Task 2.4 is complete/pass. The native C17 PLE address engine validates the
  exact target semantics, maintains a bounded 32-byte per-stream history and
  emits 16 canonical member/row intents per token. All original and expanded
  pinned-oracle vectors pass exactly; no PLE payload view, I/O, cache, prefetch
  or scheduler behavior was introduced. ADR 0012 is accepted.
- Task 2.5 is complete/pass. ADR 0013 accepts a scalar `/fp:strict` CPU
  reference layer for all seven registered storage decoders, bounded
  block-by-block row-dot and the characterized generic F32 primitive set.
  Independent synthetic calibration/holdout passes, and nine bounded real
  blocks cover all seven formats in 612 payload bytes without committing raw
  weights or implementing a model operator.
- Task 2.6 is complete/pass. ADR 0014 accepts the first model-specific scalar
  reference operator: canonical Qwen3.8 GDN prefill/decode and explicit state
  pass independent calibration, holdout and transition vectors. All 36 real
  GDN descriptors validate, 12 QSA IDs reject and real payload touched is zero.
- Task 2.7 is complete/pass. ADR 0015 accepts the scalar Qwen3.8 QSA reference
  operator with explicit bounded K/V/raw-index state, transactional prefill and
  decode, and exact sparse block/token selection. Independent calibration,
  holdout, cache-state and threshold-crossing vectors pass; all 12 real QSA
  descriptors validate, 36 GDN IDs reject and real payload touched is zero.
- Task 2.8 is complete/pass. ADR 0016 accepts a scalar C17 MoE reference:
  independent reduced calibration/holdout and 512/top-10 routing vectors pass,
  routing IDs/order are exact, routed/shared/final paths pass their calibrated
  contracts, and all 48 real target bindings validate with zero payload bytes
  touched. Packed selected-expert footprints are structural facts, not an
  I/O-per-token claim.
- Task 2.9 is complete/pass. ADR 0017 accepts a scalar C17 PLE value reference:
  exact Task 2.4 address intents drive a storage-neutral row provider, explicit
  transactional value history and canonical projection/gate/dilated-conv
  execution. Independent calibration, holdout, state and address-integration
  evidence passes. Real bounded IQ4_NL plumbing touches 1,440 logical packed
  bytes; no final cache, prefetch or scheduling policy was introduced.
- Task 2.10 is complete/pass. ADR 0018 accepts the scalar one-layer reference:
  exact layer-owned GR semantics compose PLE/GDN/QSA/MoE for all three layer
  families, persistent substates commit transactionally, 48/48 target configs
  validate, and real model payload touched is zero.
- Task 2.11 is complete/pass. ADR 0019 accepts the semantic target-weight
  provider: the same scalar layer equations execute directly from bounded GGUF
  views for ordinary GDN, QSA and PLE-GDN prefill/decode. Independent Class-Q
  decode plus canonical equations, exact route/access/PLE decisions, 48/48
  provider preflight and rollback pass under the 768 MiB correctness ceiling.
- Task 2.12 is complete/pass. ADR 0020 accepts a short-context, greedy-only
  native model executor over bounded real embedding/head access and all 48
  target layers. `KQ-PROMPT-001` produces token 271 (`\n\n`), exactly matching
  the pinned independent llama.cpp Class-Q oracle; three model-level fault
  recoveries pass. Logical payload is 40,208,768,960 bytes under the 64 GiB M1
  safety ceiling and is not physical-I/O evidence.
- Task 2.13 is complete/pass. ADR 0021 accepts prompt-prefill-once plus direct
  one-token greedy continuation. Pinned llama.cpp and native Kestrel-Q both
  produce `[271, 248068, 198, 760]`; a failed later step rolls back all
  completed layers and retry reproduces the oracle token. The successful path
  accounts 59,212,012,160 logical packed bytes under 96 GiB, not physical I/O.
- R1/Epic 2 is complete/pass after the governance-only closure review. No
  scheduler, cache/prefetch optimization, sampling or CUDA model kernel was
  started by the review.

## R2 — Compute correctly
Minimal CPU reference inference and reference-vector parity.

- Low-level scalar storage and arithmetic reference boundary: **COMPLETE /
  PASS via Task 2.5**.
- GDN scalar reference execution: **COMPLETE / PASS via Task 2.6**.
- QSA scalar reference execution: **COMPLETE / PASS via Task 2.7**.
- MoE scalar reference execution: **COMPLETE / PASS via Task 2.8**.
- PLE value scalar reference execution: **COMPLETE / PASS via Task 2.9**.
- Complete one-layer scalar composition: **COMPLETE / PASS via Task 2.10**.
- Real target-quantized single-layer execution: **COMPLETE / PASS via Task
  2.11**.
- Real target embedding, 48-layer execution, final mixer, LM head/logits,
  greedy argmax and native decode: **COMPLETE / PASS via Task 2.12**.
- True incremental bounded greedy continuation and per-step transaction:
  **COMPLETE / PASS via Task 2.13**.
- Epic 3 inception reconstructed the remaining R2 scope without changing
  runtime behavior. Sampling is the only unfinished Phase 2/R2 workstream;
  tokenizer, scalar operators, real quantized execution, logits and incremental
  greedy generation are already complete through Task 2.13. Epic 3 is now
  **COMPLETE / PASS** through Tasks 3.0-3.1.
- Task 3.0 — Native Sampling Policy & Deterministic Selection Primitives:
  **COMPLETE / PASS**. The pinned official default profile now has a separate
  C17 policy/PCG32/selection boundary with exact, calibrated-float and
  predeclared statistical evidence; ADR 0022 is accepted.
- Task 3.1 — Sampled Incremental Generation & Oracle Validation:
  **COMPLETE / PASS**. It extends the accepted prompt-prefill-once executor
  transaction to explicit sampler state, validates primary/holdout/replay and
  rollback independently, and preserves the frozen greedy sequences.
- Epic 3 is **COMPLETE / PASS** after both tasks passed their
  independent-oracle, fail-closed and clean-regression gates. This is CPU
  correctness work, not a performance, scheduler, CUDA, long-context or
  product milestone. Epic 4 remains NOT STARTED.

## R3 — Use the GPU
CUDA baseline with correctness parity.

- Defer `KQ-BACKLOG-BENCH-001`, the WDDM CUDA allocation-headroom benchmark,
  until production VRAM allocator or memory-placement capacity policy needs it.

## R4 — Fit the machine
VRAM/RAM/NVMe tiering, expert cache, async prefetch, streaming.

- Treat PLE as `PLE_DISK_BACKED_CANDIDATE`: disk-backed/mapped primary storage,
  a bounded explicit RAM page/row cache, predictive asynchronous prefetch from
  deterministic addresses and only materialized lookup results/working data in
  VRAM. Treat routed experts as candidates for active RAM cache, hot VRAM subset
  and cold disk backing. This is preliminary and does not endorse uncontrolled
  Windows paging.
- `KQ-BACKLOG-BENCH-002` is **DEFERRED / REQUIRED BEFORE FINAL SCHEDULER
  DESIGN**. It must characterize cold/warm, random/batched/prefetched PLE access,
  mmap/page faults, physical reads and amplification, explicit-cache hit rate
  and working set, prefetch lead, SATA/CPU cost and eventual NVMe comparison,
  while separating OS cache, explicit Kestrel-Q cache and cold storage. It does
  not block R1/Task 2 correctness work.

## R5 — Make it fast
Quantization, fused kernels, scheduler optimization, profiling.

## R6 — Make context practical
State management, persistence and long-context behavior.

## R7 — Make it usable
CLI, server, configuration and Windows packaging.

## R8 — Make it an agent
Native coding workflow, tools and persistent sessions.

## R9 — Make it a community project
CI, benchmark submissions, compatibility matrix, release governance.

- Define the portable CUDA release strategy (`native` versus
  multi-architecture/fat-binary policy) before the first public Windows binary
  release. `KQ-BACKLOG-CUDA-001` is deferred and does not block KQ-01 work.
