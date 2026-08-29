# Changelog

All notable changes to this project will be documented here.

The project is currently pre-alpha.

## Unreleased

### Changed

- Replaced the provisional MIT license with Apache License 2.0.
- Added contribution and dependency licensing/provenance rules.
- Documented the validated KQ-01 Windows, MSVC, CMake, CUDA and GPU baseline.
- Ignored disposable root `build-*` CMake build trees.
- Recorded deferred backlog item `KQ-BACKLOG-CUDA-001` to select and document a
  portable CUDA binary strategy before the first public binary release.
- Replaced KQ-01's theoretical-only PCIe assumption with measured pageable and
  pinned H2D/D2H results across 1–256 MiB, plus under-load Gen4 x16 evidence.
- Consolidated Foundation Tasks 0.1–0.3 as complete/pass, including the
  previously measured Task 0.2D clean-idle RAM/VRAM baseline.
- Deferred WDDM CUDA allocation-headroom measurement to separate backlog item
  `KQ-BACKLOG-BENCH-001`; it is not part of Task 0.2D.
- Pinned the official Qwen3.8-Flash-Next artifact and research revisions and
  accepted ADR 0004: official Qwen artifacts/specifications are canonical,
  while GGUF remains a derived representation.
- Documented the Apache-2.0 Kestrel-Q source boundary separately from the pinned
  Qwen Community License 1.0 model-artifact terms, including commercial MaaS
  and AI Work Assistant review triggers.
- Made `CHANGELOG.md` the canonical chronological project-change record and
  required every material mandate/iteration to update it before completion and
  commit, with task status, task evidence, ADRs and Git history retaining their
  separate responsibilities.
- Completed Task 1.1 architecture characterization against pinned official
  model/research sources and a pinned Apache-2.0 Transformers implementation
  reference; Tasks 1.2 and 1.3 remain not started.
- Accepted ADR 0005: the initial Kestrel-Q model path may support ordinary
  text-only autoregressive logits while explicitly rejecting multimodal inputs
  and deferring optional MTP speculative acceleration.
- Completed Task 1.2 canonical tensor inventory and footprint analysis: all 131
  Safetensors headers and 1,658 tensors reconcile exactly, with zero weight
  payload bytes fetched and zero unexplained classifications; Task 1.3 remains
  not started.
- Completed Task 1.3 canonical-to-GGUF mapping: all 1,658 canonical tensors and
  all 1,224 derived GGUF tensors reconcile with zero unresolved or unexplained
  entries, including exact PLE fusion/metadata, MoE/QSA splits, vision/MTP scope
  omissions and converter transforms, closing the exact 434-tensor difference.
- Accepted ADR 0006's staged container strategy: begin future correctness work
  from the verified GGUF while preserving canonical internal semantics, and
  require profiling evidence before introducing a Kestrel-Q-native container.
- Proved the local merged GGUF's 384-byte reduction from the published four
  shards is header/directory format overhead only; bounded upstream Range audit
  fetched 342 metadata bytes and zero tensor-payload bytes.
- Completed Task 1.4 and Epic 1 with separate canonical and exact-GGUF
  correctness classes, pinned Apache-2.0 Transformers and MIT llama.cpp
  oracles, and accepted ADR 0007's independence/resource-deferral policy.
- Preserved canonical BF16 and exact-GGUF full-model vectors as explicit
  capable-reference-environment gates after KQ-01 resource checks; no
  weight-dependent output was fabricated or executed.
- Recorded `PLE_DISK_BACKED_CANDIDATE` as a preliminary KQ-01 placement
  hypothesis: disk-backed/mapped PLE with a bounded explicit RAM cache,
  deterministic predictive prefetch and materialized lookup data only in VRAM;
  routed experts remain candidates for RAM-active/VRAM-hot/disk-cold tiering.
  This is not a validated performance result or final scheduler policy, and
  uncontrolled Windows paging is explicitly not the implementation mechanism.
- Added `KQ-BACKLOG-BENCH-002`, the deferred PLE disk-backed access benchmark
  required before final scheduler/residency design. It must separate OS page
  cache, explicit Kestrel-Q RAM cache and cold physical reads; it does not block
  Task 2 loader/correctness work.
- Completed Task 2.0 and accepted ADR 0008: the first production C17 runtime
  component is a Kestrel-Q-owned, Windows-native, read-only GGUF v3 layer with
  bounded logical mappings, immutable physical descriptors and no llama.cpp or
  GGML runtime dependency. Task 2.1 remains not started and
  `KQ-BACKLOG-BENCH-002` remains deferred.
- Reproduced the complete Epic 1 structural oracle through `kq-inspect` on the
  registered 111 GB artifact without a full-file hash or intentional payload
  access. File size/last-write time remained unchanged and the parser reported
  `payload_bytes_accessed = 0`.
- Fixed two intermediate MSVC C4701 warnings caused by conservative data-flow
  analysis of parser locals populated through status-returning helpers. Explicit
  initialization restored a warning-free `/W4` build; the clean CPU/CUDA build
  gate protects against recurrence.

### Added

- Initial repository structure.
- Project vision and roadmap.
- High-level project plan and task backlog.
- AI-agent operating contract.
- Initial architecture direction.
- Benchmark policy.
- Initial architecture decision records.
- Minimal C/CMake build skeleton.
- Optional `KQ_ENABLE_CUDA` smoke backend with a C ABI, checked CUDA operations,
  device diagnostics and host-validated kernel result.
- CPU-only and CUDA-enabled CTest coverage for the smoke paths.
- CUDA-only `kq_cuda_bandwidth` benchmark with host and CUDA-event timing,
  byte-complete transfer validation, setup metrics and stable CSV output.
- PowerShell evidence harness with robust `nvidia-smi` resolution and PCIe link
  sampling, plus immutable Task 0.2E KQ-01 raw CSV/text evidence.
- Reproducible Task 1.0 model-source manifest with a fail-closed, explicit
  non-weight metadata allowlist and hashes/sizes for the complete upstream file
  inventory.
- Model source baseline, model-license boundary and completed artifact register;
  `KQ-MODEL-ARTIFACT-001` records the exact merged Unsloth UD-Q4_K_XL GGUF
  size, SHA-256, GGUF v3 metadata, tensor-type evidence and pinned provenance.
- Completed Task 1.0 after read-only registration of the maintainer's local
  derived GGUF; no model artifact was copied into or tracked by the repository.
- Implementation-grade Qwen3.8-Flash-Next architecture and runtime-state
  baselines covering the exact 48-layer GDN/QSA schedule, Gated Residual, MoE,
  deterministic n-gram/PLE addressing, MTP boundaries, and prefill/decode state
  transitions.
- Machine-readable Task 1.1 evidence with stable `KQ-ARCH-*` claim IDs, exact
  source revisions, source locations, licenses and implementation file hashes.
- Fail-closed, standard-library Task 1.2 research tooling for bounded
  Safetensors-header Range capture and deterministic offline tensor analysis,
  with exact static, idealized-quantization, per-layer/expert and persistent
  runtime-state evidence; failure audits are preserved against later overwrite
  or cleanup, and each inventory row's `classification_rule` records its exact
  versioned component rule rather than only the global rule version.
- Canonical tensor-inventory and KQ-01 footprint documents, including exact
  text/vision/MTP/PLE/expert families and explicit placement-candidate limits.
- Fail-closed, standard-library Task 1.3 tooling for read-only GGUF structure,
  bounded upstream split-header audit and deterministic canonical mapping, plus
  five machine-readable evidence artifacts and exact packed quantization/type,
  family, per-layer and per-expert footprints and embedded imatrix/calibration
  provenance findings without unsupported sensitivity claims.
- A deterministic original/synthetic ten-case reference prompt suite and exact
  weight-independent tokenizer, chat-template and PLE address goldens, plus
  machine-readable GR/GDN/QSA/MoE/final checkpoint and full-model plans.
- Fail-closed Task 1.4 generation/validation tooling with pinned metadata,
  source-revision/hash and dependency checks. The checks caught a one-nibble
  source-hash transcription error and a Transformers `BatchEncoding` handling
  error before a valid manifest; a final scan also removed 19 zero-tensor
  llama.cpp vocabulary GGUF fixtures from ignored cache and added source-tree
  fixture rejection. Corrected regeneration is byte-identical.
- Production `kq_status`, Win32 `kq_file`/bounded-view and target-first `kq_gguf`
  APIs with checked 64-bit arithmetic, bounded file-backed strings, seven
  target tensor block geometries, duplicate/alignment/span validation and
  deterministic cleanup on every tested failure path.
- `kq-inspect` structural summary CLI plus deterministic runtime-created GGUF
  fixtures covering one valid seven-type container and 21 malformed cases,
  including truncation, strings/arrays, overflow, rank/dimensions, type,
  duplicate name, alignment/span and quantized-geometry failures.
- Opt-in `KQ_GGUF_PATH` integration coverage asserting the exact artifact size,
  version, architecture, metadata/tensor counts, packed bytes, overhead and all
  tensor-type counts without committing or modifying model payload.
