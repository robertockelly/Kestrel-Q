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
- Completed Task 2.1 and accepted ADR 0009: a target-specific immutable
  Qwen3.8-Flash-Next semantic registry now validates the complete `qwen4exp`
  identity/topology and reconciles 1,294 initial-text meanings with all 1,224
  registered physical GGUF tensors and three metadata-derived PLE meanings.
- Preserved the exact non-one-to-one representation boundary: 48 MoE gate/up
  and 12 QSA index query/key splits, 128 logical PLE table members fused into
  one physical tensor, and explicit PLE address metadata. Routed expert stacks
  retain their 512-expert axis without fabricating per-expert tensors.
- Confirmed the complete deterministic native registry against every
  initial-text row and physical name in the pinned Epic 1 mapping oracle, with
  zero unknown physical tensors, zero unbound required semantics and zero
  payload bytes accessed. `KQ-BACKLOG-BENCH-002` remains deferred and placement
  values remain annotations rather than scheduler policy.
- Fixed two Task 2.1 implementation findings before the clean gate: explicit
  pointer initialization resolved conservative MSVC C4701/C4703 diagnostics,
  and semantic CLI dispatch was moved after successful GGUF parsing after an
  initial null-GGUF ordering failure. Synthetic and CLI-oracle regressions cover
  both corrected paths.
- Completed Task 2.2 and accepted ADR 0010: the C17 runtime now exposes
  immutable bounded read-only quantized tensor views with checked F32, BF16,
  Q5_1, Q8_0, Q4_K, Q5_K and IQ4_NL block geometry, packed-size and logical
  element/range-to-block helpers.
- Preserved semantic/physical layout boundaries through exact whole-tensor
  views, ordered non-concatenated MoE/QSA split parts, provably contiguous
  routed-expert members and one-member-only views over the fused 128-member PLE
  tensor. Metadata-derived requests and transformed-canonical misuse fail
  explicitly; mapping adds no allocator, cache, prefetch or scheduler policy.
- Added deterministic guarded payload fixtures and real-artifact view coverage.
  Synthetic tests dereference only their 1,192 known fixture bytes; dense,
  all-seven-type, layer-2 expert, split and PLE 0/64/127 real mappings open and
  close with zero model-payload bytes touched by the test.
- Added a deterministic physical-geometry inspector dump and research-only
  Task 1.3 validator. All 1,224 physical descriptors and 111,323,630,080 packed
  bytes match the pinned inventory; production has no CSV/JSON or Python
  dependency.
- Corrected an intermediate Task 2.2 descriptor-contract omission found during
  implementation review: the first draft separated requested elements from
  physical block bytes but omitted canonical unpacked bytes. The final view
  info records `logical_unpacked_bytes` and a copied stable semantic ID, with
  synthetic regression assertions. `KQ-BACKLOG-BENCH-002` remains deferred.
- Fixed an intermediate Task 2.2 MSVC C4701 diagnostic caused by conservative
  data-flow analysis of a canonical element count returned through a
  status-reporting helper. Explicit initialization preserves fail-closed
  control flow and restores the warning-free project build.
- Corrected the synthetic non-contiguous-expert mutation so it preserves the
  canonical element total; the original mutation was validly rejected by the
  earlier shape gate and therefore did not exercise the intended contiguity
  branch. The corrected regression now isolates the slowest-axis requirement.

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
- Production `kq_model` C17 API with stable semantic IDs, canonical
  component/role/layer descriptors, explicit binding cardinality, expert/fused
  geometry, runtime scope and preliminary placement annotations. Construction
  reads only Task 2.0 metadata/descriptors and has no global mutable state.
- `kq-inspect --semantic-summary`, exact semantic lookup and deterministic TSV
  dump modes, all retaining `payload_bytes_accessed = 0`.
- A complete in-memory Task 2.1 target fixture with 20 fail-closed mutations,
  real-artifact semantic integration, and a standard-library Python validator
  that uses the Epic 1 CSV only as a test/research oracle; production has no
  research-file or Python dependency.
