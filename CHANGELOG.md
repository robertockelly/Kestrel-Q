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
