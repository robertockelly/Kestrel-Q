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
