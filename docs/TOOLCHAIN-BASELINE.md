# Task 0.3 — Windows/CUDA Build Toolchain Baseline

Status: **TASK 0.3 COMPLETE / PASS — TASK 0.3B CUDA SMOKE VALIDATED**

Reference machine: **KQ-01**

Validation date: **2026-08-28**

## 1. Objective

Establish a reproducible Windows-native C17 toolchain and validate the first
optional CUDA execution path without adding inference behavior.

## 2. Observed KQ-01 toolchain

The following values were observed during the Task 0.3B clean builds:

- Visual Studio 2022 Build Tools
- MSVC 19.44.35228.0, x64
- MSVC toolset 14.44.35207
- MSBuild 17.14.51
- Windows SDK 10.0.26100.0
- CMake 4.4.3
- CUDA Toolkit 13.3
- nvcc V13.3.73
- NVIDIA display driver 595.97
- NVIDIA GeForce RTX 3080, 10240 MiB reported by `nvidia-smi`

The CMake configure selected Windows SDK 10.0.26100.0 to target Windows
10.0.26200.

## 3. Source-language and CMake policy

The runtime core is C17:

```cmake
project(kestrel_q LANGUAGES C)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
```

`KQ_ENABLE_CUDA` defaults to `OFF`. When explicitly enabled, CMake enables the
CUDA language and builds the smoke implementation as CUDA C++17. The backend
exports only the C-compatible interface declared in `include/kq_cuda.h`.

The CUDA smoke target uses CMake's `native` CUDA architecture selection. On
KQ-01 this produced an `sm_86` cubin, verified with `cuobjdump`. This avoids a
dependency on a developer-specific path or a hard-coded GPU architecture.

## 4. CUDA smoke observations

The passing smoke run reported:

- CUDA runtime API version: 13.2 (raw 13020)
- CUDA driver API version: 13.2 (raw 13020)
- CUDA device count: 1
- selected device index: 0
- GPU: NVIDIA GeForce RTX 3080
- compute capability: 8.6
- CUDA-reported total global memory: 10,736,893,952 bytes (10239 MiB)
- free global memory at the observed run: 9,538,895,872 bytes (9097 MiB)
- multiprocessor count: 68
- warp size: 32
- kernel result: 42, host validated

Free memory is a point-in-time value and is not a capacity guarantee.

The smoke path checked device discovery and properties, memory information,
device allocation, H2D copy, kernel launch status, device synchronization, D2H
copy, result validation and device release. Any CUDA error returns non-zero.

## 5. Compatibility finding and root cause

The first CUDA build compiled successfully but the CUDA CTest failed closed at
kernel launch with `cudaErrorUnsupportedPtxVersion`.

CMake/NVCC's default architecture was `compute_75` with both `sm_75` code and
PTX. The RTX 3080 could not use that cubin, and driver 595.97's CUDA 13.2 JIT
could not consume PTX emitted by the CUDA 13.3 toolchain. Configuring the smoke
target for CMake's `native` architecture produced an `sm_86` cubin and removed
the unintended PTX-JIT dependency on KQ-01. The clean CUDA test then passed.

This is a build-target compatibility fix only. It is not evidence of inference
correctness or performance.

## 6. Validation results

CPU-only validation:

```powershell
cmake -S . -B build-cpu -G "Visual Studio 17 2022" -A x64 -DKQ_ENABLE_CUDA=OFF
cmake --build build-cpu --config Release
ctest --test-dir build-cpu -C Release --output-on-failure
```

Result: configure PASS, Release build PASS, `kq_smoke` PASS (1/1).

CUDA validation:

```powershell
cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 -DKQ_ENABLE_CUDA=ON
cmake --build build-cuda --config Release
ctest --test-dir build-cuda -C Release --output-on-failure
```

Result: configure PASS, Release build PASS, `kq_smoke`, `kq_cuda_smoke` and the
CUDA bandwidth functional self-test PASS (3/3) in the final consolidated tree.

An additional fail-closed check ran the CUDA smoke executable with no visible
CUDA device. It reported `cudaErrorNoDevice` and returned exit code 1.

MSVC emitted warning C4211 from NVCC's generated temporary `.stub.c` file while
compiling with `/W4`. No warning originated in repository source.

## 7. Disposable CMake build-tree finding

CMake build trees contain absolute repository, compiler and toolkit paths and
are not portable between machines or installations. `build/`, `build-cpu/`,
`build-cuda/` and other root `build-*` directories are disposable generated
state and are ignored by Git. Reconfigure from source rather than copying or
committing a build tree.

## 8. Scope and limitations

Task 0.3B validates only CUDA discovery, one allocation, two copies and one
trivial kernel on KQ-01. It does not validate inference, tensor math,
quantization, model loading, tokenization, scheduling, performance, additional
GPUs or cross-machine CUDA architecture portability.

Because the smoke target uses `native` architecture selection, a CUDA-enabled
build requires a visible supported NVIDIA device at configure/build time and is
specific to the detected architecture. CPU-only builds have no CUDA dependency.

## 9. Acceptance gate

- [x] Visual Studio 2022 Build Tools detected
- [x] MSVC v143 detected
- [x] Windows SDK detected
- [x] CMake 4.4.3 stable detected
- [x] CUDA Toolkit 13.3 detected
- [x] nvcc V13.3.73 detected
- [x] clean CPU-only Release build and CTest pass
- [x] clean CUDA Release build and both CTests pass
- [x] CUDA runtime, driver API and device values captured
- [x] no developer-specific absolute path added to source or configuration
- [x] generated build directories ignored by Git

**TASK 0.3 — WINDOWS/CUDA BUILD TOOLCHAIN BASELINE: COMPLETE / PASS**
