# PLAN-TASK-0.3B-CUDA-SMOKE.md

Status: **COMPLETE / PASS**

## Objective
Add the first CUDA execution path to Kestrel-Q without introducing model inference.

## Scope
Implement a minimal CUDA smoke backend that detects the device, reports basic runtime/device info, allocates memory, performs H2D copy, launches a trivial kernel, synchronizes, performs D2H copy, validates results, and returns deterministic success/failure.

## Architectural constraints
- Core runtime remains C17.
- CUDA implementation lives in `.cu` files.
- CUDA exposes a C-compatible ABI.
- CUDA stays optional via `KQ_ENABLE_CUDA`.
- CPU-only build must remain green.
- No inference/model/tensor/tokenizer/quantization/scheduler logic.
- No llama.cpp/GGML/PyTorch/Python runtime dependency.
- No developer-specific absolute paths.
- No generated build trees committed.

## Expected file direction
```text
include/kq_cuda.h
src/cuda/kq_cuda_smoke.cu
tests/cuda_smoke.c
```

## CUDA smoke functionality
Report at minimum:
- CUDA runtime version
- CUDA driver version
- selected device index
- GPU name
- compute capability
- total global memory
- multiprocessor count
- warp size

Prove:
- cudaMalloc
- H2D copy
- trivial kernel launch
- cudaDeviceSynchronize
- D2H copy
- host-side result validation

## Error policy
Fail closed. Every CUDA API call must be checked. Failures return non-zero and produce a useful diagnostic.

## CMake requirements
Default CPU-only:
`KQ_ENABLE_CUDA=OFF`

CUDA build:
`KQ_ENABLE_CUDA=ON`

CUDA must be enabled only when requested and no CUDA path may be hard-coded.

## Validation matrix

CPU-only:
```powershell
cmake -S . -B build-cpu -G "Visual Studio 17 2022" -A x64 -DKQ_ENABLE_CUDA=OFF
cmake --build build-cpu --config Release
ctest --test-dir build-cpu -C Release --output-on-failure
```

CUDA:
```powershell
cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 -DKQ_ENABLE_CUDA=ON
cmake --build build-cuda --config Release
ctest --test-dir build-cuda -C Release --output-on-failure
```

## Documentation
Update:
- `docs/TOOLCHAIN-BASELINE.md`
- `CHANGELOG.md`

Observed KQ-01 baseline:
- MSVC 19.44.35228 x64
- MSVC toolset 14.44.35207
- Windows SDK 10.0.26100.0
- CMake 4.4.3
- CUDA Toolkit 13.3
- nvcc 13.3.73
- NVIDIA driver 595.97
- RTX 3080 10 GiB

Do not claim runtime compatibility until the CUDA smoke test passes.

## Definition of done
- CPU-only build/tests green
- CUDA-enabled build/tests green
- CUDA smoke executes on RTX 3080
- kernel result validated
- runtime/driver/device details recorded
- all CUDA errors checked
- no absolute local path introduced
- docs and changelog updated
- working tree contains only intentional changes
