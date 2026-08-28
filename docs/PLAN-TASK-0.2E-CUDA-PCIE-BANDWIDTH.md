# PLAN-TASK-0.2E-CUDA-PCIE-BANDWIDTH.md

Status: **COMPLETE / PASS**

## Objective

Measure the real CUDA host/device transfer behavior of KQ-01 and replace the remaining theoretical PCIe assumptions in the memory/bandwidth baseline with reproducible evidence.

This task is a benchmark/instrumentation task only.

It must not introduce model inference, model loading, tensor math, quantization, expert scheduling, or production caching logic.

## Context

Validated KQ-01 baseline:

- GPU: NVIDIA GeForce RTX 3080 10 GiB
- compute capability: 8.6
- 68 SMs
- warp size: 32
- PCIe maximum capability: Gen4 x16
- driver: 595.97
- CUDA Toolkit: 13.3
- nvcc: 13.3.73
- CUDA runtime/driver API observed by smoke: 13.2
- CMake: 4.4.3
- MSVC: 19.44.35228 x64
- RAM WinSAT: ~25.63 GB/s
- SATA sequential read WinSAT: ~0.44 GB/s
- Task 0.3B CPU/CUDA tests: PASS

Important existing finding:

- default PTX execution failed with `cudaErrorUnsupportedPtxVersion`;
- CMake `native` architecture generated an `sm_86` cubin and passed;
- do not regress this validated behavior.

## Scope

Build a dedicated CUDA bandwidth benchmark for KQ-01 that measures:

1. pageable host RAM → VRAM;
2. pinned host RAM → VRAM;
3. VRAM → pageable host RAM;
4. VRAM → pinned host RAM;
5. concurrent/bidirectional transfer when supported;
6. transfer behavior across multiple block sizes;
7. allocation/pinning setup cost;
8. CUDA device capabilities relevant to asynchronous copy;
9. PCIe link generation/width under transfer load through a Windows-side sampling harness where practical.

## Benchmark sizes

Use a bounded default matrix appropriate for KQ-01:

- 1 MiB
- 4 MiB
- 16 MiB
- 64 MiB
- 256 MiB

Do not allocate enough memory to create pressure on the ~10 GiB GPU or ~32 GiB host.

The implementation may choose adaptive iteration counts, but each size must have:

- warm-up iterations;
- multiple measured iterations;
- reported iteration count.

A reasonable initial default is at least 3 warm-up and 10 measured iterations, unless measurement cost demonstrates a need for a documented adaptive policy.

## Timing requirements

Prefer two timing views where meaningful:

### End-to-end host timing

Use a high-resolution host timer around the complete transfer operation.

This captures observable application cost, especially for pageable memory where staging can matter.

### CUDA-event timing

Use CUDA events for device/stream-side elapsed time where valid.

Do not conflate CUDA-event time with complete host-visible latency.

Report both when available.

## Transfer-mode requirements

### H2D and D2H

Measure pageable and pinned memory separately.

Pinned memory must use CUDA-supported page-locked allocation/registration, not undocumented tricks.

### Bidirectional

If device capabilities support useful concurrent copy:

- use separate CUDA streams;
- run H2D and D2H concurrently;
- synchronize explicitly;
- report aggregate and directional throughput clearly.

If the hardware/runtime cannot support the planned mode, fail/skip explicitly with a diagnostic rather than reporting misleading zero/success data.

## Allocation/setup metrics

Capture separately where possible:

- `cudaMalloc` latency;
- pinned host allocation/registration latency;
- stream creation latency if material.

Do not include one-time allocation cost inside steady-state bandwidth unless the metric is explicitly named end-to-end setup-inclusive.

## Correctness

Every benchmark transfer must validate copied data.

Performance numbers are invalid if correctness is not demonstrated.

Use deterministic small-pattern validation or hashes/checks appropriate to the transfer size without dominating benchmark time.

## Error policy

Fail closed:

- check every CUDA API call;
- check kernel/async errors where applicable;
- non-zero process exit on benchmark failure;
- distinguish unsupported/skip from PASS;
- never silently downgrade pinned to pageable behavior.

## Output requirements

The benchmark must support machine-readable output.

Preferred:

- CSV to stdout and/or explicit output file;
- stable columns suitable for comparison across commits.

Include at minimum:

- timestamp
- commit if available through build metadata or wrapper
- machine id `KQ-01`
- GPU name
- compute capability
- driver API version
- runtime version
- transfer direction
- host memory type
- transfer size bytes
- warm-up count
- measured iteration count
- host elapsed mean/median if implemented
- CUDA elapsed mean/median if implemented
- effective GB/s
- correctness status

Do not invent fields that cannot be measured reliably.

## Canonical evidence

Store small raw evidence under:

```text
bench/results/raw/KQ-01/2026-08-28/
```

Suggested names:

```text
cuda-bandwidth.csv
cuda-bandwidth-console.txt
pcie-link-samples.csv
```

Raw evidence is immutable after capture.

Summaries belong in Markdown documentation.

Respect the repository benchmark-evidence `.gitignore` policy.

## Windows PCIe sampling harness

Because CUDA runtime properties do not directly provide a reliable current negotiated PCIe generation under load, add a small Windows/PowerShell orchestration script if useful.

Suggested behavior:

1. launch the bandwidth benchmark;
2. while it runs, sample:
   - `pcie.link.gen.current`
   - `pcie.link.width.current`
   - GPU pstate
   - memory usage
3. use `C:\Windows\System32\nvidia-smi.exe` explicitly or resolve it safely;
4. write timestamped CSV evidence;
5. avoid requiring `nvidia-smi` to be globally visible in PATH.

The script is benchmark tooling, not production runtime.

If reliable sampling cannot be completed without adding an undesirable dependency, document the limitation and keep it pending rather than fabricating a value.

## CMake integration

The benchmark must only build when CUDA is enabled.

CPU-only default remains unaffected.

Suggested target naming:

```text
kq_cuda_bandwidth
```

Do not turn performance benchmarking into a required CTest test because runtime variance makes bandwidth thresholds unsuitable for ordinary correctness CI.

A small functional self-test of the benchmark code may be added if useful.

## Baseline document updates

Update:

- `docs/MEMORY-BANDWIDTH-BUDGET.md`
- `docs/TOOLCHAIN-BASELINE.md` only if new toolchain facts emerge
- `docs/TASKS.md`
- `CHANGELOG.md`

Task 0.2E must move from deferred/pending to PASS only after canonical KQ-01 measurements exist.

## Interpretation requirements

The final report must compare measured results with:

- ~25.63 GB/s measured RAM copy baseline;
- ~0.44 GB/s measured SATA read baseline;
- ~31.5 GB/s theoretical PCIe 4.0 x16 payload ceiling.

Do not describe theoretical bandwidth as measured.

Explain the implications for:

- pageable versus pinned staging;
- expected expert prefetch design;
- RAM cache versus SSD misses;
- feasibility of overlapping transfer with compute.

No model throughput claim is allowed from this benchmark alone.

## Backlog prerequisite

As part of this mandate, also apply the project backlog update specified in:

`docs/BACKLOG-UPDATE-CUDA-PORTABILITY.md`

This must add `KQ-BACKLOG-CUDA-001` without making it a current blocker.

## Acceptance gates

### Backlog

- [x] `KQ-BACKLOG-CUDA-001` recorded in `docs/TASKS.md`
- [x] portable CUDA strategy recorded in `docs/ROADMAP.md`
- [x] changelog updated

### Build/correctness

- [x] CPU-only clean configure/build/test remains PASS
- [x] CUDA clean configure/build/test remains PASS
- [x] bandwidth benchmark builds with CUDA enabled
- [x] every supported transfer mode validates data
- [x] CUDA errors are checked/fail-closed

### Measurements

- [x] pageable H2D measured
- [x] pinned H2D measured
- [x] pageable D2H measured
- [x] pinned D2H measured
- [x] bidirectional explicitly documented as unsupported/skip on KQ-01
- [x] multiple transfer sizes measured
- [x] setup/allocation costs recorded where implemented
- [x] PCIe load-state sampling captured

### Evidence

- [x] canonical raw CSV/text evidence created
- [x] evidence path follows KQ-01/date convention
- [x] `docs/MEMORY-BANDWIDTH-BUDGET.md` updated with measured values
- [x] no performance claim lacks evidence

### Repository hygiene

- [x] no generated build trees tracked
- [x] no large profiler artifacts committed
- [x] no local secrets
- [x] no hard-coded user-specific paths in production code
- [x] working tree state reported explicitly

## Definition of done

Task 0.2E is complete only when the benchmark is correct, repeatable, evidence is captured, baseline documentation is updated, and CPU/CUDA correctness tests remain green.
