# Task 0.2 — Memory & Bandwidth Budget

Status: **COMPLETE / PASS**

Reference machine: **KQ-01**

## 1. Clean-idle memory baseline and design envelope

### System RAM

Windows reported:

- total visible memory: **33,431,200 KiB** (~31.88 GiB);
- free physical memory: **23,755,864 KiB** (~22.66 GiB);
- memory already in use: **~9.22 GiB**.

### GPU VRAM

`nvidia-smi` reported at clean idle:

- total VRAM: **10,240 MiB**;
- used VRAM: **888 MiB**;
- free VRAM: **9,166 MiB** (~8.95 GiB);
- power state: **P8**;
- current PCIe link: **Gen1 x16**.

The Gen1 link is the idle power-management state. Task 0.2E separately verified
Gen4 x16 while the CUDA transfer benchmark was active.

### Validated planning budgets

Current planning budgets remain:

- physical VRAM: 10 GiB
- initial Kestrel-Q VRAM working budget: ~8 GiB
- physical RAM: 32 GiB
- initial Kestrel-Q host model/cache budget: ~20 GiB
- Windows page file is **not** counted as an intentional model-memory tier

The ~20 GiB host budget leaves approximately 2.66 GiB of observed clean-idle
headroom. The ~8 GiB VRAM budget leaves approximately 0.95 GiB beyond the
observed clean-idle budget while preserving memory already used by the desktop.
These are validated conservative starting budgets.

Task 0.2D records clean-idle capacity only. Maximum safe CUDA allocation
headroom under WDDM is a separate deferred benchmark,
`KQ-BACKLOG-BENCH-001`, and is not part of Task 0.2D.

## 2. Measured RAM bandwidth

WinSAT memory assessment:

- reported bandwidth: **25,628.78759 MB/s**
- equivalent: **~25.63 GB/s**
- assessment: success
- test: `MemCopy_128_SSE_UCW_BPF16K_NBT`
- threads used: 8
- logical processors visible: 16

Theoretical DDR4-2133 dual-channel bandwidth:

- 2133 MT/s × 8 bytes × 2 channels
- **~34.1 GB/s**

Observed efficiency against that simple theoretical ceiling:

- **~75.1%**

### Interpretation

The current DDR4-2133 configuration provides a real memory-copy baseline of roughly 25.6 GB/s.

For Kestrel-Q this is materially below the GPU's PCIe 4.0 x16 theoretical payload ceiling, so host-memory behavior can become a first-order constraint for:

- expert-cache churn
- copy-heavy staging
- dequantization pipelines
- RAM→pinned-RAM→VRAM movement

The installed DDR4-3000-capable modules are intentionally left at the current 2133 MT/s configuration for the KQ-01 baseline. Any XMP/DOCP change must create a new benchmark baseline rather than silently replacing this result.

## 3. Measured SATA SSD sequential-read bandwidth

WinSAT sequential read assessment on the Crucial BX500:

- primary reported average throughput: **439.60875 MB/s**
- equivalent: **~0.440 GB/s**
- assessment: success
- I/O size: 64 KiB

Observed zone-level throughput varied substantially, with examples from approximately:

- low: **235.95 MB/s**
- high: **504.01 MB/s**

### Interpretation

The initial planning assumption of “~0.5 GB/s SATA-class sequential read” was directionally correct, but the measured system-level baseline is closer to **0.44 GB/s**.

This confirms that SSD access must be treated as a scarce operation.

At 0.44 GB/s, reading a hypothetical 3 GB cold working set would require approximately:

- 3 / 0.44 ≈ **6.8 seconds**

before compute or PCIe transfer.

Therefore per-token cold expert loading from SATA is not a viable execution strategy.

## 4. Measured CUDA/PCIe transfers

Task 0.2E used the dedicated `kq_cuda_bandwidth` CUDA executable on KQ-01.
Each size received three warm-up transfers and ten measured transfers.

Two timing views are reported:

- **host GB/s** uses a high-resolution host clock around enqueue plus explicit
  stream synchronization;
- **CUDA GB/s** uses CUDA events around the copy in its stream.

Allocation, pattern generation and validation are outside both timed regions.
Every warm-up and measured transfer uses a changing deterministic byte pattern
and receives a byte-complete host comparison. Setup seeding is independently
copied back and validated.

All bandwidth values below use decimal GB/s. The size column uses MiB.

| Size | Pageable H2D host / CUDA | Pinned H2D host / CUDA | Pageable D2H host / CUDA | Pinned D2H host / CUDA |
|---:|---:|---:|---:|---:|
| 1 MiB | 7.352 / 7.666 | 17.303 / 19.374 | 5.804 / 6.014 | 14.264 / 15.678 |
| 4 MiB | 8.179 / 8.282 | 22.313 / 23.033 | 6.976 / 7.049 | 20.762 / 21.520 |
| 16 MiB | 7.593 / 7.608 | 24.781 / 24.968 | 7.303 / 7.398 | 23.733 / 23.886 |
| 64 MiB | 8.043 / 8.045 | 25.349 / 25.395 | 8.837 / 8.843 | 25.077 / 25.453 |
| 256 MiB | 9.132 / 9.132 | 26.073 / 26.080 | 8.838 / 8.868 | 26.218 / 26.242 |

Every pageable and pinned H2D/D2H row passed correctness validation.

At 256 MiB, pinned memory improved host-visible throughput by approximately:

- **2.85×** for H2D versus pageable memory;
- **2.97×** for D2H versus pageable memory.

The large pinned-transfer results are approximately 82.8% H2D and 83.2% D2H
of the ~31.5 GB/s theoretical PCIe 4.0 x16 payload ceiling. The theoretical
ceiling remains a calculation, not a measurement.

### Bidirectional/concurrent transfer

CUDA reported `asyncEngineCount = 1` for the RTX 3080 under the observed WDDM
runtime. The benchmark requires at least two asynchronous copy engines before
claiming simultaneous H2D and D2H support, so all five bidirectional cases were
reported as **SKIP**, not as zero-bandwidth passes. No aggregate bidirectional
throughput is claimed.

### Setup and allocation observations

The canonical run allocated two 256 MiB host buffers of each host-memory type
and two 256 MiB device buffers.

| Operation | Bytes | Host-visible time |
|---|---:|---:|
| Pageable `malloc`, source | 256 MiB | 0.0103 ms |
| Pageable `malloc`, destination | 256 MiB | 0.0087 ms |
| Touch two pageable buffers | 512 MiB | 79.7660 ms |
| `cudaMallocHost`, source | 256 MiB | 26.2726 ms |
| `cudaMallocHost`, destination | 256 MiB | 28.0445 ms |
| Touch two pinned buffers | 512 MiB | 34.3182 ms |
| `cudaMalloc`, H2D buffer | 256 MiB | 0.5401 ms |
| `cudaMalloc`, D2H buffer | 256 MiB | 0.7155 ms |
| Create H2D / D2H streams | n/a | 0.0160 / 0.0353 ms |
| Create four CUDA events | n/a | 0.0092 / 0.0008 / 0.0005 / 0.0003 ms |
| Seed device source from pageable RAM | 256 MiB | 33.2408 ms |

Pageable `malloc` is lazy and its sub-0.02 ms call latency does not represent
committed physical memory. The separate touch cost is the meaningful setup
observation. Pinning two 256 MiB buffers cost approximately 54.3 ms in total,
so large page-locked pools should be long-lived and bounded rather than created
per transfer.

### PCIe link state under load

The capture harness collected 35 `nvidia-smi` samples while the benchmark
process was active:

- 1 initial sample: Gen1 x16, P8;
- 34 load samples: **Gen4 x16, P2**.

This verifies that KQ-01 negotiates its maximum Gen4 x16 link during the tested
transfer workload. It does not turn the ~31.5 GB/s theoretical ceiling into a
measured value.

## 5. Updated KQ-01 bandwidth landscape

```text
RTX 3080 VRAM
    ↕
PCIe 4.0 x16, measured large pinned copies
~26.1 GB/s H2D / ~26.2 GB/s D2H
~31.5 GB/s theoretical payload ceiling per direction
    ↕
DDR4-2133
~25.63 GB/s measured WinSAT copy bandwidth
    ↕
Crucial BX500 SATA SSD
~0.44 GB/s measured sequential read
```

The RAM and PCIe measurements use different tools and transfer semantics, so
their close values should not be interpreted as a precise ranking. They do show
that pinned PCIe transfers can consume a host-memory-bandwidth-scale resource.
Pageable staging reaches only about 9 GB/s for large transfers.

Storage remains the dominant external bottleneck: a 3 GB SATA miss is roughly
6.8 seconds at 0.44 GB/s, while a 3 GB pinned H2D transfer is roughly 0.12
seconds at 26.1 GB/s before any compute. These transport times are not model
throughput predictions.

## 6. Architectural consequences

1. **SSD misses must be rare.** SATA is roughly 59× slower than the measured
   large pinned PCIe path.
2. RAM should act as the large expert/tensor cache; reactive per-token SATA
   loading is not viable.
3. Reusable, bounded pinned staging pools are justified. Per-transfer pinning
   would add tens of milliseconds at 256 MiB.
4. Pageable transfers leave roughly a 3× throughput penalty at large sizes, so
   hot prefetch paths should not rely on implicit runtime staging.
5. Transfer sizes should be batched where possible: pinned throughput rises
   from ~14–17 GB/s host-visible at 1 MiB to ~26 GB/s at 256 MiB.
6. KQ-01 cannot currently demonstrate simultaneous bidirectional copies through
   this CUDA/WDDM path. Future overlap design must not assume two copy engines.
7. H2D prefetch may still overlap with compute where supported, but that must be
   measured with representative kernels rather than inferred from PCIe alone.
8. Windows paging must remain separate from intentional mmap/streaming metrics.

The likely Kestrel-Q data path remains:

```text
cold tensor / expert on SATA
        ↓
mapped / streamed read
        ↓
RAM cache
        ↓
bounded pinned staging pool
        ↓
VRAM hot set
```

Every transition should eventually expose counters and timing.

## 7. Development findings and corrected baseline

Two normal implementation issues were found before canonical capture:

1. CUDA 13.3 no longer exposes the legacy `cudaDeviceProp::deviceOverlap`
   member. The benchmark now uses the supported `asyncEngineCount` capability
   and fails/skips explicitly when concurrent bidirectional copy is unavailable.
2. An initial scalar byte-by-byte validator created long GPU-idle gaps and
   depressed the 256 MiB pinned D2H result to ~6.7 GB/s. Replacing it with an
   optimized byte-complete comparison removed the measurement artifact. Clean
   builds/tests and the full matrix were rerun before canonical evidence was
   captured.

Task 0.3B's `native` CUDA architecture policy remains unchanged; the clean build
produced native code and did not reintroduce the incompatible PTX-only path.

## 8. Baseline evidence policy

Task 0.2E canonical evidence is immutable under:

```text
bench/results/raw/KQ-01/2026-08-28/
  cuda-bandwidth.csv
  cuda-bandwidth-console.txt
  pcie-link-samples.csv
```

The bandwidth CSV records commit `56ac196e19038fa71f48073b1493849de22a9fab-dirty`,
the complete setup/transfer matrix, runtime/device metadata, iteration counts,
timing labels and correctness status. Large traces and profiler databases remain
ignored by Git.

## 9. Deferred follow-up outside Task 0.2

`KQ-BACKLOG-BENCH-001` tracks a future WDDM CUDA allocation-headroom benchmark.
It must establish a bounded capacity policy before the production VRAM allocator
or memory-placement policy depends on a maximum allocatable value. Task 0.2E's
256 MiB setup observations are not an allocation-pressure test.

## 10. Current status

**TASK 0.2A — THEORETICAL MEMORY/BANDWIDTH BUDGET: PASS**

**TASK 0.2B — RAM BANDWIDTH: PASS**

**TASK 0.2C — SATA SEQUENTIAL READ: PASS**

**TASK 0.2D — CLEAN-IDLE MEMORY BASELINE: PASS**

**TASK 0.2E — CUDA/PCIe MEASURED TRANSFERS: PASS**

**TASK 0.2 — MEMORY & BANDWIDTH BUDGET: COMPLETE / PASS**
