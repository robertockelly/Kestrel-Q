# Task 0.1 — Reference Hardware Target — KQ-01

Status: **COMPLETE / PASS**

This document defines the initial reference machine used to design, validate and benchmark Kestrel-Q.

## 1. Reference machine

Identifier: **KQ-01**

Purpose:

- primary development workstation;
- constrained-memory reference target;
- initial CUDA performance baseline;
- reference platform for VRAM/RAM/storage tiering experiments.

## 2. Operating system

- OS: **Windows 11 Pro 25H2**
- Build: **26200.9168**
- Architecture: **x64**
- Note: some legacy Windows APIs/registry fields report `Windows 10 Pro`; build and release information identify the installed platform as Windows 11 25H2.

## 3. CPU

- Processor: **AMD Ryzen 7 5800X**
- Architecture: x86-64
- Physical cores: **8**
- Logical processors: **16**
- Reported maximum clock: **3801 MHz**

## 4. Motherboard

- Manufacturer: **ASUSTeK COMPUTER INC.**
- Model: **TUF GAMING B550-PLUS WIFI II**
- Revision: **Rev X.0x**

## 5. GPU

- GPU: **NVIDIA GeForce RTX 3080**
- VRAM: **10,240 MiB**
- Driver model: **WDDM**
- NVIDIA driver: **595.97**
- Maximum CUDA version reported by driver: **13.2**
- CUDA Toolkit: **not installed/detected at baseline**
- `nvcc`: **not available at baseline**

### PCIe link

- GPU bus: `00000000:06:00.0`
- Maximum PCIe generation: **Gen 4**
- Maximum link width: **x16**
- Idle observed generation: **Gen 1**
- Idle observed width: **x16**

The observed Gen 1 link state is an idle power-management state and must not be treated as the sustained transfer capability of the GPU. Transfer benchmarks must verify the negotiated link state under load.

## 6. System memory

Total installed physical memory: **32 GiB** (reported 34,233,548,800 bytes)

Modules:

- 4 × 8 GiB
- Manufacturer: Corsair
- Part number: `CMK16GX4M2B3000C15`
- Rated family: DDR4-3000
- Reported configured clock: **2133 MT/s**
- Reported speed: **2133 MT/s**

### Important baseline finding

The modules are currently operating at 2133 MT/s rather than their nominal 3000 MT/s profile.

This configuration is intentionally preserved as the initial KQ-01 baseline. Any future XMP/DOCP or memory-frequency change must be benchmarked as a separate hardware baseline rather than silently replacing these results.

## 7. Storage

### Primary SSD

- Model: **Crucial BX500 1 TB**
- Media type: SSD
- Bus: **SATA**
- Capacity: approximately 1 TB

### Secondary disk

- Model: **TOSHIBA HDWT840**
- Media type: HDD
- Bus: SATA
- Capacity: approximately 4 TB

### NVMe

- **No NVMe device detected in the initial baseline**

This changes the initial storage-tier assumption for KQ-01.

Actual baseline:

```text
RTX 3080 10 GiB VRAM
        |
      PCIe 4.0 x16
        |
32 GiB DDR4 @ 2133 MT/s
        |
     SATA SSD
```

Kestrel-Q must therefore benchmark storage streaming against SATA SSD before assuming NVMe-class behavior.

## 8. Page file

- Location: `C:\pagefile.sys`
- Allocated size at capture: **30,546 MiB**
- Current usage at capture: **571 MiB**
- Peak usage reported: **30,703 MiB**

Paging must be monitored explicitly during inference benchmarks. A run that relies materially on Windows paging must be distinguished from intentional Kestrel-Q storage streaming.

## 9. Development toolchain at baseline

- CMake: **4.3.0-rc1**
- MSVC `cl.exe`: **not detected**
- Clang: **not detected**
- CUDA Toolkit / `nvcc`: **not detected**

The machine is therefore not yet considered build-ready.

Toolchain installation/version pinning belongs to a subsequent foundation task and must preserve this pre-installation baseline.

## 10. Initial architectural implications

KQ-01 establishes the following constraints:

1. 10 GiB VRAM is insufficient for full model residency.
2. 32 GiB RAM is also a constrained resource relative to the complete model.
3. Host-memory bandwidth is likely to matter materially.
4. PCIe 4.0 x16 provides a strong GPU/host transport ceiling, but real transfer throughput must be measured.
5. SATA SSD bandwidth/latency may become a major bottleneck for cold tensor/expert streaming.
6. Windows page-file activity must not be confused with intentional tiered-memory behavior.
7. Model-aware placement, prefetch and eviction are core architecture concerns rather than optional optimizations.

## 11. Baseline status

**TASK 0.1 — KQ-01 HARDWARE BASELINE: COMPLETE / PASS**

Next task:

**Task 0.2 — Memory & Bandwidth Budget**

The next phase will quantify:

- usable VRAM budget;
- usable RAM budget;
- reserve required for Windows and runtime overhead;
- theoretical and measured PCIe bandwidth;
- RAM bandwidth;
- SATA SSD throughput;
- expected model footprint by quantization;
- feasibility of VRAM/RAM/storage placement strategies.
