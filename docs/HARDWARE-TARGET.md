# Reference Hardware Target

This file must be completed with the exact development machine before serious optimization begins.

## Current target class

- OS: Windows 11
- GPU vendor: NVIDIA
- GPU VRAM: ~10 GB
- System RAM: 32 GB
- Storage: NVMe SSD
- CPU: x86-64

## Fill before Phase 1

- GPU exact model:
- GPU compute capability:
- PCIe generation/link width:
- NVIDIA driver:
- CUDA toolkit:
- CPU exact model:
- physical cores:
- logical cores:
- RAM speed/configuration:
- NVMe exact model:
- NVMe sequential read:
- Windows version/build:
- compiler:
- CMake:
- power profile:

## Why this matters

Kestrel-Q will optimize data movement. GPU model alone is insufficient: PCIe topology, host-memory bandwidth and SSD behavior can materially affect the best scheduler.
