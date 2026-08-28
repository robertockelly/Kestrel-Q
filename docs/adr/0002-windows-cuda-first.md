# ADR 0002 — Windows and NVIDIA CUDA first

Status: Accepted

## Context

The reference development machine is Windows with an NVIDIA GPU and constrained VRAM.

## Decision

The first accelerated backend will target NVIDIA CUDA on Windows.

The core runtime will be C17. CUDA implementation files may use the language mode required by NVCC, but will expose a C ABI to the core.

## Consequences

- Windows behavior can be optimized directly rather than inherited from a port.
- CUDA-specific opportunities can be explored early.
- Cross-platform abstraction is deliberately postponed.
