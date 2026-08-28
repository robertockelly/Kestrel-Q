# Roadmap

## R0 — Know the model
Architecture, tensor inventory, tokenizer, reference outputs, hardware baseline.

- Foundation Tasks 0.1–0.3 are complete: KQ-01 hardware, memory/bandwidth and
  Windows/CUDA toolchain baselines are recorded and validated.
- KQ-01 CUDA/PCIe transfer baseline captured with correctness-checked pageable
  and pinned H2D/D2H measurements; simultaneous bidirectional copy remains an
  explicit unsupported/skip case on the observed WDDM runtime.

## R1 — Read the model
Safe mapping, metadata, tensor inspection, diagnostics.

## R2 — Compute correctly
Minimal CPU reference inference and reference-vector parity.

## R3 — Use the GPU
CUDA baseline with correctness parity.

- Defer `KQ-BACKLOG-BENCH-001`, the WDDM CUDA allocation-headroom benchmark,
  until production VRAM allocator or memory-placement capacity policy needs it.

## R4 — Fit the machine
VRAM/RAM/NVMe tiering, expert cache, async prefetch, streaming.

## R5 — Make it fast
Quantization, fused kernels, scheduler optimization, profiling.

## R6 — Make context practical
State management, persistence and long-context behavior.

## R7 — Make it usable
CLI, server, configuration and Windows packaging.

## R8 — Make it an agent
Native coding workflow, tools and persistent sessions.

## R9 — Make it a community project
CI, benchmark submissions, compatibility matrix, release governance.

- Define the portable CUDA release strategy (`native` versus
  multi-architecture/fat-binary policy) before the first public Windows binary
  release. `KQ-BACKLOG-CUDA-001` is deferred and does not block KQ-01 work.
