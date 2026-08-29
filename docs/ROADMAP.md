# Roadmap

## R0 — Know the model
Architecture, tensor inventory, tokenizer, reference outputs, hardware baseline.

- Foundation Tasks 0.1–0.3 are complete: KQ-01 hardware, memory/bandwidth and
  Windows/CUDA toolchain baselines are recorded and validated.
- KQ-01 CUDA/PCIe transfer baseline captured with correctness-checked pageable
  and pinned H2D/D2H measurements; simultaneous bidirectional copy remains an
  explicit unsupported/skip case on the observed WDDM runtime.
- Task 1.0 pinned the official Hugging Face artifact revision and official Qwen
  research revision, captured a reproducible no-weight metadata manifest,
  documented the model-license boundary and accepted ADR 0004.
- `KQ-MODEL-SOURCE-001` is verified and the maintainer's merged Unsloth
  UD-Q4_K_XL GGUF is registered as `KQ-MODEL-ARTIFACT-001` with exact size,
  SHA-256, GGUF metadata, quantization evidence and pinned provenance. Task 1.0
  is complete/pass.
- Task 1.1 is complete/pass: the exact 48-layer GDN/QSA schedule, GR, MoE,
  PLE, MTP boundary, prefill/decode flow and persistent runtime state are
  evidence-backed against pinned sources. ADR 0005 is accepted for an initial
  text-only ordinary autoregressive path without vision or MTP acceleration.
- Task 1.2 is complete/pass: bounded HTTP Range capture reconciles all 131
  canonical shard headers and 1,658 tensors with zero weight-payload bytes,
  exact subsystem/static footprints, idealized quantization floors and
  batch-1 persistent-state scaling through 262,144 context tokens.
- Task 1.3 is complete/pass: all 1,658 canonical tensors and all 1,224 GGUF
  tensors reconcile with zero unexplained mappings; exact packed footprints,
  converter/quantizer provenance and the 384-byte split/merge overhead are
  evidence-backed. ADR 0006 accepts a staged direct-GGUF-first strategy while
  keeping canonical model semantics independent of the container.

## R1 — Read the model
Safe mapping, metadata, tensor inspection, diagnostics.

- The initial container decision is complete in ADR 0006. Production loader,
  mapping and malformed-artifact implementation remain future R1 work.

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
