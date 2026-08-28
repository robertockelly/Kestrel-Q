# Vision

## Purpose

Kestrel-Q exists to test a specific proposition:

> A large modern MoE model can be made practically usable on a Windows consumer PC by designing the entire inference stack around that model and around a tiered memory hierarchy rather than treating GPU VRAM as the only useful memory.

The project is not primarily a portability exercise. It is an engineering investigation into specialization.

## Reference target

The first optimization target is a machine in the following class:

- Windows 11
- NVIDIA GPU, approximately 10 GB VRAM
- 32 GB system RAM
- fast NVMe SSD
- mainstream x86-64 CPU

Performance on larger machines is welcome, but optimizations must not quietly move the minimum target upward.

## Model target

Initial target: **Qwen3.8-Flash-Next**.

The architecture includes properties that make a specialized runtime especially interesting:

- mixture-of-experts execution;
- limited active parameters per token compared with total parameter count;
- hybrid attention mechanisms;
- sparse attention behavior;
- n-gram embeddings that are candidates for host-memory placement and asynchronous prefetch.

All model-specific assumptions must be verified against the official model artifacts and technical documentation before implementation.

## Success criteria

The project succeeds progressively.

### Level 1 — Correct

The runtime can reproduce reference outputs for a minimal supported configuration.

### Level 2 — Runnable

A quantized model can generate useful text on the reference Windows machine without exhausting VRAM or RAM.

### Level 3 — Usable

Interactive decode speed and startup behavior are acceptable for local development use.

### Level 4 — Efficient

Memory transfers, expert loading and SSD traffic are scheduled intentionally and measured.

### Level 5 — Integrated

CLI, local API, persistent sessions and coding-agent workflows form one coherent vertical stack.

## Philosophy

The runtime should be small enough that a contributor can eventually understand the complete inference path.

The project prefers:

- specialization over abstraction;
- explicit data movement over invisible framework behavior;
- reproducible evidence over performance folklore;
- model-aware scheduling over generic graph execution;
- graceful degradation over hard memory cutoffs.

## Long-term possibility

If the model-specific architecture proves successful, additional models may eventually be supported through separate specialized execution profiles. That is not a Phase-1 objective and must not compromise the original narrow design.
