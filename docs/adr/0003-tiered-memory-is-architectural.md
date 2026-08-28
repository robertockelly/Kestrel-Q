# ADR 0003 — Tiered memory is architectural

Status: Accepted

## Context

The target model is much larger than reference GPU VRAM. System RAM is also constrained relative to the complete model representation.

## Decision

VRAM, pinned RAM, pageable/mapped RAM and NVMe storage are treated as explicit execution tiers.

Tensor placement, prefetch and eviction are model-execution concerns and will be observable by benchmarks.

## Consequences

- scheduler design becomes central;
- storage performance can affect token throughput;
- cold/warm states must be benchmarked separately;
- correctness must survive dynamic movement of tensors.
