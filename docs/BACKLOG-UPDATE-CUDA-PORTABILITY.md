# BACKLOG-UPDATE-CUDA-PORTABILITY.md

## Purpose

Record the CUDA binary portability decision as an explicit deferred project item so it cannot be lost while Kestrel-Q remains focused on KQ-01.

## Canonical backlog item

**ID:** `KQ-BACKLOG-CUDA-001`
**Title:** Portable CUDA Binary Strategy
**Status:** DEFERRED
**Trigger:** Before the first public binary release
**Priority now:** Non-blocking

## Problem

Task 0.3B validated CUDA on KQ-01 only after switching the CUDA architecture policy to CMake `native`, producing an `sm_86` cubin for the RTX 3080.

This is correct for the current development machine, but `native` alone is not a public distribution strategy.

Before shipping public Windows binaries, Kestrel-Q must explicitly choose between:

- per-machine native builds;
- prebuilt binaries containing multiple CUDA architectures;
- a hybrid release strategy;
- another evidence-based approach.

## Required decision scope

When the trigger is reached, evaluate:

1. minimum supported NVIDIA compute capabilities;
2. `native` builds versus multi-architecture fat binaries;
3. cubin versus PTX fallback policy;
4. driver/toolkit compatibility implications;
5. binary-size impact;
6. startup/load impact;
7. build/release complexity;
8. benchmark/performance implications;
9. reproducibility and CI hardware constraints;
10. failure behavior on unsupported GPUs.

## Required governance

Before the first public binary release:

- create an ADR for the selected CUDA distribution architecture policy;
- define the supported GPU/compute-capability matrix;
- define CI/release validation;
- document whether source builds remain `native` by default;
- ensure unsupported hardware fails explicitly.

## Integration instructions

Update the current repository without replacing newer content.

### `docs/TASKS.md`

Add under the CUDA/community/release backlog, preserving the current structure:

```markdown
- [ ] `KQ-BACKLOG-CUDA-001` — Define portable CUDA binary strategy before first public binary release:
  - evaluate per-machine `native` builds;
  - evaluate multi-architecture fat binaries;
  - define minimum supported NVIDIA compute capabilities;
  - define cubin/PTX fallback policy;
  - measure binary-size/startup/performance trade-offs;
  - document the selected release policy in an ADR.
```

### `docs/ROADMAP.md`

Add to the release/community hardening phase:

```markdown
- Define portable CUDA release strategy (`native` vs multi-architecture/fat-binary policy) before the first public Windows binary release.
```

### `CHANGELOG.md`

Record the backlog/governance addition under Unreleased.

## Acceptance

The item is considered recorded when:

- it exists in `docs/TASKS.md`;
- it exists in `docs/ROADMAP.md`;
- `CHANGELOG.md` records the governance addition;
- the item remains explicitly DEFERRED and does not block KQ-01 development.
