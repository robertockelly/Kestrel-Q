# PLAN-EPIC-3-CPU-CORRECTNESS-ENGINE.md

Status: **IN PROGRESS — IMPLEMENTATION NOT STARTED**

## Objective

Close the remaining R2 / Project Plan Phase 2 correctness workstream by adding
a small, model-specific native CPU sampling boundary and then integrating that
boundary with the already accepted incremental model executor.

Epic 3 does not recreate the CPU forward path completed in Epic 2. It builds on
the exact registered GGUF, native tokenizer, scalar numerics and operators,
48-layer executor, full logits, greedy selection, EOG handling, explicit model
state and per-token transaction already accepted through Task 2.13.

Published entry baseline:
`b4a2565bd49684fd6b9bd81724d667a7bf7cc11c`.

## Governed scope evidence

The committed roadmap and project plan establish the following boundary:

- `docs/PROJECT-PLAN.md` Phase 2 is the CPU correctness runtime. Its workstreams
  include sampling, and its exit gate requires reference-logit parity plus
  controlled greedy-generation parity.
- `docs/ROADMAP.md` R2 is **Compute correctly**. Every listed scalar, target
  execution, logits and incremental-greedy item is complete/pass; sampling is
  the only explicitly remaining Epic 3 / R2 item.
- `docs/TASKS.md` already records tokenizer, embedding, model blocks,
  GDN/QSA/MoE/PLE, logits and reference-vector validation as complete. Its only
  unchecked Epic 3 execution item is sampling.
- ADR 0021 explicitly leaves stochastic sampling outside the accepted greedy
  executor while preserving prompt-prefill-once, direct one-token decode and
  per-token rollback as the integration substrate.

The pinned official `generation_config.json` is already present in the Task
1.0 source manifest with SHA-256
`e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`.
It records `do_sample=true`, `temperature=1.0`, `top_k=20`, `top_p=0.95`, EOG
IDs 248046 and 248044, and pad ID 248044. These values establish a mandatory
model-specific profile to characterize; they do not by themselves define
processor order, floating arithmetic, random-number generation or token
eligibility.

## Entry criteria

All entry criteria are satisfied at the published baseline:

1. Epic 0, Epic 1 and Epic 2 are COMPLETE/PASS.
2. Tasks 2.0-2.13 are COMPLETE/PASS and ADRs 0008-0021 retain their accepted
   status.
3. M1 First Correct Native Token and the exact four-token incremental greedy
   continuation are PASS against the pinned independent Class-Q oracle.
4. The executor exposes full finite F32 logits, deterministic lower-ID-tie
   greedy argmax, explicit bounded model state, EOG identity and transactional
   prefill/decode.
5. The two-oracle/no-self-oracle policy in ADR 0007 is accepted.
6. The working tree was clean at inception and both backlog benchmarks remain
   deferred.

## Already satisfied versus remaining gaps

| Capability | Epic 2 status | Epic 3 consequence |
| --- | --- | --- |
| Native tokenizer and token decode | Complete via Task 2.3 | Reuse unchanged |
| Scalar storage, numerics and target operators | Complete via Tasks 2.5-2.11 | Do not recreate |
| Embedding, 48 layers, final mix and all logits | Complete via Task 2.12 | Logits are sampler input |
| Greedy argmax and exact token parity | Complete via Tasks 2.12-2.13 | Preserve as regression/fallback |
| Incremental state and rollback | Complete via Task 2.13 | Extend transaction to sampler state |
| EOG IDs and max/context stop | Complete for greedy | Revalidate through sampled integration |
| Sampling transformations and selection | Absent | Task 3.0 |
| Explicit portable RNG ownership/state | Absent | Task 3.0 |
| Sampled incremental generation | Absent | Task 3.1 |
| Sampling-specific independent evidence | Absent | Tasks 3.0-3.1 |

Sampling is a correctness gap because the official model generation profile
selects sampling by default while the public executor currently exposes only
greedy argmax and has no RNG or sampling API. It is not a performance gap.

## Task sequence

1. **Task 3.0 — Native Sampling Policy & Deterministic Selection Primitives**
   — **NOT STARTED**.
2. **Task 3.1 — Sampled Incremental Generation & Oracle Validation** —
   **PLANNED / NOT STARTED**. Its task-specific package is created only after
   Task 3.0 is reviewed and checkpointed.

Dependency graph:

```text
Epic 2 accepted logits + tokenizer + incremental state
                         |
                         v
Task 3.0 sampling transforms + explicit RNG + selection evidence
                         |
                         v
Task 3.1 sampled executor integration + stop/rollback/oracle evidence
                         |
                         v
              Epic 3 / R2 closure review
```

The two tasks are separate because logit processing, probability construction,
RNG and categorical selection can be validated without loading model weights,
while sampled generation changes the transaction boundary of the real model
session. Combining them would make model-level failures obscure sampler-core
defects.

## Task 3.0 acceptance gate

Task 3.0 must:

1. Characterize the exact pinned Class-C generation path before production
   code, including supported processor order, score dtype, temperature, top-k,
   top-p, softmax/categorical semantics, token eligibility and special/padded
   ID behavior.
2. Preserve the official Qwen3.8 default sampling profile and reject
   unsupported processors/options rather than silently ignoring them.
3. Select and pin a portable deterministic RNG algorithm, seed expansion,
   state layout, uniform mapping and draw-consumption rule. CRT `rand()`, host
   locale and hidden global state are forbidden.
4. Implement an immutable C17 sampling configuration plus explicit
   caller-owned mutable RNG state, separate from model execution.
5. Compare candidate membership/order and selected IDs exactly where discrete;
   compare floating transformed scores/probabilities under independently
   calibrated per-field contracts; validate RNG words/uniform mapping exactly.
6. Use a predeclared statistical corpus and acceptance rule for distributional
   behavior that is not validly token-for-token comparable. Calibration and
   holdout cases must be disjoint.
7. Preserve the existing greedy argmax behavior and EOG identities unchanged.
8. Fail closed on invalid/non-finite logits, unsupported settings, invalid
   probability mass, invalid token domain, insufficient capacity, arithmetic
   overflow, invalid/corrupted RNG state and forbidden aliasing.
9. Add no model payload access, executor/CLI integration, performance kernel,
   scheduler or cache behavior.
10. Finalize ADR 0022 only after the contract, implementation and independent
    evidence pass.

## Task 3.1 acceptance gate

Task 3.1 must:

1. Integrate the accepted Task 3.0 sampler after full-logit production without
   moving model arithmetic into the sampler.
2. Preserve one prompt prefill, direct generated-ID decode, bounded context,
   EOG/max-token stopping and no prompt replay.
3. Treat model state, sampler RNG state and caller-visible token publication as
   one per-generated-token transaction. A failure after RNG consumption must
   restore the pre-step RNG and model state; retry must reproduce the same
   independently expected decision.
4. Preserve the frozen greedy M1 and four-token Task 2.13 sequences exactly.
5. Demonstrate same-input/config/seed reproducibility exactly and compare every
   sampling decision to an independent implementation over the identical
   recorded logit input. Native logits may be test input to the independent
   sampler, but Kestrel-Q may not generate its own expected selection rule or
   replace the existing Class-Q model oracle.
6. Use the pinned Class-Q runtime for model/logit diagnostics where supported.
   Built-in llama.cpp sampled-token equality is not an acceptance gate unless
   its processor and RNG semantics are first proven identical.
7. Cover the official default profile across a bounded governed prompt/seed
   matrix, plus synthetic EOG, context exhaustion and injected post-selection
   failure paths. Evidence must predeclare exact versus statistical fields.
8. Record bounded state, logical payload and memory counters as
   characterization only; no performance claim or new payload ceiling may hide
   prompt replay or complete-matrix materialization.
9. Add no batching, product session abstraction, long-context persistence,
   optimization, cache/prefetch, SIMD or CUDA model execution.

## Epic 3 exit criteria

Epic 3 is COMPLETE/PASS only when:

- Tasks 3.0 and 3.1 are COMPLETE/PASS;
- the official Qwen3.8 default sampling profile is characterized and supported
  by the native CPU path;
- sampler configuration and RNG state have explicit ownership, lifetime,
  reset/snapshot and fail-closed contracts;
- exact processor masks/order, exact RNG vectors and exact fixed-input sampled
  decisions pass independent validation;
- calibrated floating and predeclared statistical holdout gates pass without
  widening a contract after observing failure;
- sampled incremental generation preserves prompt-prefill-once, state
  continuity, EOG/context/max-token semantics and transactional rollback;
- greedy Task 2.12/2.13 evidence remains unchanged and passing;
- production retains no Python, Transformers, llama.cpp or hidden global RNG
  dependency;
- clean CPU and CUDA regression suites pass with no new Kestrel-Q warning;
- no runtime/model artifact, secret or user-specific path is committed; and
- Epic 3 closure is recorded separately after its task gates pass.

These are correctness gates, not speed, quality or product-readiness claims.

## Oracle and comparison policy

### Class C

Canonical sampling transforms are governed by the official model revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, its pinned
`generation_config.json`, and Transformers revision
`805a9e939fa8c1bff8d8ffdf041c051b71a914aa`. Task 3.0 must pin the exact
executed source locations and hashes before creating expected vectors.

- processor inclusion/order and retained token IDs: `EXACT_DISCRETE`;
- RNG algorithm vectors and sampled ID from a fixed distribution/state:
  `EXACT_BITS` / `EXACT_DISCRETE`;
- transformed logits and probabilities: field-specific `CALIBRATED_FLOAT`;
- stochastic population behavior: predeclared statistical comparison only.

### Class Q

The exact registered GGUF and pinned
`llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a` remain the model/logit
reference. GGUF `general.sampling.*` metadata is compatibility evidence, not a
replacement for canonical semantics. A different sampler/RNG must not be
declared wrong merely because its random token sequence differs; exact token
equality is required only when processor, RNG and input-logit semantics are
proven identical.

Kestrel-Q output never defines expected transforms, RNG vectors or reference
token decisions.

## Explicitly excluded future work

| Work | Existing governed destination |
| --- | --- |
| CUDA optimized model kernels and device allocation | Epic 4 / R3 |
| `KQ-BACKLOG-BENCH-001` | Deferred until production WDDM VRAM capacity policy |
| Expert cache, PLE cache/prefetch and memory tiering | Epic 5 / R4 |
| Final scheduler/residency design | Epic 5 / R4-R5, gated by BENCH-002 |
| SIMD and other performance optimization | R5; no numbered task defined |
| Long-context production execution and session serialization | Epic 7 / R6 |
| Server/product UI and production CLI surface | Epic 8 / R7 |
| Batching | Future work; no governed task currently defined |
| MTP/speculative decoding | Future capability outside initial scope |
| Vision | Future capability outside initial text-only scope |

`KQ-BACKLOG-BENCH-001` remains **DEFERRED** and is not required by either Epic
3 task. `KQ-BACKLOG-BENCH-002` remains **DEFERRED / REQUIRED BEFORE FINAL
SCHEDULER DESIGN** and is also not required by either Epic 3 task. Neither
benchmark is run during Epic 3 inception.

## Initial ADR

`docs/adr/0022-native-sampling-policy-boundary.md` is **PROPOSED** because the
separation between model-logit production, selection policy and caller-owned
RNG state is a durable runtime boundary. It is not accepted by this planning
iteration and must be finalized from Task 3.0 evidence.

`EPIC_3_INITIAL_ADR = ADR 0022 / PROPOSED`.
