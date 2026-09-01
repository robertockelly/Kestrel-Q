# ADR 0023 — Sampled generation state transaction

Status: **Accepted**

## Context

Task 3.0 establishes native sampling policy and portable caller-owned RNG.
Epic 2 already establishes transactional incremental model decode. Sampled
generation must compose these state machines without duplicating sampling
semantics or allowing partial model/RNG advancement.

## Decision

1. Task 3.0 remains the sole sampling-mathematics/RNG authority.
2. The model executor invokes the sampler through its native API.
3. Prompt prefill occurs exactly once.
4. A later sampled step stages model state and RNG state together.
5. A failed step rolls back both while preserving the accepted prefix.
6. EOG remains a valid sampler output; generation integration owns stop.
7. Padded-ID selection remains a transactional failure.
8. Explicit seed/stream generation replays deterministically.
9. Expected integration selections are computed by an independent sampler over
   identical recorded native logits and pre-step RNG state.
10. Performance, scheduler, cache, SIMD and CUDA work remain outside Epic 3.

## Evidence gate

Require:
- primary real sampled trace PASS
- disjoint holdout seed PASS
- exact independent per-step selection/RNG equality
- deterministic replay PASS
- early/middle/late and sampler-failure rollback PASS
- EOG control PASS
- greedy regression PASS
- CPU/CUDA regressions PASS

The real primary case `(seed=0, stream=5427223837140668492)` and disjoint
holdout `(seed=18446744073709551615, stream=9223372036854775807)` each return
`[271, 248068, 198, 760]` from one prompt prefill and three incremental
decodes. Their PCG words/states differ and an exact reset replay of the primary
case matches model/RNG/request-order state. A separate implementation over
the temporary full logits validates every retained-order hash, RNG transition
and selected ID. Temporary logits remain ignored.
The bounded deterministic evidence manifest is
`sampled-generation-manifest.json`, SHA-256
`2506073d9e9c4d0733cd109429e74b420e755a161193f815ea3bb6bbb951097a`.

Injected early, middle, late, post-decode sampler, padding-ID, corrupt-RNG and
output-alias failures leave model state, RNG and caller output unchanged. EOG
controls commit exactly one selection and reject subsequent sampled decode;
context exhaustion has the same combined rollback. Greedy M1 and Task 2.13
evidence remain unchanged.

The decision is accepted for the model-specific scalar CPU correctness path.
It creates neither a product session format nor a scheduler, cache, SIMD or
CUDA model policy.

## Consequences

- Reproducibility requires the caller to retain and supply the explicit RNG
  state alongside the model state.
- A sampled call may use additional caller sampling scratch but owns no hidden
  mutable state.
- Model arithmetic and sampling remain independently testable, and failures
  can never publish a token whose associated state did not commit.
- EOG loop termination and maximum-token policy remain caller concerns; the
  sampled decode API rejects EOG input to catch stopped-state reuse.
