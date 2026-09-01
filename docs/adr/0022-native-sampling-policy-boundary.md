# ADR 0022 — Native sampling policy boundary

Status: **Proposed**

## Context

Epic 2 produces all target logits and supports exact incremental greedy
generation with explicit transactional model state. R2 still lists sampling as
its only unfinished workstream. Sampling introduces logit transforms,
probability construction and random state whose ownership and comparison rules
must not be hidden inside model arithmetic.

The official Qwen3.8 generation configuration enables sampling with
temperature 1.0, top-k 20 and top-p 0.95. That configuration does not select a
portable C RNG contract, and pinned Transformers/llama.cpp may use different
random algorithms and floating paths.

## Proposed decision

1. Keep model/logit production separate from token-selection policy.
2. Introduce a narrow native C17 sampler that consumes an explicit complete
   logit vector, immutable validated policy and caller-owned mutable RNG state.
3. Require no GGUF/model owner in the sampler core and no global mutable RNG.
4. Make the official Qwen3.8 default profile mandatory while rejecting
   uncharacterized processors/options.
5. Preserve the accepted greedy argmax as a stable regression and avoid a
   second drifting greedy definition.
6. Select and pin a portable deterministic RNG algorithm, seed expansion,
   state version and uniform/categorical mapping during Task 3.0 before native
   validation. ADR acceptance requires those exact details and independent
   vectors.
7. Commit sampler state only after successful selection. Task 3.1 will include
   that state in the whole-token executor transaction.
8. Validate discrete processor/RNG/selection results exactly, floating
   transforms with calibrated holdout contracts and repeated stochastic draws
   with a predeclared statistical test.
9. Keep production free of Python, Transformers and llama.cpp dependencies;
   they remain pinned offline oracle tools only.
10. Do not turn this boundary into a generic arbitrary-model generation
    framework.

## Alternatives considered

### Put sampling directly in the model executor

Rejected because it couples model-state rollback to unvalidated RNG and makes
synthetic sampler failures require a 111 GB model run.

### Delegate to CRT/OS randomness

Rejected because hidden platform state and implementation-defined distributions
would break reproducibility and snapshot/rollback semantics.

### Require llama.cpp or Transformers random token equality

Rejected unless their full processor, floating and RNG semantics are proven
identical. Different valid RNG algorithms do not imply an incorrect sampling
distribution.

### Treat GGUF sampling metadata as the sole authority

Rejected because it records default numbers but not the complete canonical
processor, token-domain or RNG contract.

## Consequences

Positive:

- sampler math and randomness can be validated without model payload;
- explicit state supports deterministic replay and transactional rollback;
- greedy behavior remains isolated and stable; and
- later CUDA/model optimization cannot silently change selection policy.

Negative:

- Kestrel-Q must govern a portable RNG and its compatibility surface;
- exact random token equality is valid only under tightly matched inputs and
  algorithms; and
- sampled generation requires a second integration task after sampler-core
  acceptance.

## Acceptance evidence required

ADR 0022 remains proposed until Task 3.0 records:

- the complete pinned sampling contract and supported option set;
- the exact RNG algorithm/state/seed/mapping decision and provenance;
- independent exact transform/RNG/selection vectors;
- calibrated floating and disjoint statistical holdout results;
- a fail-closed native C17 API with transactional state; and
- clean regressions proving the greedy executor is unchanged.
