# ADR 0022 — Native sampling policy boundary

Status: **Accepted**

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

## Decision

1. Keep model/logit production separate from token-selection policy.
2. Introduce a narrow native C17 sampler that consumes an explicit complete
   logit vector, immutable validated policy and caller-owned mutable RNG state.
3. Require no GGUF/model owner in the sampler core and no global mutable RNG.
4. Make the official Qwen3.8 default profile mandatory while rejecting
   uncharacterized processors/options.
5. Preserve the accepted greedy argmax as a stable regression and avoid a
   second drifting greedy definition.
6. Use an independently implemented PCG-XSH-RR 64/32 generator, pinned to
   `pcg-c-basic@bc39cd76ac3d541e618606bcc6e1e5ba5e5e6aa3` (Apache-2.0).
   State format v1 records the 64-bit seed, 63-bit stream, internal state,
   stream-derived odd increment, successful draw count and an integrity word.
   Seeding uses the published set-sequence expansion and unsigned arithmetic
   wraps modulo its width.
7. Commit sampler state only after successful selection. Task 3.1 will include
   that state in the whole-token executor transaction.
8. Validate discrete processor/RNG/selection results exactly, floating
   transforms with calibrated holdout contracts and repeated stochastic draws
   with a predeclared statistical test.
9. Keep production free of Python, Transformers and llama.cpp dependencies;
   they remain pinned offline oracle tools only.
10. Do not turn this boundary into a generic arbitrary-model generation
    framework.

The supported model-specific processor surface is finite positive F32
temperature, top-k `0..248320` (zero disables it) and top-p `[0,1]` (one
disables it). The official values are temperature 1.0, top-k 20 and top-p
0.95. Processing is temperature division, top-k threshold filtering with all
threshold ties retained, then ascending-cumulative top-p filtering with the
equality boundary removed, followed by scalar F32 softmax in token-ID order.

Every successful selection consumes exactly one PCG word. It maps to binary64
`word / 2^32`; categorical traversal uses ascending token IDs and half-open
intervals, so an exact internal boundary selects the following token. The 243
model-capacity padding IDs participate because the pinned canonical path does
not suppress them. Selecting one fails with `INVALID_TOKEN_ID` and commits
neither output nor RNG state. EOG IDs remain ordinary eligible outputs;
generation stopping belongs to Task 3.1.

The immutable config may be shared for concurrent read-only use. RNG state is
caller-owned and may not be mutated concurrently. Selection borrows config,
logits, scratch, state and result only for the call. No global mutable state,
GGUF/model owner, tokenizer, stopping policy or executor dependency is part of
this boundary.

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

## Acceptance evidence

Task 3.0 accepted this ADR after recording:

- the complete pinned sampling contract and supported option set;
- the exact RNG algorithm/state/seed/mapping decision and provenance;
- independent exact transform/RNG/selection vectors;
- calibrated floating and disjoint statistical holdout results;
- a fail-closed native C17 API with transactional state; and
- clean regressions proving the greedy executor is unchanged.

The independent evidence is under
`research/sampling/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/`.
Seven calibration and seven disjoint holdout cases pass exact processor,
PCG-state and selected-ID comparisons. Processed retained logits are bit exact;
probabilities observed at most 1 ULP error against the pinned CPU PyTorch
oracle under a frozen 3 ULP and `5.960464477539063e-08` absolute contract. A
predeclared 100,000-draw-per-case Hoeffding gate over 25 category assertions
passes independently and natively. Fail-closed tests cover invalid policies,
non-finite inputs, padding-ID selection, capacity/aliasing errors, corrupted
state and incompatible rounding mode while preserving caller state.

Task 3.0 does not alter the accepted greedy executor, access model payload or
integrate sampling with generation. That whole-token model-plus-RNG transaction
remains Task 3.1.
