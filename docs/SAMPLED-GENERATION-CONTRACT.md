# Sampled incremental generation contract

Status: **TASK 3.1 COMPLETE / PASS**

This contract composes the accepted Task 2.13 incremental model lifecycle and
the accepted Task 3.0 sampler. It does not change either model arithmetic or
sampling mathematics. The supported path is batch 1, text only, scalar CPU,
bounded context and the exact Qwen3.8 default sampling profile.

## Governed input and profile

The real integration uses `KQ-PROMPT-001`, UTF-8 `Hello, Kestrel-Q.`, with the
already frozen canonical IDs:

```text
[9419, 11, 710, 467, 3621, 27325, 13]
```

Context capacity is 16 and each trace returns at most four sampled tokens.
The immutable policy is temperature 1.0, top-k 20 and top-p 0.95. All 248,320
finite F32 model logits participate. EOG IDs 248044 and 248046 remain eligible;
IDs 248077 through 248319 remain eligible model-capacity padding but are not
valid tokenizer outputs.

The following real cases were selected before observing any Task 3.1 model
logits or sampled output:

| Case | Seed | Stream |
|---|---:|---:|
| primary | `0` | `5427223837140668492` (`KQ_SAMPLING_DEFAULT_STREAM`) |
| disjoint holdout | `18446744073709551615` | `9223372036854775807` |

The primary case is repeated from reset with the same seed/stream. Replay must
match every token, decoded fragment, PCG word/state, survivor-order hash and
model-state summary exactly.

## Lifecycle

The prompt is tokenized and prefilled exactly once. Prefill commits model
position 7 and produces the first complete logits vector. The first sampled
token is selected from those logits; it is returned but is not consumed by the
model until another token is requested.

Each later step consumes exactly the preceding accepted sampled ID through the
Task 2.13 one-token decode path. The resulting logits are sampled directly.
Generated text is decoded for publication only; accumulated text is never
retokenized and the prompt or earlier generated IDs are never replayed.

For `N` non-EOG returned tokens the exact counts are:

```text
prompt prefill = 1
incremental decode = N - 1
PCG draws = N
prompt replay = 0
```

Model position equals seven after the first selection and advances by one for
each later sampled step because that step first consumes the preceding token.
GDN, QSA K/V/raw-index, PLE address and PLE value state advance together with
that position.

## Whole-token transaction

RNG state is explicit caller-owned state and is not embedded in the model
state. The integration nevertheless treats both as one transaction.

### First sampled token

1. Snapshot the reset caller RNG.
2. Prefill the canonical prompt once into private caller-output staging.
3. Run Task 3.0 selection against the complete prefill logits and the private
   RNG copy.
4. Validate/decode the selected canonical token.
5. Commit model state, RNG state, logits, token diagnostics and decoded bytes
   together.

If any stage fails, all model layers are reset to the prefill pre-state,
position remains zero, caller RNG remains unchanged and no caller output is
published.

### Later sampled token

1. Snapshot caller RNG and the externally observable model-state summary.
2. Decode the preceding accepted sampled ID through all 48 layers.
3. Produce the complete next-logit vector.
4. Run Task 3.0 selection with the private RNG copy.
5. Validate/decode the selected canonical token.
6. Commit model position/layer slots, RNG and caller outputs together.

The Task 2.13 layer transaction remains open through selection. A model,
sampling, selected-ID or decode failure rolls every completed layer back in
reverse order. The public model position, RNG and caller outputs remain at the
pre-step values. Provider counters and traces remain monotonic audit data and
are not semantic state.

## Stop and invalid-token behavior

An EOG result is returned and decoded with the accepted keep-special policy.
The successful model transition (if this is a later token) and exactly one RNG
draw commit, then generation stops. EOG is never fed back to the model and no
additional draw occurs. `max_new_tokens` counts returned tokens and stops
without another decode or draw. Context exhaustion is checked before starting
a later step and fails without changing state.

If Task 3.0 selects a padded ID, it returns `INVALID_TOKEN_ID` without
committing its private RNG copy. The surrounding model transaction also rolls
back. Padding IDs are not masked or silently reinterpreted.

## Independent per-step oracle

For governed real evidence, each exact native full-logit vector is written only
to an ignored temporary cache. A separate standard-library Python
implementation consumes that file plus the exact pre-step PCG state and the
frozen Task 3.0 policy. It independently reproduces:

- top-k membership including threshold ties;
- ascending score/ID top-p order and cutoff boundary;
- final F32 probability construction under the pinned scalar contract;
- the PCG32 word and next state;
- categorical traversal in ascending token-ID order; and
- the selected token ID.

The exact discrete comparison fields are the survivor count and ordered-u32
FNV-1a hash, PCG word, selected ID, draw count and complete next RNG semantic
state. Compact evidence stores only those fields and bounded top candidates;
full logits are never governed or committed. Native output never supplies the
independent expected decision rule.

## Lower-level invariants

Every successful consumed token must preserve the accepted Task 2.13 facts:

- all 12 QSA lengths equal model position, with complete blocks
  `floor(position / 4)` and tail `position mod 4`;
- every layer emits exactly ten unique ordered routed expert IDs and no
  unselected expert-member request;
- every consumed token emits exactly 16 PLE intents/row requests; and
- no complete target F32 weight matrix is materialized.

## Failure controls

Real integration injects provider failures at early, middle and late layers
and a post-logits sampler failure after an already successful generated
prefix. Controlled sampling logits exercise padded-ID rollback and EOG stop
without changing the production distribution. Every failure compares semantic
model summaries, RNG snapshots and caller-output sentinels before/after; retry
from the same pre-step state is deterministic.

## Scope

This contract adds no penalty processor, stochastic policy beyond the frozen
Task 3.0 surface, batching, product session object, MTP/speculation, vision,
long-context persistence, cache/prefetch/scheduler policy, SIMD or CUDA model
kernel. Logical payload and timing are characterization only.
