# SAMPLED-INCREMENTAL-GENERATION-GUIDE.md

## Task 3.0 is the sampling authority

Do not copy temperature/top-k/top-p/softmax/RNG logic into the model executor.
The executor supplies logits and explicit RNG state to Task 3.0.

## Transaction = model state + RNG state

For every incremental sampled step:

```text
stage model state
stage RNG state
decode previous accepted token
sample next token
validate token
commit both
```

Any failure rolls both back.

## The first sampled token is special

Prompt prefill produces the first logits. Sampling that distribution advances
the RNG but does not decode the selected token yet. The token is decoded only
if another generation step is needed.

## EOG is selectable

Do not mask EOG. Generation integration stops after EOG is successfully
selected. Do not decode EOG and do not draw again after stop.

## Padded IDs still fail

Task 3.0 intentionally allows padded IDs to participate in the distribution.
If selected, the combined sampled generation step must fail transactionally.

## Independent selection oracle

Feed the exact temporary native logits plus pre-step RNG state to a separate
Task 3.0 reference implementation. Commit compact hashes/results, not full
logits arrays.

## Preserve greedy

Sampling is additive. M1 and Task 2.13 greedy evidence must remain unchanged.
