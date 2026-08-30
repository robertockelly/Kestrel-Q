# Native QSA Reference Operator

## Purpose

Task 2.7 adds the scalar CPU correctness path for the Qwen3.8-Flash-Next QSA
layers.  It is deliberately a reference implementation: explicit state,
straightforward loops, strict validation, deterministic selection, and no
SIMD, CUDA, cache scheduling, or full-model execution.

The implementation is split between `kq_qsa.c`, which owns immutable
configuration and bounded stream state, and `kq_qsa_qwen38.c`, which owns the
model-specific scalar equations.  Generic Task 2.5 arithmetic remains in
`kq_numeric.c`; QSA semantics were not moved into that generic module.

## Independent validation

`tools/generate-qsa-reference.py` imports the pinned offline Transformers
checkout and executes the unmodified canonical QSA attention/indexer with
synthetic deterministic data.  It creates:

- a reduced Tier-A F32 configuration for prefill, decode, state and floating
  checkpoint validation;
- a Tier-B configuration with 512 and 513 completed blocks to cross the real
  selection threshold without loading model weights;
- calibration, disjoint holdout, exact selection and state-transition assets.

`tools/validate-native-qsa.py` drives `kq_qsa_probe`, treats selection as
`EXACT_DISCRETE`, calibrates each floating category independently, and checks
all holdout values against the frozen calibration envelope.  The validation
covers five calibration cases, six disjoint holdout/decode cases, and three
Tier-B selection cases.  Kestrel-Q never generates the expected values.

The native validation categories are operator output, K state, V state, raw
index-key state, candidate scores and attention probabilities.  Their exact
measured `atol`, `rtol`, ULP ceilings and holdout maxima are recorded in
`qsa-native-validation.json`; no Task 2.5 or GDN tolerance is reused.

## State and failure model

A stream state is created with an explicit token capacity.  It owns K, V and
raw index-key arrays plus a current length.  The scalar executable path uses
F32 arrays.  A target structural config uses BF16 descriptors and reports the
canonical 2,304 bytes/token/layer semantic growth.

Execution validates all arguments, weights, current state, capacity and
workspace before work.  New K/V/index state and outputs are staged in bounded
caller scratch.  They are committed only after the complete call succeeds, so
failure does not mutate visible state or output.  Reset changes semantic length
to zero; it does not hide allocator capacity or construct scheduler policy.

State export/import exists for tests and controlled reference continuation.
Callers compare semantic elements and length, never raw C object padding.

## Sparse selection

The indexer emits exact candidate block IDs, candidate scores, selected block
IDs, selected token positions and tail count through a callback.  The reusable
checked selector is also exposed so the real 512-block threshold can be tested
without performing an impractical 2,052-token dense reference attention run.

The Tier-B evidence proves:

- 512 candidates / limit 512: blocks `0..511` selected;
- 513 all-tied candidates / limit 512: blocks `0..511` selected and block 512
  excluded;
- 513 distinct near-tie candidates: exact oracle block order reproduced.

The selected-token order is four ascending positions per selected block,
followed by the incomplete causal tail.  Membership and top-k order are tested
separately because the canonical boolean mask preserves membership but not the
local top-k gather order.

## Real-artifact structural gate

With `KQ_GGUF_PATH`, `qsa_integration.c` constructs all 12 QSA target configs,
constructs all 12 matching F32-reference descriptors, and requires all 36 GDN
layer IDs to reject.  It validates every required semantic tensor, the ordered
split index-Q/K binding, canonical/physical shapes and types, head/block/state
geometry, and exact layer schedule.

No real tensor view is opened.  The result is:

```text
QSA configs                    12 / 12 PASS
GDN IDs rejected               36 / 36 PASS
semantic bytes/token/layer     2304
semantic bytes/token/12 layers 27648
real payload bytes touched     0
real payload blocks touched    0
```

## Characterization metrics

Observed Release values for the reduced F32 test configuration are
characterization, not performance guarantees:

- immutable config: 184 bytes;
- capacity-16 stream state: 832 owned bytes;
- semantic state growth: 48 bytes/token;
- 12-token calibration-prefill scratch: 1,556 bytes;
- eight-token holdout-prefill scratch: 1,236 bytes;
- one-token decode after a five-token prefix scratch: 676 bytes;
- eight-token holdout prefill: median 20,900 ns over seven runs;
- one-token decode after a five-token prefix: median 3,300 ns over seven runs.

The real target capacity-one structural state occupied 2,368 bytes including
the state object and 2,304 semantic payload bytes.  No throughput claim is made.
Timing uses the Release CPU probe's no-observer path and is a descriptive
single-machine characterization, not a performance guarantee or target.

## Limits

The executable scalar API is batch one, F32, text-only, contiguous causal
positions and no padding.  Real quantized weight execution is not connected in
Task 2.7.  The target BF16 config is structural only.  Final KV allocation,
residency, paging and scheduling are separate future work.  MoE, PLE values,
complete transformer layers, multi-layer execution, logits and generation are
out of scope.
