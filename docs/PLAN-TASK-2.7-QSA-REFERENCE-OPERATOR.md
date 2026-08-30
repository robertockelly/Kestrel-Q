# PLAN-TASK-2.7-QSA-REFERENCE-OPERATOR.md

Status: **COMPLETE / PASS**

## Objective

Implement the scalar CPU reference operator for Qwen3.8-Flash-Next QSA, with
independent Class-C vectors covering prefill, incremental decode, sparse
block/token selection, K/V growth and indexer state.

Task 2.7 starts only after Task 2.6 is checkpointed.

## Baseline expectations

Task 2.6: COMPLETE/PASS.
ADR 0014: ACCEPTED.
Epic 2: IN PROGRESS.
`KQ-BACKLOG-BENCH-002`: DEFERRED.

Epic 1 QSA facts are leads to revalidate, not implementation constants:

- 12 QSA layers, expected layer IDs `3 mod 4`;
- context-growing K/V;
- high-level K/V lead `[B,2,T,256]`;
- raw index-key lead `[B,T,128]`;
- 4-token block grouping;
- up to 512 selected blocks / 2048 selected tokens;
- incomplete causal tail included.

## Mandatory characterization

Before native QSA code, create `docs/QSA-OPERATOR-CONTRACT.md` from the pinned
canonical Qwen/Transformers source.

Pin exactly:

- canonical module/functions;
- 12 QSA layer IDs;
- hidden/Q/K/V/output projection shapes;
- head counts/dimensions;
- normalization and RoPE semantics;
- attention scale and causal masking;
- K/V cache shape/dtype/append semantics;
- index query/key projections;
- raw index-key state;
- block size and block-summary rule;
- incomplete-tail behavior;
- candidate-score equation;
- top-block/token selection limit;
- tie-breaking and ordering;
- selected-token gather order;
- attention softmax/output order;
- prefill/decode state transitions;
- position/cache-length semantics;
- target split `index_qk` bindings.

Do not infer behavior from generic sparse-attention literature.

## Independent oracle first

Use:

- Qwen `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Transformers `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`

Do not download the full BF16 checkpoint.

Use deterministic synthetic weights/configuration. Reduced dimensions are
allowed only when they preserve the canonical algorithm. Add a bounded
threshold-crossing configuration if needed to exercise sparse selection beyond
the selection cap.

Kestrel-Q must never define expected values.

## Evidence

Create a governed QSA evidence set under `research/operators/...`, including:

- contract/config evidence;
- calibration corpus;
- disjoint holdout;
- exact selection vectors;
- state vectors;
- native validation;
- manifest/SHA.

Discrete selection artifacts are exact:

- candidate blocks;
- selected blocks;
- selected tokens;
- ordering;
- tie decisions;
- counts and causal exclusions.

Floating checkpoints use `EXACT_BITS`, `EXACT_DISCRETE` or
`CALIBRATED_FLOAT` with QSA-specific calibration and holdout.

## Native implementation

Suggested:

```text
include/kq_qsa.h
src/kq_qsa.c
src/kq_qsa_qwen38.c
tests/qsa_test.c
tests/qsa_integration.c
```

Implement:

- immutable validated QSA layer configuration;
- explicit resettable QSA stream state;
- scalar CPU prefill;
- scalar CPU one-token decode;
- state snapshot/query needed by tests;
- exact sparse selection outputs.

Reuse Task 2.5 numerics. Do not move QSA semantics into generic numeric code.

## QSA state

Characterization determines exact state. Expected categories include:

- K cache;
- V cache;
- raw index-key history;
- current sequence/cache length;
- partial block state if canonical decode requires it.

State is a first-class correctness output.

Use bounded explicit capacity. Do not design the final KV-cache allocator or
scheduler here.

Failure behavior must be explicit; prefer no externally visible state mutation
on failure.

## Sparse threshold cases

Oracle/tests must cover:

- no completed historical block;
- one completed block;
- multiple blocks below limit;
- candidate count equal to limit;
- candidate count above limit;
- incomplete tail;
- decode at and inside block boundaries;
- deterministic ties/near-ties;
- repeated inputs.

Also cover any canonical mandatory local/recent context rule discovered during
characterization.

## Target integration

Validate all real target layers:

- QSA configs: 12/12 valid;
- GDN layer IDs: 36/36 rejected;
- all semantic weights present;
- split `index_qk` binding valid;
- target shapes/types/layouts valid;
- state geometry valid;
- no hard-coded GGUF offsets.

Optional real payload use is bounded to:

`KQ_TASK27_REAL_PAYLOAD_SAMPLE_BUDGET <= 8 MiB`

Prefer zero if structural validation is sufficient.

## State-growth reconciliation

Re-derive target batch-1 QSA state growth from the exact contract.

Reconcile with Epic 1's high-level footprint lead of approximately
`27,648 bytes/token` across all 12 QSA layers.

Document per-layer K, V, index-key and total bytes/token plus fixed partial-block
overhead. Any discrepancy requires root cause before PASS.

## Fail closed

Cover:

- QSA constructor on GDN layer;
- missing/split semantic weights;
- wrong projection/head dimensions;
- wrong cache/index state shape/dtype;
- invalid capacity/context/position;
- invalid block/selection geometry;
- non-finite input where forbidden;
- split/transformed misuse;
- aliasing violations;
- partial mutation on failure.

## Hard boundaries

Do NOT implement:

- MoE routing/expert execution;
- PLE value lookup;
- full transformer layer;
- multi-layer/full-model execution;
- LM head/sampling;
- final KV-cache scheduler;
- SIMD/CUDA QSA kernels;
- cache/prefetch policy.

## Documentation

Create:

- `docs/QSA-OPERATOR-CONTRACT.md`
- `docs/NATIVE-QSA-REFERENCE.md`
- `docs/KQ-QSA-API.md`
- `docs/adr/0015-native-qsa-reference-operator.md`

Update architecture, model/runtime-state, golden docs, TASKS, ROADMAP, Epic 2,
Task 2.7 checklist, tools provenance and CHANGELOG.

## Acceptance

PASS requires:

- exact characterized QSA contract;
- independent reduced and threshold-crossing Class-C vectors;
- calibration + disjoint holdout;
- exact discrete selection evidence;
- native scalar prefill/decode;
- K/V/indexer state transitions PASS;
- 12/12 target QSA layers valid;
- 36/36 GDN IDs reject;
- state-growth reconciliation PASS;
- no self-oracle;
- CPU/CUDA regressions PASS;
- ADR 0015 ACCEPTED;
- next operator task NOT STARTED;
- `KQ-BACKLOG-BENCH-002` DEFERRED;
- no tracked model weights/secrets/local paths;
- `git diff --check` PASS.

Do not commit or push automatically.
