# PLAN-TASK-2.9-PLE-VALUE-REFERENCE-OPERATOR.md

Status: **COMPLETE / PASS**

## Objective

Implement Kestrel-Q's scalar CPU reference operator for the
Qwen3.8-Flash-Next PLE/N-gram **value path**.

Task 2.4 already computes exact logical PLE address intents. Task 2.2 exposes
bounded PLE member views. Task 2.5 decodes IQ4_NL and supplies scalar numerics.

Task 2.9 connects those layers and validates the canonical PLE value/history
semantics independently.

Out of scope:
- complete transformer-layer composition;
- full-model execution;
- LM head/sampling;
- final PLE disk cache/prefetch/scheduler;
- SIMD/CUDA model kernels.

Start only after Task 2.8 is checkpointed and pushed.

## Mandatory characterization gate

Before native code, create `docs/PLE-VALUE-OPERATOR-CONTRACT.md` from the
pinned Qwen/Transformers implementation.

Pin exactly:

- canonical module/functions;
- exact layer placement;
- relationship to Task 2.4 intents;
- logical member ordering;
- row/value width;
- heads/order lookup semantics;
- six dense PLE tensor roles/shapes;
- lookup and accumulation dtypes;
- normalization/projections;
- nine-position value/history state;
- exact dilation/convolution semantics;
- activation/gating/final combine;
- operation order relative to GDN/residual flow;
- prefill/decode/update/reset semantics;
- required semantic bindings.

Revalidate, do not assume, prior leads:
- zero-based layer 1;
- orders 2/3, 8 heads each;
- 128 logical members;
- 2,500,012 rows/member;
- one fused IQ4_NL physical table;
- six dense semantics;
- separate address and value state;
- nine-position value/history state.

## Independent oracle first

Pinned:
- Qwen `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Transformers `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`

Do not download the full BF16 checkpoint.

Generate expected values before native validation.

Use:
- Tier A: reduced synthetic tables/weights/state for value equations;
- Tier B: real Task 2.4 canonical address streams with synthetic canonical
  tables/weights;
- Tier C: real target structure and bounded IQ4_NL plumbing.

Kestrel-Q is never its own oracle.

## Evidence

Create under the governed operator research root, e.g.:

- `ple-value-contract.json`
- `ple-value-calibration.json`
- `ple-value-holdout.json`
- `ple-value-state-vectors.json`
- `ple-value-address-integration.json`
- `ple-value-native-validation.json`
- `ple-value-manifest.json`

Record deterministic configs/seeds, synthetic table/weights/input, address
intents, lookup outputs, intermediate checkpoints, final value output, state
before/after, shapes/dtypes/contracts, and deterministic hashes.

No real model-weight bytes may be committed.

## Production design

Suggested:

```text
include/kq_ple_value.h
src/kq_ple_value.c
src/kq_ple_value_qwen38.c
tests/ple_value_test.c
tests/ple_value_integration.c
```

Implement:

- immutable validated PLE value config;
- explicit mutable value-side state;
- reset/snapshot as needed;
- logical member+row lookup-provider interface;
- scalar prefill;
- scalar one-token decode.

Keep Task 2.4 address logic separate. Reuse Task 2.5 numerics/dequantization.

## Lookup provider boundary

The PLE value operator asks only for:

```text
logical PLE member + row -> value vector
```

Provide:

1. deterministic synthetic provider for oracle/native tests;
2. bounded real GGUF provider:
   semantic member -> Task 2.2 bounded view -> Task 2.5 IQ4_NL decode.

Do not hard-code file offsets. Do not implement persistent cache.

## Value state

Task 2.9 value state is separate from Task 2.4's 32-byte address state.

Characterize exact:
- state shape/dtype;
- history horizon;
- dilation positions;
- update order;
- reset semantics.

State is first-class golden evidence. Do not compare raw C struct padding.

## Address consumption is exact

Require exact equality for:

- intents/token;
- intent order;
- logical member IDs;
- row IDs;
- order/head -> member mapping.

Do not reorder lookup requests for locality in the scalar reference path.

## Prefill / decode

Prefill:

```text
input sequence + Task 2.4 intents + initial value state + lookup provider
-> PLE value sequence + final value state
```

Decode:

```text
current input + current intents + prior value state
-> PLE value output + updated value state
```

Require independent canonical agreement, prefill/decode continuity and
reset/replay.

## Numerical contracts

Create Task 2.9-specific calibration and disjoint holdout.

Potential checkpoints, only if canonical:
- raw lookup vectors;
- per-head/order combination;
- dense projection outputs;
- history/convolution input/output;
- activation/gating;
- final PLE output;
- value-state contents.

Use:
- `EXACT_DISCRETE`
- `EXACT_BITS`
- `CALIBRATED_FLOAT`

Do not reuse GDN/QSA/MoE tolerances.

## Real target reconciliation

Validate:

- 128 logical members;
- fused IQ4_NL binding;
- six dense PLE tensors;
- three metadata-derived address semantics compatibility;
- exact value-state target geometry.

Re-derive and reconcile:
- rows/member lead: `2500012`;
- Task 2.2 member span lead: `225001080` bytes;
- Task 1.3 aggregate PLE packed bytes lead: `28835240960`.

Root-cause any discrepancy. Do not encode these as unchecked production
constants.

## Bounded real payload

Allow:

`KQ_TASK29_REAL_PAYLOAD_SAMPLE_BUDGET <= 8 MiB`

Prefer far less.

Use deterministic Task 2.4 addresses to touch a small number of real rows.
Record only member/row/type/bytes touched and derived hashes/statistics.

No raw packed model bytes committed. No throughput/residency claim.

## Fail closed

Cover:

- wrong PLE layer/member count;
- missing fused/dense binding;
- wrong dense shape;
- bad IQ4_NL/member geometry;
- invalid member/row;
- wrong address count/order;
- incompatible address config;
- wrong value-state shape/dtype;
- provider failure;
- non-finite input where forbidden;
- capacity/count overflow;
- aliasing;
- partial state mutation on failure.

Prefer transactional state: failure leaves externally visible state unchanged.

## Hard boundaries

Do NOT implement:
- final PLE disk cache;
- async I/O/prefetch;
- scheduler/residency policy;
- complete transformer-layer composition;
- full model forward;
- LM head/sampling;
- SIMD/CUDA model kernels.

`KQ-BACKLOG-BENCH-002` remains DEFERRED.

## Documentation

Create:
- `docs/PLE-VALUE-OPERATOR-CONTRACT.md`
- `docs/NATIVE-PLE-VALUE-REFERENCE.md`
- `docs/KQ-PLE-VALUE-API.md`
- `docs/adr/0017-native-ple-value-reference-operator.md`

Update architecture/runtime/quantization/goldens/TASKS/ROADMAP/Epic 2,
Task 2.9 checklist, tools provenance and CHANGELOG.

## Acceptance

PASS requires:

- exact characterized PLE value contract;
- independent reduced vectors;
- Task 2.4 address-integration vectors;
- calibration + disjoint holdout;
- value-state vectors;
- native immutable config/value state/provider;
- scalar prefill/decode;
- exact address consumption;
- state transition/reset PASS;
- real target 128-member/fused-IQ4_NL/dense bindings PASS;
- bounded real sample <= 8 MiB;
- no self-oracle;
- no final PLE cache/full-layer/full-forward implementation;
- CPU/CUDA regressions PASS;
- ADR 0017 ACCEPTED;
- next integration task NOT STARTED;
- KQ-BACKLOG-BENCH-002 DEFERRED;
- no tracked model weights/secrets/local paths;
- `git diff --check` PASS.

Do not commit or push automatically.
