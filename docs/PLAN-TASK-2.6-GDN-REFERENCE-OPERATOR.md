# PLAN-TASK-2.6-GDN-REFERENCE-OPERATOR.md

Status: **COMPLETE / PASS**

## Objective

Implement Kestrel-Q's first model-specific execution operator:

**Qwen3.8-Flash-Next GDN scalar CPU reference operator**, with independently
generated golden vectors for prefill, incremental decode and state transitions.

Task 2.6 must not implement QSA, MoE execution, PLE value lookup, a full
transformer layer, a full model forward pass, sampling, SIMD/CUDA model kernels,
or scheduler/cache/prefetch policy.

## Baseline

Reviewed checkpoint:

`0af1cca7709f70236a213c451aa00d39783f2e0f`

Task 2.0-2.5 are COMPLETE/PASS.
Task 2.6 is NOT STARTED.
ADR 0008-0013 are ACCEPTED.
`KQ-BACKLOG-BENCH-002` remains DEFERRED.

Epic 1 high-level leads to revalidate before coding:

- 36 GDN layers among 48 text layers;
- repeating layer pattern `GDN,GDN,GDN,QSA`;
- per-layer FP32 recurrent state approximately `[B,48,128,128]`;
- causal-convolution state approximately `[B,10240,4]`;
- causal prefill plus one state update/token in decode.

These are not implementation constants until the canonical contract confirms
them.

## Mandatory characterization gate

Before production operator code, create:

`docs/GDN-OPERATOR-CONTRACT.md`

Re-open the pinned Qwen/Transformers canonical implementation and document:

- exact canonical module/class/function names;
- exact GDN layer IDs;
- input/output hidden shape;
- normalization placement/epsilon;
- every weight/parameter role and shape;
- projection split/fusion semantics;
- causal-convolution channels/kernel/state;
- activations/gates;
- recurrent-state shape/dtype;
- recurrent update equations;
- exact operation order;
- residual interaction owned by this operator;
- prefill scan semantics;
- one-token decode semantics;
- state initialization/mutation order;
- masking/padding behavior;
- batch assumptions;
- dtype transitions and accumulation dtypes;
- GR semantics internal to GDN versus future separate GR work;
- sequence-length edge cases;
- required semantic tensor bindings.

Do not derive equations from older Gated DeltaNet papers or implementations if
the pinned Qwen implementation differs.

## Independent oracle first

Use the pinned Class-C source:

- Qwen model revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Transformers oracle:
  `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`

Do not download the full BF16 checkpoint.

Prefer an isolated canonical GDN module with deterministic synthetic weights and
inputs.

Use two evidence tiers:

### Tier A — reduced-shape canonical vectors

Use reduced dimensions only if the canonical implementation permits this
without changing the algorithm.

Cover:

- sequence length 1;
- below/equal/above convolution-history width;
- repeated and alternating inputs;
- reset;
- prefill followed by decode;
- multiple decode steps;
- supplied non-zero initial states.

### Tier B — target-shape structural integration

Validate the real Qwen3.8 target:

- layer IDs;
- hidden/config dimensions;
- projection shapes;
- recurrent/conv state shapes;
- real semantic bindings;
- quantized physical types/layouts.

Tier B does not require a full target-shape canonical forward if that would
require unavailable full BF16 weights.

## Golden evidence

Create a new namespace such as:

```text
research/operators/Qwen3.8-Flash-Next/
  de4b8e4d43b917e7706784d8bb445c9af86a3540/
    gdn-contract.json
    gdn-calibration.json
    gdn-holdout.json
    gdn-state-vectors.json
    gdn-manifest.json
```

Record:

- source revisions;
- deterministic config/seed;
- inputs and synthetic weights;
- initial states;
- outputs;
- final states;
- selected intermediate checkpoints;
- dtype/shape metadata;
- comparison contracts;
- SHA-256 manifest.

Do not commit real model weight bytes.

## Production boundary

Suggested modules:

```text
include/kq_gdn.h
src/kq_gdn.c
src/kq_gdn_qwen38.c
tests/gdn_test.c
tests/gdn_integration.c
```

Generic numeric primitives remain in Task 2.5 code.

## Immutable layer config

Create a validated immutable descriptor for one GDN layer.

It must:

- accept only canonical GDN layer IDs;
- reject QSA layer IDs;
- resolve every required semantic tensor;
- validate shapes/types/layout relations;
- store state/conv geometry;
- avoid copying model payload at construction.

## Explicit stream state

Define the exact canonical mutable per-stream GDN state.

Requirements:

- deterministic initialization/reset;
- exact recurrent-state dtype;
- exact convolution/history-state dtype;
- bounded state;
- no hidden global state;
- state-size query;
- semantic element comparison in tests, never raw `memcmp` over C struct padding.

## Scalar prefill

Implement:

```text
hidden sequence + initial GDN state + resolved weights
    -> output sequence + final GDN state
```

Requirements:

- scalar correctness-first path;
- Task 2.5 numeric primitives reused;
- no fast math;
- bounded scratch;
- deterministic operation order;
- documented transactional behavior on error.

## Incremental decode

Implement one-token decode:

```text
hidden token + previous GDN state
    -> output token + updated GDN state
```

Require canonical agreement between prefill continuation and incremental decode
under the calibrated numerical contract.

## State/checkpoint evidence

State is a first-class correctness output.

Compare canonical and native, where applicable:

- normalized input;
- projection outputs;
- convolution input/output;
- gates/decays;
- recurrent update terms;
- recurrent state;
- convolution state;
- final output.

Use only canonical checkpoints confirmed by the source.

## Numerical calibration

Do not inherit one generic Task 2.5 tolerance.

Create separate GDN:

- calibration corpus;
- disjoint holdout corpus.

For every output/checkpoint/state class:

1. run independent canonical oracle;
2. run native scalar implementation;
3. measure ULP/absolute/relative differences as appropriate;
4. establish justified contract;
5. validate holdout.

Use `EXACT_BITS`, `EXACT_DISCRETE`, or `CALIBRATED_FLOAT` as appropriate.

## Optional bounded real-GGUF plumbing

Task 2.6 may read a small amount of real payload only to prove binding/numeric
plumbing, not as the complete GDN oracle.

Set:

`KQ_TASK26_REAL_PAYLOAD_SAMPLE_BUDGET <= 8 MiB`

Prefer less.

Path must be:

```text
semantic descriptor
-> Task 2.2 bounded view
-> Task 2.5 numeric reference
```

No hard-coded physical offsets and no raw model payload committed.

## Fail closed

Cover at least:

- GDN construction on QSA layer;
- missing required semantic tensor;
- wrong projection shape;
- wrong recurrent-state shape;
- wrong convolution-state shape;
- wrong required dtype;
- invalid sequence length/count overflow;
- invalid state;
- non-finite input when forbidden;
- split/transformed binding misuse;
- forbidden aliasing;
- bounded-view failures;
- partial-failure state policy.

## Memory / performance scope

Record:

- immutable config bytes;
- batch-1 state bytes;
- scratch bytes/token;
- prefill scratch;
- temporary dequant buffer;
- synthetic prefill/decode latency.

Do not optimize.

No SIMD, CUDA, thread pool or scheduler.

## Hard boundaries

Do not implement:

- QSA;
- MoE routing/expert execution;
- PLE value lookup/convolution;
- full transformer layer;
- multiple-layer execution;
- LM head;
- sampling;
- full model forward;
- model CUDA kernels.

## Documentation

Create:

- `docs/GDN-OPERATOR-CONTRACT.md`
- `docs/NATIVE-GDN-REFERENCE.md`
- `docs/KQ-GDN-API.md`

Finalize:

- `docs/adr/0014-native-gdn-reference-operator.md`

Update ARCHITECTURE, MODEL-RUNTIME-STATE, GOLDEN-VECTORS, TASKS, ROADMAP,
Epic 2 plan/status, Task 2.6 checklist, tools provenance and CHANGELOG.

## Acceptance

Task 2.6 PASS requires:

- exact canonical GDN contract;
- independent reduced-shape Class-C oracle;
- deterministic calibration/holdout/state evidence;
- immutable native layer config;
- explicit native stream state;
- scalar prefill;
- scalar one-token decode;
- state transitions validated;
- intermediate checkpoints validated;
- prefill/decode consistency;
- all 36 target GDN layers structurally valid;
- QSA layer construction rejected;
- bounded real artifact plumbing validated if used;
- no full BF16 checkpoint download;
- no self-oracle;
- CPU/CUDA regressions PASS;
- ADR 0014 ACCEPTED;
- next operator task NOT STARTED;
- KQ-BACKLOG-BENCH-002 DEFERRED;
- no tracked model weights/secrets/local paths;
- `git diff --check` PASS.

Do not commit or push automatically.
