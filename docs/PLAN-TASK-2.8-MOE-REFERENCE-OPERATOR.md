# PLAN-TASK-2.8-MOE-REFERENCE-OPERATOR.md

Status: **COMPLETE / PASS**

## Objective

Implement Kestrel-Q's scalar CPU reference operator for the Qwen3.8-Flash-Next
Mixture-of-Experts block with independent Class-C golden vectors.

Task 2.8 covers:
- router logits/probabilities;
- exact top-k expert selection/order;
- top-k normalization;
- selected routed-expert SwiGLU execution;
- shared expert;
- shared-expert gate;
- final routed/shared combination.

Out of scope:
- PLE value lookup;
- complete transformer-layer composition;
- full model forward;
- sampling;
- SIMD/CUDA MoE kernels;
- expert cache/prefetch/scheduler/residency policy.

## Baseline

Start only after Task 2.7 is checkpointed and pushed.

Expected:
- Task 2.0-2.7 COMPLETE/PASS
- ADR 0015 ACCEPTED
- Epic 2 IN PROGRESS
- KQ-BACKLOG-BENCH-002 DEFERRED

## Mandatory characterization

Before native MoE code create `docs/MOE-OPERATOR-CONTRACT.md`.

From the pinned canonical Qwen/Transformers implementation, pin:
- exact module/class/functions;
- router input/projection shape;
- router logits/probability dtype;
- routed expert count;
- selected top-k count;
- top-k membership/order/tie semantics;
- exact selected-weight normalization;
- routed expert gate/up/down equations;
- activation;
- expert intermediate width;
- expert-axis ordering;
- shared expert gate/up/down;
- shared-expert gate source/equation;
- routed/shared combination;
- output/accumulation dtype;
- residual boundary;
- inference-vs-training-only branches;
- required semantic bindings/shapes.

Revalidate Epic 1 leads:
512 experts, top-10, one shared expert, width 640, FP32 router softmax,
normalized top-10 routed outputs, separately sigmoid-gated shared expert.

Do not implement from generic MoE knowledge.

## Independent Class-C oracle first

Pinned:
- Qwen `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Transformers `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`

Do not download the full BF16 checkpoint.

Use deterministic synthetic weights/configs and generate expected values BEFORE
native validation.

Use two tiers:

### Tier A — reduced expert-count semantics
Small expert count/top-k if supported without changing algorithm semantics.

### Tier B — canonical 512/top-10 routing
Use 512 experts and top-10 with small hidden/intermediate sizes if supported.

Exercise:
- equal logits;
- kth-boundary ties;
- near-ties;
- expert 0;
- expert 511;
- one dominant expert;
- exactly ten dominant experts;
- more than ten plausible experts.

Kestrel-Q is never its own oracle.

## Evidence

Create governed Task 2.8 evidence under `research/operators/...`, including:
- `moe-contract.json`
- `moe-calibration.json`
- `moe-holdout.json`
- `moe-routing-vectors.json`
- `moe-native-validation.json`
- `moe-manifest.json`

Record deterministic configs/seeds, inputs/weights, router logits/probabilities,
selected expert IDs/order, normalized weights, per-expert outputs, routed sum,
shared path, final output, shapes/dtypes/contracts and manifest hashes.

No real model-weight bytes may be committed.

## Native implementation

Suggested:
```text
include/kq_moe.h
src/kq_moe.c
src/kq_moe_qwen38.c
tests/moe_test.c
tests/moe_integration.c
```

Implement scalar CPU:
- immutable validated config;
- router;
- exact top-k;
- selected routed-expert execution;
- shared expert;
- shared gate;
- final combination.

Reuse Task 2.5 numerics. Keep MoE semantics out of generic numeric code.

## Exact routing contract

Expert selection is `EXACT_DISCRETE`.

Require exact:
- membership;
- order;
- count;
- tie-break;
- boundary behavior.

Floating closeness never excuses different selected experts.

Pin exact canonical normalization semantics rather than assuming a standard
softmax/top-k recipe.

## Routed expert path

For each selected expert:
```text
expert ID
-> bounded Task 2.2 member views
-> gate/up/down
-> Task 2.5 scalar numerics
-> expert output
```

Do not map/dequantize all 512 experts.

Preserve split gate/up order.

## Shared expert path

Characterize and implement separately:
- shared gate/up/down;
- activation;
- shared-expert gate;
- sigmoid/gating placement;
- final routed+shared combination.

Do not make the shared expert part of top-k unless canonical source says so.

## Floating calibration

Create MoE-specific calibration and disjoint holdout.

Calibrate independently:
- router logits;
- router probabilities;
- normalized selected weights;
- expert outputs;
- routed weighted sum;
- shared output;
- shared gate;
- gated shared output;
- final output.

Use `EXACT_DISCRETE`, `EXACT_BITS`, or `CALIBRATED_FLOAT` as justified.

No generic epsilon.

## Real target integration

Validate the target semantic/physical structure:
- all target MoE configurations;
- 512 expert axis;
- routed gate/up/down stacks;
- split gate/up physical relation;
- shared expert and shared gate;
- physical quant types/layouts.

Reconcile representative per-expert packed spans with Epic 1 evidence:
- 3,072,000 bytes in 43 layers;
- 3,993,600 bytes in layer 2;
- 3,584,000 bytes in layers 4,30,46,47.

Re-derive; do not blindly copy.

Optional bounded real-payload plumbing:
`KQ_TASK28_REAL_PAYLOAD_SAMPLE_BUDGET <= 16 MiB`

Prefer less. No raw packed model bytes committed, no hard-coded offsets.

Do not treat selected-parameter footprint as actual I/O/token.

## Fail closed

Cover:
- wrong expert count/top-k;
- missing router/routed stack/shared expert/shared gate;
- wrong expert axis;
- wrong hidden/intermediate shape;
- broken split gate/up;
- invalid expert ID;
- invalid/non-finite router input where forbidden;
- capacity/count overflow;
- aliasing violations;
- bounded-view failures.

MoE should be stateless across tokens unless the canonical source proves
otherwise. Do not invent persistent routing state.

## Memory / performance

Record characterization only:
- config bytes;
- router workspace;
- top-k workspace;
- one-expert workspace;
- routed accumulation workspace;
- shared workspace;
- reduced router/expert/shared/total latency.

No optimization, parallel experts, SIMD, CUDA, cache or prefetch.

## Documentation

Create:
- `docs/MOE-OPERATOR-CONTRACT.md`
- `docs/NATIVE-MOE-REFERENCE.md`
- `docs/KQ-MOE-API.md`
- `docs/adr/0016-native-moe-reference-operator.md`

Update ARCHITECTURE, MODEL-ARCHITECTURE if sharpened, GOLDEN-VECTORS,
MODEL-QUANTIZATION-FOOTPRINT if evidence adds value, TASKS, ROADMAP, Epic 2,
Task 2.8 checklist, tools provenance and CHANGELOG.

## Acceptance

Task 2.8 PASS requires:
- exact canonical MoE contract;
- reduced independent Class-C vectors;
- canonical 512/top-10 routing vectors;
- calibration + disjoint holdout;
- exact routing IDs/order;
- native scalar routed path;
- native shared path;
- final combination PASS;
- real target configs/bindings PASS;
- per-expert packed geometry reconciliation PASS;
- optional payload budget respected;
- no self-oracle;
- no PLE value/full-layer/full-forward work;
- CPU/CUDA regressions PASS;
- ADR 0016 ACCEPTED;
- next operator task NOT STARTED;
- KQ-BACKLOG-BENCH-002 DEFERRED;
- no tracked model weights/secrets/local paths;
- `git diff --check` PASS.

Do not commit or push automatically.
