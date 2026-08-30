# PLAN-TASK-2.10-CANONICAL-LAYER-COMPOSITION.md

Status: **COMPLETE / PASS**

## Objective

Implement Kestrel-Q's scalar CPU reference composition for one complete
Qwen3.8-Flash-Next text transformer layer.

Task 2.10 composes the already validated operators into the exact canonical
layer order, including layer-owned GR/residual semantics and the special
PLE-enabled layer behavior.

Existing operators:
- Task 2.6 GDN
- Task 2.7 QSA
- Task 2.8 MoE
- Task 2.9 PLE value

Task 2.10 must cover three layer families:
1. ordinary GDN + MoE;
2. QSA + MoE;
3. the canonical PLE-enabled GDN layer.

It must NOT implement embedding lookup, a 48-layer executor, final norm, LM
head, logits, sampling, final cache/scheduler policy, SIMD or CUDA model
kernels.

## Entry gate

Start only after Task 2.9 is checkpointed and pushed.

Expected prior state:
- Task 2.0–2.9 COMPLETE/PASS
- ADR 0017 ACCEPTED
- Task 2.10 NOT STARTED
- KQ-BACKLOG-BENCH-002 DEFERRED

## Mandatory characterization

Before native layer code create:

`docs/TRANSFORMER-LAYER-CONTRACT.md`

From the pinned Qwen/Transformers implementation, pin exactly:
- decoder-layer class/functions;
- layer-family classification rule;
- norm placement and epsilons;
- GR/residual class/functions;
- GR parameter roles/shapes;
- rank and branch count;
- GR read/write equations and operation order;
- whether residual branches mix;
- exact mixer input/output composition;
- exact PLE placement and combine semantics;
- GDN/QSA invocation position;
- MoE preparation/call/final combine;
- dtype transitions;
- persistent state owned by GDN/QSA/PLE;
- forward-only GR activations;
- prefill/decode semantics;
- mask/position/state inputs;
- layer-owned semantic bindings.

Do not derive composition by simply chaining standalone operators.

## GR leads to revalidate

Epic 1 leads:
- four residual branches;
- target width 2560 each;
- rank-320 elementwise read gating;
- per-branch scalar writes;
- no branch-mixing H_res;
- GR branches are forward activations, not context cache.

Canonical source wins if it differs.

## Independent Class-C oracle first

Pinned:
- Qwen `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Transformers `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`

Do not download the full BF16 checkpoint.

Generate deterministic complete-layer expected values before native validation.
Reduced dimensions are allowed only when canonical semantics remain unchanged.

Required oracle families:
- ordinary GDN layer;
- QSA layer;
- PLE-enabled GDN layer.

For each family cover prefill, prefill+decode, multi-step decode, reset/replay,
and non-zero initial state where canonical APIs permit.

## Evidence

Create governed Task 2.10 evidence under `research/operators/...`, for example:
- `layer-contract.json`
- `layer-calibration.json`
- `layer-holdout.json`
- `layer-state-vectors.json`
- `layer-family-vectors.json`
- `layer-native-validation.json`
- `layer-manifest.json`

Record deterministic config/seeds, synthetic inputs/weights, GR/residual
checkpoints, suboperator boundary inputs/outputs, final layer output, final
persistent states, shapes/dtypes/contracts and manifest hashes.

No real model-weight bytes may be committed.

## Production design

Suggested:
```text
include/kq_layer.h
src/kq_layer.c
src/kq_layer_qwen38.c
tests/layer_test.c
tests/layer_integration.c
```

Reuse GDN/QSA/MoE/PLE APIs and Task 2.5 numerics. Do not duplicate suboperator
equations in the layer module unless canonical code proves the operation belongs
to the layer itself.

## Immutable layer config

Construct one config per text layer.

Validate:
- layer ID/family;
- layer-owned norm/GR bindings;
- correct GDN or QSA config;
- MoE config;
- PLE address/value configs only on the canonical PLE layer;
- model dimensions.

All 48 real text layers must classify and validate exactly.

## Runtime state

Compose only persistent suboperator state:
- GDN recurrent/conv state;
- QSA K/V/index state;
- PLE address state;
- PLE value state.

Do not persist forward-only GR branches if canonical source says they are
forward activations.

Provide reset/init/state-size and explicit ownership.

## GR/residual implementation

Implement exact layer-owned scalar C17 GR/residual equations.

Use Task 2.5 primitives where appropriate.
No fast math, SIMD or CUDA.
Keep branch/update order canonical.
Expose layer-composition checkpoints for validation.

## Layer families

### Ordinary GDN layer
Compose exact canonical norm/GR/GDN/MoE/residual order.

### QSA layer
Compose exact canonical norm/GR/QSA/MoE/residual order and preserve QSA state.

### PLE-enabled layer
Compose Task 2.4 address + Task 2.9 value + GDN + GR/residual + MoE in the exact
canonical order.

Do not assume PLE is a generic pre-layer add.

## Layer checkpoints

Focus on composition boundaries, e.g. when canonical:
- layer input;
- GR read/branch inputs;
- PLE-enhanced hidden;
- mixer input/output;
- GR write/update branches;
- MoE input/output;
- final layer output.

Suboperator internals remain validated by their own evidence.

## Floating contracts

Create Task 2.10-specific calibration and disjoint holdout.

Do not reuse one generic suboperator tolerance for final layer output.

Use:
- EXACT_DISCRETE
- EXACT_BITS
- CALIBRATED_FLOAT

Persistent suboperator state must also remain within its existing operator
contracts.

## Real target integration

Validate 48/48 target layer configurations.

Re-derive expected family counts from the canonical contract. The prior leads
are:
- 36 GDN layers;
- 12 QSA layers;
- exactly one PLE-enabled GDN layer.

Require:
- exact family classification;
- exact norm/GR bindings;
- all suboperator configs valid;
- no missing/extra semantic bindings;
- no hard-coded GGUF offsets.

Optional real payload budget:
`KQ_TASK210_REAL_PAYLOAD_SAMPLE_BUDGET <= 16 MiB`

Prefer zero/minimal use.

## Transactional state

A composed call may touch multiple persistent states.

Preferred policy:
if the layer call fails, all externally visible persistent suboperator states
remain unchanged.

Implement/test bounded staging/rollback. If another policy is necessary,
document it explicitly.

## State-footprint reconciliation

Compute exact persistent state by family and reconcile with
`MODEL-RUNTIME-STATE.md`.

GR forward activations must not become persistent state unless canonical source
requires it.

## Fail closed

Cover:
- wrong layer family/ID;
- PLE on wrong layer or missing on canonical layer;
- missing/wrong norm/GR binding;
- wrong GR branch count/rank/shape;
- invalid suboperator config/state;
- insufficient QSA capacity;
- PLE address/provider failure;
- MoE failure;
- non-finite input where forbidden;
- count/context overflow;
- aliasing;
- insufficient scratch/output capacity;
- transactional rollback.

## Hard boundaries

Do NOT implement:
- embedding;
- 48-layer executor;
- final norm;
- LM head;
- logits;
- sampling;
- final PLE/expert/KV scheduler;
- cache/residency optimization;
- SIMD/CUDA model kernels.

## Documentation

Create:
- `docs/TRANSFORMER-LAYER-CONTRACT.md`
- `docs/NATIVE-LAYER-REFERENCE.md`
- `docs/KQ-LAYER-API.md`
- `docs/adr/0018-native-transformer-layer-reference.md`

Update ARCHITECTURE, MODEL-ARCHITECTURE, MODEL-RUNTIME-STATE, GOLDEN-VECTORS,
TASKS, ROADMAP, Epic 2 plan/status, Task 2.10 checklist, tools provenance and
CHANGELOG.

## Acceptance

PASS requires:
- exact canonical layer/GR contract;
- independent vectors for all three layer families;
- calibration + disjoint holdout;
- native scalar layer composition;
- exact family classification;
- persistent state transitions PASS;
- transactional failure policy PASS;
- 48/48 target configs PASS;
- exact canonical PLE placement;
- state-footprint reconciliation PASS;
- no self-oracle;
- no embedding/48-layer/full-forward/logits implementation;
- CPU/CUDA regressions PASS;
- ADR 0018 ACCEPTED;
- next model-executor task NOT STARTED;
- KQ-BACKLOG-BENCH-002 DEFERRED;
- no tracked model weights/secrets/local paths;
- `git diff --check` PASS.

Do not commit or push automatically.
