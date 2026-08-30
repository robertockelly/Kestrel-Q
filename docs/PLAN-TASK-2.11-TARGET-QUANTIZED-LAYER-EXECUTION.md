# PLAN-TASK-2.11-TARGET-QUANTIZED-LAYER-EXECUTION.md

## Objective

Bridge the proven scalar layer semantics to the real quantized
Qwen3.8-Flash-Next GGUF.

Task 2.10 proves complete-layer equations using independent synthetic F32
Class-C vectors and validates real target structure without executing real
target weights.

Task 2.11 must make a complete canonical layer executable directly from the
verified quantized artifact through bounded on-demand weight access.

Required representative families:

1. ordinary GDN layer;
2. QSA layer;
3. PLE-enabled GDN layer.

This is the final bridge before the first 48-layer native model executor.

Out of scope:
- token embedding model entry;
- 48-layer iteration;
- final norm;
- LM head/logits;
- sampling;
- final scheduler/cache policy;
- SIMD/CUDA model kernels.

## Entry gate

Start only after Task 2.10 is checkpointed and pushed.

Expected:
- Task 2.0–2.10 COMPLETE/PASS
- ADR 0018 ACCEPTED
- 48/48 layer configs structurally valid
- Task 2.11 NOT STARTED
- KQ-BACKLOG-BENCH-002 DEFERRED

## Architectural bridge

Connect:

```text
semantic weight request
-> target weight provider
-> Task 2.2 bounded view
-> Task 2.5 quantized row-dot/dequant
-> existing scalar operator/layer equations
```

No full target weight matrix may be materialized in F32.

## Weight-provider contract

Create `docs/TARGET-WEIGHT-PROVIDER-CONTRACT.md`.

Document:
- GR/GDN/QSA/MoE/PLE access patterns;
- semantic row orientation;
- split bindings;
- transformed layouts;
- expert-member access;
- PLE row access;
- storage types;
- scratch/lifetime/error policy;
- payload accounting.

The provider exposes semantic operations, not raw offsets.

## Provider API

Create a narrow C17 provider that can perform semantic linear operations and
selected-expert/member access through bounded views.

Requirements:
- no caller-side GGUF offset arithmetic;
- no whole-matrix F32 allocation;
- row/chunk access only;
- Task 2.5 numerics reused;
- deterministic accounting;
- explicit capacity/scratch/lifetime.

## One algorithm, multiple providers

Prefer:

```text
synthetic F32 provider ─┐
                        ├─> same GDN/QSA/MoE/PLE/layer equations
real quantized provider ┘
```

Do not create a separate target-only model algorithm.

All Task 2.6–2.10 synthetic/reference tests must remain green.

## Split/transformed storage

Provider must handle:
- MoE gate/up split;
- QSA index_qk split;
- transformed layouts.

Unsupported transforms fail closed.

Do not concatenate giant physical tensors.

## Independent target-quantized oracle

Build a test/research-only independent target oracle:

1. independently decode required GGUF rows/blocks using the pinned Class-Q
   llama.cpp helper;
2. execute frozen Task 2.10 canonical layer equations in separate Python/NumPy
   tooling;
3. use deterministic synthetic hidden inputs and initial states;
4. compare Kestrel-Q real-quantized execution.

Pinned Class-Q:
`llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a`

Pinned canonical semantics:
- Qwen `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Transformers `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`

Production must not depend on Python/NumPy/llama.cpp.
Kestrel-Q is not its own oracle.

## Representative target layers

Choose and document one real representative for:
- ordinary GDN;
- QSA;
- PLE-GDN layer 1.

No production special-casing by file offset.

## Real cases

For each family validate:
- reset/one-token case;
- short prefill where feasible;
- prefill + one decode step.

Use deterministic finite synthetic hidden inputs.

For QSA require exact sparse selection.
For MoE require exact routed expert IDs/order.
For PLE require exact logical member/row intents.

## Target-specific calibration

Create calibration and disjoint holdout for quantized target execution.

Do not reuse Task 2.10's synthetic F32 tolerance blindly.

Discrete decisions remain exact.
Floating outputs/states use calibrated target-specific contracts.

## Evidence

Create governed evidence under the operator research root, e.g.:
- `target-layer-contract.json`
- `target-layer-calibration.json`
- `target-layer-holdout.json`
- `target-layer-state-vectors.json`
- `target-layer-native-validation.json`
- `target-layer-manifest.json`

Record:
- GGUF identity;
- oracle revisions;
- representative layer IDs;
- deterministic input/state generation;
- semantic tensors accessed;
- payload bytes/blocks touched;
- exact discrete selections;
- outputs/states/checkpoints;
- calibrated contracts;
- SHA manifest.

Never commit raw model payload bytes.

## Access invariants

MoE:
- only selected routed experts may be touched;
- shared expert is accessed separately;
- no unselected routed expert may be requested.

PLE:
- only canonical requested rows may be touched;
- no speculative/prefetch reads;
- no whole member/table materialization.

Instrument and test both invariants.

## Real-payload budget

Hard correctness-test ceiling:

`KQ_TASK211_REAL_PAYLOAD_SAMPLE_BUDGET <= 768 MiB logical packed bytes`

Prefer substantially less.

Record:
- logical bytes touched;
- blocks touched;
- unique semantic tensors touched;
- selected expert requests;
- PLE row requests.

This is not a storage-performance benchmark.

## Full 48-layer provider preflight

Even though only representative layers execute numerically, validate all 48
layer configs against the target provider.

Require:
- 48/48 provider-compatible configs;
- no unsupported semantic role;
- no unsupported transformed layout in full-model path;
- valid selected-expert member access for every routed stack;
- valid PLE provider for layer 1.

This is the gate for the first-token task.

## Transactional failure

Task 2.10 transactional state guarantees remain mandatory with real providers.

Inject provider failures at several stages and prove no visible partial advance
of GDN/QSA/PLE state when a complete layer call fails.

## Memory discipline

Record:
- provider object bytes;
- row/dequant scratch;
- layer/operator scratch;
- persistent target state;
- maximum simultaneously materialized F32 weight bytes.

Hard requirement:
**no complete target weight matrix in F32**.

## First-token readiness

Create `docs/FIRST-TOKEN-READINESS.md`.

After PASS, list concrete remaining work.

Expected:
- token embedding lookup;
- 48-layer state orchestration;
- 48-layer prefill/decode loop;
- final model norm;
- LM head/logits;
- greedy argmax;
- native decode.

Any additional blocker must be evidence-backed.

## Hard boundaries

Do NOT implement:
- embedding model input;
- full 48-layer execution;
- final norm;
- LM head/logits;
- argmax/sampling;
- final scheduler/cache optimization;
- SIMD/CUDA model kernels.

The next task is expected to target First Correct Native Token.

## Documentation

Create:
- `docs/TARGET-WEIGHT-PROVIDER-CONTRACT.md`
- `docs/NATIVE-QUANTIZED-LAYER-EXECUTION.md`
- `docs/KQ-WEIGHT-PROVIDER-API.md`
- `docs/FIRST-TOKEN-READINESS.md`
- `docs/adr/0019-target-quantized-layer-execution.md`

Update architecture/quantization/runtime/goldens/TASKS/ROADMAP/Epic 2,
Task 2.11 checklist, tools provenance and CHANGELOG.

## Acceptance

PASS requires:
- semantic weight-provider contract/API;
- bounded target row/chunk execution;
- no duplicate dequant implementation;
- independent Class-Q + canonical-equation target oracle;
- calibration + holdout;
- real ordinary GDN layer PASS;
- real QSA layer PASS;
- real PLE-GDN layer PASS;
- exact QSA/MoE/PLE decisions;
- selected-expert and PLE-row access invariants PASS;
- payload <= 768 MiB;
- 48/48 provider preflight PASS;
- transactional failure PASS;
- FIRST-TOKEN-READINESS has no unexplained blocker;
- CPU/CUDA regressions PASS;
- ADR 0019 ACCEPTED;
- next first-token task NOT STARTED;
- KQ-BACKLOG-BENCH-002 DEFERRED;
- no tracked model weights/secrets/local paths;
- `git diff --check` PASS.

Do not commit or push automatically.
