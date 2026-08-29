# PLAN-TASK-2.1-CANONICAL-TENSOR-REGISTRY.md

Status: **COMPLETE / PASS — 2026-08-29**

## Objective

Implement Kestrel-Q's canonical tensor registry and semantic descriptor layer.

Task 2.0 exposes physical GGUF facts:

```text
name / GGUF type / dimensions / offset / packed bytes
```

Task 2.1 must transform them into stable Qwen3.8-Flash-Next semantic identities
that future runtime code can use without understanding GGUF names or
converter-specific representation choices.

No tensor payload access is allowed.

## Baseline

Reviewed checkpoint:

`5db71945806b483c7d8306144a7987ae9f17f320`

Entry status at reviewed checkpoint:

- Epic 1 COMPLETE/PASS
- Epic 2 IN PROGRESS
- Task 2.0 COMPLETE/PASS
- Task 2.1 NOT STARTED
- ADR 0008 ACCEPTED
- `KQ-BACKLOG-BENCH-002` DEFERRED

Epic 1 established:

```text
canonical total tensors       1658
vision excluded                333
MTP excluded                    31
initial-text semantics        1294
verified GGUF physical        1224
```

Representation reconciliation:

```text
1294 initial-text semantics
- 3 PLE address semantics represented as metadata
- 127 net reduction from 128 canonical PLE tables fused into one tensor
+ 48 MoE gate_up splits
+ 12 QSA index_qk splits
= 1224 physical GGUF tensors
```

This arithmetic is integration-test evidence, not production parsing logic.

## Architectural boundary

Maintain two separate models:

### Physical GGUF descriptor
Owned by Task 2.0:
- GGUF name
- physical type
- shape
- offset
- packed span

### Canonical semantic descriptor
Owned by Task 2.1:
- component
- layer type/id
- projection/role
- binding relation
- initial runtime scope
- preliminary placement hint
- physical binding(s)

Future GDN/QSA/MoE/PLE code consumes semantic descriptors, never raw GGUF names.

## Model identity

The adapter is explicitly Qwen3.8-Flash-Next-specific.

Validate, using parsed GGUF metadata/structure plus the canonical model spec:

- architecture `qwen4exp`
- hidden size 2560
- vocabulary 248320
- 48 text layers
- 36 GDN + 12 QSA
- context 262144 where represented
- 512 routed experts
- top-k 10
- shared expert
- target PLE/N-gram configuration

A near-match qwen4exp with incompatible topology must fail closed.

Canonical model-spec constants are allowed in this model-specific adapter.
Artifact-specific structural oracle values must not be hidden in the generic
GGUF parser.

## Semantic key

Define stable model-runtime keys independent of GGUF naming.

Required role families include:

```text
TOKEN_EMBEDDING
FINAL_NORM
LM_HEAD
GDN
QSA
QSA_INDEXER
GATED_RESIDUAL
MOE_ROUTER
ROUTED_EXPERT_STACK
SHARED_EXPERT
SHARED_EXPERT_GATE
PLE_TABLE
PLE_DENSE
PLE_ADDRESS_METADATA
LAYER_NORM
OTHER_REVIEWED_TEXT
```

Exact enum names may differ.

Keys must support layer and projection identity, and PLE logical-member
identity where needed.

## Layer topology

Expose all 48 text layers with explicit type.

Pattern:

```text
12 × [GDN, GDN, GDN, QSA]
```

Required:

- 48 unique contiguous layers
- 36 GDN
- 12 QSA
- all required per-layer roles present exactly as canonical semantics demand

## Binding relations

Explicitly represent relations equivalent to:

- `DIRECT_ONE_TO_ONE`
- `RENAMED_ONE_TO_ONE`
- `ONE_CANONICAL_TO_MULTIPLE_PHYSICAL`
- `MULTIPLE_CANONICAL_TO_ONE_PHYSICAL`
- `TRANSFORMED_LAYOUT`
- `METADATA_DERIVED`
- `ABSENT_INITIAL_SCOPE`

### Split relations

Task 1.3 proved:
- 48 canonical MoE gate_up semantics split into GGUF parts
- 12 canonical QSA index_qk semantics split into GGUF parts

A semantic descriptor must bind to multiple physical parts with deterministic
part role/order.

### Fused PLE

128 canonical PLE table semantics are represented by one physical GGUF tensor.

Preserve all 128 logical identities and their structural relationship to the
shared physical tensor.

Task 2.1 must not open payload slices, but must preserve enough geometry for
Task 2.2 to derive bounded views.

### Metadata-derived PLE

Three canonical PLE address semantics are encoded in metadata rather than as
payload tensors.

Represent them explicitly as metadata-derived semantics. Never invent a payload
offset.

## Registry coverage

Expected real-artifact integration results:

```text
initial semantic entries       1294
metadata-derived entries          3
physical tensors covered        1224
unknown physical                   0
unbound required semantic          0
```

Vision/MTP are recorded as intentionally excluded by ADR 0005, not accidentally
missing.

## Expert semantics

Do not create 512×48 fake tensor objects if canonical expert weights are
stacked tensors.

Represent:
- expert-stack projection
- 512 expert axis
- top-k 10 configuration
- proven expert-axis location
- split gate/up relationship

Per-expert bounded payload views belong to Task 2.2.

## Structural validation

Mapping must validate more than names.

As applicable verify:
- rank
- dimensions
- hidden/intermediate geometry
- expert axis
- split part count/order/shape
- PLE fused geometry/member count
- physical-type compatibility

Correct-looking name + wrong shape = fail closed.

## No fuzzy matching

Do not:
- classify ambiguous tensors through loose substring fallback
- silently ignore extras
- silently make missing required semantics optional

Every physical tensor must be explained by a reviewed deterministic rule.

Unknown/ambiguous mapping is a hard error.

## Placement hints

Expose only preliminary annotations:

```text
ALWAYS_NEEDED_CANDIDATE
ROUTED_EXPERT_CACHE_CANDIDATE
PLE_DISK_BACKED_CANDIDATE
EXCLUDED_INITIAL_SCOPE
NEUTRAL
```

Use:
- dense/shared always-used text weights → ALWAYS_NEEDED_CANDIDATE
- routed experts → ROUTED_EXPERT_CACHE_CANDIDATE
- PLE → PLE_DISK_BACKED_CANDIDATE
- vision/MTP capability → EXCLUDED_INITIAL_SCOPE

These hints do not allocate memory or implement scheduler/cache/prefetch.
`KQ-BACKLOG-BENCH-002` remains unfulfilled.

## API direction

Suggested concepts:

```c
typedef struct kq_model kq_model;
typedef struct kq_semantic_tensor kq_semantic_tensor;
typedef struct kq_tensor_binding kq_tensor_binding;
```

Desired operations:

```c
kq_status kq_model_open_from_gguf(const kq_gguf *, kq_model *);
void      kq_model_close(kq_model *);

uint32_t kq_model_layer_count(const kq_model *);
uint64_t kq_model_semantic_tensor_count(const kq_model *);

const kq_semantic_tensor *kq_model_semantic_tensor_at(...);
const kq_semantic_tensor *kq_model_find_semantic_tensor(...);
```

Exact API may differ.

Requirements:
- immutable registry after construction
- no global mutable mapping state
- explicit ownership/lifetime
- stable semantic IDs
- no payload pointer requirement

## Semantic diagnostics

Extend `kq-inspect` with semantics, preferably:

```text
--semantic-summary
--semantic <semantic-id>
```

Summary must expose:
- target model identity
- semantic count
- physical coverage
- metadata-derived count
- 48 / 36 / 12 layer counts
- 512 experts / top-k 10
- relation counts
- placement-hint counts
- unknown/unbound counts

No payload touch.

## Synthetic tests

Positive:
- direct/renamed relation
- split relation
- fused relation
- metadata-derived relation
- layer topology
- stacked expert-axis validation

Fail closed:
- correct architecture string but wrong topology
- wrong hidden/vocab/layer/expert counts
- missing required tensor
- unknown/ambiguous physical tensor
- duplicate semantic binding
- wrong rank/shape/expert axis
- missing/duplicate split part
- inconsistent split shapes
- wrong PLE fusion/member geometry
- missing PLE metadata
- invalid layer ID/type

Every discovered bug gets regression coverage.

## Real-artifact oracle

When `KQ_GGUF_PATH` exists:

```text
physical tensors            1224
semantic entries            1294
metadata-derived               3
unique physical coverage    1224
unknown physical               0
unbound required               0

layers                         48
GDN                            36
QSA                            12
experts                       512
top-k                          10
payload_bytes_accessed          0
```

Also compare the complete native mapping against Epic 1:

```text
research/model-gguf/Qwen3.8-Flash-Next/
c8b5954a88c2775c546b92593eda40ea041d3176/
canonical-gguf-mapping.csv
```

This comparison is test/research-only.

Production code must never read research CSV/JSON.

A deterministic native registry dump plus a Python comparison tool is allowed.

## Payload boundary

Registry construction uses only:
- GGUF metadata
- GGUF descriptors
- model-specific mapping rules

No tensor payload view is opened.

No dequantization.

Task 2.0's `payload_bytes_accessed = 0` contract must remain true.

## Documentation

Create:

```text
docs/MODEL-SEMANTIC-REGISTRY.md
docs/KQ-SEMANTIC-API.md
docs/adr/0009-canonical-semantic-tensor-registry.md
```

Update:
- `docs/ARCHITECTURE.md`
- `docs/TASKS.md`
- `docs/ROADMAP.md`
- Epic 2 plan/status
- Task 2.1 checklist
- `CHANGELOG.md`

## Acceptance

Task 2.1 PASS requires:

- clean CPU Release tests PASS
- clean CUDA Release tests PASS
- Task 2.0 regressions remain green
- qwen4exp target/topology validation
- 1294/1294 semantic entries
- 1224/1224 physical coverage
- 3 metadata-derived
- unknown physical = 0
- unbound required = 0
- full Epic 1 mapping-oracle comparison PASS
- 48 layers / 36 GDN / 12 QSA
- 512 experts / top-k 10
- split/fused/metadata-derived semantics validated
- payload bytes accessed = 0
- no runtime research-file dependency
- no dequant/inference/tokenizer/PLE/scheduler implementation
- ADR 0009 accepted
- `KQ-BACKLOG-BENCH-002` still deferred
- `CHANGELOG.md` updated
- tracked model weights = 0
- no secrets/local artifact path
- `git diff --check` PASS

## Definition of done

Task 2.1 is COMPLETE/PASS when the verified 1,224 physical GGUF tensors are
deterministically reconciled into the 1,294 canonical initial-text semantic
set, including split/fused/metadata-derived relations, and future runtime code
can consume stable semantic descriptors without knowing GGUF names.

Do not commit or push automatically.
