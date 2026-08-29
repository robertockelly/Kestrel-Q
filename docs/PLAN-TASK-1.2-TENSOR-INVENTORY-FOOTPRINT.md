# PLAN-TASK-1.2-TENSOR-INVENTORY-FOOTPRINT.md

## Objective

Build an exact, reproducible static-weight and runtime-state footprint model for Qwen3.8-Flash-Next from the pinned canonical checkpoint metadata, without downloading model weight payloads.

Task 1.2 must answer:

1. What are the exact canonical tensor names, shapes, dtypes and payload sizes?
2. How many bytes belong to each subsystem/family?
3. What exact weight set is required by the ADR-0005 initial text-only path?
4. What is excluded initially because it belongs only to vision or optional MTP?
5. What is the exact routed-expert footprint per expert/layer and the top-10 parameter-selection footprint?
6. How large is PLE/N-gram storage?
7. How does persistent runtime state scale with context?
8. What are the idealized Q8/Q6/Q5/Q4/Q3 payload lower bounds?
9. How do these facts compare with KQ-01's ~8 GiB VRAM and ~20 GiB managed host-RAM budgets?

This task does not decide the final scheduler, eviction policy, GGUF packing or production tensor format.

## Baseline

Reviewed Kestrel-Q checkpoint:

`82ba9b7c3d1a74725606f90718888e7e85292f54`

Status:

- Task 1.0 COMPLETE/PASS
- Task 1.1 COMPLETE/PASS
- ADR 0004 ACCEPTED
- ADR 0005 ACCEPTED
- Task 1.2 NOT STARTED
- Task 1.3 NOT STARTED

Canonical HF revision:

`de4b8e4d43b917e7706784d8bb445c9af86a3540`

Task 1.0 established 131 Safetensors shards and 1,658 tensor names in the pinned index. Task 1.1 established the initial text-only/no-vision/no-MTP execution scope.

## Critical Safetensors rule

`model.safetensors.index.json` maps tensor names to shards but does not provide every tensor shape and dtype.

Task 1.2 may retrieve **only Safetensors header bytes** from the 131 pinned official shards.

For each shard:

1. derive the exact pinned URL from the pinned index/revision;
2. request bytes `0-7`;
3. require a bounded/partial response;
4. parse the 8-byte little-endian header length;
5. enforce a strict maximum header size;
6. request exactly bytes `8..8+header_length-1`;
7. verify the response is bounded and exact;
8. parse the UTF-8 JSON header;
9. close before any payload byte is requested;
10. record requested range and bytes actually received.

If the server ignores Range or attempts a full shard response, abort immediately and fail closed.

Never save a file with a `.safetensors` extension.

Research tooling must enforce:

- exact repository/revision;
- exact 131-shard allowlist;
- maximum header bytes per shard;
- maximum total bytes for the run;
- maximum response size;
- allowed hostnames;
- non-zero exit on any safety violation.

## Required machine-readable outputs

Create:

```text
research/model-tensors/Qwen3.8-Flash-Next/
  de4b8e4d43b917e7706784d8bb445c9af86a3540/
    shard-header-manifest.json
    tensor-inventory.csv
    tensor-summary.json
    network-range-audit.json
```

`tensor-inventory.csv` must contain at least:

- tensor_name
- shard
- dtype
- shape
- rank
- parameter_count
- payload_bytes
- subsystem
- component
- layer_id
- expert_id
- initial_text_scope
- classification_rule
- notes

Output ordering must be deterministic.

## Research tooling

Research-only Python tooling under `tools/` is allowed and expected, for example:

```text
tools/capture-safetensors-headers.py
tools/analyze-model-tensors.py
```

It must not become a production dependency.

Requirements:

- pinned revision mandatory;
- explicit allowlist;
- bounded Range-only transfer;
- deterministic output;
- clear exit codes;
- no credentials;
- no user-specific absolute paths;
- documented in `tools/README.md`;
- recorded in `CHANGELOG.md`.

## Canonical reconciliation

Reconcile three independent views:

1. pinned `model.safetensors.index.json`;
2. parsed headers for all 131 shards;
3. pinned repository/shard size metadata.

Required checks:

- exactly 131 expected shards;
- all 1,658 index tensor names accounted for unless pinned evidence proves an exact different count;
- each tensor appears exactly once;
- no unexpected non-metadata header tensor;
- valid dtype and shape;
- `parameter_count = product(shape)`;
- payload bytes consistent with offsets and dtype semantics;
- data offsets valid and non-overlapping;
- final data span + header region consistent with remote shard size where verifiable;
- aggregate payload bytes reconcile with index `metadata.total_size`, or the exact evidence-backed difference is documented.

Do not silently normalize a mismatch.

## Deterministic classification

Classify every canonical tensor using canonical names plus Task-1.1 architecture.

Top-level categories must distinguish:

- text backbone
- PLE/N-gram
- vision
- MTP
- multimodal connector/projector if distinct
- embedding/final head
- reviewed canonical other

Text components must distinguish:

- token embedding
- final norm
- LM head
- GDN
- QSA attention
- QSA indexer
- Gated Residual parameters
- router
- routed experts
- shared expert
- layer norms
- other text dense

For routed experts derive layer ID, expert ID and subcomponent where names permit.

Do not use GGUF tensor names as canonical classification evidence.

Task 1.2 PASS requires zero unexplained `UNKNOWN` or `REVIEW` rows.

## Required documentation

Create:

```text
docs/MODEL-TENSOR-INVENTORY.md
docs/MODEL-FOOTPRINT.md
```

Report exact stored bytes and parameter counts for:

- full released checkpoint;
- initial text-only ordinary-autoregressive scope;
- excluded vision;
- excluded MTP;
- PLE/N-gram;
- token embeddings;
- LM head;
- GDN;
- QSA/indexer;
- Gated Residual parameters;
- routers;
- routed experts;
- shared experts;
- remaining dense/norm parameters.

Use actual stored tensors. If conceptual parameter counts differ because of tying/sharing or omitted aliases, say so explicitly.

## Per-layer and per-expert analysis

For every text layer report:

- layer type;
- non-expert bytes;
- router bytes;
- routed-expert bytes;
- shared-expert bytes;
- total bytes;
- parameter count.

For routed experts derive:

- exact bytes per expert/layer;
- uniformity or min/max/median;
- exact top-10 selected expert parameter payload per layer;
- aggregate selected routed-expert parameter footprint across all 48 layers for one token.

This is a **parameter-selection footprint**, not measured memory traffic and not a tokens/s claim.

## PLE / N-gram analysis

Report:

- exact canonical PLE tensor families;
- exact parameters;
- exact stored bytes;
- share of full checkpoint;
- share of initial text-only static weights;
- partitions/tables visible in canonical tensor structure;
- idealized Q8/Q6/Q5/Q4/Q3 lower bounds.

Relate these numbers to Task 1.1's early-address/prefetch finding without designing cache policy.

## Idealized quantization lower bounds

For meaningful families and totals calculate:

- 8 bits/parameter
- 6 bits/parameter
- 5 bits/parameter
- 4 bits/parameter
- 3 bits/parameter

These are theoretical payload floors only.

Explicitly exclude scales, zero points, block metadata, padding/alignment, mixed-precision exceptions, format headers and converter-specific packing.

Do not call them GGUF sizes. Actual Unsloth/GGUF representation is Task 1.3.

## Initial text-only weight set

Every tensor receives exactly one scope:

- `REQUIRED_INITIAL_TEXT`
- `EXCLUDED_INITIAL_VISION`
- `EXCLUDED_INITIAL_MTP`
- `REVIEW`

Task 1.2 PASS requires `REVIEW = 0`.

This quantifies ADR 0005; it does not redefine the canonical model.

## Runtime-state footprint

Using `docs/MODEL-RUNTIME-STATE.md` and Task-1.1 evidence, quantify batch=1 persistent state independently from static weights.

At minimum:

- all 36 GDN recurrent matrix states;
- all 36 GDN causal-convolution states;
- all 12 QSA K/V caches;
- all 12 QSA raw index-key states;
- any other source-backed persistent QSA state;
- PLE token-history/dilated-convolution state;
- position bookkeeping.

Report context lengths:

- 1
- 4,096
- 16,384
- 65,536
- 262,144

Use source-backed dtypes.

GR residual branches are current-forward activations, not context-persistent cache.

Keep optional vision/MTP runtime state outside the initial text scope.

## KQ-01 feasibility

Compare exact results against:

```text
VRAM working budget       ~8 GiB
Host model/cache budget   ~20 GiB
RAM bandwidth             ~25.63 GB/s measured
Pinned PCIe H2D/D2H       ~26 GB/s measured at large blocks
SATA sequential read      ~0.44 GB/s measured
```

Compare canonical BF16 and idealized Q8/Q6/Q5/Q4/Q3 for at least:

- full checkpoint;
- initial text-only static weights;
- PLE;
- routed experts;
- dense/shared/non-routed text weights.

Do not claim that a runtime exists because an idealized payload floor fits a tier.

## Preliminary placement candidates

Task 1.2 may assign analysis-only labels:

- `ALWAYS_NEEDED_CANDIDATE`
- `ROUTED_EXPERT_CACHE_CANDIDATE`
- `PLE_PREFETCH_CANDIDATE`
- `COLD_BACKING_CANDIDATE`
- `EXCLUDED_INITIAL_SCOPE`

These are not final scheduler policy. Do not define cache sizes, eviction, prefetch depth or final VRAM residency.

## Network audit

`network-range-audit.json` must record for every request:

- shard filename;
- pinned revision;
- requested byte range;
- HTTP status;
- Content-Range/equivalent;
- expected bytes;
- actual bytes;
- header length;
- result.

Final report must include:

- total bytes fetched across header requests;
- largest single response;
- `weight_payload_bytes_fetched = 0`.

If payload bytes may have transferred, preserve evidence and mark the run DIRTY/INVALID until root-caused.

## Required project updates

Update:

- `docs/TASKS.md`
- `docs/ROADMAP.md`
- `docs/TASKS-EPIC-1-MODEL-CHARACTERIZATION.md`
- `docs/TASKS-TASK-1.2-TENSOR-INVENTORY-FOOTPRINT.md`
- `tools/README.md` if tools are added
- `CHANGELOG.md`

The canonical CHANGELOG rule is mandatory.

Task 1.3 remains NOT STARTED.

## Acceptance gates

### Header safety
- [ ] pinned revision enforced
- [ ] exact 131-shard allowlist
- [ ] no unrestricted/full model download
- [ ] 131/131 headers captured
- [ ] every response bounded/audited
- [ ] zero weight payload bytes fetched
- [ ] no `.safetensors` file saved

### Tensor inventory
- [ ] all canonical tensors reconciled
- [ ] no duplicate names
- [ ] dtype/shape/parameters/bytes for every tensor
- [ ] aggregate bytes reconciled
- [ ] zero unexplained classification rows

### Footprint
- [ ] exact full static footprint
- [ ] exact initial text-only footprint
- [ ] exact excluded vision
- [ ] exact excluded MTP
- [ ] exact PLE
- [ ] exact routed experts
- [ ] exact shared/dense
- [ ] per-layer/expert summary
- [ ] idealized Q8/Q6/Q5/Q4/Q3

### Runtime state
- [ ] GDN fixed state quantified
- [ ] QSA context-growing state quantified
- [ ] PLE/position state quantified
- [ ] context table through 262,144

### KQ-01/governance
- [ ] 8 GiB VRAM / 20 GiB RAM comparison
- [ ] no theoretical quantized size called implemented
- [ ] no performance/token-rate claim
- [ ] CHANGELOG updated
- [ ] TASKS/ROADMAP synchronized
- [ ] Task 1.3 NOT STARTED
- [ ] tracked `.safetensors` = 0
- [ ] tracked `.gguf` = 0
- [ ] repository model weights = 0
- [ ] no secrets
- [ ] no model/runtime implementation added
- [ ] `git diff --check` PASS

## Definition of done

Task 1.2 is COMPLETE/PASS only when Kestrel-Q has an exact, reproducible canonical tensor inventory and quantitative memory model sufficient to begin Task 1.3 GGUF mapping without guessing about checkpoint composition or footprint.

Do not commit or push automatically. Leave the intentional Task 1.2 delta ready for review.
