# PLAN-TASK-1.3-CANONICAL-TO-GGUF-MAPPING.md

## Objective

Build a complete, reproducible mapping between the canonical Qwen3.8-Flash-Next Safetensors checkpoint and the maintainer's verified local Unsloth `UD-Q4_K_XL` GGUF representation.

Task 1.3 must explain, quantitatively and structurally, how:

```text
Canonical Safetensors
1,658 tensors
359,999,963,128 stored bytes
```

becomes:

```text
Local merged GGUF v3
1,224 tensors
111,334,654,400 bytes
```

without treating tensor-count differences as unexplained omissions.

This remains a research/model-format task. It must not implement inference, a production GGUF loader, tensor kernels, scheduling, or quantization kernels.

## Baseline

Reviewed checkpoint: `2ae53be4c38a6dd995f3f060bb90408e160d5eec`

- Task 1.0 COMPLETE/PASS
- Task 1.1 COMPLETE/PASS
- Task 1.2 COMPLETE/PASS
- Task 1.3 NOT STARTED
- ADR 0004 ACCEPTED
- ADR 0005 ACCEPTED

Canonical HF revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`

Canonical inventory:
- 131 shards
- 1,658 tensors
- payload 359,999,963,128 bytes

Local GGUF:
- artifact `KQ-MODEL-ARTIFACT-001`
- `Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf`
- 111,334,654,400 bytes
- SHA-256 `8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2`
- GGUF v3
- architecture `qwen4exp`
- 67 metadata keys
- 1,224 tensors
- source `unsloth/Qwen3.8-Flash-Next-GGUF`
- pinned Unsloth revision `c8b5954a88c2775c546b92593eda40ea041d3176`

## 1. GGUF access rule

Inspect the local GGUF read-only. Parse only:
- GGUF header
- metadata directory
- tensor directory
- tensor names/dimensions/types
- offsets/alignment
- data-section start
- packed spans derived from offsets and format semantics

Do not read tensor payload contents by default. Do not copy, rewrite, rename or mutate the file.

The path must come from `KQ_GGUF_PATH`; no user-specific path may be committed.

## 2. Required outputs

Create:

```text
research/model-gguf/Qwen3.8-Flash-Next/
  c8b5954a88c2775c546b92593eda40ea041d3176/
    gguf-metadata.json
    gguf-tensor-inventory.csv
    canonical-gguf-mapping.csv
    gguf-summary.json
    mapping-evidence.json
```

Create:

```text
docs/MODEL-GGUF-MAPPING.md
docs/MODEL-QUANTIZATION-FOOTPRINT.md
docs/adr/0006-initial-model-container-strategy.md
```

## 3. Research tooling

Research-only tooling may be added under `tools/`, e.g.:

```text
tools/inspect-gguf.py
tools/map-canonical-to-gguf.py
```

Requirements:
- read-only
- deterministic
- fail closed
- no production dependency
- no user-specific absolute path
- no embedded credentials
- `KQ_GGUF_PATH` required
- document usage in `tools/README.md`
- update `CHANGELOG.md`

Prefer a small purpose-built parser unless a dependency is clearly justified and license-reviewed.

## 4. GGUF structural validation

Validate:
- magic/version
- metadata count
- tensor count
- architecture
- alignment
- data-section offset
- unique tensor names
- valid ranks/dimensions
- recognized GGUF/GGML types
- aligned/non-overlapping tensor spans
- all spans within file size
- split/merge metadata
- file-level overhead

Compute:

```text
file_size - total_tensor_packed_bytes = GGUF metadata/header/alignment/other overhead
```

Explain the remainder precisely.

## 5. Quantization-type accounting

Task 1.0 observed:
- F32
- Q5_1
- Q8_0
- Q4_K
- Q5_K
- IQ4_NL
- BF16

Reproduce independently the exact count, parameters and packed bytes by GGUF type.

For each type used:
- block size
- packed bytes/block
- effective bits/parameter where meaningful
- validation against actual tensor spans

Pin and license-review any GGUF/GGML format source used.

## 6. Canonical-to-GGUF mapping

Every canonical tensor from Task 1.2 must receive exactly one status:

- `ONE_TO_ONE`
- `RENAMED_ONE_TO_ONE`
- `FUSED_INTO_GGUF`
- `SPLIT_IN_GGUF`
- `TRANSFORMED_LAYOUT`
- `OMITTED_INITIAL_SCOPE_VALID`
- `OMITTED_FORMAT_DERIVED`
- `ALIAS_OR_TIED`
- `NO_DIRECT_TENSOR_EQUIVALENT`
- `UNRESOLVED`

Every GGUF tensor must map back to one or more canonical meanings.

PASS requires:
- canonical `UNRESOLVED = 0`
- unexplained GGUF tensors = 0

## 7. Deterministic mapping rules

Build versioned mapping rules for:
- embeddings/head
- final norm
- 48 layers
- GDN
- QSA
- QSA indexer
- GR
- routers
- routed experts
- shared experts
- PLE/N-gram
- vision
- MTP
- connector/projector if distinct

Document:

```text
canonical name pattern -> transformation -> GGUF name pattern
```

Separate semantic transformation from dimension-order/layout convention.

## 8. Converter/provenance evidence

Use pinned converter/format sources only when needed to explain transformations.

For each source record:
- exact revision
- license
- exact file/function/symbol
- evidence role

Separate:
- observed artifact structure
- proven converter behavior
- strong inference
- unresolved provenance

Do not copy implementation code.

## 9. Actual packed footprint by canonical family

Compute actual GGUF packed bytes for the same families defined in Task 1.2:
- full artifact
- initial text-only
- vision
- MTP
- PLE/N-gram
- token embeddings
- LM head
- GDN
- QSA/indexer
- GR
- routers
- routed experts
- shared experts
- dense/norm text

For each family report:
- canonical BF16 bytes
- Task-1.2 idealized Q4 floor where useful
- actual GGUF packed tensor bytes
- effective bits/parameter
- share of GGUF
- delta vs idealized floor

Keep file/header/alignment overhead separate and avoid double counting fused tensors.

## 10. Precision policy analysis

Report per canonical family:
- tensor count by GGUF type
- parameter-weighted type distribution
- packed-byte distribution
- effective weighted bits/parameter

Especially analyze:
- PLE
- routed experts
- routers
- shared experts
- GDN
- QSA/indexer
- embeddings/head
- norms/GR/gates
- vision/MTP

Use labels such as:
- `OBSERVED_HIGHER_PRECISION`
- `OBSERVED_LOWER_PRECISION`
- `MIXED`
- `UNQUANTIZED`

Do not infer sensitivity merely from higher precision.

## 11. Imatrix analysis

Use embedded metadata and pinned upstream repository metadata to determine what is actually proven about the importance-matrix pipeline.

Record:
- imatrix source metadata
- calibration dataset metadata
- whether scores are embedded/recoverable
- which precision choices can be tied directly to evidence

Do not download large imatrix payload unless clearly necessary.

## 12. Reconcile the 434-tensor-count difference

Canonical: 1,658 tensors
GGUF: 1,224 tensors

Provide exact arithmetic by cause, e.g. aliases, fusions, splits, valid omissions, format-derived/no-direct tensors.

Final arithmetic must reconcile exactly to 1,224 GGUF tensors.

## 13. Investigate the 384-byte merged-file delta

Published sharded total: 111,334,654,784 bytes
Local merged GGUF: 111,334,654,400 bytes
Difference: 384 bytes smaller

Investigate metadata/layout semantics only. Do not redownload upstream GGUF payloads.

Use bounded upstream metadata/header access if needed.

Classify the delta as proven metadata/split overhead removal, proven alignment/layout effect, evidence-backed combination, or unresolved. Do not force an explanation.

## 14. KQ-01 relevance

Compare actual GGUF packed footprints with:
- ~8 GiB VRAM working budget
- ~20 GiB host model/cache budget
- ~25.63 GB/s RAM bandwidth
- ~26 GB/s pinned PCIe bandwidth
- ~0.44 GB/s SATA read

Derive actual:
- initial-text GGUF size
- PLE size
- routed-expert size
- dense/shared size
- per-expert packed size
- top-10 selected packed parameter footprint/layer
- selected packed parameter footprint across 48 layers

Selected packed parameter footprint is not measured physical I/O. No tokens/s claims.

## 15. ADR 0006 — initial container strategy

Evaluate:

### A. Direct GGUF first runtime
Pros/cons: verified quantized artifact, mature metadata, compatibility, mapping cost, random access, PLE/expert locality, mmap suitability.

### B. Kestrel-Q-native converted container first
Pros/cons: model-specific layout/indexes versus converter complexity, extra storage and validation burden.

### C. Staged strategy
Example: prototype/correctness path from GGUF, canonical internal semantics, introduce native container later only if profiling proves benefit.

Accept one only if Task 1.3 evidence supports it. Do not implement a container here.

## 16. Required project updates

Update:
- `docs/MODEL-ARTIFACT-REGISTER.md`
- `docs/TASKS.md`
- `docs/ROADMAP.md`
- `docs/TASKS-EPIC-1-MODEL-CHARACTERIZATION.md`
- `docs/TASKS-TASK-1.3-CANONICAL-TO-GGUF-MAPPING.md`
- `tools/README.md`
- `CHANGELOG.md`

The canonical CHANGELOG rule is mandatory.

## 17. Acceptance gates

### Integrity
- [ ] exact artifact from `KQ_GGUF_PATH`
- [ ] identity verified against register
- [ ] read-only inspection
- [ ] GGUF v3 structure valid
- [ ] 1,224 unique tensors
- [ ] metadata count validated
- [ ] offsets/alignment/spans valid
- [ ] all tensor types recognized

### Mapping
- [ ] 1,658 canonical tensors covered
- [ ] 1,224 GGUF tensors covered
- [ ] canonical UNRESOLVED = 0
- [ ] unexplained GGUF = 0
- [ ] exact 434-count reconciliation
- [ ] deterministic mapping rules

### Quantization
- [ ] exact packed bytes by type
- [ ] exact packed bytes by canonical family
- [ ] effective bits/parameter
- [ ] actual PLE footprint
- [ ] actual routed-expert footprint
- [ ] actual dense/shared footprint
- [ ] per-expert and top-10 findings

### Provenance
- [ ] Unsloth revision pinned
- [ ] format/converter source pinned/licensed where used
- [ ] imatrix evidence documented
- [ ] observed/inferred/proven distinctions explicit

### Merge delta
- [ ] 384-byte difference investigated
- [ ] explanation proven or explicitly unresolved
- [ ] no upstream GGUF payload download

### ADR/governance
- [ ] ADR 0006 evidence-backed outcome
- [ ] CHANGELOG updated
- [ ] TASKS/ROADMAP synchronized
- [ ] no production model/container implementation

### Safety
- [ ] tracked `.gguf` = 0
- [ ] tracked `.safetensors` = 0
- [ ] repository model weights = 0
- [ ] no local artifact path committed
- [ ] no secrets
- [ ] no incompatible source copied
- [ ] `git diff --check` PASS

## Definition of done

Task 1.3 is COMPLETE/PASS only when the canonical checkpoint and local GGUF are completely reconciled at tensor-semantic and packed-footprint levels, with no unexplained mapping gaps, and ADR 0006 has an evidence-backed outcome.

Do not commit or push automatically. Leave the Task 1.3 delta ready for review.
