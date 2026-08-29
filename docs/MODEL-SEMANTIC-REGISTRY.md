# Qwen3.8-Flash-Next semantic tensor registry

Status: **TASK 2.1 COMPLETE / PASS**

Task 2.1 establishes the model-semantic boundary above Task 2.0's physical
GGUF descriptors. Future model execution addresses stable Kestrel-Q semantic
IDs; it does not parse or construct raw GGUF names.

## Identity and topology gate

The production adapter is intentionally specific to Qwen3.8-Flash-Next. It
requires `qwen4exp` and validates the target metadata before constructing any
semantic entry:

- hidden size 2,560 and tokenizer vocabulary 248,320;
- context length 262,144;
- 48 layers with the exact `GDN, GDN, GDN, QSA` repeating pattern;
- 36 GDN and 12 QSA layers;
- 512 routed experts, top-k 10 and shared intermediate width 640;
- GDN, QSA/indexer and four-branch Gated Residual geometry;
- the single layer-1-zero-based PLE, n-gram size 3, eight heads per order,
  width 160 and exact address arrays.

Architecture spelling alone is insufficient. Missing/mistyped metadata,
near-match topology, a right-looking name with a wrong rank/shape/type, an
invalid expert axis or any unexplained physical tensor fails closed.

## Stable identities

Semantic IDs are generated independently from GGUF names. Representative IDs
are:

```text
text.token_embedding
text.final_gr.down
layer.00.gdn.qkv
layer.03.qsa.indexer.qk
layer.00.moe.routed.gate_up
layer.01.ple.table.127
layer.01.ple.address.head_offsets
```

Each immutable descriptor records:

- component and projection/role;
- zero-based layer ID and validated layer type where applicable;
- canonical dtype, rank and dimensions;
- runtime scope and preliminary placement annotation;
- binding relation and ordered physical/metadata binding parts;
- canonical and physical expert axes for routed expert stacks;
- PLE fused-member identity without opening a payload slice.

The official canonical tensor name is retained as provenance/diagnostic data.
Runtime lookup uses the stable semantic ID.

## Complete reconciliation

The registered initial text scope contains exactly 1,294 semantic entries.
Vision's 333 and MTP's 31 canonical tensors remain intentionally excluded by
ADR 0005; they are not silently missing registry entries.

| Binding relation | Semantic entries |
|---|---:|
| `DIRECT_ONE_TO_ONE` | 0 |
| `RENAMED_ONE_TO_ONE` | 847 |
| `TRANSFORMED_LAYOUT` | 256 |
| `ONE_CANONICAL_TO_MULTIPLE_PHYSICAL` | 60 |
| `MULTIPLE_CANONICAL_TO_ONE_PHYSICAL` | 128 |
| `METADATA_DERIVED` | 3 |
| `ABSENT_INITIAL_SCOPE` | 0 |
| **Total** | **1,294** |

The 60 split semantics comprise 48 routed-expert `gate_up` projections and 12
QSA index `qk` projections. Part order is explicit: gate/up and index-query/
index-key respectively.

The 128 PLE table semantics all bind to the one fused
`per_layer_token_embd.weight` descriptor. Every entry records its member index
0–127 and member count 128. Task 2.1 validates the exact fused shape
`[160, 320001536]`; Task 2.2, not this task, may derive bounded member views.

Three PLE address semantics bind to the exact uint64 metadata arrays rather
than inventing physical tensors or offsets:

- `qwen4exp.ple.layer_multipliers`;
- `qwen4exp.ple.head_offsets`;
- `qwen4exp.ple.head_vocab_sizes`.

Task 2.4 consumes these three bindings and validates all 128 fused member
descriptors when constructing its independent immutable PLE address config.
The config copies only the exact address constants; it opens no fused-member
view and emits canonical member/row identities rather than physical offsets.
The semantic-registry ownership contract is therefore unchanged.

Unique physical coverage is 1,224/1,224, unknown physical tensors are zero and
unbound required semantics are zero.

## Component and placement annotations

| Component | Entries | Placement annotation |
|---|---:|---|
| Token embedding | 1 | `ALWAYS_NEEDED_CANDIDATE` |
| LM head | 1 | `ALWAYS_NEEDED_CANDIDATE` |
| Final Gated Residual | 3 | `ALWAYS_NEEDED_CANDIDATE` |
| Layer Gated Residual | 384 | `ALWAYS_NEEDED_CANDIDATE` |
| GDN | 324 | `ALWAYS_NEEDED_CANDIDATE` |
| QSA attention | 72 | `ALWAYS_NEEDED_CANDIDATE` |
| QSA indexer | 36 | `ALWAYS_NEEDED_CANDIDATE` |
| MoE routers | 48 | `ALWAYS_NEEDED_CANDIDATE` |
| Routed expert stacks | 96 | `ROUTED_EXPERT_CACHE_CANDIDATE` |
| Shared experts/gates | 192 | `ALWAYS_NEEDED_CANDIDATE` |
| PLE table members | 128 | `PLE_DISK_BACKED_CANDIDATE` |
| PLE dense | 6 | `PLE_DISK_BACKED_CANDIDATE` |
| PLE address metadata | 3 | `PLE_DISK_BACKED_CANDIDATE` |

Totals are 1,061 always-needed candidates, 96 routed-expert cache candidates
and 137 PLE disk-backed candidates. These are annotations only. No allocation,
residency, cache, prefetch or scheduler behavior is implemented, and
`KQ-BACKLOG-BENCH-002` remains deferred.

## Routed expert boundary

Each layer has two canonical routed-stack semantics: `down` and `gate_up`.
They retain the canonical 512-expert axis without creating 512 × 48 fabricated
tensor objects. The physical expert axis is validated at dimension 2 for each
stack and the gate/up halves are separate ordered physical bindings. Per-expert
payload views remain Task 2.2.

## Evidence and tests

The deterministic in-memory fixture constructs the complete target topology
and physical descriptor set without a GGUF payload. It covers positive renamed,
split, fused and metadata-derived bindings, topology and expert axes, plus 20
fail-closed mutations: identity/topology/count errors, missing/unknown/
ambiguous tensors, rank/type/expert-axis errors, missing/duplicate/invalid split
parts, invalid PLE fusion, missing PLE metadata, impossible layer IDs and a
non-zero payload-access boundary.

The opt-in real integration uses only `KQ_GGUF_PATH`. The native dump is
generated twice and required to be byte-identical. The research-only validator
compares all 1,294 initial-text rows and all 1,224 unique physical names against
the pinned Epic 1 `canonical-gguf-mapping.csv`. Production code never opens that
CSV or another research artifact.

The verified native dump SHA-256 is
`a214013005b4600ade0d4284169d1ab4f70a9329069aed73568720a927898d49`.
It is diagnostic evidence for this registered artifact, not a model-payload
hash.

## Payload and lifetime boundary

Registry construction reads only parsed metadata and immutable physical
descriptors. It opens no file view, exposes no tensor payload pointer and
requires Task 2.0's `payload_bytes_accessed` counter to remain zero.

`kq_model` owns its semantic arrays and coverage bookkeeping. Physical and
metadata bindings point into the caller-owned `kq_gguf`; close the model before
the GGUF. After successful construction the registry has no mutation API and
uses no global mutable mapping state.

No dequantization, tensor execution, tokenizer, PLE lookup, scheduler or cache
is introduced by Task 2.1.
