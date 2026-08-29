# PLAN-TASK-2.2-QUANTIZED-TENSOR-VIEWS.md

Status: **COMPLETE / PASS — 2026-08-29**

## Objective

Implement bounded, read-only quantized tensor views on top of Task 2.0's GGUF
container layer and Task 2.1's canonical semantic registry.

Task 2.2 is the first production task allowed to create logical views over
model payload regions.

It must support:

- complete physical tensor views;
- split semantic bindings;
- routed-expert member views;
- fused PLE member views;
- exact quantized block geometry.

It must NOT implement:

- dequantization;
- inference;
- tokenizer;
- PLE address execution;
- cache/prefetch policy;
- scheduler;
- model CUDA kernels.

Reviewed baseline:

`dfc9fea7c9320b691a444ff47d7b2cb692e558ec`

Task 2.0 and 2.1 are COMPLETE/PASS.
ADR 0008/0009 are ACCEPTED.
`KQ-BACKLOG-BENCH-002` remains DEFERRED.

## Architecture

```text
semantic descriptor
    -> physical binding
        -> bounded tensor view
            -> quant block geometry
```

Future code must not manually add GGUF offsets.

## Quantized storage geometry

Support and validate:

- F32
- BF16
- Q5_1
- Q8_0
- Q4_K
- Q5_K
- IQ4_NL

For every type derive:

- elements per block;
- packed bytes per block;
- checked block count;
- checked packed size;
- logical-range to block-range mapping.

No dequantization.

## Tensor view

Define an immutable read-only view containing enough information to distinguish:

- canonical semantic/logical shape;
- physical GGUF shape/layout;
- physical type;
- logical element range;
- physical quantized block span;
- logical byte span;
- Windows mapped view.

No implicit copy or dequantization.

## Whole tensor

A direct physical binding may open a bounded view of exactly its payload span,
subject only to Windows allocation-granularity mapping mechanics.

The public logical span must never include mapping padding.

## Split semantics

The 48 MoE `gate_up` and 12 QSA `index_qk` canonical split relations remain
segmented views.

Do not concatenate them into temporary buffers.

Each physical part has deterministic role/order.

## Routed experts

Support:

```text
expert-stack semantic + expert_id 0..511 -> bounded member view
```

Requirements:

- validate expert axis from physical geometry;
- prove member contiguity from layout;
- checked member stride/span;
- map only the selected expert member when representable;
- reject out-of-range expert IDs;
- fail closed or expose explicit segmented form if member storage is not
  contiguous.

Do not create 512×48 permanent expert tensor objects.

## Fused PLE

Support all 128 logical PLE members over the one fused physical tensor.

Requirements:

- validate fused geometry;
- first/middle/last member;
- checked physical member span;
- no full 26.855 GiB PLE mapping just to access one member;
- no PLE hash/address execution;
- no cache/prefetch policy.

## Metadata-derived semantics

The three PLE address semantics have no payload.

A payload-view request must return an explicit no-payload error.

Do not create fake zero-length payloads.

## Transformed layout

Preserve the distinction between canonical logical ordering and physical GGUF
ordering.

Do not present transformed physical bytes as canonical-contiguous memory.

Carry explicit layout/transform metadata where necessary.

## Block addressability

Provide checked helpers for:

```text
logical element index -> containing quant block
logical element range -> minimal block-aligned physical byte span
```

Use checked 64-bit arithmetic and enforce tensor bounds.

## Mapping versus payload touch

Distinguish:

- logical payload exposed;
- virtual mapped logical bytes;
- payload bytes actually dereferenced by a test.

For the real GGUF integration test, prefer:

`payload_bytes_touched_by_test = 0`

Synthetic fixtures may safely dereference their tiny payloads to prove
boundary behavior.

Do not equate MapViewOfFile with physical disk reads.

## Synthetic fixtures

Use tiny generated payload-bearing GGUF fixtures with known bytes.

Test:

- whole tensor exact bytes;
- guard bytes before/after;
- non-allocation-granularity offsets;
- block-range helper;
- split part views;
- expert member view;
- PLE member view;
- lifecycle close/reopen.

## Fail-closed coverage

Include:

- out-of-range request;
- arithmetic overflow;
- span past EOF;
- invalid block geometry;
- unrepresentable logical range;
- expert ID out of range;
- invalid expert axis;
- non-contiguous expert member;
- PLE member out of range;
- invalid PLE fusion;
- missing/duplicate split part;
- metadata-derived payload request;
- transformed layout requested as canonical-contiguous;
- invalid ownership/lifetime where detectable.

## Real artifact integration

With `KQ_GGUF_PATH`, validate bounded view construction for representative:

- always-needed dense tensor;
- Q8_0;
- Q4_K;
- IQ4_NL;
- Q5_1/Q5_K;
- routed-expert members from representative layers, including layer 2;
- split MoE gate_up;
- split QSA index_qk;
- first/middle/last PLE member.

For each derive all offsets/spans from production registry/parser. No hard-coded
physical offsets.

Prefer zero payload dereference on the real model.

## Research oracle

A test/research-only validator may compare native view geometry to Task 1.3
evidence.

Production code must not read research CSV/JSON.

## Inspector

Optional development diagnostics:

```text
--view <semantic-id>
--expert-view <semantic-id> <expert-id>
--ple-view <member-id>
```

Print geometry only. Do not dump payload.

## Documentation

Create:

- `docs/QUANTIZED-TENSOR-VIEWS.md`
- `docs/KQ-TENSOR-VIEW-API.md`

Finalize:

- `docs/adr/0010-bounded-quantized-tensor-views.md`

Update ARCHITECTURE, semantic API, TASKS, ROADMAP, Epic 2 status and
`CHANGELOG.md`.

## Acceptance

Task 2.2 PASS requires:

- clean CPU/CUDA regressions;
- all seven target quant types;
- whole/split/expert/PLE views;
- metadata-derived explicit failure;
- transformed-layout safety;
- synthetic payload boundary tests;
- real-artifact view geometry validation;
- Task 1.3 oracle comparison;
- real-artifact payload bytes touched = 0;
- no dequantization/inference/tokenizer/PLE/scheduler;
- ADR 0010 ACCEPTED;
- `KQ-BACKLOG-BENCH-002` still DEFERRED;
- no tracked model weights/secrets/local paths;
- `git diff --check` PASS.

Do not commit or push automatically.
