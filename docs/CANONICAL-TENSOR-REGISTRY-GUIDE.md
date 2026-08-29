# CANONICAL-TENSOR-REGISTRY-GUIDE.md

Status: **IMPLEMENTED / TASK 2.1 COMPLETE**

## Core rule

GGUF names are physical-container identifiers.

Kestrel-Q semantic keys are model-runtime identifiers.

Future execution code depends on semantic keys, not GGUF naming.

## Count model

Epic 1 established:

```text
1658 canonical total
-333 vision
-31 MTP
=1294 initial-text semantics
```

The verified GGUF contains 1224 physical tensors because of proven
fusion/split/metadata transformations.

## Binding cardinality

A semantic descriptor may bind to:

- one physical tensor;
- multiple physical parts;
- one member of a fused physical tensor;
- metadata rather than payload.

One physical tensor may participate in multiple semantic descriptors when a
proven fused representation requires it.

## No fuzzy mapping

Mapping is deterministic and fail-closed.

A physical tensor not matched by a reviewed rule is an error.

## Stable semantics

Semantic keys should remain stable if a future Kestrel-Q-native container uses
different physical names, order or layout.

## Placement hints

Placement hints communicate architectural hypotheses only.

They are not allocator or scheduler commands.

`PLE_DISK_BACKED_CANDIDATE` remains unvalidated until
`KQ-BACKLOG-BENCH-002` is completed.
