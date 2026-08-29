# QUANTIZED-TENSOR-VIEW-GUIDE.md

Status: **IMPLEMENTED / TASK 2.2 COMPLETE**

Semantic code requests a logical tensor/member; the view layer resolves the
validated physical storage.

Opening a view must not dequantize, copy a complete tensor, concatenate split
parts, execute PLE addressing or allocate cache residency.

Distinguish:

```text
logical requested range
quantized block-aligned physical range
Windows mapped range
```

Mapping padding is not tensor payload.

Expert members are only treated as contiguous when physical layout proves it.

PLE member views represent one logical member of the fused physical PLE tensor
and do not imply full-table residency.

Memory mapping does not prove residency or disk I/O. PLE storage behavior will
be measured later by `KQ-BACKLOG-BENCH-002`.

The implemented API is documented in `docs/KQ-TENSOR-VIEW-API.md`; verified
behavior and evidence are in `docs/QUANTIZED-TENSOR-VIEWS.md`.
