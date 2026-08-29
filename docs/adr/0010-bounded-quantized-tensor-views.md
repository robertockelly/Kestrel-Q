# ADR 0010 — Bounded quantized tensor views

Status: **Accepted — 2026-08-29**

## Decision

1. All production payload access uses bounded read-only tensor views.
2. Views are built from canonical semantic descriptors plus validated physical
   bindings.
3. Quantized block geometry is explicit and checked.
4. Split semantics remain segmented physical views with no implicit concat.
5. Expert members are bounded views derived from proven stacked geometry.
6. Fused PLE members are bounded logical member views.
7. Metadata-derived semantics have no payload view.
8. Transformed layouts preserve physical-vs-canonical ordering metadata.
9. Opening a view performs no implicit copy or dequantization.
10. Memory mapping is not cache/residency/prefetch policy.
11. Final scheduler decisions remain deferred.

Task 2.2 must validate this with synthetic payload tests and real-artifact
geometry checks before ADR acceptance.

## Evidence result

Task 2.2 satisfies the gate:

- checked geometry covers F32, BF16, Q5_1, Q8_0, Q4_K, Q5_K and IQ4_NL;
- deterministic payload-bearing fixtures read exact guarded bytes through
  whole, split, expert and PLE-member mappings, including an unaligned logical
  Win32 offset and close/reopen lifecycle;
- malformed range/geometry/ownership, split/fusion, expert-axis,
  metadata-derived and transformed-canonical requests fail explicitly;
- real-artifact views cover dense tensors, all seven types, layer-2 and boundary
  expert IDs, MoE/QSA splits and PLE members 0/64/127;
- the native geometry dump matches all 1,224 Task 1.3 physical descriptors and
  111,323,630,080 packed bytes; and
- real view construction performs zero payload dereferences in the test.

## Consequences

Future dequantization and execution code can consume a bounded physical span
without constructing GGUF offsets or conflating physical representation with
canonical layout. It must still explicitly interpret each quant type and
transformation under later correctness gates.

Mapping a view is not a cache, prefetch, placement or physical-I/O policy.
`KQ-BACKLOG-BENCH-002` therefore remains deferred and mandatory before final
PLE scheduler/residency decisions.
