# QSA-REFERENCE-OPERATOR-GUIDE.md

## QSA correctness has two output classes

Floating outputs/states use calibrated numerical contracts.

Sparse selection is discrete and must be exact.

A numerically close attention result does not excuse a different selected block
or token set.

## Characterize sparse semantics before coding

Do not assume:
- block size;
- number of selected blocks;
- tail handling;
- local context rules;
- tie-breaking;
- gather ordering.

Pin all of them from the canonical Qwen3.8 implementation.

## State is first-class evidence

K/V cache and index-key/indexer state transitions are correctness outputs, not
implementation details.

## Keep allocation policy separate

Task 2.7 establishes semantic cache layout and bounded reference storage. It
does not design the final production KV-cache scheduler or residency system.

## Separate oracle roles

Synthetic Class-C vectors prove QSA algorithm semantics.

Real GGUF integration proves that target QSA semantic bindings and layouts are
correct.

Do not use the quantized artifact as a substitute for the independent canonical
operator oracle.

## Correctness before optimization

No SIMD, CUDA or multithreaded sparse selector in Task 2.7.
