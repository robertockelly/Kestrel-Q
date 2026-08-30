# CPU-REFERENCE-NUMERIC-GUIDE.md

## Purpose

Task 2.5 creates the slow, simple numerical implementation against which later optimized CPU/CUDA kernels are checked.

Correctness and observability matter more than speed.

## Storage versus math

Keep two concepts separate:

```text
packed GGUF storage -> scalar decode to F32
F32/quantized inputs -> reference primitive math
```

A future optimized kernel may fuse them; Task 2.5 should not.

## No generic tolerance

Never choose a tolerance because it is conventional. Calibrate each floating primitive using an independent reference corpus, then validate on a separate holdout corpus.

## Real model reads

This is the first task allowed to touch model-weight bytes, but only through Task 2.2 bounded views and within a predeclared small sample budget.

Do not commit raw sampled model bytes.

## Reference path is intentionally scalar

No SIMD, no architecture intrinsics, no fast-math, no model CUDA kernel.

Later optimized kernels must be compared against this path.

## Do not drift into model operators

RMSNorm/softmax/etc. are low-level primitives. GDN recurrence, QSA sparse attention, MoE routing and PLE value execution are separate operator tasks with their own independent vectors.
