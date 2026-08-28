# Benchmark Policy

Kestrel-Q treats benchmarks as evidence, not marketing.

## Primary metrics

### Startup

- file open/map time
- metadata parse time
- warm vs cold startup
- time to first token

### Prefill

- tokens/s
- effective model bandwidth
- host-to-device traffic
- disk traffic
- peak VRAM
- peak RAM

### Decode

- tokens/s
- milliseconds/token
- expert cache hit ratio
- PCIe transfer/token
- disk read/token
- GPU utilization

### Context scaling

Record behavior across increasing context sizes.

### Quality

Performance optimization is invalid if model quality or deterministic reference behavior regresses beyond accepted tolerance.

## Reference hardware file

Exact machine details belong in `docs/HARDWARE-TARGET.md`.

## Result storage

Suggested future structure:

```text
bench/results/
  <date>-<commit>-<profile>.json
```

Raw results should remain machine-readable. Human summaries may be generated from them.

## Comparison rule

A benchmark comparison must hold constant all material variables except the variable under test.

If that is impossible, state the difference explicitly.

## Minimum run policy

The exact statistical method will be defined after the first runnable backend. Until then:

- record at least 3 measured runs for performance conclusions;
- separate cold and warm tests;
- do not discard outliers without explanation.
