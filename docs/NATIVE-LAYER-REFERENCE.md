# Native scalar transformer-layer reference

Status: **TASK 2.10 COMPLETE / PASS**

The C17 implementation in `src/kq_layer.c` and `src/kq_layer_qwen38.c`
composes the existing GDN, QSA, MoE, PLE-address and PLE-value APIs. It owns
only the canonical GR equations and layer order; it does not duplicate a
suboperator's internal math.

The independent generator imports the pinned
`Qwen4ExpTextDecoderLayer`/`Qwen4ExpTextGatedResidual`, uses deterministic
reduced F32 weights and creates ordinary-GDN, QSA and PLE-GDN calibration and
disjoint holdout outputs before native validation. Maximum complete-layer
absolute error is below `1.28e-6`; the frozen per-family acceptance ceiling is
`2e-6`. Existing suboperator contracts continue to gate their internal
checkpoints, states and exact-discrete selections.

Synthetic native tests cover all three families, four-token prefill,
prefill plus two decode steps, reset/replay and a forced MoE numeric failure after the
mixer state has advanced in staging. The failure leaves the active state and
position unchanged, and the next successful decode matches a failure-free
control stream. An implementation bug found during this test returned
from GDN state cloning before copying PLE address/value state; the regression
now proves PLE prefill/decode continuation.

The opt-in real-artifact integration validates 48/48 immutable configs:

```text
ordinary GDN  35
QSA           12
PLE-GDN        1  (zero-based layer 1)
GR branches    4
GR rank      320
config bytes   488 ordinary GDN / 496 QSA / 1,696 PLE-GDN
payload bytes touched 0
```

Target construction resolves all eight GR semantics per layer and the correct
GDN/QSA, MoE and optional PLE configs. It reads metadata/descriptors only and
uses no physical offsets. The complete target-width real layer is deliberately
not executed; doing so would require a bounded weight-loader/executor design
beyond this task.

Observed reduced Release characterization values are measurements, not
performance guarantees. For the deterministic fixtures, owned transactional
state was 2,608 bytes (GDN), 2,640 bytes (QSA at capacity 16), and 6,064 bytes
(PLE-GDN); caller scratch was respectively 4,000, 3,252 and 8,648 bytes for a
four-token prefill. GR workspace is included in those scratch totals.

One final clean-Release characterization run observed the following scalar
fixture timings; they are single-run diagnostics, not performance guarantees:

| Family | four-token prefill | one decode step |
|---|---:|---:|
| ordinary GDN | 60,100 ns | 11,700 ns |
| QSA | 34,200 ns | 8,900 ns |
| PLE-GDN | 69,700 ns | 21,000 ns |

The production library has no Python/Transformers dependency, opens no tensor
payload view, and adds no embedding, model loop, final norm, LM head, logits,
sampling, storage policy, scheduler, SIMD or CUDA model kernel.
