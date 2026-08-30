# ADR 0017 — Native PLE value reference operator

Status: **Accepted**

## Decision

1. Keep PLE address generation and PLE value execution as separate operators.
2. PLE value execution consumes exact logical address intents.
3. Resolve table rows through a logical member/row provider abstraction.
4. Allow the scalar CPU reference provider to fetch bounded rows synchronously.
5. Keep value/history state explicit and separate from address state.
6. Define correctness against independent pinned Class-C vectors.
7. Validate real fused IQ4_NL/member geometry separately.
8. Defer disk cache/prefetch/residency policy.
9. Do not introduce complete transformer-layer composition.
10. Require future optimized/disk-backed PLE paths to validate against this
    scalar reference.

## Evidence gate

Require:
- exact characterized value contract;
- calibration/holdout/state vectors PASS;
- Task 2.4 address integration PASS;
- real target member/dense binding PASS;
- bounded real row plumbing PASS;
- CPU/CUDA regressions PASS.

## Evidence

Task 2.9 met the evidence gate with pinned independent Class-C calibration,
disjoint holdout, value-state and Task 2.4 address-integration vectors. The
native validator passed all four calibration cases, three holdout cases and
three exact address-integration cases. Prefix prefill plus decode matched
one-shot prefill, including the final nine-position value state.

The real target validated 128 logical members, one fused IQ4_NL table, six
dense semantic bindings and the exact table/member geometry. The bounded GGUF
provider touched 1,440 logical packed bytes in 80 blocks, within the 8 MiB
Task 2.9 guard. Clean CPU and CUDA regression suites passed.

## Consequences

Future PLE storage implementations must consume the same logical member/row
requests and validate against this scalar reference. Physical layout and
cache policy cannot become semantic identity. The current synchronous provider
is an integration mechanism, not a final cache or performance design.

The F32 scalar path is deliberately separate from the released BF16 semantic
state contract and from quantized storage. Complete layer composition and
optimized CPU/CUDA implementations require later governed work.
