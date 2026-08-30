# Native PLE value reference

Status: **Task 2.9 COMPLETE / PASS**

The Task 2.9 scalar CPU reference consumes the exact logical address intents
produced by Task 2.4 and evaluates the characterized Qwen3.8 PLE value path.
It does not derive addresses and does not identify a row by a physical file
offset. The authoritative equations, shapes, ordering and state transition are
fixed in `PLE-VALUE-OPERATOR-CONTRACT.md`.

## Production boundary

`kq_ple_value_config_create` validates the supported model, layer 1 zero-based,
the compatible Task 2.4 address configuration, all 128 logical table members,
the fused IQ4_NL table binding and the six dense PLE semantics. The immutable
configuration borrows the model and address configuration; both must outlive
it. Construction opens no payload view and copies no model payload.

The scalar executable path is explicitly F32. It receives canonical F32 test
weights, uses Task 2.5 reference numerics and exposes checkpoint observations.
The target structural configuration retains the released BF16 semantic state
contract; it is not silently executed through the F32 entry point.

The lookup-provider interface is storage-neutral:

```text
logical member + member row -> exactly 160 F32 values
```

The real GGUF provider resolves the semantic member through Task 2.2, opens a
bounded member view, decodes five IQ4_NL blocks through Task 2.5 and closes the
view. It has an explicit logical-byte budget and counters. It has no cache,
prefetch, asynchronous I/O or residency policy.

## State and execution

The batch-1 value state is the last nine normalized gated-value positions,
`[10240,9]`. The released semantic state is BF16 (184,320 bytes); the scalar
F32 reference container is 368,640 bytes. It is independent of Task 2.4's
32-byte address state.

Prefill processes tokens in canonical order. Decode is the same transition for
one token. The output and state from prefix prefill plus decode match a single
prefill over the concatenated sequence. State reset is zero initialization.
Import/export compare semantic F32 elements, not C object padding.

Execution stages the complete value state in caller-provided scratch. State is
committed only after all intents, lookups and numeric operations succeed.
Lookup, intent, capacity, arithmetic, finite-domain and aliasing failures are
therefore transactional.

## Independent evidence

The expected calibration, holdout, state and Task 2.4 integration values are
generated from pinned Transformers revision
`805a9e939fa8c1bff8d8ffdf041c051b71a914aa`, before native comparison. The
native validator applies per-checkpoint calibration contracts unchanged to a
disjoint holdout. Lookup/embedding identity is exact bits; projected,
normalized, gated, convolution, output and state comparisons are separately
calibrated F32 contracts. Kestrel-Q never supplies expected values.

The real KQ-01 integration validates 128 members, 2,500,012 rows/member,
225,001,080 packed bytes/member, 28,800,138,240 packed table bytes and
28,835,240,960 total packed PLE bytes including the six dense tensors. Sixteen
deterministic rows touch 1,440 logical packed bytes across 80 IQ4_NL blocks,
well below the 8 MiB guard. No throughput, residency or page-cache claim is
made from those reads.

Release characterization on KQ-01 reported a 1,208-byte immutable target
configuration, 184,320 semantic state bytes, 368,640 F32 reference-state bytes
and 676,480 target scratch bytes. The small reduced synthetic case used 1,096
scratch bytes and observed 22,400 ns for a three-token prefill and 2,900 ns for
one decode step. These single-run timings characterize the reference path only;
they are neither benchmarks nor performance guarantees.

## Scope

This is a correctness implementation, not a storage policy or optimized
kernel. It does not implement a disk cache, prefetch, scheduling, complete
transformer-layer composition, full-model execution, LM head, sampling, SIMD
or CUDA model kernels. `KQ-BACKLOG-BENCH-002` remains deferred and required
before a final disk-backed PLE policy.
