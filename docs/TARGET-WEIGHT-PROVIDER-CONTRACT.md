# Target weight-provider contract

Status: **CHARACTERIZED / VERIFIED FOR TASK 2.11**

The target provider is the semantic boundary between the immutable registry
and the verified Qwen3.8-Flash-Next GGUF. Callers request a canonical semantic,
binding part and optional routed-expert ID. They never provide a file offset or
reinterpret a physical tensor.

## Data path and ownership

```text
semantic descriptor
-> validated Task 2.2 bounded physical view
-> Task 2.5 block decoder or row-dot
-> existing scalar operator equation
```

The file, GGUF and model registry outlive the provider and every borrowed
descriptor/view. The provider is mutable only for per-run accounting and
bounded diagnostic traces; model semantics and physical bindings remain
immutable. It owns no tensor payload and never retains a mapped view after a
call.

Every semantic linear request supplies explicit rows, columns, capacities and
scratch. The provider checks those values against the selected physical member
before dereferencing it. It stages one F32 output vector and commits it only on
success. The largest small-vector materialization in the accepted target run
is the 40,960-element GDN convolution vector, or 163,840 bytes. A complete
target matrix is never materialized in F32.

## Physical forms

- ordinary one-to-one matrices use physical GGUF row-major rows, with GGUF
  dimension 0 as canonical linear input width;
- MoE `gate_up` remains two ordered physical parts, `GATE` then `UP`;
- QSA `index_qk` remains two ordered physical parts, `QUERY` then `KEY`;
- routed stacks require an explicit expert ID and open only that contiguous
  expert member; IDs outside `0..511` fail;
- fused PLE opens only the requested logical member and exactly five IQ4_NL
  blocks for its 160-value row;
- metadata-derived PLE address semantics have no payload and are never accepted
  as weight requests.

The verified Unsloth converter layout is inverted behind the provider:

- GDN value-head rows use the canonical `(key-group, repeat, feature)` order
  while GGUF stores `(repeat, key-group, feature)`;
- GDN output-projection columns receive the inverse activation permutation;
- convolution value channels and alpha/beta/gate/A/dt rows receive the same
  proven head permutation;
- stored GDN `ssm_a = -exp(A_log)` is converted back to canonical `A_log`;
- PLE norm weights stored with the converter's `+1` convention are restored to
  canonical zero-centered weights.

No fuzzy name matching or generic transform fallback exists. An unrecognized
relation, part, type, rank, shape, expert geometry or transformed role fails
closed.

## Access and accounting

The provider counts logical packed bytes and quantized blocks immediately
before each decode/row-dot, rejects a request that would exceed its caller-set
budget, and records unique semantics, linear/vector requests, selected-expert
requests and PLE rows. The accepted calibration run reports:

```text
provider object                         4,232 bytes
main representative execution    765,493,568 logical packed bytes
two rollback injections             7,332,736 logical packed bytes
aggregate correctness run         772,826,304 logical packed bytes
quantized blocks (main run)        32,652,832
unique semantics (main run)               109
selected-expert matrix requests            180
PLE row requests                             32
maximum F32 weight materialization      163,840 bytes
hard ceiling                         805,306,368 bytes (768 MiB)
```

The 180 expert matrix requests are exactly three projections for ten selected
experts, two tokens, across three representative layers. No unselected expert
member is opened. The 32 PLE requests are exactly 16 canonical intents for
each of the layer-1 prefill/decode tokens; no speculative row is opened.

Mapping a Win32 view is not a claim about physical residency or disk reads.
These counters are logical payload accounting, not I/O or performance data.
Final caching, prefetch and scheduling remain deferred, and
`KQ-BACKLOG-BENCH-002` remains required before that policy.

## Failure policy

Arguments, capacities, pointer ranges and checked arithmetic are validated
before use. Input/output/scratch overlap fails explicitly. Output is copied
from staging only after all rows succeed. The complete layer continues to use
Task 2.10's inactive state slot, so provider failure before the first request
or after the GR reads leaves output, active GDN/QSA/PLE state and layer position
unchanged.
