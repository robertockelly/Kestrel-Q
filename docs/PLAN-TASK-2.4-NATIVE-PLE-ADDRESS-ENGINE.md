# PLAN-TASK-2.4-NATIVE-PLE-ADDRESS-ENGINE.md

Status: **COMPLETE / PASS — 2026-08-29**

## Objective

Implement the native C17 Qwen3.8-Flash-Next PLE/N-gram address engine and its
minimal incremental state.

Task 2.4 consumes canonical token IDs and emits exact logical PLE address
intents. It must reproduce the independent Epic 1 PLE goldens.

It must NOT read PLE payload, perform PLE lookup/value math, dequantize, run
inference, access disk for PLE, implement cache/prefetch/scheduler policy, or
add model CUDA kernels.

## Baseline

Reviewed checkpoint:

`e5b4331b178d3ba3ef918fac0f96a5df215bbc1d`

Task 2.0-2.3: COMPLETE/PASS.
Task 2.4: NOT STARTED.
ADR 0008-0011: ACCEPTED.
`KQ-BACKLOG-BENCH-002`: DEFERRED.

Independent oracle:

`research/goldens/Qwen3.8-Flash-Next/canonical/ple-address-vectors.json`

SHA-256:

`495ef70f091e8d61caac99bb14ad8cea0fdb77940ec4dc6e8ce9811a144da3b6`

Coverage: 7 sequence cases + 4 decode steps.

## Mandatory characterization gate

Before production code, re-open the pinned canonical Qwen/reference evidence
and create `docs/PLE-ADDRESS-CONTRACT.md`.

Record exactly:

- PLE layer placement;
- n-gram orders;
- head/order relationship;
- logical table/member ordering;
- canonical member count;
- token-history requirements;
- exact bigram/trigram composition;
- primes/constants;
- XOR/multiply/modulo ordering;
- integer widths, casts and wrap semantics;
- table-row/modulo base;
- beginning-of-sequence behavior;
- prefill transition;
- incremental decode transition;
- state required for addresses versus state required only for later PLE value
  computation;
- metadata-derived configuration used by the supported model.

Do not implement from memory or from plausible N-gram hashing patterns.

## Architecture

```text
canonical token IDs
      |
      v
bounded PLE stream state
      |
      v
Qwen3.8 canonical address engine
      |
      +-- sequence position
      +-- n-gram order
      +-- head
      +-- logical PLE member/table
      +-- row/index
```

Exact address fields follow the characterized contract.

No physical GGUF offset belongs in semantic address identity.

## Production modules

Suggested:

```text
include/kq_ple.h
src/kq_ple.c
src/kq_ple_qwen38.c
tests/ple_test.c
tests/ple_integration.c
```

Keep Qwen-specific PLE arithmetic out of generic GGUF/tensor-view/tokenizer code.

## API / state

Provide:

- immutable PLE configuration;
- explicit resettable per-stream state;
- deterministic prefill API;
- deterministic one-token decode-step API;
- stable logical address descriptor;
- checked output-capacity/required-size behavior;
- no global mutable state.

State must be bounded.

Do not implement layer-value state unless it is proven necessary for address
calculation.

## Integer exactness

Address generation is discrete.

Use fixed-width integer types and explicitly define:

- signed/unsigned semantics;
- cast order;
- multiplication;
- XOR order;
- wrap behavior;
- modulo behavior.

Never rely on C signed-overflow undefined behavior.

Every result must be checked within canonical table-row bounds.

## PLE member mapping

Task 2.1 exposes 128 canonical logical PLE members.

Map canonical order/head/table semantics to those stable logical members only
after proving the ordering from the canonical evidence.

Do not derive physical file offsets in the PLE engine.

## Prefill

For a tokenized prompt, emit the exact canonical address sequence:

- exact address count per position;
- exact beginning/history semantics;
- exact order/head/member ordering;
- no I/O-oriented reordering.

## Incremental decode

Implement:

```text
previous bounded state + new token
    -> exact next addresses
    -> updated bounded state
```

Require incremental output to equal full canonical recomputation.

The 4 committed decode-step golden cases are mandatory.

## Tokenizer integration

Add at least one native path:

```text
UTF-8
-> Task 2.3 native tokenizer
-> Task 2.4 native PLE engine
-> exact independent PLE golden
```

Production code must not invoke Python.

## Model compatibility

Construct PLE config from the supported semantic/model registry.

Validate all evidence-backed model/configuration values exposed by the artifact.

Fail closed on incompatible:

- model identity;
- logical member count;
- n-gram order/head geometry;
- row/modulo geometry;
- required metadata-derived PLE semantics.

Canonical model constants may live in the Qwen3.8 adapter, but contradictory
artifact metadata must not be ignored.

## Synthetic positive tests

Cover:

- shortest valid histories;
- repeated tokens;
- alternating tokens;
- low and high canonical token IDs;
- history/order boundaries;
- head/member boundaries;
- first/last logical member;
- near-zero and near-maximum row index;
- reset/replay;
- prefill followed by decode;
- incremental versus recomputation.

## Fail-closed tests

Cover:

- invalid token ID;
- null/invalid arguments;
- arithmetic/count overflow;
- insufficient output capacity;
- corrupted/invalid state;
- incompatible configuration;
- wrong logical member count;
- wrong head/order configuration;
- invalid row/modulo size;
- missing metadata-derived PLE semantics;
- lifetime misuse where detectable.

## Golden/oracle validation

Require exact:

- 7/7 sequence cases;
- 4/4 decode steps;
- exact address values;
- exact ordering;
- exact state transitions where represented.

Re-run independent canonical golden validation/regeneration and preserve SHA:

`495ef70f091e8d61caac99bb14ad8cea0fdb77940ec4dc6e8ce9811a144da3b6`

Kestrel-Q is never its own oracle.

## Expanded differential evidence

Add independent canonical cases targeting:

- minimum history;
- repeated/alternating tokens;
- token IDs near 0 and near canonical upper bound;
- every relevant history/dilation boundary identified by the contract;
- bounded long prefill + decode;
- reset/replay equivalence.

Record deterministic evidence and SHA-256.

## Prefetch-readiness boundary

Expose deterministic address intents suitable for future prefetch.

Do NOT implement:

- asynchronous I/O;
- threads;
- disk reads;
- mmap prefetch;
- RAM/VRAM cache;
- eviction;
- storage batching;
- scheduler policy.

This task does not complete `KQ-BACKLOG-BENCH-002`.

## Payload boundary

Real-artifact validation must retain:

```text
PLE_payload_views_opened = 0
model_tensor_payload_bytes_touched = 0
```

Task 2.4 computes addresses only.

## Metrics

Record characterization only:

- config construction time;
- config-owned bytes;
- per-stream state bytes;
- addresses emitted/token;
- prefill address-generation time;
- decode-step address-generation time.

No storage throughput benchmark.

## Documentation

Create:

- `docs/PLE-ADDRESS-CONTRACT.md`
- `docs/NATIVE-PLE-ADDRESS-ENGINE.md`
- `docs/KQ-PLE-API.md`

Finalize:

- `docs/adr/0012-native-ple-address-engine.md`

Update ARCHITECTURE, TASKS, ROADMAP, Epic 2 plan/status, GOLDEN-VECTORS if
useful, Task 2.4 checklist and CHANGELOG.

## Acceptance

Task 2.4 PASS requires:

- characterized exact PLE contract;
- immutable config + bounded explicit state;
- exact prefill + decode;
- 7/7 sequence goldens exact;
- 4/4 decode goldens exact;
- golden hash unchanged;
- independent oracle PASS;
- expanded differential corpus exact;
- native tokenizer->PLE integration exact;
- incompatible configs fail closed;
- PLE payload views opened = 0;
- tensor payload touched = 0;
- no PLE lookup/I/O/cache/prefetch/scheduler;
- CPU/CUDA regressions PASS;
- ADR 0012 ACCEPTED;
- Task 2.5 NOT STARTED;
- KQ-BACKLOG-BENCH-002 DEFERRED;
- no tracked weights/secrets/local paths;
- `git diff --check` PASS.

Do not commit or push automatically.
