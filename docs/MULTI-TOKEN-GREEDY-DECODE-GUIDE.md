# MULTI-TOKEN-GREEDY-DECODE-GUIDE.md

## Core invariant

Prefill the prompt once. Then advance by one selected token at a time:

```text
selected token N
-> decode-one
-> new persistent state
-> selected token N+1
```

Never replay the complete prompt to simulate generation.

## Preserve M1

For `KQ-PROMPT-001`, `max_new_tokens=1` must still produce token 271 and
decoded bytes `\n\n`.

Historical M1 evidence is immutable.

## Exact sequence

Task 2.13 passes only if the complete governed greedy continuation matches the
independent full-model oracle.

For the accepted case the exact sequence is `[271, 248068, 198, 760]`, with
decoded fragments `["\n\n", "<think>", "\n", "The"]`. The explicit context
capacity is 16, prompt prefill count is one and incremental decode count is
three.

## Per-step rollback

A failed later decode step must restore the state immediately before that step
without discarding earlier successful generated tokens.

## Still correctness-only

Record timings and logical payload counters, but do not add caching, prefetch,
SIMD or CUDA kernels in this task.

The successful logical packed-byte accounting is 40,208,768,960 bytes for
prefill and 6,334,414,400 bytes for each decode step, or 59,212,012,160 bytes
total. These are semantic access counters, not physical disk or page-residency
measurements.
