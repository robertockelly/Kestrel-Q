# NATIVE-PLE-ADDRESS-GUIDE.md

## Core rule

Task 2.4 computes **where** PLE data will be needed; it does not fetch or
interpret that data.

```text
token IDs
  -> bounded PLE history state
  -> deterministic logical address intents
```

## Characterize before coding

Do not infer the arithmetic from common N-gram hashing patterns. Use the pinned
Qwen3.8 canonical implementation/evidence and write the exact contract first.

## Exact integer semantics

PLE addresses are discrete outputs. Use explicit fixed-width arithmetic and
never rely on signed C overflow.

## Logical identity only

Output identifies canonical logical table/member and row/index, plus
order/head/position fields required by the contract.

It does not contain:
- cache slots;
- file offsets as semantic identity;
- mapped pointers;
- scheduler priority.

## Prefetch-ready, not prefetching

Address intents should be consumable by a future prefetch/storage layer, but
Task 2.4 itself starts no I/O and makes no residency decision.

`KQ-BACKLOG-BENCH-002` remains required before final PLE disk-backed policy.

## State

Keep address state explicit and bounded. Do not accidentally implement PLE
value/layer math merely because the architecture has related recurrent state.
