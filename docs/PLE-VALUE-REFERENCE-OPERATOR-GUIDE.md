# PLE-VALUE-REFERENCE-OPERATOR-GUIDE.md

## Addressing and values remain separate

Task 2.4 determines **which logical PLE member and row** are needed.

Task 2.9 determines **what canonical value is computed from those rows and the
value-side state**.

Do not merge those contracts.

## Lookup-provider boundary

The value operator requests logical member/row data through a narrow provider.

This allows the same equations to use:
- synthetic tables;
- bounded real GGUF rows;
- a future disk/cache/prefetch subsystem.

Do not bake storage policy into the equations.

## Value state is first-class evidence

The value-side history/dilation state is separate from Task 2.4's address
history and must be independently validated through prefill/decode transitions.

## No full table residency

Real tests map/decode only required rows. Never map the complete fused PLE
tensor merely to run the reference operator.

## Correctness before storage optimization

KQ-BACKLOG-BENCH-002 remains deferred. Task 2.9 proves value semantics and
bounded row plumbing, not storage performance.
