# MOE-REFERENCE-OPERATOR-GUIDE.md

## Router selection is exact

Selected expert IDs and ordering are discrete correctness outputs.

A numerically close final result is not a pass if routing differs.

## Routed and shared paths stay separate

Validate independently:

```text
router -> top-k -> selected routed experts -> routed weighted sum

shared expert -> shared gate -> gated shared output
```

Combine only according to the canonical final equation.

## Access only selected experts

The real target path uses Task 2.2 expert-member views and Task 2.5 scalar
numerics.

Do not dequantize or map all 512 experts for one token.

## No cache policy yet

Task 2.8 proves semantic correctness and bounded expert access.

Expert cache/residency/prefetch/scheduling remains later work.

## Separate oracle roles

Synthetic Class-C vectors prove the MoE algorithm.

Real GGUF integration proves Kestrel-Q resolves the quantized expert stacks and
shared weights correctly.
