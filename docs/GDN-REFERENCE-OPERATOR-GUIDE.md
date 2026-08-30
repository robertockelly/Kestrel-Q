# GDN-REFERENCE-OPERATOR-GUIDE.md

## First model operator

Task 2.6 is the first place where Kestrel-Q executes a model-specific operator
rather than generic storage/numeric infrastructure.

The native path must therefore be treated as a **reference implementation**,
not a performance implementation.

## Characterize before coding

Do not use generic Gated DeltaNet knowledge as the source of truth.

The pinned Qwen3.8 implementation defines:
- equations;
- state layout;
- projection structure;
- dtype transitions;
- prefill/decode ordering.

Write the contract first.

## Independent oracle first

Generate Class-C canonical vectors before accepting native outputs.

Use reduced deterministic configurations if the canonical module supports them.
Do not download the full BF16 checkpoint merely to validate operator semantics.

## State is output

GDN correctness is not just final hidden output.

Convolution and recurrent state transitions are first-class golden evidence.

Never compare C structs with raw `memcmp` when padding may exist; compare the
semantic state elements.

## Keep algorithm and artifact validation separate

Canonical synthetic vectors answer:
"Did we implement GDN correctly?"

Real GGUF structural/bounded checks answer:
"Did we bind the real artifact correctly?"

Do not collapse these into one oracle.

## Correctness before optimization

No SIMD, CUDA, fast math, thread pool or scheduling work in Task 2.6.

Future optimized GDN implementations must validate against this scalar
reference.
