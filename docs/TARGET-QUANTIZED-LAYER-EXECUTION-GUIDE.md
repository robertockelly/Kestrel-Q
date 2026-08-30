# TARGET-QUANTIZED-LAYER-EXECUTION-GUIDE.md

## Why this task exists

The scalar layer mathematics are proven, and the quantized GGUF can be decoded
row-by-row, but those capabilities have not yet been joined for a complete real
target layer.

Task 2.11 is that bridge.

## One algorithm, two providers

Prefer:

```text
synthetic F32 provider ─┐
                        ├─> same reference operators/layer equations
real quantized provider ┘
```

Do not fork the model equations into a separate real-model implementation.

## Never materialize full matrices

A target linear operation consumes bounded rows/chunks and produces
activations. The weight matrix itself must not become a full F32 allocation.

## Independent target correctness

Use independently decoded quantized weights plus a separate implementation of
the frozen canonical equations. Do not compare the target layer only against
another Kestrel-Q path.

## Selected experts only

A real MoE token must not touch all 512 routed experts. Instrument this
invariant.

## Exact PLE rows only

The scalar reference requests exactly the canonical PLE intents with no extra
prefetch rows.

## After this task

If Task 2.11 passes, remaining First Correct Native Token work should be
model-level plumbing: embedding, 48-layer loop/state orchestration, final norm,
LM head/logits, argmax and decode.
