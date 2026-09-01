# Native Sampling Task 3.0 Guide

Status: **PLANNING GUIDE — IMPLEMENTATION NOT STARTED**

## Purpose

Task 3.0 establishes a small correctness boundary:

```text
complete finite F32 logits
+ validated Qwen3.8 sampling profile
+ explicit caller-owned RNG state
-> one selected token ID
+ committed next RNG state
```

The model executor is deliberately absent. Task 3.1 performs integration only
after this boundary is independently validated.

## Source hierarchy

1. The pinned official `generation_config.json` defines the required model
   profile.
2. The pinned Transformers revision defines the canonical transform path to
   characterize.
3. A separately specified portable RNG defines Kestrel-Q reproducibility.
4. The exact GGUF and pinned llama.cpp remain model/logit evidence; their
   sampling defaults or RNG are not silently adopted.
5. Kestrel-Q output is comparison-only and never expected-data authority.

## Mandatory profile

The first supported profile must reproduce the evidence-backed official values
`temperature=1.0`, `top_k=20` and `top_p=0.95`, with sampling enabled and EOG
IDs 248046/248044. Task 3.0 must still characterize processor order, ties,
floating arithmetic, padded IDs and RNG; the JSON values alone are
insufficient.

## Comparison matrix

| Field | Contract |
| --- | --- |
| Active processors and order | EXACT_DISCRETE |
| Retained token IDs/order | EXACT_DISCRETE |
| RNG state/output/uniform mapping | EXACT_BITS / EXACT_DISCRETE |
| Selected ID for fixed logits/state | EXACT_DISCRETE |
| Transformed logits/probabilities | CALIBRATED_FLOAT per field |
| Repeated-draw population | Predeclared statistical test |

Token-for-token equality is not a substitute for a statistical test unless
the exact RNG, processor arithmetic, draw order and input logits are identical.

## Common traps

- Treating GGUF sampling metadata as canonical semantic authority.
- Reusing MoE top-k code without characterizing generation top-k semantics.
- Using CRT `rand()`, `std::random_device`, host locale or hidden global state.
- Sorting ties differently from the pinned processor.
- Applying top-p before top-k or softmax at the wrong stage.
- Silently masking padded IDs or silently exposing them as ordinary tokens.
- Consuming RNG state before a failure and not rolling it back.
- Calibrating and evaluating on the same corpus.
- Demanding exact random token sequences from a reference with a different RNG.
- Adding penalties, grammar, stop-string or generic plugin processors that the
  official profile does not require.

## Expected Task 3.0 outputs

- a pinned sampling contract;
- an accepted evidence-backed ADR 0022;
- a C17 sampler API and implementation;
- deterministic independent evidence and manifest;
- focused unit/oracle/fail-closed coverage;
- production/API documentation; and
- unchanged model executor and greedy behavior.

Task 3.0 is correctness work. It contains no model run, payload access,
performance target, CLI integration or future memory/CUDA policy.
