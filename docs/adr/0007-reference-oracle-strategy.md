# ADR 0007 — Reference oracle and golden-vector strategy

Status: **Accepted — 2026-08-29**

## Context

Kestrel-Q targets a verified mixed-quantized GGUF while canonical model semantics are defined by the pinned official Qwen checkpoint.

A single oracle is insufficient because quantization introduces legitimate drift. KQ-01 also cannot safely execute the full canonical BF16 model and may not be suitable for a conventional full run of the 111 GB GGUF.

## Decision

Adopt a two-oracle strategy.

### Oracle C — canonical semantics

Use pinned official Qwen artifacts/research revisions plus
`huggingface/transformers@805a9e939fa8c1bff8d8ffdf041c051b71a914aa`
under Apache-2.0 for executable canonical text-model behavior.

Use it for tokenizer/template semantics, operator/state semantics, routing/index checkpoints, canonical hidden/logit vectors and greedy canonical generation.

### Oracle Q — quantized artifact

Use
`ggml-org/llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a`
under MIT for the exact verified GGUF SHA. This revision is accepted only for
`KQ-MODEL-ARTIFACT-001`; it does not create a generic GGUF support promise.

Use it for quantized final-output behavior, logits where exposed, greedy token sequences and validation that Kestrel-Q correctly consumes the artifact.

### Resource rule

Do not force reference execution onto KQ-01 when it would create unsafe memory pressure or uncontrolled paging.

Full-model vectors may be generated later on a capable reference environment
and remain mandatory before the corresponding future correctness gate is
passed. The canonical and GGUF plans are versioned machine-readable assets and
must be followed without silently substituting a different artifact, runtime,
dtype, prompt rendering, offload policy or speculation mode.

### Independence rule

Kestrel-Q must never use its own implementation as its only oracle.

### Comparison rule

Discrete/token/address outputs use exact equality. Floating outputs use documented evidence-calibrated tolerances. Do not widen tolerances to hide unexplained mismatch.

## Evidence and acceptance gate

Before acceptance:

- the canonical executable reference is pinned and executes the official
  tokenizer/template offline; its Qwen4-Exp model source is present and pinned;
- the quantized runtime is pinned, builds on KQ-01, registers `qwen4exp` and
  supports all seven tensor types in the exact artifact;
- safe tokenizer/chat/PLE goldens are generated and regenerate byte-identically;
- the prompt suite and exact/numeric schemas are final;
- full-model generation contracts are reproducible; and
- resource-deferred vectors are explicitly registered, not fabricated.

## Consequences

Positive: quantization drift is separated from implementation defects; future CPU/CUDA paths have explicit correctness gates; lack of oversized local hardware does not encourage fabricated goldens.

Negative: some reference capture requires external hardware; two oracle
identities add governance; tolerances require careful calibration.

- The generated safe vectors are permanent pre-implementation gates.
- Class-C full-model vectors remain
  `DEFERRED_CAPABLE_REFERENCE_ENV` because KQ-01 cannot safely load the
  359,999,963,128-byte canonical payload.
- Class-Q full-model vectors remain
  `DEFERRED_CAPABLE_REFERENCE_ENV` because the 111,334,654,400-byte GGUF would
  exceed KQ-01 RAM/VRAM and risk uncontrolled SATA-backed paging.
- Operator floating vectors remain `PLANNED_REFERENCE_REQUIRED`; future
  implementation work cannot replace that independent capture with
  self-generated expected values.
- Task 1.4 may pass with these explicit deferrals, but the corresponding later
  correctness claim may not.
