# ADR 0005 — Initial text-only inference scope

Status: **Accepted**

## Context

Qwen3.8-Flash-Next is released as a multimodal `Qwen4ExpForConditionalGeneration` model and includes a text backbone, a vision subsystem, multimodal integration and a Multi-Token Prediction (MTP) component.

Kestrel-Q's first target is a constrained Windows workstation with 10 GiB VRAM and 32 GiB system RAM. Implementing multimodal inference and speculative MTP before a correct ordinary text path would substantially enlarge the initial implementation and memory surface.

However, Kestrel-Q must not exclude components merely because they appear optional by name. The initial scope must preserve canonical ordinary autoregressive text-generation semantics.

## Decision

For the first Kestrel-Q model-inference milestone:

1. support text-token input and ordinary autoregressive next-token generation;
2. do not support image/video input initially;
3. do not execute or load the vision path for the initial text-only runtime;
4. do not implement MTP speculative acceleration initially;
5. preserve tokenizer/chat-template semantics needed for text operation;
6. fail explicitly on unsupported multimodal inputs rather than silently ignoring them.

## Evidence

Task 1.1 proved the two boundaries from pinned sources:

### Vision

In Hugging Face Transformers at pinned commit
`805a9e939fa8c1bff8d8ffdf041c051b71a914aa`,
`Qwen4ExpModel.forward` calls `get_image_features` only when `pixel_values` is
non-null and calls `get_video_features` only when `pixel_values_videos` is
non-null. Otherwise token embeddings pass unchanged to the language model.
`Qwen4ExpForCausalLM` is a text-only class whose load boundary explicitly ignores
`model.visual.*`. Vision/projector weights do not participate in ordinary text
logits. See [KQ-ARCH-WRAPPER-001], [KQ-ARCH-WRAPPER-002] and
[KQ-ARCH-SCOPE-001].

### MTP

The pinned Qwen technical report describes and evaluates MTP under four-step
speculative decoding and QSA index reuse. The pinned Transformers
`Qwen4ExpForCausalLM` computes logits directly from the main text model and LM
head and explicitly ignores `mtp.*` checkpoint keys. Ordinary next-token logits
therefore do not invoke MTP. See [KQ-ARCH-MTP-002] and [KQ-ARCH-MTP-004].

## Consequences

Positive:
- smaller first implementation surface;
- smaller required runtime footprint;
- easier correctness validation;
- focus on Kestrel-Q's initial text/coding-agent goal.

Negative:
- no image/video capability initially;
- no MTP speculative acceleration initially;
- future multimodal/MTP support requires additional architecture/state/test work.

## Non-consequence

This ADR does not redefine the canonical model. It only defines which canonical execution path Kestrel-Q implements first.

It also does not authorize silently accepting multimodal placeholder tokens,
discarding MTP weights from the canonical artifact inventory, or claiming that
the full released multimodal/speculative feature set is implemented.
