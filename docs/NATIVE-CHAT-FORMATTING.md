# Native Qwen3.8 chat formatting

Status: **IMPLEMENTED / VERIFIED — Task 2.3 initial text-only subset**

`include/kq_chat.h` defines a separate model-specific formatter. Structured
messages are rendered to exact canonical UTF-8 and may then be passed to the
native tokenizer. Kestrel-Q does not embed Jinja and does not treat the
Unsloth-modified GGUF template as semantic authority.

## Supported subset

- optional first `system` message;
- alternating `user` and `assistant` text messages;
- final `user` message;
- Unicode-whitespace trimming equivalent to the pinned template's text path;
- `add_generation_prompt=false`; and
- `add_generation_prompt=true` with the exact disabled-thinking suffix
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`.

Each message is rendered as
`<|im_start|><role>\n<trimmed-content><|im_end|>\n`. Rendering uses explicit
pointer/length inputs and deterministic required-size/capacity behavior with no
silent truncation.

## Explicitly unsupported

Developer/tool roles, tool results/calls, multimodal or iterable content,
reasoning modes, preserved reasoning, tool declarations, image/video options
and nonzero reserved options return stable errors. A system message after the
first position, non-alternating roles, invalid UTF-8 or a conversation outside
the initial final-user contract also fails closed.

These are deferred capabilities, not ignored template branches. Extending the
subset requires new independent canonical vectors and documentation.

## Oracle and safety

Both Task 1.4 chat cases match exact rendered bytes and token IDs with
generation prompt off/on (4 vectors). The independent Task 2.3 differential
adds two supported chat cases and three explicit canonical rejection cases.
Expected data is generated only by the pinned canonical Transformers oracle;
native output never becomes expected output.

Formatting allocates only the bounded rendered prompt and does not read GGUF
metadata or tensor payload. Tokenization preserves the zero-payload boundary.
