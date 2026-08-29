# Qwen3.8-Flash-Next reference prompt suite

Status: **GENERATED / VERIFIED**

Canonical asset:

`research/goldens/Qwen3.8-Flash-Next/prompt-suite.json`

Exact file SHA-256:

`ffee472cac6e57f85df5f50104535b6e4e2d801c4ae6cac1f775840a29b7ed15`

The suite contains only original synthetic text. It has no model excerpts,
multimodal inputs, tool calls or quality-evaluation prompts.

| ID | Input form | Coverage |
|---|---|---|
| `KQ-PROMPT-001` | raw text | short ASCII |
| `KQ-PROMPT-002` | raw text | leading/trailing spaces, tab, LF and CRLF |
| `KQ-PROMPT-003` | raw text | punctuation and numbers |
| `KQ-PROMPT-004` | raw text | Italian/Spanish accented UTF-8 |
| `KQ-PROMPT-005` | raw text | Greek, Japanese, Arabic and Devanagari Unicode |
| `KQ-PROMPT-006` | raw text | original C source fragment |
| `KQ-PROMPT-007` | messages | system + user chat |
| `KQ-PROMPT-008` | messages | system/user/assistant/user multi-turn chat |
| `KQ-PROMPT-009` | raw text | repeated prefix and token-boundary punctuation |
| `KQ-PROMPT-010` | raw text | deterministic bounded 64-line longer prefill |

Raw prompts are identified by the SHA-256 of their exact UTF-8 bytes. Structured
chat prompts are identified by UTF-8 canonical JSON with sorted object keys and
compact separators. The JSON asset records the identity serialization, byte
count and hash for every prompt.

The suite is shared input for both correctness classes. A shared prompt does
not imply that canonical BF16 and exact-GGUF output must be numerically equal.
Tokenizer and chat-template vectors remain Class C because they use the pinned
official tokenizer/template semantics. Future Class-Q runs consume the same
rendered prompt bytes under their separately pinned runtime.
