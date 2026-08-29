# PLAN-TASK-2.3-NATIVE-TOKENIZER-CHAT.md

Status: **COMPLETE / PASS — 2026-08-29**

## Objective

Implement Kestrel-Q's native Qwen3.8-Flash-Next tokenizer and initial text-only
chat formatter in C17.

Task 2.3 must reproduce the already-governed Epic 1 tokenizer/chat golden
vectors exactly. It must not implement inference, tensor math, dequantization,
PLE execution, scheduler/cache/prefetch, multimodal input, or model CUDA
kernels.

Reviewed baseline:

`202fb5190f53e27610cdd0bbcbe50005f86cf802`

Status:
- Epic 1 COMPLETE/PASS
- Epic 2 IN PROGRESS
- Task 2.0/2.1/2.2 COMPLETE/PASS
- Task 2.3 COMPLETE/PASS after the governed 2.3A continuation
- ADR 0008/0009/0010 ACCEPTED
- `KQ-BACKLOG-BENCH-002` DEFERRED

## Golden evidence

Prompt suite:
`research/goldens/Qwen3.8-Flash-Next/prompt-suite.json`
SHA-256:
`ffee472cac6e57f85df5f50104535b6e4e2d801c4ae6cac1f775840a29b7ed15`

Tokenizer vectors:
`research/goldens/Qwen3.8-Flash-Next/canonical/tokenizer-vectors.json`
SHA-256:
`cbe1290d84a7a61113cf201edaf5034893eb1e70ba5a3c4bc9f4ea50bdcaf153`

Coverage:
- 10 synthetic prompts
- 14 tokenizer segments
- exact token IDs
- exact decode/round-trip evidence

Chat vectors:
`research/goldens/Qwen3.8-Flash-Next/canonical/chat-template-vectors.json`
SHA-256:
`72858a105a0f009b94eaea5dec24ae9f45ec0a400b9ce7325efc9341f0fbd1d6`

Coverage:
- 2 chat cases
- `add_generation_prompt=false/true`
- exact rendered UTF-8
- exact token IDs

Production runtime must not read these JSON files.

## Mandatory characterization gate

Before implementing tokenizer logic, inspect:
1. pinned canonical tokenizer assets used by Task 1.4;
2. tokenizer metadata in the verified GGUF;
3. pinned Transformers reference implementation;
4. pinned canonical chat template/config.

Create `docs/TOKENIZER-CONTRACT.md` recording, as applicable:
- exact tokenizer algorithm/model;
- vocabulary cardinality and ID assignment;
- merge/rank representation;
- normalization;
- pre-tokenization;
- byte-level transformation/fallback;
- Unicode and whitespace behavior;
- special-token inventory/IDs;
- BOS/EOS policy;
- unknown-token behavior;
- decoder pipeline;
- chat-template semantics/defaults/options;
- which required semantics exist in GGUF metadata.

Do not infer the algorithm from older Qwen releases or familiarity.

## Production source

Preferred production source is tokenizer data embedded in the verified GGUF,
but only after proving it is sufficient for exact canonical behavior.

If GGUF metadata is sufficient:
- construct the production tokenizer from it.

If insufficient:
- fail closed;
- document the exact missing contract;
- do not invent an undocumented sidecar;
- do not silently substitute another tokenizer.

No user-specific tokenizer path may become a runtime requirement.

## Architecture

```text
GGUF tokenizer metadata
        |
        v
native immutable kq_tokenizer
        |
        +-- encode UTF-8 -> token IDs
        +-- decode token IDs -> UTF-8

structured messages
        |
        v
native kq_chat_qwen38
        |
        v
exact rendered UTF-8
        |
        v
kq_tokenizer
```

Tokenizer core and chat formatting remain separate modules.

## Suggested modules

```text
include/kq_tokenizer.h
include/kq_chat.h

src/kq_tokenizer.c
src/kq_tokenizer_qwen38.c
src/kq_chat.c
src/kq_chat_qwen38.c

tests/tokenizer_test.c
tests/tokenizer_integration.c
tests/chat_test.c
```

Exact names may differ.

## API requirements

Tokenizer API must:
- use explicit byte lengths;
- not require NUL-terminated input;
- provide deterministic output-size/capacity behavior;
- never silently truncate;
- use checked arithmetic;
- expose explicit special-token options;
- be immutable after construction;
- use explicit ownership/lifetime;
- have no global mutable state.

## UTF-8 and preprocessing

Public text input is UTF-8 according to the characterized contract.

Requirements:
- locale-independent;
- Windows-language-setting-independent;
- exact canonical Unicode/whitespace/punctuation/number behavior;
- invalid UTF-8 fails explicitly unless canonical evidence proves a different
  public byte contract;
- no unsafe `strlen` on untrusted `(ptr,len)` input.

If the tokenizer uses byte-level internals, implement them exactly without
confusing them with the public UTF-8 contract.

If Unicode properties are required, pin their semantics rather than silently
using host locale behavior.

## Vocabulary / merges / ranks

Build all required production lookup structures from the verified production
source.

Validate:
- exact vocabulary cardinality;
- token IDs;
- duplicate/conflicting entries;
- merge/rank consistency if applicable;
- special-token consistency;
- bounded allocation sizes.

Production must not use golden JSON as vocabulary input.

## Special tokens

Characterize:
- exact strings/bytes;
- exact IDs;
- ordinary-text handling;
- decode behavior;
- BOS/EOS defaults;
- chat-generation boundary tokens.

No guessed BOS/EOS insertion.

## Encode/decode

Exact-match coverage must include:
- ASCII;
- whitespace/newlines/tabs;
- punctuation/numbers;
- Italian/Spanish accents;
- non-Latin Unicode;
- code;
- repeated prefixes;
- bounded longer prefill;
- token/merge boundary cases;
- special-token text policy.

All 10 prompt and 14 segment goldens must match exactly.

Decode must test:
- exact round trips;
- multi-byte Unicode;
- whitespace-sensitive sequences;
- special tokens according to options;
- invalid token IDs;
- insufficient output capacity.

Approximate/human-equivalent comparison is not acceptable where exact bytes
exist.

## Chat formatting

Do not implement a generic Jinja interpreter merely because the canonical
template is expressed in Jinja.

Implement the exact pinned Qwen3.8-Flash-Next template semantics required by
the initial text-only runtime.

At minimum:
- support roles/content exercised by Task 1.4 goldens;
- support generation prompt false/true exactly;
- exact rendered UTF-8 bytes;
- exact resulting token IDs.

Characterize every template branch/option first. Unsupported roles, content
types, multimodal structures, tools/tool-results, or options must fail
explicitly rather than being ignored.

If important canonical branches are outside Task 2.3 scope, record them as
explicit deferred capabilities.

## Fail-closed tests

Cover:
- missing/malformed tokenizer metadata;
- vocabulary mismatch;
- duplicate token IDs;
- conflicting vocabulary entries;
- bad merge/rank relation;
- invalid special-token mapping;
- invalid UTF-8;
- invalid decode token ID;
- insufficient buffer;
- count/allocation overflow;
- unsupported chat role/content/option;
- malformed chat structure;
- lifetime misuse where detectable.

Every internal bug discovered requires root cause plus regression coverage.

## Oracle discipline

Preserve ADR 0007:
- re-run independent canonical golden validation/regeneration as appropriate;
- committed golden hashes must remain unchanged;
- compare native output to goldens;
- never use Kestrel-Q output to define expected output.

## Payload boundary

Tokenizer/chat construction may read GGUF metadata only.

It must not open model tensor payload views.

Real integration must retain:
`model_tensor_payload_bytes_touched = 0`

## Metrics

Record characterization metrics:
- tokenizer construction time;
- tokenizer-owned heap bytes;
- vocab/merge structure bytes;
- encode temporary allocation if practical;
- chat temporary allocation if practical.

Do not optimize before exact correctness.

## Build/regression

- C17
- CPU-only tokenizer implementation
- clean CPU Release + CTest PASS
- clean CUDA Release + CTest PASS
- Task 2.0/2.1/2.2 regressions PASS
- no new Kestrel-Q `/W4` warnings
- existing external NVCC C4211 may remain documented
- `git diff --check` PASS

## Documentation

Create:
- `docs/TOKENIZER-CONTRACT.md`
- `docs/NATIVE-TOKENIZER.md`
- `docs/KQ-TOKENIZER-API.md`
- `docs/NATIVE-CHAT-FORMATTING.md`

Finalize:
- `docs/adr/0011-native-tokenizer-and-chat-formatting.md`

Update:
- `docs/ARCHITECTURE.md`
- `docs/TASKS.md`
- `docs/ROADMAP.md`
- Epic 2 plan/status
- Task 2.3 checklist
- `docs/GOLDEN-VECTORS.md` if useful
- `CHANGELOG.md`

## Acceptance

Task 2.3 PASS requires:
- tokenizer contract characterized before algorithm implementation;
- GGUF production-source sufficiency proven;
- 10/10 prompt goldens exact;
- 14/14 tokenizer segments exact;
- exact decode/round-trip;
- 2/2 chat cases exact;
- generation prompt OFF/ON exact;
- exact rendered bytes and IDs;
- independent golden validation PASS;
- golden hashes unchanged;
- fail-closed suite PASS;
- model tensor payload touched = 0;
- CPU/CUDA regressions PASS;
- ADR 0011 ACCEPTED;
- Task 2.4 NOT STARTED;
- `KQ-BACKLOG-BENCH-002` DEFERRED;
- no production Python/Transformers/tokenizers dependency;
- no tracked model weights/secrets/local paths;
- `git diff --check` PASS.

Do not commit or push automatically.

## Original characterization outcome

The mandatory pre-code gate found that the GGUF vocabulary and merge arrays are
complete, but the metadata is not sufficient to reproduce every canonical
semantic without a new override policy. Exact findings are recorded in
`docs/TOKENIZER-CONTRACT.md`.

Per this plan's production-source rule, the original run correctly stopped
before production algorithm or API work. The blocker remains part of the Task
2.3 evidence.

## Governed continuation and completion

The maintainer selected the explicit canonical override adapter in
`PLAN-TASK-2.3A-CANONICAL-TOKENIZER-OVERRIDE.md`. The completed implementation:

- validates exact GGUF token/ID, merge/rank and type streams before use;
- uses the oracle's distinct Unicode 9.0.0 NFC and Unicode 16.0.0 property
  contracts through a generated age gate plus utf8proc 2.10.0;
- implements the pinned marks-excluding byte-level BPE contract;
- enforces 248,044 base + 33 canonical added + 243 unused padded IDs;
- applies canonical BOS/EOS and skip-special semantics;
- separately implements the supported official text-only chat subset;
- matches 10/10 prompts, 14/14 segments, 4/4 original chat vectors and the
  expanded independent 22-encode/5-decode/2-chat corpus exactly; and
- preserves zero model tensor payload access and no production Python
  dependency.

ADR 0011 is accepted. Task 2.4 remains NOT STARTED and
`KQ-BACKLOG-BENCH-002` remains DEFERRED.
