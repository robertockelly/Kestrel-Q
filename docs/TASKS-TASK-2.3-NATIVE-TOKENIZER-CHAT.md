# TASKS-TASK-2.3-NATIVE-TOKENIZER-CHAT.md

Status: **COMPLETE / PASS — 2026-08-29**

## Baseline
- [x] Record actual HEAD/worktree.
- [x] Confirm Task 2.2 COMPLETE/PASS.
- [x] Confirm Task 2.3 NOT STARTED before edits.
- [x] Keep `KQ-BACKLOG-BENCH-002` DEFERRED.

## Characterization gate
- [x] Inspect pinned canonical tokenizer artifacts.
- [x] Inspect GGUF tokenizer metadata.
- [x] Inspect pinned Transformers tokenizer implementation.
- [x] Record algorithm/model.
- [x] Record vocab/ID scheme.
- [x] Record pretokenization.
- [x] Record normalization.
- [x] Record byte/Unicode behavior.
- [x] Record decoder behavior.
- [x] Record special tokens/BOS/EOS.
- [x] Record chat-template semantics/options.
- [x] Create `docs/TOKENIZER-CONTRACT.md`.
- [x] Prove whether GGUF metadata is sufficient production source: **NO —
  FAIL-CLOSED**, then apply the separately governed canonical override.

## Tokenizer model
- [x] Immutable tokenizer object.
- [x] Explicit lifetime/ownership.
- [x] Vocabulary validation.
- [x] ID -> token lookup.
- [x] Required token/sequence -> ID lookup.
- [x] Merge/rank structures as required.
- [x] Duplicate/conflict detection.
- [x] Bounded initialization memory.
- [x] No research JSON runtime dependency.

## UTF-8 / preprocessing
- [x] Explicit UTF-8 contract.
- [x] Locale-independent behavior.
- [x] Exact whitespace behavior.
- [x] Exact punctuation/number behavior.
- [x] Exact Unicode behavior.
- [x] Invalid UTF-8 fails explicitly unless canonical evidence requires otherwise.
- [x] No unsafe implicit C-string reads.

## Special tokens
- [x] Exact inventory.
- [x] Exact IDs.
- [x] Exact ordinary-text policy.
- [x] Exact decode policy.
- [x] BOS policy.
- [x] EOS policy.
- [x] Chat-generation boundary tokens.
- [x] Inconsistent metadata fails closed.

## Encode
- [x] Deterministic size/capacity contract.
- [x] Empty input contract.
- [x] ASCII.
- [x] whitespace/newlines/tabs.
- [x] punctuation/numbers.
- [x] Italian accents.
- [x] Spanish accents.
- [x] non-Latin Unicode.
- [x] code.
- [x] repeated prefixes.
- [x] bounded longer prefill.
- [x] special-token text policy.
- [x] 10/10 prompt goldens exact.
- [x] 14/14 segment goldens exact.

## Decode
- [x] Exact golden decode/round-trip.
- [x] Multi-byte Unicode.
- [x] Whitespace-sensitive sequences.
- [x] Special-token decode options.
- [x] Invalid token ID fails.
- [x] Insufficient buffer fails cleanly.
- [x] Required-size query deterministic.

## Chat formatter
- [x] Separate module/API from tokenizer core.
- [x] Explicit message/role structures.
- [x] Text-only content contract.
- [x] Pinned template characterized.
- [x] No generic Jinja engine unless separately justified.
- [x] generation prompt false.
- [x] generation prompt true.
- [x] 2/2 golden chats exact.
- [x] exact rendered bytes.
- [x] exact token IDs.
- [x] unknown role fails.
- [x] unsupported content fails.
- [x] unsupported template options fail.

## Golden/oracle
- [x] Prompt-suite hash unchanged.
- [x] Tokenizer-vector hash unchanged.
- [x] Chat-vector hash unchanged.
- [x] Independent canonical validation/regeneration PASS.
- [x] Native output compared to committed goldens.
- [x] Kestrel-Q not used as oracle.

## Fail closed
- [x] Missing tokenizer metadata.
- [x] Malformed tokenizer metadata.
- [x] Vocab count mismatch.
- [x] Duplicate token ID.
- [x] Conflicting vocab entry.
- [x] Bad merge/rank.
- [x] Invalid special token.
- [x] Invalid UTF-8.
- [x] Invalid decode ID.
- [x] Count/allocation overflow.
- [x] Unsupported chat role/content/option.
- [x] Lifetime misuse where detectable.

## Payload boundary
- [x] Characterization reads GGUF metadata only.
- [x] Tensor payload views not required.
- [x] Real model tensor payload touched = 0.

## Metrics
- [x] Construction time.
- [x] Tokenizer-owned heap bytes.
- [x] Vocab/merge structure sizes.
- [x] Temporary encode allocation reviewed; not retained in the immutable
  tokenizer because it is input-dependent.
- [x] Chat temporary bytes are bounded by the rendered-size query and released
  on every path; no persistent allocation is retained.
- [x] No optimization before correctness.

## Build/regression
- [x] Clean CPU Release PASS.
- [x] CPU CTests PASS: 11/11.
- [x] Clean CUDA Release PASS.
- [x] CUDA CTests PASS: 13/13.
- [x] Task 2.0/2.1/2.2 remain PASS.
- [x] No new Kestrel-Q warnings; existing external NVCC C4211 only.
- [x] `git diff --check` PASS.

## Docs/governance
- [x] `docs/TOKENIZER-CONTRACT.md`.
- [x] `docs/NATIVE-TOKENIZER.md`.
- [x] `docs/KQ-TOKENIZER-API.md`.
- [x] `docs/NATIVE-CHAT-FORMATTING.md`.
- [x] ADR 0011 finalized.
- [x] ARCHITECTURE updated.
- [x] TASKS/ROADMAP/Epic 2 updated.
- [x] CHANGELOG updated.
- [x] Task 2.4 remains NOT STARTED.

## Safety
- [x] no inference/tensor math/dequantization.
- [x] no PLE execution/cache/prefetch/scheduler.
- [x] no model CUDA kernels.
- [x] no production Python/Transformers/tokenizers dependency.
- [x] no tracked model weights.
- [x] no secrets/local paths.
- [x] generated build trees untracked.

## Final report
Report actual HEAD, exact tokenizer characterization, production source-of-truth
decision, production files/API, vocab/special-token properties, tokenizer and
decode golden results, chat golden results, supported/deferred template
branches, fail-closed results, memory/init metrics, payload boundary, CPU/CUDA
regression, ADR 0011, findings/root causes/fixes, limitations, Task 2.3 status
and worktree status.

## Blocking evidence

- Entry HEAD: `202fb5190f53e27610cdd0bbcbe50005f86cf802`; only the four
  maintainer-provided governed Task 2.3 files were untracked at entry.
- Official base vocabulary: 248,044 IDs; official added tokens: 33; official
  merge ranks: 247,587. GGUF strings/IDs and merges match exactly.
- GGUF also carries 243 unique `[PAD<id>]` entries at 248077–248319, marked
  unused.
- Canonical active oracle: NFC plus the pinned `Qwen2Tokenizer` regex without
  `\p{M}` in its letter classes. GGUF provides no NFC metadata and selects
  `qwen35`, whose pinned reference expression includes `\p{M}`.
- Removing NFC changes decomposed `e` + U+0301 from canonical `[933]` to
  `[68, 52033]`.
- GGUF token types mark IDs 248060–248065 as control; canonical skip-special
  decode preserves those six FIM/repository tokens.
- Official chat-template SHA-256 is
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`;
  GGUF template SHA-256 is
  `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`.
- The GGUF template independently matches both committed chat cases with
  generation prompt off/on, but differs in other branches.
- Header/metadata inspection stopped before tensor payload; model tensor bytes
  touched: zero.
- Independent validation reports 8/8 manifest assets valid. Fresh offline
  regeneration produced eight files byte-identical to the committed golden
  tree; prompt, tokenizer, chat and manifest SHA-256 values remain unchanged.
- The original blocked delta's clean CPU Release configure/build/CTest passed
  7/7 and CUDA passed 9/9; that evidence was preserved before continuation.

## Continuation evidence

- The maintainer-selected 2.3A continuation introduced a governed compiled
  canonical override, not a silent substitution or sidecar.
- Exact construction digests cover the 248,320 token/ID stream, 247,587
  merge/rank stream and 248,320 token-type stream. Near-match mutations of
  counts, strings, merges, special semantics and padding fail closed.
- Unicode 9.0.0 NFC and Unicode 16.0.0 regex properties are reproduced through
  a generated assigned-range gate plus utf8proc 2.10.0, under the recorded
  MIT/Unicode-data provenance boundary.
- Native/oracle validation passed 10/10 prompts, 14/14 segments, 4/4 original
  chat vectors, 22 differential encode, five differential decode and two
  supported differential chat cases; three unsupported chat cases rejected.
- Differential evidence SHA-256:
  `f383e213ccc4cde06b47a9855ca1eabb54d0b8911acbd43b2078be5fc546b463`.
- Real construction reported 24,945,728 owned heap bytes (12,361,728 vocabulary
  lookup; 12,582,912 merge lookup) and 114,608,200 ns in one Release
  characterization run. These observations are not a performance baseline.
- Final clean CPU Release passes 11/11 and CUDA Release passes 13/13; the only
  warning is the already documented external NVCC C4211.
- ADR 0011 is accepted; Task 2.3 is COMPLETE/PASS. Task 2.4 remains NOT
  STARTED and `KQ-BACKLOG-BENCH-002` remains DEFERRED.
