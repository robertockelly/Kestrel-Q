# PLAN-TASK-2.3A-CANONICAL-TOKENIZER-OVERRIDE.md

Status: **COMPLETE / PASS — 2026-08-29**

## Purpose

Continue the blocked Task 2.3 using the governed canonical tokenizer override
adapter. This is a continuation of Task 2.3, not a new Epic task.

Baseline committed HEAD:
`202fb5190f53e27610cdd0bbcbe50005f86cf802`

The current dirty Task 2.3 characterization delta must be preserved and
continued. Do not reset it.

## Architecture

```text
verified GGUF tokenizer substrate
(vocab / IDs / ordered merges)
        |
        v
strict compatibility validation
        |
        v
Qwen3.8 canonical semantic adapter
        |
        v
immutable native kq_tokenizer
```

The generic GGUF parser remains generic.

## Production compatibility gate

Before constructing the tokenizer, validate:
- expected tokenizer substrate/model;
- exact canonical-used token strings and IDs;
- exact ordered merges/ranks;
- expected padded/unused GGUF IDs;
- required special-token substrate;
- no duplicate/conflicting entries.

Any mismatch fails closed.

## Canonical override

Implement exactly from `docs/TOKENIZER-CONTRACT.md`:
1. NFC normalization.
2. Exact pinned canonical pre-tokenization semantics.
3. Exact byte-level BPE behavior.
4. Canonical special-token classification/skip-special behavior.
5. Canonical BOS absence.
6. Canonical EOS identity and `add_eos_token=false`.
7. Official pinned chat-template semantics for the supported text-only subset.

## Unicode/NFC

Do not implement a partial NFC hack merely to pass current goldens.

Choose and document a deterministic governed method:
- pinned small Unicode library, or
- generated tables with pinned Unicode version, or
- an equally reproducible implementation.

No host-locale dependence. Update NOTICE/provenance if a third-party dependency
is introduced.

## Canonical pre-tokenization

The GGUF `qwen35` marks-inclusive pre-tokenization is not authoritative.

Reproduce the pinned canonical marks-excluding behavior exactly.

If using a Unicode-property dependency, pin version/license.
If implementing a dedicated scanner, prove it against an expanded independent
canonical corpus, not only the original goldens.

## Padded/unused IDs

Explicitly represent the proven distinction:
- 248044 base tokens;
- 33 canonical added tokens;
- canonical tokenizer length 248077;
- model vocab capacity 248320;
- 243 padded/unused GGUF IDs.

Unused/padded IDs must not silently become ordinary canonical tokenizer output.

## Chat

Use the pinned official canonical template semantics, not the embedded Unsloth
template.

Implement the already-characterized initial text-only subset:
- generation prompt OFF;
- generation prompt ON.

Unsupported multimodal/tools/developer/tool/reasoning branches remain explicit
fail-closed/deferred.

## Expanded differential corpus

Beyond the original goldens, add independent canonical cases targeting the
actual discovered divergences:
- decomposed Unicode/NFC;
- combining marks and whitespace+marks;
- IDs 248060-248065;
- BOS absence;
- EOS not auto-added;
- skip-special decode;
- safe canonical-vs-Unsloth template divergences.

Expected outputs must come from the pinned independent canonical oracle.

## Original golden gates

Preserve unchanged:
- prompt suite:
  `ffee472cac6e57f85df5f50104535b6e4e2d801c4ae6cac1f775840a29b7ed15`
- tokenizer vectors:
  `cbe1290d84a7a61113cf201edaf5034893eb1e70ba5a3c4bc9f4ea50bdcaf153`
- chat vectors:
  `72858a105a0f009b94eaea5dec24ae9f45ec0a400b9ce7325efc9341f0fbd1d6`
- manifest:
  `aa572756672f288957d429a60d7180650ffb2d603a792b21cd72def0a14ec0c4`

## ADR

Revise existing ADR 0011. Do not create ADR 0012 unless a genuinely broader
decision emerges.

ADR 0011 becomes ACCEPTED only after implementation passes, and must record:
- GGUF-only tokenizer source is insufficient;
- validated GGUF vocab/IDs/merges remain physical tokenizer substrate;
- canonical behavior comes from a compiled model-specific adapter;
- provenance pins;
- no GGUF mutation;
- no tokenizer sidecar;
- incompatibility fails closed.

## Completion

Task 2.3 becomes COMPLETE/PASS only when:
- native tokenizer exists;
- native text-only chat formatter exists;
- original goldens exact-match;
- expanded divergence corpus exact-match;
- real GGUF substrate compatibility passes;
- incompatible synthetic substrate fails closed;
- model tensor payload touched = 0;
- no production Python/Transformers/tokenizers dependency;
- CPU/CUDA regressions pass;
- ADR 0011 ACCEPTED;
- production API/docs exist;
- Task 2.4 remains NOT STARTED;
- KQ-BACKLOG-BENCH-002 remains DEFERRED.

Do not commit or push automatically.

## Completion evidence

- Exact GGUF substrate validation PASS on the registered artifact; tensor
  payload touched = 0.
- Unicode 9.0.0 NFC plus Unicode 16.0.0 properties reproduced through a
  generated age gate and utf8proc 2.10.0; provenance recorded.
- Original hashes unchanged; native comparison PASS for 10 prompts, 14
  segments and four chat vectors.
- Independent divergence corpus PASS; SHA-256
  `f383e213ccc4cde06b47a9855ca1eabb54d0b8911acbd43b2078be5fc546b463`.
- Near-match, invalid input/capacity and unsupported chat paths fail closed.
- Clean CPU 11/11 and CUDA 13/13 PASS; ADR 0011 ACCEPTED.
