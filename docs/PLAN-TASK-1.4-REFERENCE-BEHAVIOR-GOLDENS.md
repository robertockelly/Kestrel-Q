# PLAN-TASK-1.4-REFERENCE-BEHAVIOR-GOLDENS.md

Status: **COMPLETE / PASS — 2026-08-29**

## Objective

Define and validate Kestrel-Q's reference-oracle strategy and golden-vector contract before any model loader or inference implementation begins.

Task 1.4 is the final characterization task of Epic 1. It must establish two distinct correctness oracles:

- **Class C — canonical semantics** for the pinned official Qwen model.
- **Class Q — quantized artifact semantics** for the verified local Unsloth `UD-Q4_K_XL` GGUF.

The task must generate all safe, weight-independent goldens that KQ-01 can produce now, and define exact executable contracts for full-model vectors that require more capable hardware.

## Baseline

- Kestrel-Q checkpoint: `f5e5a1052db89e56aa81439efbce08c63e831bf2`
- Tasks 1.0–1.3: COMPLETE/PASS
- ADR 0004/0005/0006: ACCEPTED
- Task 1.4: NOT STARTED
- Canonical Qwen revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Canonical research revision: `69885871a64393807d988b27b1b5e380e8f28526`
- Previously pinned Transformers reference: `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`, Apache-2.0
- Verified GGUF SHA-256: `8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2`
- Unsloth revision: `c8b5954a88c2775c546b92593eda40ea041d3176`

## Resource guard

KQ-01 has 32 GiB RAM, 10 GiB VRAM and SATA storage. Therefore Task 1.4 must not:

- download or execute the ~360 GB canonical BF16 checkpoint on KQ-01;
- deliberately trigger Windows pagefile thrashing;
- run the 111 GB GGUF only to satisfy a checklist unless a bounded feasibility check proves it safe;
- fabricate full-model outputs.

Weight-dependent full-model vectors may be recorded as `DEFERRED_CAPABLE_REFERENCE_ENV` when generation requires a stronger reference machine. This does not block Task 1.4 PASS if the oracle, schema, corpus, generation contract and future acceptance gate are complete.

## Oracle hierarchy

### Oracle C — canonical semantics

Use, in order:

1. pinned official Qwen artifacts/config/tokenizer;
2. pinned official Qwen architecture/research material;
3. a pinned independent executable reference implementation when exact execution behavior is needed.

Verify whether the already pinned Transformers Qwen4-Exp revision is suitable for the required checkpoints. If another revision is needed, pin it explicitly and document why.

### Oracle Q — exact GGUF artifact

Select and pin an independent runtime that supports:

- `qwen4exp`;
- all GGUF tensor types used by `KQ-MODEL-ARTIFACT-001`;
- text-only ordinary autoregressive generation;
- MTP/speculation disabled;
- deterministic greedy operation.

A llama.cpp revision is a candidate, not an assumption. Verify support and record revision/license/build settings.

Kestrel-Q itself must never be its only oracle.

## Required documents

Create:

```text
docs/REFERENCE-ORACLE-STRATEGY.md
docs/GOLDEN-VECTORS.md
docs/REFERENCE-PROMPT-SUITE.md
docs/adr/0007-reference-oracle-strategy.md
```

## Required machine-readable assets

Create:

```text
research/goldens/Qwen3.8-Flash-Next/
  manifest.json
  prompt-suite.json
  canonical/
    tokenizer-vectors.json
    chat-template-vectors.json
    ple-address-vectors.json
    operator-vector-plan.json
    full-model-vector-plan.json
  quantized/
    gguf-vector-plan.json
```

Allowed statuses:

- `GENERATED_VERIFIED`
- `PLANNED_REFERENCE_REQUIRED`
- `DEFERRED_CAPABLE_REFERENCE_ENV`
- `NOT_APPLICABLE`

No vector may be labeled generated without an actual artifact and SHA-256.

## Reference prompt suite

Use deterministic original/synthetic text only. Include at minimum:

1. short ASCII;
2. whitespace edge case;
3. punctuation/numbers;
4. Italian or Spanish accented UTF-8;
5. non-Latin Unicode;
6. short C source fragment;
7. system + user chat;
8. multi-turn chat;
9. repeated-prefix/token-boundary case;
10. bounded longer prefill case.

No copyrighted excerpts, multimodal inputs, tool-use requirements or quality benchmarking. Give stable IDs `KQ-PROMPT-001...` and record exact UTF-8 SHA-256.

## Safe goldens required now

### Tokenizer

Generate exact token IDs, token counts, special-token behavior and round-trip/decode checks where meaningful. Include ordinary text, Unicode, whitespace, code punctuation and chat/special tokens.

Tokenizer IDs are exact-match goldens.

### Chat template

Using the pinned official `chat_template.jinja`, record:

- structured messages;
- exact rendered string;
- rendered UTF-8 SHA-256;
- exact token IDs;
- `add_generation_prompt` behavior;
- special-token placement.

Rendered bytes and token IDs are exact-match goldens.

### PLE/N-gram addresses

Generate exact integer vectors for sequences of length 1, 2, 3 and >3, repeated tokens and large valid vocabulary IDs. Record bigram/trigram addresses, partition/head data, rolling history and decode-step addressing.

No model weights are required.

## Future operator/checkpoint contract

Define stable fields for future comparison.

### Gated Residual
- read gates
- branch read result
- write coefficients
- resulting four branches

### GDN
- selected projection outputs
- recurrent state before/after
- convolution state before/after
- layer output

### QSA
- indexer scores/selection representation
- selected block IDs/positions
- K/V state growth
- attention output

### MoE
- router logits/probabilities
- top-10 expert IDs
- normalized expert weights
- shared-expert gate
- combined output

### Final
- selected hidden-state checkpoints
- selected/full logits
- top-k token IDs/logits
- greedy next-token ID
- short greedy generation sequence

Where a pinned independent implementation exposes a small synthetic operator without loading full model weights, generate the vector now. Otherwise mark it `PLANNED_REFERENCE_REQUIRED`. Do not implement an operator yourself and call its own output an independent golden.

## Numeric comparison policy

Use exact equality for discrete values such as token IDs, rendered UTF-8, PLE addresses, deterministic selected IDs and greedy token IDs.

Floating values use a tolerance contract containing:

- dtype;
- exact shape;
- `atol`;
- `rtol`;
- NaN/Inf policy;
- tolerance calibration status.

If no independent full-model runs exist yet, use `TO_BE_CALIBRATED_FROM_REFERENCE_RUNS`. Do not select loose tolerances to make future tests pass.

## Canonical full-model vector plan

Define an exact reproducible capture plan for a capable machine:

- canonical model/revision;
- pinned reference implementation/revision;
- canonical released dtype;
- batch size 1;
- text-only;
- no vision;
- no MTP/speculation;
- `do_sample=false` / greedy;
- fixed prompt IDs;
- fixed max-new-token counts;
- exact dependency versions and deterministic flags;
- checkpoint hooks.

Plan at least 3 prompt cases and include checkpoints at early GDN, first QSA, PLE injection region, middle layer, late layer, selected routers/states, final logits and short greedy output.

If not generated now, status is `DEFERRED_CAPABLE_REFERENCE_ENV`.

## Quantized GGUF full-model plan

Pin the independent reference runtime and define deterministic invocation for the exact GGUF SHA. Record:

- text-only/no-speculation;
- batch 1;
- greedy settings;
- fixed context and prompts;
- exact build/runtime revision;
- offload/mmap configuration;
- device/backend.

Perform a bounded resource-feasibility check before any local full-model run. If unsafe on KQ-01, mark `DEFERRED_CAPABLE_REFERENCE_ENV` rather than forcing it.

## Golden manifest

The manifest must record:

- schema version;
- canonical model identity/revision;
- GGUF identity/hash;
- oracle revisions/licenses;
- prompt-suite hash;
- each golden asset ID/class/status/hash/comparison mode/dependencies/generation tool or command.

## Research tooling

Research-only tooling under `tools/` is allowed, e.g.:

```text
tools/generate-tokenizer-goldens.py
tools/generate-ple-goldens.py
tools/validate-golden-manifest.py
```

Requirements:

- deterministic outputs;
- pinned inputs;
- no model-weight download;
- no runtime dependency;
- no secrets/local user paths in committed output;
- documented in `tools/README.md`;
- recorded in `CHANGELOG.md`.

## ADR 0007

Finalize `docs/adr/0007-reference-oracle-strategy.md` covering:

- canonical oracle;
- quantized GGUF oracle;
- exact-vs-tolerance policy;
- capable-reference-environment deferral;
- prohibition on self-oracle-only validation;
- requirement that future implementation stages satisfy applicable golden gates.

## Epic 1 exit gate

Epic 1 may become COMPLETE/PASS when:

- Tasks 1.0–1.4 are COMPLETE/PASS;
- ADR 0005/0006/0007 outcomes are explicit;
- the oracle strategy and schemas are pinned;
- safe tokenizer/chat/PLE goldens exist and regenerate deterministically;
- missing full-model vectors are explicit future gates rather than fabricated values;
- Task 2 implementation work has NOT started.

## Required project updates

Update:

- `docs/TASKS.md`
- `docs/ROADMAP.md`
- `docs/TASKS-EPIC-1-MODEL-CHARACTERIZATION.md`
- `docs/TASKS-TASK-1.4-REFERENCE-BEHAVIOR-GOLDENS.md`
- `tools/README.md` when tooling is added
- `CHANGELOG.md`

## Acceptance gates

### Oracle/provenance
- [x] canonical oracle pinned/licensed
- [x] GGUF oracle pinned/licensed or precise blocker documented
- [x] Kestrel-Q is not sole oracle

### Safe generated vectors
- [x] prompt suite finalized/hashed
- [x] tokenizer vectors generated/verified
- [x] chat-template vectors generated/verified
- [x] PLE address vectors generated/verified
- [x] manifest hashes/statuses valid
- [x] deterministic regeneration PASS

### Future full-model contract
- [x] canonical full-model schema/plan finalized
- [x] quantized full-model schema/plan finalized
- [x] operator/checkpoint contract finalized
- [x] exact/tolerance policy documented
- [x] capable-reference-environment requirements documented
- [x] no unavailable output fabricated

### Resource/safety
- [x] no BF16 checkpoint download
- [x] no unsafe BF16 execution on KQ-01
- [x] no unbounded GGUF execution/paging
- [x] tracked `.safetensors` = 0
- [x] tracked `.gguf` = 0
- [x] repository model weights = 0
- [x] no secrets
- [x] no runtime/model implementation added
- [x] `git diff --check` PASS

### Governance
- [x] ADR 0007 has evidence-backed outcome
- [x] CHANGELOG updated
- [x] TASKS/ROADMAP synchronized
- [x] Epic 1 status accurate
- [x] Task 2 NOT STARTED

## Definition of done

Task 1.4 is COMPLETE/PASS when Kestrel-Q has a pinned two-oracle correctness strategy, deterministic prompt suite, generated safe tokenizer/chat/PLE goldens, machine-readable manifest and precise full-model checkpoint contracts for canonical and quantized behavior.

Full-model vectors may remain `DEFERRED_CAPABLE_REFERENCE_ENV` on KQ-01 but become mandatory gates before the corresponding future correctness claim.

Do not commit or push automatically. Leave the complete Task 1.4 delta ready for review.
