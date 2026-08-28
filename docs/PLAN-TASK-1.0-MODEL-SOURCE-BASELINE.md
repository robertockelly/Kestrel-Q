# PLAN-TASK-1.0-MODEL-SOURCE-BASELINE.md

Status: **COMPLETE / PASS**

All Task 1.0 definition-of-done items are complete. `KQ-MODEL-ARTIFACT-001`
was resolved through the persistent User-scope `KQ_GGUF_PATH`, inspected
read-only and registered with exact identity, hash and provenance.

## Objective

Create the canonical source and artifact baseline for Qwen3.8-Flash-Next before any model implementation begins.

The task must answer:

1. Which upstream artifacts define the model?
2. Which exact upstream revision is Kestrel-Q studying?
3. Which metadata files exist and what are their hashes/sizes?
4. Which files may be downloaded now without accidentally pulling the 360 GB checkpoint?
5. What license applies to the model artifacts?
6. What role does the user's GGUF Q4 play relative to the canonical checkpoint?
7. What must never be committed to the Kestrel-Q source repository?

## Upstream source hierarchy

### Tier A — canonical model artifacts

Official Hugging Face repository:

`Qwen/Qwen3.8-Flash-Next`

Use its pinned revision as the canonical source for:

- `config.json`;
- generation config;
- tokenizer/processor files;
- chat template;
- Safetensors index;
- official model license;
- official released tensor/shard inventory.

### Tier A — canonical architecture description

Official Qwen repository and technical report:

`https://github.com/QwenLM/Qwen3.8-Flash-Next`

Official release article:

`https://qwen.ai/blog?id=qwen3.8-flash-next`

Use them for architecture claims that are not fully specified by config/tensor metadata.

### Tier B — reference implementations

Transformers, SGLang, vLLM, llama.cpp or other runtimes may be used to understand behavior only after their exact revision and license are recorded.

Do not make another runtime's naming/layout the canonical model definition.

## Licensing boundary

Kestrel-Q source code is Apache License 2.0.

The Qwen3.8-Flash-Next model repository currently identifies the model artifacts under **Qwen Community License 1.0**, not Apache-2.0.

The task must create a factual licensing boundary document.

Important requirements:

- do not copy Qwen's LICENSE into Kestrel-Q as if it relicensed Kestrel-Q;
- do not state that the model weights are Apache-2.0;
- do not state that Qwen Community License 1.0 is OSI-approved unless authoritative evidence says so;
- record the license's special commercial conditions relevant to Model-as-a-Service / AI Work Assistant use;
- do not provide legal advice; document terms and flag commercial deployment for license review.

Because Kestrel-Q may eventually include a coding agent, the AI Work Assistant clause is materially relevant to future commercial distribution/hosting and must be tracked.

## No-weight-download guardrail

During Task 1.0:

> **Do not download any `.safetensors` weight shard.**

The official checkpoint is approximately 360 GB and uses 131 Safetensors shards.

The task may retrieve:

- repository API/file listing;
- small metadata/config/tokenizer files;
- `model.safetensors.index.json`;
- README/license/technical report metadata as needed.

A download command must use explicit allowlists/includes.

Never use an unrestricted repository clone/download command during this task.

## Metadata capture

Create a reproducible manifest containing at minimum:

- upstream repository ID;
- pinned upstream commit/revision SHA;
- capture timestamp UTC;
- file path;
- byte size;
- SHA-256 where downloaded;
- upstream ETag/Xet hash if available;
- role/classification;
- license/provenance note.

Recommended committed output:

```text
research/model-baseline/Qwen3.8-Flash-Next/<revision>/manifest.json
```

Do not commit downloaded upstream source files unless separately reviewed for redistribution and attribution. Prefer the derived manifest.

Temporary downloaded metadata should live under an ignored cache directory.

## Metadata inventory

Do not assume a fixed list. First enumerate the official repository.

Classify at least:

- configuration;
- tokenizer;
- processor/multimodal metadata;
- chat template;
- model index;
- license/readme;
- weight shards;
- other auxiliary artifacts.

Expected examples include `config.json`, `generation_config.json`, `chat_template.jinja`, tokenizer assets and `model.safetensors.index.json`, but the actual pinned repository listing wins.

## Canonical model facts baseline

Extract a small set of high-confidence facts into:

```text
docs/MODEL-SOURCE-BASELINE.md
```

At minimum verify from pinned authoritative evidence:

- repository/model ID;
- model type;
- declared architecture class;
- multimodal/text-only flag;
- dtype of canonical weights;
- vocabulary size;
- hidden size;
- number of text layers;
- layer-type pattern;
- configured context length;
- number of experts;
- experts per token;
- MoE intermediate size;
- shared-expert configuration;
- N-gram configuration;
- MTP configuration;
- vision configuration presence;
- tokenizer/chat-template files available.

Do not turn Task 1.0 into a full architecture analysis. Detailed equations belong to Task 1.1.

## GGUF artifact registration

The user's existing Q4 GGUF is a **derived runtime artifact**, not the canonical model source.

Assign:

`KQ-MODEL-ARTIFACT-001`

Required fields:

- exact filename;
- local path or logical location;
- file size bytes;
- SHA-256;
- GGUF version if inspectable;
- architecture metadata;
- quantization label exactly as encoded;
- source/download repository URL if known;
- source revision if known;
- converter/quantizer provenance if discoverable;
- model license/provenance;
- status.

Do not copy the GGUF into Git.

### Discovery safety

Use the GGUF path only if:

1. the human supplies `KQ_GGUF_PATH`, or
2. the artifact is present under the repository's already-ignored local `models/` directory.

Do not recursively search the user's entire drive.

If the exact GGUF path is unavailable:

- complete the official upstream baseline;
- create the registry entry as `AWAITING_LOCAL_PATH`;
- report Task 1.0 as complete except for the explicit artifact-identification subgate, or as PARTIAL according to the repository's task-status convention;
- request only the exact path/filename from the maintainer.

## Artifact register

Create:

```text
docs/MODEL-ARTIFACT-REGISTER.md
```

Minimum entries:

### KQ-MODEL-SOURCE-001
Official `Qwen/Qwen3.8-Flash-Next` Safetensors checkpoint/repository.

Role: `CANONICAL_SOURCE`

### KQ-MODEL-ARTIFACT-001
User's local GGUF Q4.

Role: `DERIVED_RUNTIME_ARTIFACT`

The register must explicitly state that matching model identity does not imply identical tensor naming, packing, precision, layout or component inclusion.

## ADR

Create an ADR:

```text
docs/adr/0004-model-source-of-truth-and-artifact-boundary.md
```

Decision:

- official pinned Qwen artifacts/specification define canonical model semantics;
- Safetensors official tensor inventory is canonical for weight/tensor identity;
- GGUF is treated as derived representation;
- Kestrel-Q must not infer canonical architecture solely from GGUF;
- model weights remain external to the source repository.

## Research/download tooling

Small research-only tooling may be added under `tools/` if it improves reproducibility.

Requirements:

- explicit allowlist;
- never download `*.safetensors`;
- fail closed if an unrestricted/full-checkpoint download would occur;
- no production runtime dependency on Python/Hugging Face tooling;
- record the upstream revision used.

PowerShell or Python is acceptable for offline research tooling.

## Project documentation updates

Update:

- `docs/TASKS.md`;
- `docs/ROADMAP.md`;
- `CHANGELOG.md`;
- relevant Epic 1 documents.

Do not mark Task 1.1 started merely because Task 1.0 records some config values.

## Validation

Before completion verify:

- no `.safetensors` shard was downloaded into the repository;
- no model weight is tracked by Git;
- no GGUF is tracked by Git;
- manifest is reproducible;
- upstream revision is pinned;
- source URLs are official where labeled canonical;
- model license is correctly separated from Apache-2.0 project code;
- hashes are recorded for downloaded metadata;
- no secrets/tokens are committed;
- working tree contains only intentional research/governance changes.

## Definition of done

Task 1.0 is fully PASS when:

1. official upstream revision is pinned;
2. metadata manifest exists;
3. model source baseline exists;
4. licensing boundary exists;
5. artifact register exists;
6. ADR 0004 exists;
7. GGUF `KQ-MODEL-ARTIFACT-001` has exact size/hash/quantization/provenance;
8. no model weights are committed/downloaded accidentally;
9. project status docs are updated;
10. validation passes.

If item 7 cannot be completed because the maintainer has not supplied the local GGUF path, all other work should still be completed and the exact missing input reported rather than searching the workstation broadly.

Resolved 2026-08-28: `KQ_GGUF_PATH` identifies the exact maintained local GGUF.
