# TASKS-TASK-1.0-MODEL-SOURCE-BASELINE.md

Status: **COMPLETE / PASS**

## A. Establish baseline
- [x] Read all governance/foundation docs.
- [x] Record current HEAD and clean/dirty working-tree state.
- [x] Confirm Task 0.x foundation baseline remains intact.

## B. Official upstream
- [x] Query official Hugging Face repository metadata.
- [x] Pin exact upstream revision SHA.
- [x] Enumerate all repository files.
- [x] Classify files without downloading weight shards.
- [x] Record repository total reported size and shard count.
- [x] Record official Qwen GitHub research repository/revision where practical.
- [x] Record official release article and technical-report source.

## C. Safe metadata capture
- [x] Create explicit metadata-download allowlist.
- [x] Prohibit all `.safetensors` shard downloads.
- [x] Download only required small metadata into ignored temporary cache.
- [x] Compute SHA-256 for downloaded metadata.
- [x] Create derived `manifest.json` ready for review.
- [x] Verify manifest can be reproduced from pinned upstream revision.

## D. Source baseline
- [x] Create `docs/MODEL-SOURCE-BASELINE.md`.
- [x] Verify model type and architecture class.
- [x] Verify multimodal wrapper.
- [x] Verify canonical dtype.
- [x] Verify vocab/hidden/layer counts.
- [x] Verify layer pattern.
- [x] Verify configured context length.
- [x] Verify MoE config.
- [x] Verify N-gram config.
- [x] Verify MTP presence/config.
- [x] Verify vision config presence.
- [x] Inventory tokenizer/chat-template assets.
- [x] Keep detailed equations for Task 1.1.

## E. Licensing/provenance
- [x] Create `docs/MODEL-LICENSE-BOUNDARY.md`.
- [x] Separate Kestrel-Q Apache-2.0 code from Qwen model license.
- [x] Record Qwen Community License 1.0 factual conditions relevant to project scope.
- [x] Flag commercial MaaS / AI Work Assistant use for future license review.
- [x] Do not give legal conclusions beyond the source text.

## F. Artifact register
- [x] Create `docs/MODEL-ARTIFACT-REGISTER.md`.
- [x] Add `KQ-MODEL-SOURCE-001` as canonical source.
- [x] Resolve the exact GGUF through persistent User-scope `KQ_GGUF_PATH` and
  fail closed if the value or file is unavailable.
- [x] Do not recursively search the workstation.
- [x] Register `KQ-MODEL-ARTIFACT-001` as `REGISTERED_VERIFIED`.
- [x] Record exact file name.
- [x] Record byte size.
- [x] Compute SHA-256 over the complete merged file.
- [x] Inspect GGUF metadata without modifying the file.
- [x] Record exact quantization/type evidence.
- [x] Record origin URL/revision/converter provenance.
- [x] Verify merged-artifact identity from embedded and pinned upstream evidence,
  not the filename alone.

### F.1 Artifact evidence — 2026-08-28

- The already-running Codex process did not inherit the newly persistent
  variable and failed closed on its first Process-scope check. The exact value
  was then read from persistent User-scope `KQ_GGUF_PATH`; Machine scope was
  empty and there was no conflicting value.
- `Test-Path -LiteralPath <resolved value> -PathType Leaf`: PASS.
- Exact file: `Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf`.
- Exact size: 111,334,654,400 bytes.
- Complete-file SHA-256:
  `8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2`.
- Read-only parse: GGUF v3, `qwen4exp`, 67 metadata keys, 1,224 tensors,
  merged split count zero.
- Pinned Unsloth distribution revision:
  `c8b5954a88c2775c546b92593eda40ea041d3176`; four published
  `UD-Q4_K_XL` shards totaling 111,334,654,784 bytes.
- Post-inspection file size and last-write timestamp exactly matched the values
  captured before inspection.

## G. ADR/governance
- [x] Accept `docs/adr/0004-model-source-of-truth-and-artifact-boundary.md`.
- [x] Update `docs/TASKS.md`.
- [x] Update `docs/ROADMAP.md`.
- [x] Update `CHANGELOG.md`.
- [x] Keep Task 1.1 NOT STARTED until Task 1.0 gate is resolved.

## H. Safety validation
- [x] `git ls-files` contains no `*.safetensors`.
- [x] `git ls-files` contains no `*.gguf`.
- [x] No unrestricted model-download command was used.
- [x] No model-weight file exists under tracked source directories.
- [x] No token/credential appears in evidence.
- [x] No copied incompatible source implementation.
- [x] Temporary metadata cache is ignored.

## I. Final report
Report:
- Kestrel-Q HEAD;
- official HF revision;
- official research-repo revision/source;
- upstream metadata files captured;
- manifest path;
- source baseline facts;
- model-license boundary;
- artifact register status;
- GGUF exact identity/hash/quantization or exact missing input;
- ADR status;
- files changed;
- safety checks;
- unresolved issues;
- final Task 1.0 status;
- working-tree status.
