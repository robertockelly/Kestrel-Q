# Developer Tools

Offline tooling may live here, including:

- model metadata inspection
- tensor inventory generation
- reference-vector generation
- conversion experiments
- benchmark analysis

Production inference must not depend on Python tooling unless an ADR explicitly changes that policy.

`capture-model-baseline.ps1` is a Task 1.0 research-only metadata capture tool.
It requires an exact Hugging Face revision, downloads only its source-controlled
allowlist, and rejects any allowlisted path ending in `.safetensors`. Its cache
lives under the ignored `.research-cache/` directory.

## Task 1.2 Safetensors header inventory

`capture-safetensors-headers.py` is a research-only network tool. It enforces
the exact Qwen repository/revision and 131-shard allowlist from the pinned
index. For each shard it retrieves only the eight-byte Safetensors length prefix
and exact JSON-header range. HTTP 206 status, `Content-Range`, response length,
remote total, host allowlist and global byte budget are fail-closed. It never
saves a `.safetensors` file. A failed run writes a separate audit and later
runs refuse to overwrite or delete that evidence.

```powershell
python tools/capture-safetensors-headers.py `
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 `
  --index .research-cache/model-baseline/de4b8e4d43b917e7706784d8bb445c9af86a3540/model.safetensors.index.json `
  --baseline-manifest research/model-baseline/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/manifest.json `
  --output-dir research/model-tensors/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540 `
  --failure-audit .research-cache/task-1.2-invalid-network-audit.json
```

`analyze-model-tensors.py` is an offline deterministic reconciler. It validates
all index/header names, dtypes, shapes, offsets, shard sizes and aggregate
payload, then emits the canonical CSV and summary. Classification rules use
official tensor names and Task 1.1 architecture only; they do not inspect GGUF.

```powershell
python tools/analyze-model-tensors.py `
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 `
  --index .research-cache/model-baseline/de4b8e4d43b917e7706784d8bb445c9af86a3540/model.safetensors.index.json `
  --header-manifest research/model-tensors/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/shard-header-manifest.json `
  --output-dir research/model-tensors/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540
```

These tools use only the Python standard library and are not production runtime
dependencies.

## Task 1.3 canonical-to-GGUF mapping

`inspect-gguf.py` is a read-only, fail-closed GGUF v3 parser for the exact
registered `KQ-MODEL-ARTIFACT-001`. It requires `KQ_GGUF_PATH`, verifies exact
filename, size and full-file SHA-256, parses only metadata/tensor descriptors,
validates type blocks and spans, and never reads tensor payload values. Its
development-only `--skip-full-sha256` mode is rejected unless output remains
under `.research-cache/`; canonical evidence always requires the full hash.

```powershell
python tools/inspect-gguf.py `
  --output-dir research/model-gguf/Qwen3.8-Flash-Next/c8b5954a88c2775c546b92593eda40ea041d3176
```

`audit-gguf-split-headers.py` pins the Unsloth repository, revision, exact four
shard names and sizes. It reads only exact HTTP Range bytes for fixed GGUF
headers and the three scalar split keys on secondary shards. A non-206 response,
incorrect `Content-Range`, revision/count/size mismatch or unexpected key fails
closed. It saves no GGUF data and records zero tensor-payload bytes fetched.

```powershell
python tools/audit-gguf-split-headers.py `
  --output .research-cache/task-1.3-upstream-splits.json
```

`map-canonical-to-gguf.py` is an offline deterministic reconciler. It requires
the fully verified local metadata/inventory, Task-1.2 canonical inventory and
the bounded split audit. Versioned rules cover all renames, converter
transforms, splits, fusion and accepted scope/format omissions. It fails unless
all 1,658 canonical and 1,224 GGUF tensors reconcile, all parameter/packed-byte
invariants hold and the 384-byte split/merge delta closes exactly.

```powershell
python tools/map-canonical-to-gguf.py `
  --canonical-inventory research/model-tensors/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/tensor-inventory.csv `
  --gguf-metadata research/model-gguf/Qwen3.8-Flash-Next/c8b5954a88c2775c546b92593eda40ea041d3176/gguf-metadata.json `
  --gguf-inventory research/model-gguf/Qwen3.8-Flash-Next/c8b5954a88c2775c546b92593eda40ea041d3176/gguf-tensor-inventory.csv `
  --upstream-split-audit .research-cache/task-1.3-upstream-splits.json `
  --output-dir research/model-gguf/Qwen3.8-Flash-Next/c8b5954a88c2775c546b92593eda40ea041d3176
```

These standard-library tools are research-only and are not production runtime
dependencies. Pinned third-party sources are inspected for evidence under their
original license; no implementation source is incorporated.

## Task 1.4 reference goldens

`generate-reference-goldens.py` generates only weight-independent Task 1.4
assets. It requires the Task 1.0 allowlisted metadata directory plus exact
Transformers and llama.cpp source checkouts. It verifies revision and source
hashes, rejects `.safetensors`/`.gguf` inputs, forces local-only tokenizer use,
and fails unless the pinned Python/tokenizer/template dependency versions match.
It does not import PyTorch, load a model or resolve a remote model ID.

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python tools/generate-reference-goldens.py `
  --model-dir .research-cache/model-baseline/de4b8e4d43b917e7706784d8bb445c9af86a3540 `
  --transformers-source .research-cache/task-1.4/transformers `
  --llama-source .research-cache/task-1.4/llama.cpp `
  --output-dir research/goldens/Qwen3.8-Flash-Next
```

`validate-reference-goldens.py` parses the manifest and every referenced JSON
asset, rehashes all paths, checks the two oracle classes and allowed statuses,
and requires full prompt/tokenizer/chat/PLE coverage.

```powershell
python tools/validate-reference-goldens.py `
  --golden-dir research/goldens/Qwen3.8-Flash-Next
```

Both tools are research-only. Transformers/tokenizers/Jinja are isolated
generation dependencies; neither tool is a production runtime dependency.

## Task 2.1 semantic-registry oracle

`validate-semantic-registry.py` runs the native `kq-inspect --semantic-dump`
mode twice, requires byte-identical output, and compares every initial-text
canonical identity/relation/physical name against the pinned Task 1.3
`canonical-gguf-mapping.csv`. It also requires 1,224 unique physical names and
three metadata-derived semantics. The local model is resolved only from
`KQ_GGUF_PATH` unless an explicit test path is supplied.

```powershell
python tools/validate-semantic-registry.py `
  --inspect build-cpu/Release/kq-inspect.exe `
  --mapping research/model-gguf/Qwen3.8-Flash-Next/c8b5954a88c2775c546b92593eda40ea041d3176/canonical-gguf-mapping.csv
```

The validator is test/research-only. `kq_model` and the production runtime do
not read CSV/JSON evidence and do not depend on Python.

## Task 2.2 tensor-view geometry oracle

`validate-tensor-view-geometry.py` runs
`kq-inspect --view-geometry-dump` twice, requires byte-identical output and
compares all 1,224 physical names, ranks, dimensions, types, block geometry,
element counts, offsets and packed bytes with the pinned Task 1.3
`gguf-tensor-inventory.csv`.

```powershell
python tools/validate-tensor-view-geometry.py `
  --inspect build-cpu/Release/kq-inspect.exe `
  --inventory research/model-gguf/Qwen3.8-Flash-Next/c8b5954a88c2775c546b92593eda40ea041d3176/gguf-tensor-inventory.csv
```

The model resolves only from an explicit `--model` or `KQ_GGUF_PATH`; the test
skips when neither is available. The standard-library validator reads no model
payload. Production code neither invokes it nor reads the evidence CSV.

## Task 2.3 canonical tokenizer oracle

`generate-tokenizer-differential.py` is a research-only pinned-oracle tool. It
verifies the official tokenizer asset hashes, imports the exact offline
Transformers revision and emits canonical expected behavior for the discovered
NFC, combining-mark, padded-ID, special-token and chat-template divergences.
It also emits the binary compatibility digests compiled into the production
GGUF substrate gate. It never imports Kestrel-Q output.

```powershell
$env:PYTHONPATH = (Resolve-Path '.research-cache/task-1.4/transformers/src').Path
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'
.research-cache/task-1.4/venv/Scripts/python.exe `
  tools/generate-tokenizer-differential.py `
  --model-dir .research-cache/model-baseline/de4b8e4d43b917e7706784d8bb445c9af86a3540 `
  --output research/tokenizer/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/canonical-differential.json
```

`validate-native-tokenizer.py` hashes the unchanged Task 1.4 prompt/tokenizer/
chat/manifest assets, then compares the native test probe with every original
golden and every independent differential case. The local GGUF is resolved
only by the probe from `KQ_GGUF_PATH`; the validator receives no model path.

```powershell
python tools/validate-native-tokenizer.py `
  --probe build-cpu/Release/kq_tokenizer_probe.exe `
  --golden-dir research/goldens/Qwen3.8-Flash-Next `
  --differential research/tokenizer/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/canonical-differential.json
```

Both programs are test/research tools. Python, Transformers and tokenizers are
not production runtime dependencies.

`generate-unicode9-assigned.py` verifies the official Unicode 9.0.0 UCD
`DerivedAge.txt` SHA-256 and deterministically emits the compact assigned-range
table used to reproduce the pinned tokenizer's Unicode-9 NFC boundary. The
source file stays in ignored research cache; the generated table and its source
hash are reviewable production inputs.

```powershell
python tools/generate-unicode9-assigned.py `
  --derived-age .research-cache/task-2.3-unicode9/DerivedAge-9.0.0.txt `
  --output src/kq_unicode9_assigned.inc
```

The generated table SHA-256 is
`83a57437a5785fcbe40b21f4f297c5b7cc5bc472ace9b0a57a3e9040a8a39694`.

## Task 2.4 PLE address oracle

`generate-ple-differential.py` is a research-only, standard-library generator.
It requires the exact pinned Transformers Qwen4-Exp implementation file,
official `config.json` and unchanged Task 1.4 tokenizer vectors, and verifies
all three hashes before deriving expected PLE values. It independently derives
the seeded odd multipliers and prime/head-offset sequence, then emits 12
sequence, three incremental-stream and one tokenizer-integration case. It does
not import or invoke Kestrel-Q.

```powershell
python tools/generate-ple-differential.py `
  --transformers-source .research-cache/task-1.4/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py `
  --config .research-cache/model-baseline/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json `
  --tokenizer-vectors research/goldens/Qwen3.8-Flash-Next/canonical/tokenizer-vectors.json `
  --output research/ple/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/canonical-differential.json
```

The deterministic output SHA-256 is
`b9c9be4d927d59c9ac12ba2313034cda5a1857d5484fca479327c9b771cb9671`.

`validate-native-ple.py` first requires the unchanged Task 1.4 PLE SHA and the
expanded differential SHA, then compares every native intent/state field with
the independent expectations through the test-only `kq_ple_probe`. The probe
opens the model from the explicit test argument supplied from `KQ_GGUF_PATH`;
production PLE code reads no JSON and has no Python dependency.

```powershell
python tools/validate-native-ple.py `
  --probe build-cpu/Release/kq_ple_probe.exe `
  --golden research/goldens/Qwen3.8-Flash-Next/canonical/ple-address-vectors.json `
  --differential research/ple/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/canonical-differential.json
```
