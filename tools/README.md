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

## Task 2.5 numeric oracles

`llama-dequant-oracle.cpp` is a research-only Class-Q helper built against the
ignored checkout/build of
`ggml-org/llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a` (MIT). It calls the pinned
GGML F32 decode trait for synthetic blocks; F32 uses the equivalent
bit-preserving copy because that trait intentionally has no conversion
callback. Production Kestrel-Q neither compiles nor links this helper.

```powershell
$llamaSource = (Resolve-Path .research-cache/task-1.4/llama.cpp).Path
$llamaBuild = (Resolve-Path .research-cache/task-1.4/llama.cpp/build-kq-oracle).Path
cmake -S tools/llama-dequant-oracle `
  -B .research-cache/task-2.5/llama-oracle-build `
  -G "Visual Studio 17 2022" -A x64 `
  -DLLAMA_CPP_SOURCE_DIR="$llamaSource" `
  -DLLAMA_CPP_BUILD_DIR="$llamaBuild"
cmake --build .research-cache/task-2.5/llama-oracle-build --config Release
```

`generate-numeric-evidence.py` uses that helper for dequant expected values and
Python 3.13.12/NumPy 2.5.2 (BSD-3-Clause) for explicitly ordered F32 primitive
expected values. It records native output only as an observation, derives
per-primitive calibration contracts, checks a disjoint holdout and emits the
synthetic evidence/manifest.

`capture-real-numeric-samples.py` invokes the test-only native view probe and
holds each raw block only in subprocess memory while comparing it to the pinned
llama helper. The committed file contains hashes/statistics, not packed model
bytes. It requires the exact registered artifact via `KQ_GGUF_PATH`, rechecks
its 111,334,654,400-byte size and enforces the 1 MiB guard.

```powershell
$out = 'research/numerics/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540'
$llama = '.research-cache/task-2.5/llama-oracle-build/Release/kq_llama_dequant_oracle.exe'
$native = 'build-cpu/Release/kq_numeric_probe.exe'
$real = 'build-cpu/Release/kq_numeric_integration_test.exe'
.research-cache/task-1.4/venv/Scripts/python.exe tools/generate-numeric-evidence.py `
  --llama-oracle $llama --native-probe $native --output-dir $out
.research-cache/task-1.4/venv/Scripts/python.exe tools/capture-real-numeric-samples.py `
  --real-probe $real --llama-oracle $llama --output "$out/real-gguf-samples.json"
.research-cache/task-1.4/venv/Scripts/python.exe tools/generate-numeric-evidence.py `
  --llama-oracle $llama --native-probe $native --output-dir $out
```

`validate-native-numerics.py` verifies manifest hashes, all 39 synthetic
decode vectors, seven row-dot cases and all 31 calibration/21 holdout cases
through `kq_numeric_probe`. It uses only the Python standard library and is a
CTest/research dependency, never a production dependency.

## Task 2.6 GDN Class-C oracle

`generate-gdn-reference.py` is a research-only offline Class-C generator. It
verifies the exact official config and four pinned Transformers source hashes,
imports `Qwen4ExpTextGatedDeltaNet` from revision
`805a9e939fa8c1bff8d8ffdf041c051b71a914aa`, and instantiates the module with a
bounded reduced F32 configuration supported by the canonical class. Exact
power-of-two synthetic inputs/weights feed five calibration, five disjoint
holdout and prefill/decode/reset state cases. It runs before native comparison,
does not import Kestrel-Q and downloads no checkpoint.

```powershell
$env:PYTHONPATH = (Resolve-Path '.research-cache/task-1.4/venv/Lib/site-packages').Path
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$out = 'research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540'
python tools/generate-gdn-reference.py `
  --transformers-source .research-cache/task-1.4/transformers/src `
  --config .research-cache/model-baseline/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json `
  --output-dir $out
```

`validate-native-gdn.py` sends those independent arrays to the test-only
`kq_gdn_probe`, derives per-checkpoint limits from calibration only, and
requires the untouched limits to pass holdout and state cases. It separately
requires native split-prefill/decode and reset/replay bit identity. With
`--write-validation` it emits deterministic comparison evidence and finalizes
the manifest; normal CTest mode recomputes the result and requires byte
identity. High-resolution timing printed by the probe is characterization only
and is intentionally excluded from hashed deterministic evidence.

```powershell
python tools/validate-native-gdn.py `
  --probe build-cpu/Release/kq_gdn_probe.exe `
  --evidence-dir $out `
  --write-validation
```

Python, PyTorch, NumPy and Transformers remain ignored test/research
dependencies. The C17 production GDN code reads no evidence JSON and links no
oracle framework.

## Task 2.7 QSA Class-C oracle

`generate-qsa-reference.py` is the offline independent QSA generator. It
verifies the pinned official model/config and Transformers source hashes,
imports `Qwen4ExpTextAttention`/`Qwen4ExpTextQSAIndexer`, and emits expectations
before native comparison. Tier A uses the canonical module's supported reduced
F32 dimensions for calibration, disjoint holdout and state transitions. Tier B
uses a bounded 512/513-complete-block configuration to cross the released
selection limit and records exact top-k membership/order, including ties. It
does not download or load checkpoint weights.

```powershell
$env:PYTHONPATH = (Resolve-Path '.research-cache/task-1.4/venv/Lib/site-packages').Path
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$out = 'research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540'
python tools/generate-qsa-reference.py `
  --checkout .research-cache/task-1.4/transformers `
  --config .research-cache/model-baseline/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json `
  --output-dir $out
```

`validate-native-qsa.py` drives the test-only `kq_qsa_probe`, derives separate
per-checkpoint limits from calibration only, and applies them value-by-value to
the disjoint holdout/state cases. Candidate/selected block IDs, gathered token
positions, counts and ordering compare exactly. Normal mode records
deterministic native evidence; CTest `--verify` mode requires all files
to remain byte-identical. The probe also reports one high-resolution,
no-observer execution timing for characterization; timing is deliberately
excluded from deterministic evidence and correctness decisions.

```powershell
python tools/validate-native-qsa.py `
  --probe build-cpu/Release/kq_qsa_probe.exe `
  --evidence-dir $out
```

The Python/oracle stack remains an ignored research/test dependency. Production
C17 QSA code reads no evidence JSON and links no Transformers, PyTorch, NumPy
or Python runtime.

## Task 2.8 MoE Class-C oracle

`generate-moe-reference.py` is the offline independent MoE generator. It
verifies the pinned official config and four Transformers source hashes, then
imports `Qwen4ExpTextSparseMoeBlock` from revision
`805a9e939fa8c1bff8d8ffdf041c051b71a914aa`. Tier A preserves canonical
router/routed/shared equations with hidden size 6, four experts and top-2 for
five calibration and four disjoint holdout cases. Tier B keeps 512 experts and
top-10 with bounded hidden width and captures exact tie/boundary routing. It
does not download or load checkpoint weights.

The pinned checkout requires `tokenizers==0.23.1`. The global development
environment had 0.22.2, so generation deliberately uses the already governed
Task 1.4 virtual-environment site-packages through `PYTHONPATH`; the generator
fails closed on another version and runs with Hugging Face/Transformers offline.

```powershell
$env:PYTHONPATH = (Resolve-Path '.research-cache/task-1.4/venv/Lib/site-packages').Path
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$out = 'research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540'
python tools/generate-moe-reference.py `
  --checkout .research-cache/task-1.4/transformers `
  --config .research-cache/model-baseline/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json `
  --output-dir $out
```

`validate-native-moe.py` drives the test-only `kq_moe_probe`. It derives a
separate floating contract for each routed/shared/final checkpoint from
calibration only, requires the disjoint holdout to pass, and compares all
Tier-B expert IDs/order exactly. Timing remains characterization output and is
excluded from deterministic evidence. CTest `--verify` requires byte-identical
native evidence and manifest.

```powershell
python tools/validate-native-moe.py `
  --probe build-cpu/Release/kq_moe_probe.exe `
  --evidence-dir $out `
  --verify
```

Production C17 MoE code reads no evidence file and links no Python,
Transformers, PyTorch or NumPy runtime.

## Task 2.9 PLE value Class-C oracle

`generate-ple-value-reference.py` verifies the pinned official config and
Transformers source hashes, imports `Qwen4ExpTextPLELayer` from revision
`805a9e939fa8c1bff8d8ffdf041c051b71a914aa`, and generates expected values
before native comparison. Tier A uses the canonical equations with bounded F32
dimensions for four calibration and three disjoint holdout cases. Tier B
combines real Task 2.4 canonical address streams with deterministic synthetic
tables and weights. State vectors cover prefix prefill plus decode, replay and
the nine-position dilation boundary. No checkpoint weights are downloaded or
loaded.

```powershell
$env:PYTHONPATH = (Resolve-Path '.research-cache/task-1.4/venv/Lib/site-packages').Path
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$out = 'research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540'
python tools/generate-ple-value-reference.py `
  --checkout .research-cache/task-1.4/transformers `
  --config .research-cache/model-baseline/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json `
  --output-dir $out `
  --verify
```

`validate-native-ple-value.py` drives the test-only `kq_ple_value_probe`,
derives independent per-checkpoint floating contracts from calibration only,
then applies them unchanged to disjoint holdout and state. Lookup and embedding
bits plus Task 2.4 intent fields compare exactly. Normal mode writes native
validation and updates the deterministic manifest; `--verify` requires both to
remain byte-identical.

```powershell
python tools/validate-native-ple-value.py `
  --probe build-cpu/Release/kq_ple_value_probe.exe `
  --evidence-dir $out `
  --verify
```

The real CTest integration uses `KQ_GGUF_PATH` only, resolves rows through
Task 2.2 views and Task 2.5 IQ4_NL decode, and enforces an 8 MiB logical packed
payload guard. Production C17 PLE value code reads no evidence JSON and links
no Python, Transformers, PyTorch or NumPy runtime.

## Task 2.10 complete-layer Class-C oracle

`generate-layer-reference.py` runs offline against the same pinned Qwen
Transformers checkout. It imports the canonical decoder layer and GR modules,
generates deterministic reduced ordinary-GDN, QSA and PLE-GDN expectations
before native execution, and writes the governed `layer-*.json` assets under
the existing operator evidence root. No checkpoint weights are loaded.
The offline environment must provide CPU PyTorch and `tokenizers==0.23.1`,
the exact dependency accepted by the pinned Transformers checkout. The
generator rejects a different tokenizers version before writing evidence;
neither dependency is used by production Kestrel-Q.

```powershell
$env:PYTHONPATH = (Resolve-Path '.research-cache/task-1.4/venv/Lib/site-packages').Path
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$out = 'research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540'
python tools/generate-layer-reference.py `
  --checkout .research-cache/task-1.4/transformers `
  --output-dir $out
python tools/validate-native-layer.py `
  --probe build-cpu/Release/kq_layer_probe.exe `
  --evidence-dir $out
```

The validator derives family-specific limits from calibration only, applies
them unchanged to disjoint holdout, records native validation, and supports
`--verify` for CTest byte identity. Production C17 reads none of these files
and has no Python/PyTorch/Transformers dependency.

## Task 2.11 exact-GGUF target-layer oracle

`llama-dequant-oracle.cpp` retains its Task 2.5 pinned llama.cpp provenance and
adds the research-only `dequant-file` mode. That mode decodes an explicitly
bounded physical span to an ignored F32 cache file; it rejects more than
256 MiB per invocation and is never linked into production Kestrel-Q.

`generate-target-layer-reference.py` verifies the registered GGUF size and
SHA-256, exact Epic 1 physical inventory, pinned Transformers checkout, pinned
oracle packages and official config. It uses the llama helper as the only
packed-storage decoder, applies the verified converter-layout inverses, and
executes separate canonical equations for deterministic ordinary GDN layer 0,
QSA layer 3 and PLE-GDN layer 1. It writes no packed weight bytes to governed
evidence.

```powershell
$gguf = [Environment]::GetEnvironmentVariable('KQ_GGUF_PATH', 'User')
$out = 'research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540'
python tools/generate-target-layer-reference.py `
  --gguf $gguf `
  --inventory research/model-gguf/Qwen3.8-Flash-Next/c8b5954a88c2775c546b92593eda40ea041d3176/gguf-tensor-inventory.csv `
  --llama-helper .research-cache/task-2.5/llama-oracle-build/Release/kq_llama_dequant_oracle.exe `
  --transformers-checkout .research-cache/task-1.4/transformers `
  --oracle-site-packages .research-cache/task-1.4/venv/Lib/site-packages `
  --config .research-cache/model-baseline/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json `
  --cache-dir .research-cache/task-2.11/dequant `
  --output-dir $out
```

`validate-target-layer.py` runs the native real-artifact integration for four
calibration profiles and a disjoint holdout. It derives per-family/per-phase
floating limits from calibration only, requires holdout PASS, and compares
route order, selected-expert membership/access and PLE intents exactly. Host
timings are deliberately removed from deterministic evidence.

```powershell
$env:KQ_GGUF_PATH = [Environment]::GetEnvironmentVariable('KQ_GGUF_PATH', 'User')
python tools/validate-target-layer.py `
  --probe build-cpu/Release/kq_target_layer_integration_test.exe `
  --evidence-dir $out `
  --verify
```

Production C17 target-layer code reads no research CSV/JSON and links no
llama.cpp, Python, NumPy, PyTorch or Transformers runtime.

## Task 2.12 full-model Class-Q oracle

`llama-first-token-oracle.cpp` is a research/test-only helper for pinned
`llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a` (MIT). Its separate CMake
project requires explicit `LLAMA_SOURCE_ROOT` and `LLAMA_BUILD_ROOT`; neither
the helper nor llama.cpp is linked into `kq_core` or `kq-run`. The helper loads
the verified GGUF CPU-only with mmap and lazy tensor reads, accepts explicit
canonical IDs and emits bounded top-N/greedy evidence. It deliberately bypasses
llama.cpp tokenization because Task 2.3 proved the GGUF tokenizer semantics are
not the canonical production contract.

```powershell
cmake -S tools/llama-first-token-oracle `
  -B .research-cache/task-2.12/llama-first-token-oracle-build `
  -DLLAMA_SOURCE_ROOT=<pinned-llama-checkout> `
  -DLLAMA_BUILD_ROOT=<pinned-llama-build>
cmake --build .research-cache/task-2.12/llama-first-token-oracle-build `
  --config Release
& .research-cache/task-2.12/llama-first-token-oracle-build/Release/kq_llama_first_token_oracle.exe `
  --model $env:KQ_GGUF_PATH `
  --tokens 9419,11,710,467,3621,27325,13 `
  --context 8 --top-n 20 --threads 16 --output <ignored-output.json>
```

The governed milestone stores only deterministic derived summaries under
`research/milestones/`; raw logs, full logits and all model payload remain
ignored. Kestrel-Q never supplies expected values to this oracle.
