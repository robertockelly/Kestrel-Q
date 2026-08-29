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
