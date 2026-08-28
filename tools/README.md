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
