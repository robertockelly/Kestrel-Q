# Qwen3.8-Flash-Next model artifact register

Status: **TASK 1.0 REGISTER COMPLETE / VERIFIED**

Artifact roles are intentionally distinct. Matching a model name does not imply
identical tensor names, component inclusion, precision, quantization, packing or
layout.

## KQ-MODEL-SOURCE-001

- name: Official Qwen3.8-Flash-Next Safetensors checkpoint/repository
- role: `CANONICAL_SOURCE`
- organization: Qwen
- repository ID: `Qwen/Qwen3.8-Flash-Next`
- URL: `https://huggingface.co/Qwen/Qwen3.8-Flash-Next`
- pinned revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- format: 131-shard Safetensors checkpoint plus official metadata/index
- canonical index: `model.safetensors.index.json`
- index tensor-name entries: 1,658
- index-referenced shards: 131
- license: Qwen Community License 1.0 at the pinned revision
- local state: allowlisted metadata only in ignored research cache; no weights
- evidence: `research/model-baseline/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/manifest.json`
- status: `PINNED_VERIFIED`

The official index defines canonical released tensor identity and checkpoint
composition. Detailed tensor-family analysis remains Task 1.2.

## KQ-MODEL-ARTIFACT-001

- name: Maintainer's local Qwen3.8-Flash-Next Q4 GGUF
- role: `DERIVED_RUNTIME_ARTIFACT`
- exact filename: `Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf`
- local path source: persistent User-scope `KQ_GGUF_PATH`
- local path: intentionally not recorded; resolved only from `KQ_GGUF_PATH`
- size bytes: **111,334,654,400**
- SHA-256: `8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2`
- format: GGUF, magic `GGUF`, version **3**
- encoded architecture: `qwen4exp`
- metadata keys: **67**
- tensor count: **1,224**
- model identity metadata:
  - `general.name = Qwen3.8 Flash Next`;
  - `general.description = A Preview of the Qwen4 Architecture`;
  - `general.type = model`;
  - `general.size_label = 512x56B`;
  - 48 blocks, context length 262,144, embedding length 2,560, 512 experts
    and 10 experts used, consistent with `KQ-MODEL-SOURCE-001`;
- split/merged metadata:
  - `split.no = 0`;
  - `split.count = 0`;
  - `split.tensors.count = 1224`;
- quantization identity: **UD-Q4_K_XL**, verified from the exact published
  upstream variant plus artifact metadata and merged-size evidence; the literal
  `UD-Q4_K_XL` label is not stored as a GGUF metadata value;
- GGUF quantization metadata:
  - `general.quantized_by = Unsloth`;
  - `general.quantization_version = 2`;
  - `general.file_type = 15`, which maps to `MOSTLY_Q4_K_M` in pinned
    llama.cpp revision `90c26fcd4b2114b4aa39d09d69318cb8f438d27a`;
  - `quantize.imatrix.file = Qwen3.8-Flash-Next-GGUF/imatrix_unsloth.gguf`;
  - `quantize.imatrix.dataset = unsloth_calibration_Qwen3.8-Flash-Next.txt`;
  - imatrix entries: **926**; chunks: **45**;
- tensor storage types:

| GGML type | Type ID | Tensor count |
|---|---:|---:|
| `F32` | 0 | 557 |
| `Q5_1` | 7 | 43 |
| `Q8_0` | 8 | 503 |
| `Q4_K` | 12 | 94 |
| `Q5_K` | 13 | 2 |
| `IQ4_NL` | 20 | 1 |
| `BF16` | 30 | 24 |

- source repository: `https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF`
- pinned source revision: `c8b5954a88c2775c546b92593eda40ea041d3176`
- upstream variant: `UD-Q4_K_XL`, published as four GGUF shards
- published shard total: **111,334,654,784 bytes**
- local merged-size delta: **384 bytes smaller** than the four published shards
  combined; Task 1.3 proves this is format/header-directory overhead only, with
  identical 111,323,630,080 packed tensor bytes;
- source repository metadata declares base model
  `Qwen/Qwen3.8-Flash-Next` and Qwen Community License 1.0;
- converter/quantizer provenance: Unsloth, with the embedded imatrix metadata
  above;
- canonical upstream model: `KQ-MODEL-SOURCE-001`, pinned official Qwen
  Safetensors repository;
- inspection: read-only header, metadata and tensor-descriptor parse followed by
  a complete-file SHA-256 read; tensor payloads were not interpreted or changed;
- status: `REGISTERED_VERIFIED`

Identity is supported by more than the filename: the artifact encodes Unsloth
quantizer/imatrix metadata, the expected `qwen4exp` model identity and canonical
configuration values, 1,224 merged tensors and zero active splits. The pinned
Unsloth repository explicitly declares the canonical Qwen base model and
publishes exactly four `UD-Q4_K_XL` shards whose combined size differs from the
merged local file by only 384 bytes.

The upstream shards were not downloaded or re-merged during Task 1.0, so this
registration does not claim a cryptographic derivation proof between the local
merged SHA-256 and the four individual upstream LFS hashes.

Task 1.3 maps all 1,658 canonical tensors to the 1,224 GGUF tensors or to exact
scope/format-derived dispositions with zero unresolved entries. See
`docs/MODEL-GGUF-MAPPING.md` and the pinned machine-readable evidence under
`research/model-gguf/`.

## Boundary rule

The GGUF is a conversion/quantization product. It may rename, omit, fuse, repack
or change precision/layout of canonical tensors. Kestrel-Q must map it back to
the pinned official source model and must never infer canonical architecture
solely from GGUF metadata or tensor names.
