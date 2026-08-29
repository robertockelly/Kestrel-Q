# `kq-inspect`

Status: **TASK 2.0 COMPLETE / PASS**

`kq-inspect` is the first native Kestrel-Q diagnostics executable. It opens one
GGUF through the read-only Windows file layer and prints structural facts only.

## Usage

```powershell
build-cpu\Release\kq-inspect.exe <model.gguf>
```

The path is a command-line input and is never persisted by the tool. The real
artifact integration test resolves its path only from `KQ_GGUF_PATH`.

## Default output

For `KQ-MODEL-ARTIFACT-001`, the verified output is:

```text
file_size_bytes=111334654400
gguf_version=3
architecture=qwen4exp
metadata_count=67
tensor_count=1224
alignment_bytes=32
directory_end_offset=11024307
data_section_offset=11024320
packed_tensor_bytes=111323630080
format_overhead_bytes=11024320
directory_bytes_parsed=11024307
payload_bytes_accessed=0
type.BF16=24
type.F32=557
type.IQ4_NL=1
type.Q4_K=94
type.Q5_1=43
type.Q5_K=2
type.Q8_0=503
```

Untrusted architecture bytes are printed with non-printable bytes escaped;
they are never treated as an implicit C string. A successful inspection exits
zero. Usage errors exit 2. File/format/validation errors print a bounded
diagnostic and exit non-zero.

## Safety and scope

The command does not hash the full file, dump tensor values, dereference tensor
payload, dequantize data or allocate the model footprint. Its mapped address
range is virtual; only header/directory bytes are parsed. Current Task 2.0 has a
single summary mode. Metadata/tensor listing, JSON output and canonical semantic
interpretation remain future work.
