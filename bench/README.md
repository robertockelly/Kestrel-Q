# Benchmarks

Benchmark executables, datasets and result-schema documentation belong here.

Raw benchmark outputs should eventually be stored in `bench/results/` and are ignored by Git by default unless an explicit policy later chooses selected canonical results for version control.

## CUDA/PCIe bandwidth

With `KQ_ENABLE_CUDA=ON`, CMake builds `kq_cuda_bandwidth`. Its default matrix
covers 1, 4, 16, 64 and 256 MiB with three warm-ups and ten measured iterations
for pageable/pinned H2D and D2H. Concurrent bidirectional copies run only when
the CUDA device reports at least two asynchronous copy engines.

The executable writes stable CSV to standard output. It reports host-visible
enqueue-plus-synchronize timing separately from CUDA stream-event timing and
validates every transfer outside the timed interval.

KQ-01 evidence capture is orchestrated by:

```powershell
tools\capture-cuda-bandwidth.ps1
```

The script refuses to overwrite existing raw evidence, resolves
`nvidia-smi.exe` robustly and samples PCIe link state while the benchmark is
active. Canonical evidence belongs under the machine/date layout documented in
`bench/results/README.md`.
