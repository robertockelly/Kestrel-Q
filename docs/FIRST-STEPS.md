# First Steps

Recommended order after creating the GitHub repository.

1. Rename the project if the final public name differs from Kestrel-Q.
2. Fill in `docs/HARDWARE-TARGET.md`.
3. Build the scaffold with CMake and run the smoke test.
4. Create the first Git tag only after the repository identity is final.
5. Begin **Epic 1 — model research** before writing inference kernels.
6. Record every architectural conclusion as documentation or an ADR.
7. Generate authoritative reference vectors before GPU optimization.
8. Only then start the loader and CPU correctness path.

## Suggested first technical milestone

Create a program such as:

```text
kq-inspect <model-path>
```

It should eventually print:

- model identity
- architecture/config values
- tensor count
- tensor names
- shapes
- dtypes
- byte sizes
- mapped-file offsets
- totals grouped by tensor family

This gives the project its first concrete, testable artifact without prematurely attempting inference.
