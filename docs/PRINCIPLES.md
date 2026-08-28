# Project Principles

1. **Narrow by design** — one model before a framework.
2. **Windows is not a secondary port** — it is the first platform.
3. **Constrained hardware is the target** — optimizations must be meaningful on it.
4. **Correctness is measurable** — reference vectors precede aggressive optimization.
5. **Memory movement is a first-class cost** — PCIe and NVMe are part of the execution model.
6. **The model architecture should shape the runtime** — not the reverse.
7. **Keep the core inspectable** — contributors should be able to follow a token through the engine.
8. **Benchmarks are reproducible artifacts** — not anecdotes.
9. **Fail closed** — unsupported artifacts must not appear to work.
10. **Community over ownership** — design decisions should be documented and reviewable.
