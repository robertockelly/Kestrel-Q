# ADR 0001 — Narrow single-model runtime

Status: Accepted

## Context

Generic inference runtimes carry abstraction costs and must support many architectures. Kestrel-Q is motivated by the opposite question: what becomes possible when the engine is allowed to know the model deeply?

## Decision

The first major version targets Qwen3.8-Flash-Next only.

Generic model support is explicitly out of scope.

## Consequences

Positive:

- simpler execution path;
- aggressive model-specific optimization;
- smaller codebase;
- easier reasoning about memory placement.

Negative:

- limited compatibility;
- architecture changes in future Qwen releases may require substantial work;
- less reuse across unrelated models.
