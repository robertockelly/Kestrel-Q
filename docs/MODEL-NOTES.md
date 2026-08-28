# Qwen3.8-Flash-Next — Research Notes

Status: **initial fact sheet; implementation details still require verification against official artifacts and technical report.**

## Public high-level facts

The target model is Qwen3.8-Flash-Next, released by the Qwen team in August 2026.

Public architecture descriptions identify:

- multimodal MoE architecture;
- approximately 125B parameters in the main model;
- additional n-gram embedding parameters;
- approximately 6B parameters activated per token;
- hybrid Gated DeltaNet (GDN) and Qwen Sparse Attention (QSA);
- Gated Residual mechanism;
- n-gram embeddings;
- architecture intended as an early preview of Qwen4-era design.

## Important optimization hypothesis

The official architecture description explicitly indicates that the n-gram embedding table can be offloaded to host memory and overlapped with model computation through asynchronous prefetch.

This is directly relevant to Kestrel-Q's constrained-memory target and should be investigated early.

## Research checklist

Before implementation, derive from authoritative sources:

- exact config fields
- layer count
- hidden dimensions
- head dimensions/counts
- GDN/QSA layer pattern
- expert count
- experts selected per token
- router computation
- expert dimensions
- residual branch layout
- n-gram table shape and lookup algorithm
- positional encoding
- normalization equations
- activation functions
- vocabulary/tokenizer
- chat template
- multimodal components and whether text-only execution can exclude them
- tensor naming
- tensor shapes
- supported official weight formats
- precision of released weights
- reference generation settings

## Scope decision still required

The first runnable Kestrel-Q milestone should probably be **text-only inference**, even if the model is multimodal.

That decision must be recorded in an ADR before implementation.
