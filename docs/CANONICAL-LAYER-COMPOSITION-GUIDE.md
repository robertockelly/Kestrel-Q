# CANONICAL-LAYER-COMPOSITION-GUIDE.md

Status: **TASK 2.10 COMPLETE / PASS**

## Purpose

Task 2.10 is the first point where isolated model operators become one complete
canonical transformer layer.

It is still a reference integration task, not a full model executor.

## The canonical layer owns composition

Do not infer layer order from standalone APIs.

The pinned decoder-layer implementation determines:
- norms;
- GR/residual reads/writes;
- PLE insertion;
- GDN/QSA call order;
- MoE call order;
- final layer output.

## GR is not automatically persistent state

Epic 1 indicates GR branches are forward activations, not context cache.
Revalidate this and avoid turning them into per-token persistent state.

## Three layer families

Validate separately:
- ordinary GDN;
- QSA;
- PLE-enabled GDN.

A single synthetic layer case is not sufficient.

## Transactional state matters

A layer call may update GDN/QSA/PLE states. A failure must not leave some
suboperators advanced and others unchanged.

Prefer a tested transactional policy.

## Still no full model

Embedding, 48-layer iteration, final norm, LM head and logits belong to the next
model-executor milestone.
