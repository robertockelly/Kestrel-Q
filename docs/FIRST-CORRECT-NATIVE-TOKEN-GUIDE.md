# FIRST-CORRECT-NATIVE-TOKEN-GUIDE.md

## M1 is correctness, not speed

The goal is one independently verified next token from the real quantized model.

A slow scalar run is acceptable.

## Use the shortest governed prompt

The scalar path rereads many weights for each prompt token. Use the shortest
existing canonical tokenizer-golden prompt that exercises the complete model.

Do not invent expected input IDs.

## Full-model oracle must be independent

Preferred:
- pinned llama.cpp full-model helper;
- explicit canonical token IDs;
- same GGUF;
- one short prompt;
- one next-token argmax.

Do not rely on the GGUF tokenizer because Task 2.3 already proved its semantic
metadata differs from the canonical tokenizer.

## Lower-level proofs still matter

M1 does not need to reproduce every internal activation from a second full
implementation because Tasks 2.6-2.11 already freeze operator/layer/provider
correctness independently.

Still compare full-model logits/top-N/final hidden where the independent oracle
can expose them safely.

## Keep state small

M1 needs only enough QSA capacity for the short prompt and one generated token,
not 262k context.

## No optimization work

Do not add caching, prefetch, SIMD or CUDA merely because the first scalar run
is slow. First establish the correct token.
