#ifndef KQ_TOKENIZER_INTERNAL_H
#define KQ_TOKENIZER_INTERNAL_H

#include "kq_tokenizer.h"

typedef enum kq_tokenizer_test_mutation_kind {
    KQ_TOKENIZER_TEST_MUTATION_NONE = 0,
    KQ_TOKENIZER_TEST_MUTATION_TOKEN_STRING,
    KQ_TOKENIZER_TEST_MUTATION_MERGE_STRING,
    KQ_TOKENIZER_TEST_MUTATION_TOKEN_TYPE,
    KQ_TOKENIZER_TEST_MUTATION_TOKEN_COUNT,
    KQ_TOKENIZER_TEST_MUTATION_MERGE_COUNT
} kq_tokenizer_test_mutation_kind;

typedef struct kq_tokenizer_test_mutation {
    kq_tokenizer_test_mutation_kind kind;
    uint64_t index;
    kq_string_view replacement_string;
    int32_t replacement_i32;
} kq_tokenizer_test_mutation;

kq_status kq_tokenizer_open_from_gguf_for_test(
    const kq_gguf *gguf,
    const kq_tokenizer_test_mutation *mutation,
    kq_tokenizer **out_tokenizer,
    kq_diagnostic *diagnostic);

#endif
