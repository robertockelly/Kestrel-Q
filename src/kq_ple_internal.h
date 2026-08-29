#ifndef KQ_PLE_INTERNAL_H
#define KQ_PLE_INTERNAL_H

#include "kq_ple.h"

typedef struct kq_ple_compatibility_descriptor {
    uint32_t hidden_size;
    uint32_t vocabulary_size;
    uint32_t context_length;
    uint32_t layer_count;
    uint32_t gdn_layer_count;
    uint32_t qsa_layer_count;
    uint32_t ple_layer_id;
    kq_model_layer_type ple_layer_type;
    uint32_t ngram_size;
    uint32_t heads_per_order;
    uint32_t eos_token_id;
    uint32_t logical_member_count;
    uint64_t member_rows;
    uint32_t table_width;
    uint64_t ple_table_semantic_count;
    uint64_t ple_dense_semantic_count;
    uint64_t ple_metadata_semantic_count;
    int table_semantics_valid;
    int metadata_semantics_valid;
    uint64_t payload_bytes_accessed;
    uint64_t multipliers[KQ_PLE_NGRAM_SIZE];
    uint64_t head_offsets[KQ_PLE_HEAD_COUNT];
    uint64_t head_vocab_sizes[KQ_PLE_HEAD_COUNT];
} kq_ple_compatibility_descriptor;

kq_status kq_ple_config_open_from_descriptor_for_test(
    const kq_ple_compatibility_descriptor *descriptor,
    kq_ple_config **out_config,
    kq_diagnostic *diagnostic);

#endif
