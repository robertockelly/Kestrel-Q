#ifndef KQ_PLE_VALUE_INTERNAL_H
#define KQ_PLE_VALUE_INTERNAL_H

#include "kq_ple_value.h"
#include "kq_weight_provider.h"

#define KQ_PLE_VALUE_CONFIG_MAGIC UINT32_C(0x4b515643)
#define KQ_PLE_VALUE_STATE_MAGIC UINT32_C(0x4b515653)
#define KQ_PLE_VALUE_DENSE_ROLE_COUNT 6U

typedef struct kq_ple_value_dimensions {
    uint32_t hidden_size;
    uint32_t residual_branches;
    uint32_t heads_per_order;
    uint32_t head_count;
    uint32_t row_width;
    uint32_t logical_member_count;
    uint64_t member_rows;
    uint32_t convolution_kernel;
    uint32_t convolution_dilation;
    uint32_t history_length;
    float rms_epsilon;
    kq_ple_value_activation_dtype activation_dtype;
} kq_ple_value_dimensions;

typedef struct kq_ple_value_semantic_source {
    const kq_model *model;
    const kq_ple_config *address_config;
    uint32_t hidden_size;
    kq_model_layer_type layer_type;
    kq_ple_config_info address_info;
    const kq_semantic_tensor *dense[KQ_PLE_VALUE_DENSE_ROLE_COUNT];
    const kq_semantic_tensor *tables[KQ_PLE_LOGICAL_MEMBER_COUNT];
} kq_ple_value_semantic_source;

struct kq_ple_value_config {
    uint32_t magic;
    const kq_model *model;
    const kq_ple_config *address_config;
    uint32_t layer_id;
    kq_ple_value_dimensions dimensions;
    uint64_t embedding_width;
    uint64_t branch_width;
    uint64_t state_elements;
    uint64_t state_bytes;
    uint64_t semantic_state_bytes;
    uint64_t scratch_bytes;
    const kq_semantic_tensor *dense[KQ_PLE_VALUE_DENSE_ROLE_COUNT];
    const kq_semantic_tensor *tables[KQ_PLE_LOGICAL_MEMBER_COUNT];
};

struct kq_ple_value_state {
    uint32_t magic;
    const kq_ple_value_config *config;
    uint64_t position;
    uint64_t history_count;
    float *history;
};

kq_status kq_ple_value_test_config_create(
    const kq_ple_value_dimensions *dimensions,
    kq_ple_value_config **out_config, kq_diagnostic *diagnostic);
kq_status kq_ple_value_config_create_from_source(
    const kq_ple_value_semantic_source *source,
    kq_ple_value_activation_dtype dtype,
    kq_ple_value_config **out_config, kq_diagnostic *diagnostic);
int kq_ple_value_config_valid(const kq_ple_value_config *config);
int kq_ple_value_u64_add(uint64_t a, uint64_t b, uint64_t *out);
int kq_ple_value_u64_mul(uint64_t a, uint64_t b, uint64_t *out);

kq_status kq_ple_value_execute_quantized(
    const kq_ple_value_config *config, kq_ple_value_state *state,
    kq_weight_provider *weight_provider,
    const kq_ple_value_lookup_provider *lookup_provider,
    const float *hidden_states, uint64_t token_count,
    const kq_ple_address_intent *intents, uint64_t intent_count,
    float *output, uint64_t output_capacity,
    void *scratch, uint64_t scratch_bytes,
    void *weight_scratch, uint64_t weight_scratch_bytes,
    kq_ple_value_checkpoint_observer observer, void *observer_user_data,
    kq_ple_value_run_metrics *metrics, kq_diagnostic *diagnostic);

#endif
