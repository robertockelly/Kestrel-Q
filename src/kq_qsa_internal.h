#ifndef KQ_QSA_INTERNAL_H
#define KQ_QSA_INTERNAL_H

#include "kq_qsa.h"
#include "kq_weight_provider.h"

#define KQ_QSA_CONFIG_MAGIC UINT32_C(0x4b515143)
#define KQ_QSA_STATE_MAGIC UINT32_C(0x4b515153)
#define KQ_QSA_WEIGHT_ROLE_COUNT 9U

typedef struct kq_qsa_dimensions {
    uint32_t hidden_size;
    uint32_t query_head_count;
    uint32_t key_value_head_count;
    uint32_t head_dimension;
    uint32_t index_query_head_count;
    uint32_t index_head_dimension;
    uint32_t block_size;
    uint32_t token_budget;
    uint32_t max_context;
    uint32_t rotary_dimension;
    float rms_norm_epsilon;
    float rope_theta;
    kq_qsa_activation_dtype activation_dtype;
} kq_qsa_dimensions;

typedef struct kq_qsa_semantic_source {
    const kq_model *model;
    uint32_t layer_id;
    kq_model_layer_type layer_type;
    kq_qsa_dimensions dimensions;
    const kq_semantic_tensor *tensors[KQ_QSA_WEIGHT_ROLE_COUNT];
} kq_qsa_semantic_source;

struct kq_qsa_config {
    uint32_t magic;
    const kq_model *model;
    uint32_t layer_id;
    kq_qsa_dimensions dimensions;
    uint32_t key_value_repeat;
    uint32_t block_topk;
    uint32_t selected_token_capacity;
    uint64_t key_values_per_token;
    uint64_t raw_index_values_per_token;
    uint64_t semantic_state_bytes_per_token;
    const kq_semantic_tensor *tensors[KQ_QSA_WEIGHT_ROLE_COUNT];
};

struct kq_qsa_state {
    uint32_t magic;
    const kq_qsa_config *config;
    uint64_t capacity;
    uint64_t length;
    void *key_cache;
    void *value_cache;
    void *raw_index_key_cache;
    uint64_t owned_bytes;
};

kq_status kq_qsa_config_create_from_source(
    const kq_qsa_semantic_source *source,
    int require_target_bindings,
    kq_qsa_config **out_config,
    kq_diagnostic *diagnostic);
kq_status kq_qsa_test_config_create(
    const kq_qsa_dimensions *dimensions,
    kq_qsa_config **out_config,
    kq_diagnostic *diagnostic);
int kq_qsa_config_valid(const kq_qsa_config *config);
int kq_qsa_state_valid(const kq_qsa_state *state);
int kq_qsa_u64_add(uint64_t left, uint64_t right, uint64_t *output);
int kq_qsa_u64_mul(uint64_t left, uint64_t right, uint64_t *output);
kq_status kq_qsa_calculate_scratch_bytes(
    const kq_qsa_config *config,
    const kq_qsa_state *state,
    uint64_t sequence_length,
    uint64_t *scratch_bytes,
    kq_diagnostic *diagnostic);
kq_status kq_qsa_execute_f32(
    const kq_qsa_config *config,
    const kq_qsa_weights_f32 *weights,
    const float *hidden_states,
    uint64_t sequence_length,
    float *output,
    uint64_t output_capacity,
    kq_qsa_state *state,
    void *scratch,
    uint64_t scratch_bytes,
    kq_qsa_selection_observer selection_observer,
    kq_qsa_checkpoint_observer checkpoint_observer,
    void *observer_user_data,
    kq_diagnostic *diagnostic);

kq_status kq_qsa_execute_quantized(
    const kq_qsa_config *config, kq_weight_provider *provider,
    const float *hidden_states, uint64_t sequence_length,
    float *output, uint64_t output_capacity, kq_qsa_state *state,
    void *scratch, uint64_t scratch_bytes,
    void *weight_scratch, uint64_t weight_scratch_bytes,
    kq_qsa_selection_observer selection_observer,
    kq_qsa_checkpoint_observer checkpoint_observer,
    void *observer_user_data, kq_diagnostic *diagnostic);

#endif
