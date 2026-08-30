#ifndef KQ_GDN_INTERNAL_H
#define KQ_GDN_INTERNAL_H

#include "kq_gdn.h"

#define KQ_GDN_CONFIG_MAGIC UINT32_C(0x4b514743)
#define KQ_GDN_STATE_MAGIC UINT32_C(0x4b514753)
#define KQ_GDN_WEIGHT_ROLE_COUNT 9U

typedef struct kq_gdn_dimensions {
    uint32_t hidden_size;
    uint32_t key_head_count;
    uint32_t value_head_count;
    uint32_t key_head_dimension;
    uint32_t value_head_dimension;
    uint32_t conv_kernel_size;
    float rms_norm_epsilon;
    kq_gdn_activation_dtype activation_dtype;
} kq_gdn_dimensions;

typedef struct kq_gdn_semantic_source {
    const kq_model *model;
    uint32_t layer_id;
    kq_model_layer_type layer_type;
    kq_gdn_dimensions dimensions;
    const kq_semantic_tensor *tensors[KQ_GDN_WEIGHT_ROLE_COUNT];
} kq_gdn_semantic_source;

struct kq_gdn_config {
    uint32_t magic;
    const kq_model *model;
    uint32_t layer_id;
    kq_gdn_dimensions dimensions;
    uint32_t key_dimension;
    uint32_t value_dimension;
    uint32_t conv_channels;
    uint32_t key_value_head_repeat;
    uint64_t recurrent_elements;
    uint64_t conv_state_elements;
    uint64_t token_scratch_bytes;
    uint64_t scratch_bytes;
    const kq_semantic_tensor *tensors[KQ_GDN_WEIGHT_ROLE_COUNT];
};

struct kq_gdn_state {
    uint32_t magic;
    const kq_gdn_config *config;
    int initialized;
    void *conv_state;
    float *recurrent_state;
    uint64_t owned_bytes;
};

kq_status kq_gdn_config_create_from_source(
    const kq_gdn_semantic_source *source,
    int require_target_bindings,
    kq_gdn_config **out_config,
    kq_diagnostic *diagnostic);

kq_status kq_gdn_test_config_create(
    const kq_gdn_dimensions *dimensions,
    kq_gdn_config **out_config,
    kq_diagnostic *diagnostic);

kq_status kq_gdn_execute_f32(
    const kq_gdn_config *config,
    const kq_gdn_weights_f32 *weights,
    const float *hidden_states,
    uint64_t sequence_length,
    const uint8_t *padding_mask,
    float *output,
    uint64_t output_capacity,
    kq_gdn_state *state,
    void *scratch,
    uint64_t scratch_bytes,
    kq_gdn_checkpoint_observer observer,
    void *observer_user_data,
    kq_diagnostic *diagnostic);

#endif
