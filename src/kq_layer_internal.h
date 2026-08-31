#ifndef KQ_LAYER_INTERNAL_H
#define KQ_LAYER_INTERNAL_H

#include "kq_layer.h"

#define KQ_LAYER_CONFIG_MAGIC UINT32_C(0x4b514c43)
#define KQ_LAYER_STATE_MAGIC UINT32_C(0x4b514c53)
#define KQ_LAYER_GR_BINDING_COUNT 8U

typedef struct kq_layer_dimensions {
    uint32_t layer_id;
    uint32_t hidden_size;
    uint32_t branch_count;
    uint32_t gr_rank;
    float rms_epsilon;
    kq_layer_family family;
} kq_layer_dimensions;

struct kq_layer_config {
    uint32_t magic;
    const kq_model *model;
    kq_layer_dimensions dimensions;
    int owns_subconfigs;
    kq_gdn_config *gdn;
    kq_qsa_config *qsa;
    kq_moe_config *moe;
    kq_ple_config *ple;
    kq_ple_value_config *ple_value;
    const kq_semantic_tensor *gr_bindings[KQ_LAYER_GR_BINDING_COUNT];
    uint64_t gr_workspace_bytes;
};

struct kq_layer_state {
    uint32_t magic;
    const kq_layer_config *config;
    uint64_t position;
    uint64_t qsa_capacity;
    kq_gdn_state *gdn[2];
    kq_qsa_state *qsa[2];
    kq_ple_stream_state ple[2];
    kq_ple_value_state *ple_value[2];
    uint32_t active_slot;
    float *copy_a;
    float *copy_b;
    float *copy_c;
    uint64_t copy_a_count;
    uint64_t copy_b_count;
    uint64_t copy_c_count;
    uint64_t owned_bytes;
};

typedef struct kq_layer_state_summary {
    kq_layer_family family;
    uint64_t position;
    uint64_t qsa_length;
    uint64_t ple_address_position;
    uint64_t ple_value_position;
    uint64_t ple_address_integrity;
    uint64_t gdn_state_hash;
    uint64_t qsa_state_hash;
    uint64_t ple_value_state_hash;
    uint32_t active_slot;
    int gdn_initialized;
} kq_layer_state_summary;

/* Model-executor transaction support. The immediately previous slot remains
   intact until the next successful operation on this layer. */
kq_status kq_layer_state_rollback_last(
    kq_layer_state *state, uint64_t token_count,
    kq_diagnostic *diagnostic);
kq_status kq_layer_state_get_summary(
    const kq_layer_state *state, kq_layer_state_summary *summary,
    kq_diagnostic *diagnostic);

kq_status kq_layer_test_config_create(
    const kq_layer_dimensions *dimensions, kq_gdn_config *gdn,
    kq_qsa_config *qsa, kq_moe_config *moe, kq_ple_config *ple,
    kq_ple_value_config *ple_value, kq_layer_config **out_config,
    kq_diagnostic *diagnostic);

kq_status kq_layer_gr_read_f32(
    const kq_layer_config *config, const kq_layer_gr_weights_f32 *weights,
    const float *branches, float *normalized, float *read_gate,
    float *mixed, float *rank_workspace, float *write_gate,
    kq_diagnostic *diagnostic);
kq_status kq_layer_gr_read_quantized_f32(
    const kq_layer_config *config, kq_weight_provider *provider,
    uint32_t binding_base, const float *branches, float *normalized,
    float *read_gate, float *mixed, float *rank_workspace,
    float *write_gate, void *weight_scratch, uint64_t weight_scratch_bytes,
    kq_diagnostic *diagnostic);

void kq_layer_gr_write_f32(const kq_layer_config *config,
                           const float *branches, const float *block_output,
                           const float *write_gate, float *output);

#endif
