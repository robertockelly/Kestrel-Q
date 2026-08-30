#ifndef KQ_MOE_INTERNAL_H
#define KQ_MOE_INTERNAL_H

#include "kq_moe.h"
#include "kq_weight_provider.h"

#define KQ_MOE_CONFIG_MAGIC UINT32_C(0x4b514d43)
#define KQ_MOE_WEIGHT_ROLE_COUNT 7U

typedef struct kq_moe_dimensions {
    uint32_t hidden_size;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t routed_intermediate_size;
    uint32_t shared_intermediate_size;
    kq_moe_activation_dtype activation_dtype;
} kq_moe_dimensions;

typedef struct kq_moe_semantic_source {
    const kq_model *model;
    uint32_t layer_id;
    kq_model_layer_type layer_type;
    kq_moe_dimensions dimensions;
    const kq_semantic_tensor *tensors[KQ_MOE_WEIGHT_ROLE_COUNT];
} kq_moe_semantic_source;

struct kq_moe_config {
    uint32_t magic;
    const kq_model *model;
    uint32_t layer_id;
    kq_model_layer_type layer_type;
    kq_moe_dimensions dimensions;
    uint64_t router_workspace_bytes;
    uint64_t top_k_workspace_bytes;
    uint64_t one_expert_workspace_bytes;
    uint64_t routed_accumulation_workspace_bytes;
    uint64_t shared_workspace_bytes;
    uint64_t scratch_bytes;
    const kq_semantic_tensor *tensors[KQ_MOE_WEIGHT_ROLE_COUNT];
};

kq_status kq_moe_config_create_from_source(
    const kq_moe_semantic_source *source,
    int require_target_bindings,
    kq_moe_config **out_config,
    kq_diagnostic *diagnostic);
kq_status kq_moe_test_config_create(
    const kq_moe_dimensions *dimensions,
    kq_moe_config **out_config,
    kq_diagnostic *diagnostic);

int kq_moe_config_valid(const kq_moe_config *config);
int kq_moe_u64_add(uint64_t left, uint64_t right, uint64_t *output);
int kq_moe_u64_mul(uint64_t left, uint64_t right, uint64_t *output);

kq_status kq_moe_execute_quantized(
    const kq_moe_config *config, kq_weight_provider *provider,
    const float *hidden_states, uint64_t token_count,
    float *output, uint64_t output_capacity,
    void *scratch, uint64_t scratch_bytes,
    void *weight_scratch, uint64_t weight_scratch_bytes,
    kq_moe_route_observer route_observer,
    kq_moe_checkpoint_observer checkpoint_observer,
    void *observer_user_data, kq_diagnostic *diagnostic);

#endif
