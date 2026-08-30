#ifndef KQ_MOE_H
#define KQ_MOE_H

#include <stdint.h>

#include "kq_model.h"
#include "kq_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KQ_MOE_MAX_CHECKPOINT_RANK 2U
#define KQ_MOE_NO_EXPERT UINT32_MAX

typedef struct kq_moe_config kq_moe_config;

typedef enum kq_moe_activation_dtype {
    KQ_MOE_ACTIVATION_F32 = 0,
    KQ_MOE_ACTIVATION_BF16
} kq_moe_activation_dtype;

typedef enum kq_moe_checkpoint_kind {
    KQ_MOE_CHECKPOINT_ROUTER_LOGITS = 0,
    KQ_MOE_CHECKPOINT_ROUTER_PROBABILITIES,
    KQ_MOE_CHECKPOINT_SELECTED_WEIGHTS,
    KQ_MOE_CHECKPOINT_ROUTED_GATE,
    KQ_MOE_CHECKPOINT_ROUTED_UP,
    KQ_MOE_CHECKPOINT_ROUTED_ACTIVATED,
    KQ_MOE_CHECKPOINT_ROUTED_EXPERT_OUTPUT,
    KQ_MOE_CHECKPOINT_ROUTED_WEIGHTED_OUTPUT,
    KQ_MOE_CHECKPOINT_ROUTED_WEIGHTED_SUM,
    KQ_MOE_CHECKPOINT_SHARED_GATE_PROJECTION,
    KQ_MOE_CHECKPOINT_SHARED_UP_PROJECTION,
    KQ_MOE_CHECKPOINT_SHARED_ACTIVATED,
    KQ_MOE_CHECKPOINT_SHARED_OUTPUT,
    KQ_MOE_CHECKPOINT_SHARED_SCALE_LOGIT,
    KQ_MOE_CHECKPOINT_SHARED_SCALE,
    KQ_MOE_CHECKPOINT_GATED_SHARED_OUTPUT,
    KQ_MOE_CHECKPOINT_OPERATOR_OUTPUT
} kq_moe_checkpoint_kind;

typedef struct kq_moe_checkpoint {
    kq_moe_checkpoint_kind kind;
    uint64_t token_index;
    uint32_t expert_id;
    uint32_t top_k_position;
    uint32_t rank;
    uint64_t dimensions[KQ_MOE_MAX_CHECKPOINT_RANK];
    const float *values;
    uint64_t value_count;
} kq_moe_checkpoint;

typedef void (*kq_moe_checkpoint_observer)(
    const kq_moe_checkpoint *checkpoint,
    void *user_data);

typedef struct kq_moe_route {
    uint64_t token_index;
    const float *router_logits;
    const float *router_probabilities;
    uint64_t expert_count;
    const uint32_t *selected_expert_ids;
    const float *selected_weights;
    uint32_t selected_count;
} kq_moe_route;

typedef void (*kq_moe_route_observer)(
    const kq_moe_route *route,
    void *user_data);

typedef struct kq_moe_weights_f32 {
    const float *router;
    uint64_t router_count;
    const float *routed_gate;
    uint64_t routed_gate_count;
    const float *routed_up;
    uint64_t routed_up_count;
    const float *routed_down;
    uint64_t routed_down_count;
    const float *shared_gate;
    uint64_t shared_gate_count;
    const float *shared_up;
    uint64_t shared_up_count;
    const float *shared_down;
    uint64_t shared_down_count;
    const float *shared_gate_weight;
    uint64_t shared_gate_weight_count;
} kq_moe_weights_f32;

typedef struct kq_moe_routed_expert_weights_f32 {
    const float *gate;
    uint64_t gate_count;
    const float *up;
    uint64_t up_count;
    const float *down;
    uint64_t down_count;
} kq_moe_routed_expert_weights_f32;

kq_status kq_moe_config_create(const kq_model *model,
                               uint32_t layer_id,
                               kq_moe_config **out_config,
                               kq_diagnostic *diagnostic);
kq_status kq_moe_config_create_reference_f32(
    const kq_model *model,
    uint32_t layer_id,
    kq_moe_config **out_config,
    kq_diagnostic *diagnostic);
void kq_moe_config_close(kq_moe_config *config);

uint32_t kq_moe_config_layer_id(const kq_moe_config *config);
uint32_t kq_moe_config_hidden_size(const kq_moe_config *config);
uint32_t kq_moe_config_expert_count(const kq_moe_config *config);
uint32_t kq_moe_config_top_k(const kq_moe_config *config);
uint32_t kq_moe_config_routed_intermediate_size(const kq_moe_config *config);
uint32_t kq_moe_config_shared_intermediate_size(const kq_moe_config *config);
kq_moe_activation_dtype kq_moe_config_activation_dtype(
    const kq_moe_config *config);
uint64_t kq_moe_config_owned_bytes(const kq_moe_config *config);
uint64_t kq_moe_config_router_workspace_bytes(const kq_moe_config *config);
uint64_t kq_moe_config_top_k_workspace_bytes(const kq_moe_config *config);
uint64_t kq_moe_config_one_expert_workspace_bytes(const kq_moe_config *config);
uint64_t kq_moe_config_routed_accumulation_workspace_bytes(
    const kq_moe_config *config);
uint64_t kq_moe_config_shared_workspace_bytes(const kq_moe_config *config);
uint64_t kq_moe_config_scratch_bytes(const kq_moe_config *config);

kq_status kq_moe_route_f32(
    const kq_moe_config *config,
    const float *router_weight,
    uint64_t router_weight_count,
    const float *hidden_token,
    float *router_logits,
    uint64_t router_logits_capacity,
    float *router_probabilities,
    uint64_t router_probabilities_capacity,
    uint32_t *selected_expert_ids,
    uint64_t selected_ids_capacity,
    float *selected_weights,
    uint64_t selected_weights_capacity,
    kq_diagnostic *diagnostic);

kq_status kq_moe_execute_routed_expert_f32(
    const kq_moe_config *config,
    const kq_moe_routed_expert_weights_f32 *weights,
    uint32_t expert_id,
    const float *hidden_token,
    float *output_token,
    uint64_t output_capacity,
    void *scratch,
    uint64_t scratch_bytes,
    kq_diagnostic *diagnostic);

kq_status kq_moe_execute_f32(
    const kq_moe_config *config,
    const kq_moe_weights_f32 *weights,
    const float *hidden_states,
    uint64_t token_count,
    float *output,
    uint64_t output_capacity,
    void *scratch,
    uint64_t scratch_bytes,
    kq_moe_route_observer route_observer,
    kq_moe_checkpoint_observer checkpoint_observer,
    void *observer_user_data,
    kq_diagnostic *diagnostic);

const char *kq_moe_checkpoint_kind_name(kq_moe_checkpoint_kind kind);

#ifdef __cplusplus
}
#endif

#endif
