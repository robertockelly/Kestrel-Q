#include "kq_moe_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_internal.h"
#include "kq_tensor_view.h"

typedef struct kq_moe_role_spec {
    kq_semantic_component component;
    kq_semantic_role role;
    const char *suffix;
    kq_binding_relation relation;
    uint32_t canonical_rank;
    uint64_t canonical_dimensions[3];
    uint32_t physical_rank;
    uint64_t physical_dimensions[3];
    uint64_t allowed_type_mask;
    uint32_t routed_stack;
} kq_moe_role_spec;

#define KQ_MOE_TYPE_MASK(type_id) (UINT64_C(1) << (type_id))

static const kq_moe_role_spec kq_moe_specs[KQ_MOE_WEIGHT_ROLE_COUNT] = {
    {KQ_COMPONENT_MOE_ROUTER, KQ_ROLE_MOE_ROUTER, "router",
     KQ_BINDING_RENAMED_ONE_TO_ONE, 2U, {512U, 2560U, 0U},
     2U, {2560U, 512U, 0U}, KQ_MOE_TYPE_MASK(KQ_GGUF_TYPE_F32), 0U},
    {KQ_COMPONENT_ROUTED_EXPERT_STACK, KQ_ROLE_ROUTED_DOWN, "routed.down",
     KQ_BINDING_RENAMED_ONE_TO_ONE, 3U, {512U, 2560U, 640U},
     3U, {640U, 2560U, 512U},
     KQ_MOE_TYPE_MASK(KQ_GGUF_TYPE_Q5_1) |
         KQ_MOE_TYPE_MASK(KQ_GGUF_TYPE_Q8_0), 1U},
    {KQ_COMPONENT_ROUTED_EXPERT_STACK, KQ_ROLE_ROUTED_GATE_UP,
     "routed.gate_up", KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL,
     3U, {512U, 1280U, 2560U}, 3U, {2560U, 640U, 512U},
     KQ_MOE_TYPE_MASK(KQ_GGUF_TYPE_Q4_K) |
         KQ_MOE_TYPE_MASK(KQ_GGUF_TYPE_Q5_K), 1U},
    {KQ_COMPONENT_SHARED_EXPERT, KQ_ROLE_SHARED_DOWN, "shared.down",
     KQ_BINDING_RENAMED_ONE_TO_ONE, 2U, {2560U, 640U, 0U},
     2U, {640U, 2560U, 0U}, KQ_MOE_TYPE_MASK(KQ_GGUF_TYPE_Q8_0), 0U},
    {KQ_COMPONENT_SHARED_EXPERT, KQ_ROLE_SHARED_GATE, "shared.gate",
     KQ_BINDING_RENAMED_ONE_TO_ONE, 2U, {640U, 2560U, 0U},
     2U, {2560U, 640U, 0U}, KQ_MOE_TYPE_MASK(KQ_GGUF_TYPE_Q8_0), 0U},
    {KQ_COMPONENT_SHARED_EXPERT, KQ_ROLE_SHARED_UP, "shared.up",
     KQ_BINDING_RENAMED_ONE_TO_ONE, 2U, {640U, 2560U, 0U},
     2U, {2560U, 640U, 0U}, KQ_MOE_TYPE_MASK(KQ_GGUF_TYPE_Q8_0), 0U},
    {KQ_COMPONENT_SHARED_EXPERT, KQ_ROLE_SHARED_GATE_WEIGHT,
     "shared.gate_weight", KQ_BINDING_RENAMED_ONE_TO_ONE,
     2U, {1U, 2560U, 0U}, 1U, {2560U, 0U, 0U},
     KQ_MOE_TYPE_MASK(KQ_GGUF_TYPE_F32), 0U}
};

static kq_status kq_moe_fail(kq_diagnostic *diagnostic,
                             kq_status status,
                             const char *message) {
    kq_diagnostic_set(diagnostic, status, "%s", message);
    return status;
}

int kq_moe_u64_add(uint64_t left, uint64_t right, uint64_t *output) {
    if (output == NULL || UINT64_MAX - left < right) return 0;
    *output = left + right;
    return 1;
}

int kq_moe_u64_mul(uint64_t left, uint64_t right, uint64_t *output) {
    if (output == NULL || (left != 0U && right > UINT64_MAX / left)) return 0;
    *output = left * right;
    return 1;
}

static int kq_moe_accumulate(uint64_t *total, uint64_t value) {
    return kq_moe_u64_add(*total, value, total);
}

int kq_moe_config_valid(const kq_moe_config *config) {
    return config != NULL && config->magic == KQ_MOE_CONFIG_MAGIC;
}

static kq_status kq_moe_workspace_sizes(const kq_moe_dimensions *d,
                                         kq_moe_config *config,
                                         kq_diagnostic *diagnostic) {
    uint64_t floats;
    uint64_t bytes;
    uint64_t total = 0U;
    if (d == NULL || config == NULL || d->hidden_size == 0U ||
        d->expert_count == 0U || d->top_k == 0U ||
        d->top_k > d->expert_count || d->routed_intermediate_size == 0U ||
        d->shared_intermediate_size == 0U ||
        (d->activation_dtype != KQ_MOE_ACTIVATION_F32 &&
         d->activation_dtype != KQ_MOE_ACTIVATION_BF16)) {
        return kq_moe_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MOE,
                           "invalid MoE dimensions, top-k, or dtype");
    }
    if (!kq_moe_u64_mul(d->expert_count, 2U, &floats) ||
        !kq_moe_u64_mul(floats, sizeof(float), &bytes)) {
        return kq_moe_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "MoE router workspace overflows");
    }
    config->router_workspace_bytes = bytes;
    if (!kq_moe_u64_mul(d->top_k,
                        sizeof(float) + sizeof(uint32_t), &bytes)) {
        return kq_moe_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "MoE top-k workspace overflows");
    }
    config->top_k_workspace_bytes = bytes;
    if (!kq_moe_u64_mul(d->routed_intermediate_size, 3U, &floats) ||
        !kq_moe_u64_add(floats, (uint64_t)d->hidden_size * 2U, &floats) ||
        !kq_moe_u64_mul(floats, sizeof(float), &bytes)) {
        return kq_moe_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "MoE expert workspace overflows");
    }
    config->one_expert_workspace_bytes = bytes;
    if (!kq_moe_u64_mul(d->hidden_size, sizeof(float), &bytes)) {
        return kq_moe_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "MoE routed accumulation workspace overflows");
    }
    config->routed_accumulation_workspace_bytes = bytes;
    if (!kq_moe_u64_mul(d->shared_intermediate_size, 3U, &floats) ||
        !kq_moe_u64_add(floats, (uint64_t)d->hidden_size * 2U, &floats) ||
        !kq_moe_u64_mul(floats, sizeof(float), &bytes)) {
        return kq_moe_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "MoE shared workspace overflows");
    }
    config->shared_workspace_bytes = bytes;
    if (!kq_moe_accumulate(&total, config->router_workspace_bytes) ||
        !kq_moe_accumulate(&total, config->top_k_workspace_bytes) ||
        !kq_moe_accumulate(&total, config->one_expert_workspace_bytes) ||
        !kq_moe_accumulate(&total,
                           config->routed_accumulation_workspace_bytes) ||
        !kq_moe_accumulate(&total, config->shared_workspace_bytes)) {
        return kq_moe_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "MoE total workspace overflows");
    }
    config->scratch_bytes = total;
    return KQ_STATUS_OK;
}

static kq_status kq_moe_validate_target_tensor(
    const kq_semantic_tensor *semantic,
    const kq_moe_role_spec *spec,
    uint32_t layer_id,
    kq_model_layer_type layer_type,
    kq_diagnostic *diagnostic) {
    uint32_t dimension;
    uint32_t binding;
    if (semantic == NULL) {
        return kq_moe_fail(diagnostic, KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                           "required MoE semantic tensor is missing");
    }
    if (semantic->component != spec->component ||
        semantic->role != spec->role || semantic->relation != spec->relation ||
        semantic->runtime_scope != KQ_SCOPE_REQUIRED_INITIAL_TEXT ||
        semantic->layer_id != layer_id || semantic->layer_type != layer_type ||
        semantic->canonical_dtype != KQ_CANONICAL_DTYPE_BF16 ||
        semantic->canonical_rank != spec->canonical_rank) {
        return kq_moe_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MOE,
                           "MoE semantic identity, dtype, relation, or rank mismatch");
    }
    for (dimension = 0U; dimension < spec->canonical_rank; ++dimension) {
        if (semantic->canonical_dimensions[dimension] !=
            spec->canonical_dimensions[dimension]) {
            return kq_moe_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MOE,
                               "MoE canonical tensor shape mismatch");
        }
    }
    if (spec->routed_stack != 0U &&
        (semantic->canonical_expert_axis != 0U ||
         semantic->expert_count != 512U)) {
        return kq_moe_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MOE,
                           "MoE routed expert axis/count mismatch");
    }
    if (spec->role == KQ_ROLE_ROUTED_GATE_UP) {
        if (semantic->binding_count != 2U ||
            semantic->bindings[0].part_role != KQ_BINDING_PART_GATE ||
            semantic->bindings[0].part_index != 0U ||
            semantic->bindings[0].part_count != 2U ||
            semantic->bindings[1].part_role != KQ_BINDING_PART_UP ||
            semantic->bindings[1].part_index != 1U ||
            semantic->bindings[1].part_count != 2U) {
            return kq_moe_fail(diagnostic, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                               "MoE routed gate/up split is missing or out of order");
        }
    } else if (semantic->binding_count != 1U ||
               semantic->bindings[0].part_role != KQ_BINDING_PART_WHOLE ||
               semantic->bindings[0].part_index != 0U ||
               semantic->bindings[0].part_count != 1U) {
        return kq_moe_fail(diagnostic, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                           "MoE whole-tensor binding is malformed");
    }
    for (binding = 0U; binding < semantic->binding_count; ++binding) {
        const kq_tensor_binding *part = &semantic->bindings[binding];
        const kq_gguf_tensor *physical = part->physical;
        kq_quant_geometry geometry;
        if (physical == NULL || physical->type_id >= 64U ||
            (spec->allowed_type_mask &
             (UINT64_C(1) << physical->type_id)) == 0U ||
            physical->rank != spec->physical_rank) {
            return kq_moe_fail(diagnostic, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                               "MoE physical binding type or rank mismatch");
        }
        for (dimension = 0U; dimension < spec->physical_rank; ++dimension) {
            if (physical->dimensions[dimension] !=
                spec->physical_dimensions[dimension]) {
                return kq_moe_fail(diagnostic,
                                   KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                                   "MoE physical binding shape mismatch");
            }
        }
        if (spec->routed_stack != 0U && part->physical_expert_axis != 2U) {
            return kq_moe_fail(diagnostic, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                               "MoE physical expert axis mismatch");
        }
        if (kq_quant_geometry_for_type(physical->type_id, &geometry,
                                       diagnostic) != KQ_STATUS_OK) {
            return kq_moe_fail(diagnostic, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                               "MoE physical block geometry is unsupported");
        }
    }
    return KQ_STATUS_OK;
}

kq_status kq_moe_config_create_from_source(
    const kq_moe_semantic_source *source, int require_target_bindings,
    kq_moe_config **out_config, kq_diagnostic *diagnostic) {
    kq_moe_config *config;
    uint32_t index;
    kq_status status;
    kq_diagnostic_clear(diagnostic);
    if (source == NULL || out_config == NULL) {
        return kq_moe_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "MoE semantic source and output are required");
    }
    *out_config = NULL;
    if (source->layer_type != KQ_MODEL_LAYER_GDN &&
        source->layer_type != KQ_MODEL_LAYER_QSA) {
        return kq_moe_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MOE,
                           "MoE configuration requires a canonical text layer");
    }
    if (require_target_bindings) {
        const kq_moe_dimensions *d = &source->dimensions;
        if (source->model == NULL || source->layer_id >= 48U ||
            d->hidden_size != 2560U || d->expert_count != 512U ||
            d->top_k != 10U || d->routed_intermediate_size != 640U ||
            d->shared_intermediate_size != 640U ||
            (d->activation_dtype != KQ_MOE_ACTIVATION_F32 &&
             d->activation_dtype != KQ_MOE_ACTIVATION_BF16)) {
            return kq_moe_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MOE,
                               "target Qwen3.8 MoE topology mismatch");
        }
        for (index = 0U; index < KQ_MOE_WEIGHT_ROLE_COUNT; ++index) {
            status = kq_moe_validate_target_tensor(
                source->tensors[index], &kq_moe_specs[index],
                source->layer_id, source->layer_type, diagnostic);
            if (status != KQ_STATUS_OK) return status;
        }
    }
    config = (kq_moe_config *)calloc(1U, sizeof(*config));
    if (config == NULL) {
        return kq_moe_fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                           "could not allocate immutable MoE config");
    }
    config->dimensions = source->dimensions;
    status = kq_moe_workspace_sizes(&source->dimensions, config, diagnostic);
    if (status != KQ_STATUS_OK) {
        free(config);
        return status;
    }
    config->magic = KQ_MOE_CONFIG_MAGIC;
    config->model = source->model;
    config->layer_id = source->layer_id;
    config->layer_type = source->layer_type;
    for (index = 0U; index < KQ_MOE_WEIGHT_ROLE_COUNT; ++index) {
        config->tensors[index] = source->tensors[index];
    }
    *out_config = config;
    return KQ_STATUS_OK;
}

kq_status kq_moe_test_config_create(
    const kq_moe_dimensions *dimensions, kq_moe_config **out_config,
    kq_diagnostic *diagnostic) {
    kq_moe_semantic_source source;
    if (dimensions == NULL) {
        kq_diagnostic_clear(diagnostic);
        return kq_moe_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "test MoE dimensions are required");
    }
    memset(&source, 0, sizeof(source));
    source.layer_id = KQ_MODEL_NO_LAYER;
    source.layer_type = KQ_MODEL_LAYER_GDN;
    source.dimensions = *dimensions;
    return kq_moe_config_create_from_source(&source, 0, out_config, diagnostic);
}

static kq_status kq_moe_config_create_for_model(
    const kq_model *model, uint32_t layer_id, kq_moe_activation_dtype dtype,
    kq_moe_config **out_config, kq_diagnostic *diagnostic) {
    kq_moe_semantic_source source;
    char id[KQ_SEMANTIC_ID_CAPACITY];
    uint32_t index;
    int length;
    kq_diagnostic_clear(diagnostic);
    if (model == NULL || out_config == NULL) {
        return kq_moe_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "model and MoE config output are required");
    }
    *out_config = NULL;
    if (layer_id >= kq_model_layer_count(model)) {
        return kq_moe_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MOE,
                           "requested MoE layer ID is out of range");
    }
    memset(&source, 0, sizeof(source));
    source.model = model;
    source.layer_id = layer_id;
    source.layer_type = kq_model_layer_type_at(model, layer_id);
    source.dimensions.hidden_size = kq_model_hidden_size(model);
    source.dimensions.expert_count = kq_model_expert_count(model);
    source.dimensions.top_k = kq_model_expert_top_k(model);
    source.dimensions.routed_intermediate_size = 640U;
    source.dimensions.shared_intermediate_size = 640U;
    source.dimensions.activation_dtype = dtype;
    for (index = 0U; index < KQ_MOE_WEIGHT_ROLE_COUNT; ++index) {
        length = snprintf(id, sizeof(id), "layer.%02u.moe.%s",
                          layer_id, kq_moe_specs[index].suffix);
        if (length < 0 || (size_t)length >= sizeof(id)) {
            return kq_moe_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                               "MoE semantic ID formatting overflow");
        }
        source.tensors[index] = kq_model_find_semantic_tensor(model, id);
    }
    return kq_moe_config_create_from_source(&source, 1, out_config, diagnostic);
}

kq_status kq_moe_config_create(const kq_model *model, uint32_t layer_id,
                               kq_moe_config **out_config,
                               kq_diagnostic *diagnostic) {
    return kq_moe_config_create_for_model(
        model, layer_id, KQ_MOE_ACTIVATION_BF16, out_config, diagnostic);
}

kq_status kq_moe_config_create_reference_f32(
    const kq_model *model, uint32_t layer_id,
    kq_moe_config **out_config, kq_diagnostic *diagnostic) {
    return kq_moe_config_create_for_model(
        model, layer_id, KQ_MOE_ACTIVATION_F32, out_config, diagnostic);
}

void kq_moe_config_close(kq_moe_config *config) {
    if (config == NULL) return;
    config->magic = 0U;
    free(config);
}

uint32_t kq_moe_config_layer_id(const kq_moe_config *config) {
    return kq_moe_config_valid(config) ? config->layer_id : UINT32_MAX;
}
uint32_t kq_moe_config_hidden_size(const kq_moe_config *config) {
    return kq_moe_config_valid(config) ? config->dimensions.hidden_size : 0U;
}
uint32_t kq_moe_config_expert_count(const kq_moe_config *config) {
    return kq_moe_config_valid(config) ? config->dimensions.expert_count : 0U;
}
uint32_t kq_moe_config_top_k(const kq_moe_config *config) {
    return kq_moe_config_valid(config) ? config->dimensions.top_k : 0U;
}
uint32_t kq_moe_config_routed_intermediate_size(const kq_moe_config *config) {
    return kq_moe_config_valid(config) ?
        config->dimensions.routed_intermediate_size : 0U;
}
uint32_t kq_moe_config_shared_intermediate_size(const kq_moe_config *config) {
    return kq_moe_config_valid(config) ?
        config->dimensions.shared_intermediate_size : 0U;
}
kq_moe_activation_dtype kq_moe_config_activation_dtype(
    const kq_moe_config *config) {
    return kq_moe_config_valid(config) ? config->dimensions.activation_dtype :
        KQ_MOE_ACTIVATION_F32;
}
uint64_t kq_moe_config_owned_bytes(const kq_moe_config *config) {
    return kq_moe_config_valid(config) ? sizeof(*config) : 0U;
}
uint64_t kq_moe_config_router_workspace_bytes(const kq_moe_config *config) {
    return kq_moe_config_valid(config) ? config->router_workspace_bytes : 0U;
}
uint64_t kq_moe_config_top_k_workspace_bytes(const kq_moe_config *config) {
    return kq_moe_config_valid(config) ? config->top_k_workspace_bytes : 0U;
}
uint64_t kq_moe_config_one_expert_workspace_bytes(
    const kq_moe_config *config) {
    return kq_moe_config_valid(config) ?
        config->one_expert_workspace_bytes : 0U;
}
uint64_t kq_moe_config_routed_accumulation_workspace_bytes(
    const kq_moe_config *config) {
    return kq_moe_config_valid(config) ?
        config->routed_accumulation_workspace_bytes : 0U;
}
uint64_t kq_moe_config_shared_workspace_bytes(const kq_moe_config *config) {
    return kq_moe_config_valid(config) ? config->shared_workspace_bytes : 0U;
}
uint64_t kq_moe_config_scratch_bytes(const kq_moe_config *config) {
    return kq_moe_config_valid(config) ? config->scratch_bytes : 0U;
}
