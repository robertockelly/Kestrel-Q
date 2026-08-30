#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "kq_layer_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_gdn_internal.h"
#include "kq_internal.h"
#include "kq_moe_internal.h"
#include "kq_ple_value_internal.h"
#include "kq_qsa_internal.h"

#define KQ_LAYER_TARGET_HIDDEN 2560U
#define KQ_LAYER_TARGET_BRANCHES 4U
#define KQ_LAYER_TARGET_RANK 320U
#define KQ_LAYER_TARGET_COUNT 48U
#define KQ_LAYER_TARGET_PLE_LAYER 1U

static int kq_layer_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == NULL || UINT64_MAX - a < b) return 0;
    *out = a + b;
    return 1;
}

static int kq_layer_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == NULL || (a != 0U && b > UINT64_MAX / a)) return 0;
    *out = a * b;
    return 1;
}

static int kq_layer_config_valid(const kq_layer_config *config) {
    return config != NULL && config->magic == KQ_LAYER_CONFIG_MAGIC &&
           config->dimensions.hidden_size != 0U &&
           config->dimensions.branch_count != 0U &&
           config->dimensions.gr_rank != 0U &&
           config->moe != NULL &&
           ((config->dimensions.family == KQ_LAYER_FAMILY_QSA &&
             config->qsa != NULL && config->gdn == NULL) ||
            ((config->dimensions.family == KQ_LAYER_FAMILY_GDN ||
              config->dimensions.family == KQ_LAYER_FAMILY_PLE_GDN) &&
             config->gdn != NULL && config->qsa == NULL)) &&
           ((config->dimensions.family == KQ_LAYER_FAMILY_PLE_GDN) ==
            (config->ple != NULL && config->ple_value != NULL));
}

static int kq_layer_state_valid(const kq_layer_state *state) {
    return state != NULL && state->magic == KQ_LAYER_STATE_MAGIC &&
           kq_layer_config_valid(state->config) && state->active_slot < 2U;
}

static void kq_layer_emit(kq_layer_checkpoint_observer observer, void *user,
                          kq_layer_checkpoint_kind kind, uint64_t token,
                          uint32_t rank, uint64_t d0, uint64_t d1,
                          const float *values, uint64_t count) {
    kq_layer_checkpoint checkpoint;
    if (observer == NULL) return;
    memset(&checkpoint, 0, sizeof(checkpoint));
    checkpoint.kind = kind;
    checkpoint.token_index = token;
    checkpoint.rank = rank;
    checkpoint.dimensions[0] = d0;
    checkpoint.dimensions[1] = d1;
    checkpoint.values = values;
    checkpoint.value_count = count;
    observer(&checkpoint, user);
}

static kq_status kq_layer_validate_gr_binding(
    const kq_model *model, uint32_t layer_id, const char *suffix,
    kq_semantic_role role, uint32_t rank, uint64_t d0, uint64_t d1,
    const kq_semantic_tensor **out, kq_diagnostic *diagnostic) {
    char id[KQ_SEMANTIC_ID_CAPACITY];
    const kq_semantic_tensor *tensor;
    (void)snprintf(id, sizeof(id), "layer.%02u.gr.%s", layer_id, suffix);
    tensor = kq_model_find_semantic_tensor(model, id);
    if (tensor == NULL || tensor->component != KQ_COMPONENT_GATED_RESIDUAL ||
        tensor->role != role || tensor->layer_id != layer_id ||
        tensor->canonical_rank != rank || tensor->canonical_dimensions[0] != d0 ||
        (rank > 1U && tensor->canonical_dimensions[1] != d1) ||
        tensor->binding_count != 1U) {
        kq_diagnostic_set(diagnostic, KQ_STATUS_INCOMPATIBLE_LAYER,
                          "layer %u GR binding %s has incompatible semantics",
                          layer_id, suffix);
        return KQ_STATUS_INCOMPATIBLE_LAYER;
    }
    *out = tensor;
    return KQ_STATUS_OK;
}

static kq_status kq_layer_validate_gr_bindings(kq_layer_config *config,
                                               kq_diagnostic *diagnostic) {
    static const char *suffixes[KQ_LAYER_GR_BINDING_COUNT] = {
        "attn.inject", "attn.norm", "attn.down", "attn.up",
        "moe.inject", "moe.norm", "moe.down", "moe.up"};
    static const kq_semantic_role roles[KQ_LAYER_GR_BINDING_COUNT] = {
        KQ_ROLE_HC_INJECT, KQ_ROLE_HC_NORM, KQ_ROLE_HC_DOWN, KQ_ROLE_HC_UP,
        KQ_ROLE_HC_INJECT, KQ_ROLE_HC_NORM, KQ_ROLE_HC_DOWN, KQ_ROLE_HC_UP};
    uint32_t i;
    uint64_t width = (uint64_t)config->dimensions.hidden_size *
                     config->dimensions.branch_count;
    for (i = 0U; i < KQ_LAYER_GR_BINDING_COUNT; ++i) {
        uint32_t rank = (i % 4U) == 1U ? 1U : 2U;
        uint64_t d0 = (i % 4U) == 0U ? config->dimensions.branch_count :
                      (i % 4U) == 1U ? width :
                      (i % 4U) == 2U ? config->dimensions.gr_rank : width;
        uint64_t d1 = (i % 4U) == 0U ? width :
                      (i % 4U) == 2U ? width : config->dimensions.gr_rank;
        kq_status status = kq_layer_validate_gr_binding(
            config->model, config->dimensions.layer_id, suffixes[i], roles[i],
            rank, d0, d1, &config->gr_bindings[i], diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    return KQ_STATUS_OK;
}

static kq_status kq_layer_config_create_common(
    const kq_model *model, uint32_t layer_id, int reference_f32,
    kq_layer_config **out_config, kq_diagnostic *diagnostic) {
    kq_layer_config *config;
    kq_model_layer_type type;
    kq_status status;
    if (model == NULL || out_config == NULL) return KQ_STATUS_INVALID_ARGUMENT;
    *out_config = NULL;
    if (layer_id >= kq_model_layer_count(model) ||
        kq_model_layer_count(model) != KQ_LAYER_TARGET_COUNT ||
        kq_model_hidden_size(model) != KQ_LAYER_TARGET_HIDDEN) {
        kq_diagnostic_set(diagnostic, KQ_STATUS_INCOMPATIBLE_LAYER,
                          "target model/layer identity is incompatible");
        return KQ_STATUS_INCOMPATIBLE_LAYER;
    }
    config = (kq_layer_config *)calloc(1U, sizeof(*config));
    if (config == NULL) return KQ_STATUS_OUT_OF_MEMORY;
    config->magic = KQ_LAYER_CONFIG_MAGIC;
    config->model = model;
    config->dimensions.layer_id = layer_id;
    config->dimensions.hidden_size = KQ_LAYER_TARGET_HIDDEN;
    config->dimensions.branch_count = KQ_LAYER_TARGET_BRANCHES;
    config->dimensions.gr_rank = KQ_LAYER_TARGET_RANK;
    config->dimensions.rms_epsilon = 1.0e-6f;
    config->owns_subconfigs = 1;
    type = kq_model_layer_type_at(model, layer_id);
    config->dimensions.family = type == KQ_MODEL_LAYER_QSA
        ? KQ_LAYER_FAMILY_QSA
        : layer_id == KQ_LAYER_TARGET_PLE_LAYER
            ? KQ_LAYER_FAMILY_PLE_GDN : KQ_LAYER_FAMILY_GDN;
    status = kq_layer_validate_gr_bindings(config, diagnostic);
    if (status != KQ_STATUS_OK) goto fail;
    if (type == KQ_MODEL_LAYER_QSA) {
        status = reference_f32
            ? kq_qsa_config_create_reference_f32(model, layer_id, &config->qsa,
                                                  diagnostic)
            : kq_qsa_config_create(model, layer_id, &config->qsa, diagnostic);
    } else {
        status = reference_f32
            ? kq_gdn_config_create_reference_f32(model, layer_id, &config->gdn,
                                                  diagnostic)
            : kq_gdn_config_create(model, layer_id, &config->gdn, diagnostic);
    }
    if (status != KQ_STATUS_OK) goto fail;
    status = reference_f32
        ? kq_moe_config_create_reference_f32(model, layer_id, &config->moe,
                                              diagnostic)
        : kq_moe_config_create(model, layer_id, &config->moe, diagnostic);
    if (status != KQ_STATUS_OK) goto fail;
    if (config->dimensions.family == KQ_LAYER_FAMILY_PLE_GDN) {
        status = kq_ple_config_open_from_model(model, &config->ple, diagnostic);
        if (status != KQ_STATUS_OK) goto fail;
        status = reference_f32
            ? kq_ple_value_config_create_reference_f32(
                  model, config->ple, &config->ple_value, diagnostic)
            : kq_ple_value_config_create(model, config->ple, &config->ple_value,
                                         diagnostic);
        if (status != KQ_STATUS_OK) goto fail;
    }
    config->gr_workspace_bytes =
        ((uint64_t)config->dimensions.branch_count *
             config->dimensions.hidden_size * 2U +
         config->dimensions.hidden_size + config->dimensions.gr_rank +
         config->dimensions.branch_count) * sizeof(float);
    *out_config = config;
    return KQ_STATUS_OK;
fail:
    kq_layer_config_close(config);
    return status;
}

kq_status kq_layer_config_create(const kq_model *model, uint32_t layer_id,
                                 kq_layer_config **out_config,
                                 kq_diagnostic *diagnostic) {
    return kq_layer_config_create_common(model, layer_id, 0, out_config,
                                         diagnostic);
}

kq_status kq_layer_config_create_reference_f32(
    const kq_model *model, uint32_t layer_id, kq_layer_config **out_config,
    kq_diagnostic *diagnostic) {
    return kq_layer_config_create_common(model, layer_id, 1, out_config,
                                         diagnostic);
}

kq_status kq_layer_test_config_create(
    const kq_layer_dimensions *dimensions, kq_gdn_config *gdn,
    kq_qsa_config *qsa, kq_moe_config *moe, kq_ple_config *ple,
    kq_ple_value_config *ple_value, kq_layer_config **out_config,
    kq_diagnostic *diagnostic) {
    kq_layer_config *config;
    if (dimensions == NULL || out_config == NULL || moe == NULL ||
        dimensions->hidden_size == 0U || dimensions->branch_count == 0U ||
        dimensions->gr_rank == 0U || !isfinite(dimensions->rms_epsilon) ||
        dimensions->rms_epsilon <= 0.0f) return KQ_STATUS_INVALID_ARGUMENT;
    *out_config = NULL;
    config = (kq_layer_config *)calloc(1U, sizeof(*config));
    if (config == NULL) return KQ_STATUS_OUT_OF_MEMORY;
    config->magic = KQ_LAYER_CONFIG_MAGIC;
    config->dimensions = *dimensions;
    config->gdn = gdn;
    config->qsa = qsa;
    config->moe = moe;
    config->ple = ple;
    config->ple_value = ple_value;
    config->gr_workspace_bytes =
        ((uint64_t)dimensions->branch_count * dimensions->hidden_size * 2U +
         dimensions->hidden_size + dimensions->gr_rank +
         dimensions->branch_count) * sizeof(float);
    if (!kq_layer_config_valid(config) ||
        kq_moe_config_hidden_size(moe) != dimensions->hidden_size ||
        (gdn != NULL && kq_gdn_config_hidden_size(gdn) != dimensions->hidden_size) ||
        (qsa != NULL && kq_qsa_config_hidden_size(qsa) != dimensions->hidden_size) ||
        (ple_value != NULL &&
         (kq_ple_value_config_hidden_size(ple_value) != dimensions->hidden_size ||
          kq_ple_value_config_residual_branches(ple_value) !=
              dimensions->branch_count))) {
        free(config);
        kq_diagnostic_set(diagnostic, KQ_STATUS_INCOMPATIBLE_LAYER,
                          "test layer subconfigs have incompatible geometry");
        return KQ_STATUS_INCOMPATIBLE_LAYER;
    }
    *out_config = config;
    return KQ_STATUS_OK;
}

void kq_layer_config_close(kq_layer_config *config) {
    if (config == NULL) return;
    if (config->owns_subconfigs) {
        kq_gdn_config_close(config->gdn);
        kq_qsa_config_close(config->qsa);
        kq_moe_config_close(config->moe);
        kq_ple_value_config_close(config->ple_value);
        kq_ple_config_close(config->ple);
    }
    config->magic = 0U;
    free(config);
}

uint32_t kq_layer_config_layer_id(const kq_layer_config *c) { return kq_layer_config_valid(c) ? c->dimensions.layer_id : UINT32_MAX; }
kq_layer_family kq_layer_config_family(const kq_layer_config *c) { return kq_layer_config_valid(c) ? c->dimensions.family : KQ_LAYER_FAMILY_INVALID; }
uint32_t kq_layer_config_hidden_size(const kq_layer_config *c) { return kq_layer_config_valid(c) ? c->dimensions.hidden_size : 0U; }
uint32_t kq_layer_config_branch_count(const kq_layer_config *c) { return kq_layer_config_valid(c) ? c->dimensions.branch_count : 0U; }
uint32_t kq_layer_config_gr_rank(const kq_layer_config *c) { return kq_layer_config_valid(c) ? c->dimensions.gr_rank : 0U; }
uint64_t kq_layer_config_owned_bytes(const kq_layer_config *c) {
    if (!kq_layer_config_valid(c)) return 0U;
    return sizeof(*c) + kq_gdn_config_owned_bytes(c->gdn) +
           kq_qsa_config_owned_bytes(c->qsa) + kq_moe_config_owned_bytes(c->moe) +
           kq_ple_value_config_owned_bytes(c->ple_value);
}
uint64_t kq_layer_config_gr_workspace_bytes(const kq_layer_config *c) { return kq_layer_config_valid(c) ? c->gr_workspace_bytes : 0U; }

uint64_t kq_layer_config_persistent_semantic_bytes(
    const kq_layer_config *c, uint64_t qsa_capacity) {
    uint64_t bytes = 0U;
    if (!kq_layer_config_valid(c)) return 0U;
    if (c->gdn != NULL)
        bytes = kq_gdn_config_recurrent_element_count(c->gdn) * 4U +
                kq_gdn_config_conv_state_element_count(c->gdn) * 2U;
    else if (!kq_layer_mul_u64(kq_qsa_config_semantic_state_bytes_per_token(c->qsa),
                              qsa_capacity, &bytes)) return 0U;
    if (c->ple_value != NULL) {
        uint64_t next;
        if (!kq_layer_add_u64(bytes, sizeof(kq_ple_stream_state), &next) ||
            !kq_layer_add_u64(next,
                              kq_ple_value_config_semantic_state_bytes(c->ple_value),
                              &bytes)) return 0U;
    }
    return bytes;
}

static kq_status kq_layer_copy_slot(kq_layer_state *state, uint32_t from,
                                    uint32_t to, kq_diagnostic *diagnostic) {
    kq_status status;
    int initialized = 0;
    uint64_t length = 0U;
    uint64_t history_count = 0U;
    uint64_t position = 0U;
    if (state->gdn[from] != NULL) {
        status = kq_gdn_state_export_f32(state->gdn[from], state->copy_a,
            state->copy_a_count, state->copy_b, state->copy_b_count,
            &initialized, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_gdn_state_import_f32(state->gdn[to], state->copy_a,
            state->copy_a_count, state->copy_b, state->copy_b_count,
            initialized, diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    else if (state->qsa[from] != NULL) {
        status = kq_qsa_state_export_f32(state->qsa[from], state->copy_a,
            state->copy_a_count, state->copy_b, state->copy_b_count,
            state->copy_c, state->copy_c_count, &length, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_qsa_state_import_f32(state->qsa[to], state->copy_a,
            length * state->config->qsa->key_values_per_token, state->copy_b,
            length * state->config->qsa->key_values_per_token, state->copy_c,
            length * state->config->qsa->raw_index_values_per_token, length,
            diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    if (state->config->ple_value != NULL) {
        state->ple[to] = state->ple[from];
        status = kq_ple_value_state_export_f32(state->config->ple_value,
            state->ple_value[from], state->copy_c, state->copy_c_count,
            &history_count, &position, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_ple_value_state_import_f32(state->config->ple_value,
            state->ple_value[to], state->copy_c, history_count, position,
            diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    return KQ_STATUS_OK;
}

kq_status kq_layer_state_create(const kq_layer_config *config,
                                uint64_t qsa_capacity,
                                kq_layer_state **out_state,
                                kq_diagnostic *diagnostic) {
    kq_layer_state *state;
    kq_status status = KQ_STATUS_OK;
    uint64_t bytes;
    uint32_t slot;
    if (!kq_layer_config_valid(config) || out_state == NULL ||
        (config->qsa != NULL && qsa_capacity == 0U))
        return KQ_STATUS_INVALID_ARGUMENT;
    *out_state = NULL;
    state = (kq_layer_state *)calloc(1U, sizeof(*state));
    if (state == NULL) return KQ_STATUS_OUT_OF_MEMORY;
    state->magic = KQ_LAYER_STATE_MAGIC;
    state->config = config;
    state->qsa_capacity = qsa_capacity;
    state->owned_bytes = sizeof(*state);
    for (slot = 0U; slot < 2U; ++slot) {
        if (config->gdn != NULL)
            status = kq_gdn_state_create(config->gdn, &state->gdn[slot], diagnostic);
        else
            status = kq_qsa_state_create(config->qsa, qsa_capacity,
                                         &state->qsa[slot], diagnostic);
        if (status != KQ_STATUS_OK) goto fail;
        if (config->ple_value != NULL) {
            status = kq_ple_state_reset(config->ple, &state->ple[slot], diagnostic);
            if (status != KQ_STATUS_OK) goto fail;
            status = kq_ple_value_state_create(config->ple_value,
                                               &state->ple_value[slot], diagnostic);
            if (status != KQ_STATUS_OK) goto fail;
        }
    }
    if (config->gdn != NULL) {
        state->copy_a_count = kq_gdn_config_conv_state_element_count(config->gdn);
        state->copy_b_count = kq_gdn_config_recurrent_element_count(config->gdn);
    } else {
        if (!kq_layer_mul_u64(qsa_capacity,
                              config->qsa->key_values_per_token,
                              &state->copy_a_count) ||
            !kq_layer_mul_u64(qsa_capacity,
                              config->qsa->raw_index_values_per_token,
                              &state->copy_c_count)) {
            status = KQ_STATUS_ARITHMETIC_OVERFLOW;
            goto fail;
        }
        state->copy_b_count = state->copy_a_count;
    }
    if (config->ple_value != NULL &&
        config->ple_value->state_elements > state->copy_c_count)
        state->copy_c_count = config->ple_value->state_elements;
    if (state->copy_a_count != 0U) state->copy_a = (float *)calloc((size_t)state->copy_a_count, sizeof(float));
    if (state->copy_b_count != 0U) state->copy_b = (float *)calloc((size_t)state->copy_b_count, sizeof(float));
    if (state->copy_c_count != 0U) state->copy_c = (float *)calloc((size_t)state->copy_c_count, sizeof(float));
    if ((state->copy_a_count != 0U && state->copy_a == NULL) ||
        (state->copy_b_count != 0U && state->copy_b == NULL) ||
        (state->copy_c_count != 0U && state->copy_c == NULL)) {
        status = KQ_STATUS_OUT_OF_MEMORY; goto fail;
    }
    bytes = (state->copy_a_count + state->copy_b_count + state->copy_c_count) * sizeof(float);
    state->owned_bytes += bytes;
    for (slot = 0U; slot < 2U; ++slot) {
        state->owned_bytes += kq_gdn_state_owned_bytes(state->gdn[slot]);
        state->owned_bytes += kq_qsa_state_owned_bytes(state->qsa[slot]);
        if (state->ple_value[slot] != NULL)
            state->owned_bytes +=
                kq_ple_value_config_state_bytes(config->ple_value);
    }
    *out_state = state;
    return KQ_STATUS_OK;
fail:
    kq_layer_state_close(state);
    return status;
}

void kq_layer_state_close(kq_layer_state *state) {
    uint32_t slot;
    if (state == NULL) return;
    for (slot = 0U; slot < 2U; ++slot) {
        kq_gdn_state_close(state->gdn[slot]);
        kq_qsa_state_close(state->qsa[slot]);
        kq_ple_value_state_close(state->ple_value[slot]);
    }
    free(state->copy_a); free(state->copy_b); free(state->copy_c);
    state->magic = 0U; free(state);
}

kq_status kq_layer_state_reset(kq_layer_state *state,
                               kq_diagnostic *diagnostic) {
    uint32_t slot;
    kq_status status;
    if (!kq_layer_state_valid(state)) return KQ_STATUS_INVALID_LAYER_STATE;
    for (slot = 0U; slot < 2U; ++slot) {
        if (state->gdn[slot] != NULL)
            status = kq_gdn_state_reset(state->gdn[slot], diagnostic);
        else
            status = kq_qsa_state_reset(state->qsa[slot], diagnostic);
        if (status != KQ_STATUS_OK) return status;
        if (state->config->ple != NULL) {
            status = kq_ple_state_reset(state->config->ple, &state->ple[slot], diagnostic);
            if (status != KQ_STATUS_OK) return status;
            status = kq_ple_value_state_reset(state->config->ple_value,
                                               state->ple_value[slot], diagnostic);
            if (status != KQ_STATUS_OK) return status;
        }
    }
    state->active_slot = 0U; state->position = 0U;
    return KQ_STATUS_OK;
}

uint64_t kq_layer_state_owned_bytes(const kq_layer_state *s) { return kq_layer_state_valid(s) ? s->owned_bytes : 0U; }
uint64_t kq_layer_state_position(const kq_layer_state *s) { return kq_layer_state_valid(s) ? s->position : 0U; }

kq_status kq_layer_required_scratch_bytes(
    const kq_layer_config *config, const kq_layer_state *state,
    uint64_t token_count, uint64_t *scratch_bytes,
    kq_diagnostic *diagnostic) {
    uint64_t width, float_count, base, sub = 0U, qsa = 0U;
    if (!kq_layer_config_valid(config) || !kq_layer_state_valid(state) ||
        state->config != config || token_count == 0U || scratch_bytes == NULL)
        return KQ_STATUS_INVALID_ARGUMENT;
    width = (uint64_t)config->dimensions.hidden_size * config->dimensions.branch_count;
    if (!kq_layer_mul_u64(token_count, width * 3U +
                          (uint64_t)config->dimensions.hidden_size * 4U,
                          &float_count) ||
        !kq_layer_add_u64(float_count, width * 2U +
                          config->dimensions.gr_rank +
                          config->dimensions.branch_count, &float_count) ||
        !kq_layer_mul_u64(float_count, sizeof(float), &base))
        return KQ_STATUS_ARITHMETIC_OVERFLOW;
    if (config->ple != NULL) {
        uint64_t intent_bytes;
        if (!kq_layer_mul_u64(token_count, KQ_PLE_ADDRESSES_PER_TOKEN *
                              sizeof(kq_ple_address_intent), &intent_bytes) ||
            !kq_layer_add_u64(base, intent_bytes, &base))
            return KQ_STATUS_ARITHMETIC_OVERFLOW;
        sub = kq_ple_value_config_scratch_bytes(config->ple_value);
    }
    if (config->gdn != NULL && kq_gdn_config_scratch_bytes(config->gdn) > sub)
        sub = kq_gdn_config_scratch_bytes(config->gdn);
    if (config->qsa != NULL) {
        kq_status status = kq_qsa_required_scratch_bytes(config->qsa,
                                          state->qsa[state->active_slot],
                                          token_count, &qsa,
                                          diagnostic);
        if (status != KQ_STATUS_OK) return status;
        if (qsa > sub) sub = qsa;
    }
    if (kq_moe_config_scratch_bytes(config->moe) > sub)
        sub = kq_moe_config_scratch_bytes(config->moe);
    if (!kq_layer_add_u64(base, sub, scratch_bytes))
        return KQ_STATUS_ARITHMETIC_OVERFLOW;
    return KQ_STATUS_OK;
}

typedef struct kq_layer_buffers {
    float *ple_output; float *current; float *mixed; float *block_output;
    float *after_mixer; float *moe_input; float *moe_output;
    float *normalized; float *read_gate; float *rank; float *write_gate;
    kq_ple_address_intent *intents; void *sub_scratch; uint64_t sub_bytes;
} kq_layer_buffers;

static void kq_layer_partition(const kq_layer_config *c, uint64_t tokens,
                               void *scratch, uint64_t total,
                               kq_layer_buffers *b) {
    uint64_t h = c->dimensions.hidden_size;
    uint64_t w = h * c->dimensions.branch_count;
    float *p = (float *)scratch;
    b->ple_output=p;p+=tokens*w;b->current=p;p+=tokens*w;
    b->mixed=p;p+=tokens*h;b->block_output=p;p+=tokens*h;
    b->after_mixer=p;p+=tokens*w;b->moe_input=p;p+=tokens*h;
    b->moe_output=p;p+=tokens*h;b->normalized=p;p+=w;
    b->read_gate=p;p+=w;b->rank=p;p+=c->dimensions.gr_rank;
    b->write_gate=p;p+=c->dimensions.branch_count;
    if (c->ple != NULL) {
        b->intents=(kq_ple_address_intent *)p;
        p=(float *)((unsigned char *)p + tokens*KQ_PLE_ADDRESSES_PER_TOKEN*sizeof(*b->intents));
    } else b->intents=NULL;
    b->sub_scratch=p;
    b->sub_bytes=total-(uint64_t)((unsigned char *)p-(unsigned char *)scratch);
}

static kq_status kq_layer_execute(
    const kq_layer_config *config, const kq_layer_weights_f32 *weights,
    const float *input, const uint32_t *tokens, uint64_t count,
    const uint8_t *padding_mask, float *output, uint64_t output_capacity,
    kq_layer_state *state, void *scratch, uint64_t scratch_bytes,
    int decode, kq_layer_checkpoint_observer observer, void *observer_user,
    kq_layer_metrics *metrics, kq_diagnostic *diagnostic) {
    kq_layer_buffers b;
    uint64_t required, width, total, t;
    uint32_t active, staging;
    kq_status status;
    LARGE_INTEGER frequency, start, end;
    if (!kq_layer_config_valid(config) || weights == NULL || input == NULL ||
        output == NULL || !kq_layer_state_valid(state) || state->config != config ||
        scratch == NULL || count == 0U || (decode && count != 1U) ||
        output == input || (void *)output == scratch || (const void *)input == scratch)
        return KQ_STATUS_INVALID_ARGUMENT;
    width=(uint64_t)config->dimensions.hidden_size*config->dimensions.branch_count;
    if (!kq_layer_mul_u64(count,width,&total)) return KQ_STATUS_ARITHMETIC_OVERFLOW;
    if (output_capacity < total) return KQ_STATUS_BUFFER_TOO_SMALL;
    status=kq_layer_required_scratch_bytes(config,state,count,&required,diagnostic);
    if(status!=KQ_STATUS_OK)return status;
    if(scratch_bytes<required)return KQ_STATUS_BUFFER_TOO_SMALL;
    if (config->ple != NULL && (tokens == NULL || weights->ple_value == NULL ||
                               weights->ple_provider == NULL))
        return KQ_STATUS_INCOMPATIBLE_LAYER;
    if (config->ple == NULL && tokens != NULL) { /* token IDs are irrelevant outside PLE */ }
    if (weights->moe == NULL ||
        (config->gdn != NULL && weights->gdn == NULL) ||
        (config->qsa != NULL && weights->qsa == NULL))
        return KQ_STATUS_INCOMPATIBLE_LAYER;
    active=state->active_slot;staging=1U-active;
    status=kq_layer_copy_slot(state,active,staging,diagnostic);
    if(status!=KQ_STATUS_OK)return status;
    kq_layer_partition(config,count,scratch,scratch_bytes,&b);
    QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&start);
    for(t=0U;t<count;++t)
        kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_INPUT,t,2U,
                      config->dimensions.branch_count,config->dimensions.hidden_size,
                      input+t*width,width);
    if(config->ple!=NULL){
        uint64_t needed=0U; kq_ple_run_metrics address_metrics;
        if(padding_mask!=NULL)for(t=0U;t<count;++t)if(padding_mask[t]==0U)
            return KQ_STATUS_INCOMPATIBLE_LAYER;
        status=decode?kq_ple_generate_decode_step(config->ple,&state->ple[staging],tokens[0],
                    b.intents,KQ_PLE_ADDRESSES_PER_TOKEN,&needed,&address_metrics,diagnostic)
            :kq_ple_generate_prefill(config->ple,&state->ple[staging],tokens,count,b.intents,
                    count*KQ_PLE_ADDRESSES_PER_TOKEN,&needed,&address_metrics,diagnostic);
        if(status!=KQ_STATUS_OK)return status;
        status=decode?kq_ple_value_decode_f32(config->ple_value,state->ple_value[staging],
                    weights->ple_value,weights->ple_provider,input,b.intents,needed,b.ple_output,width,
                    b.sub_scratch,b.sub_bytes,NULL,NULL,NULL,diagnostic)
            :kq_ple_value_prefill_f32(config->ple_value,state->ple_value[staging],weights->ple_value,
                    weights->ple_provider,input,count,b.intents,needed,b.ple_output,total,
                    b.sub_scratch,b.sub_bytes,NULL,NULL,NULL,diagnostic);
        if(status!=KQ_STATUS_OK)return status;
        for(t=0U;t<total;++t)b.current[t]=input[t]+b.ple_output[t];
        for(t=0U;t<count;++t){
            kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_PLE_OUTPUT,t,2U,
                config->dimensions.branch_count,config->dimensions.hidden_size,b.ple_output+t*width,width);
            kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_PLE_ENHANCED_INPUT,t,2U,
                config->dimensions.branch_count,config->dimensions.hidden_size,b.current+t*width,width);
        }
    } else memcpy(b.current,input,(size_t)(total*sizeof(float)));
    for(t=0U;t<count;++t){
        status=kq_layer_gr_read_f32(config,&weights->attention_gr,b.current+t*width,
            b.normalized,b.read_gate,b.mixed+t*config->dimensions.hidden_size,
            b.rank,b.write_gate,diagnostic);if(status!=KQ_STATUS_OK)return status;
        kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_ATTN_GR_NORMALIZED,t,2U,
            config->dimensions.branch_count,config->dimensions.hidden_size,b.normalized,width);
        kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_ATTN_GR_READ_GATE,t,2U,
            config->dimensions.branch_count,config->dimensions.hidden_size,b.read_gate,width);
        kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_MIXER_INPUT,t,1U,
            config->dimensions.hidden_size,0U,b.mixed+t*config->dimensions.hidden_size,config->dimensions.hidden_size);
    }
    if(config->gdn!=NULL)status=decode?kq_gdn_decode_f32(config->gdn,weights->gdn,b.mixed,b.block_output,
            config->dimensions.hidden_size,state->gdn[staging],b.sub_scratch,b.sub_bytes,NULL,NULL,diagnostic)
        :kq_gdn_prefill_f32(config->gdn,weights->gdn,b.mixed,count,padding_mask,b.block_output,
            count*config->dimensions.hidden_size,state->gdn[staging],b.sub_scratch,b.sub_bytes,NULL,NULL,diagnostic);
    else status=decode?kq_qsa_decode_f32(config->qsa,weights->qsa,b.mixed,b.block_output,
            config->dimensions.hidden_size,state->qsa[staging],b.sub_scratch,b.sub_bytes,NULL,NULL,NULL,diagnostic)
        :kq_qsa_prefill_f32(config->qsa,weights->qsa,b.mixed,count,b.block_output,
            count*config->dimensions.hidden_size,state->qsa[staging],b.sub_scratch,b.sub_bytes,NULL,NULL,NULL,diagnostic);
    if(status!=KQ_STATUS_OK)return status;
    for(t=0U;t<count;++t){
        status=kq_layer_gr_read_f32(config,&weights->attention_gr,b.current+t*width,
            b.normalized,b.read_gate,b.mixed+t*config->dimensions.hidden_size,
            b.rank,b.write_gate,diagnostic);if(status!=KQ_STATUS_OK)return status;
        kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_MIXER_OUTPUT,t,1U,
            config->dimensions.hidden_size,0U,b.block_output+t*config->dimensions.hidden_size,config->dimensions.hidden_size);
        kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_ATTN_GR_WRITE_GATE,t,1U,
            config->dimensions.branch_count,0U,b.write_gate,config->dimensions.branch_count);
        kq_layer_gr_write_f32(config,b.current+t*width,b.block_output+t*config->dimensions.hidden_size,
                             b.write_gate,b.after_mixer+t*width);
        kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_AFTER_MIXER_RESIDUAL,t,2U,
            config->dimensions.branch_count,config->dimensions.hidden_size,b.after_mixer+t*width,width);
        status=kq_layer_gr_read_f32(config,&weights->moe_gr,b.after_mixer+t*width,
            b.normalized,b.read_gate,b.moe_input+t*config->dimensions.hidden_size,
            b.rank,b.write_gate,diagnostic);if(status!=KQ_STATUS_OK)return status;
        kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_MOE_GR_NORMALIZED,t,2U,
            config->dimensions.branch_count,config->dimensions.hidden_size,b.normalized,width);
        kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_MOE_GR_READ_GATE,t,2U,
            config->dimensions.branch_count,config->dimensions.hidden_size,b.read_gate,width);
        kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_MOE_INPUT,t,1U,
            config->dimensions.hidden_size,0U,b.moe_input+t*config->dimensions.hidden_size,config->dimensions.hidden_size);
    }
    status=kq_moe_execute_f32(config->moe,weights->moe,b.moe_input,count,b.moe_output,
        count*config->dimensions.hidden_size,b.sub_scratch,b.sub_bytes,NULL,NULL,NULL,diagnostic);
    if(status!=KQ_STATUS_OK)return status;
    for(t=0U;t<count;++t){
        status=kq_layer_gr_read_f32(config,&weights->moe_gr,b.after_mixer+t*width,
            b.normalized,b.read_gate,b.moe_input+t*config->dimensions.hidden_size,
            b.rank,b.write_gate,diagnostic);if(status!=KQ_STATUS_OK)return status;
        kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_MOE_OUTPUT,t,1U,
            config->dimensions.hidden_size,0U,b.moe_output+t*config->dimensions.hidden_size,config->dimensions.hidden_size);
        kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_MOE_GR_WRITE_GATE,t,1U,
            config->dimensions.branch_count,0U,b.write_gate,config->dimensions.branch_count);
        kq_layer_gr_write_f32(config,b.after_mixer+t*width,b.moe_output+t*config->dimensions.hidden_size,
                             b.write_gate,output+t*width);
        kq_layer_emit(observer,observer_user,KQ_LAYER_CHECKPOINT_OUTPUT,t,2U,
            config->dimensions.branch_count,config->dimensions.hidden_size,output+t*width,width);
    }
    if(UINT64_MAX-state->position<count)return KQ_STATUS_ARITHMETIC_OVERFLOW;
    state->position+=count;state->active_slot=staging;
    QueryPerformanceCounter(&end);
    if(metrics!=NULL){memset(metrics,0,sizeof(*metrics));metrics->tokens_processed=count;
        metrics->scratch_bytes=required;metrics->gr_workspace_bytes=config->gr_workspace_bytes;
        metrics->transaction_staging_bytes=state->owned_bytes;
        if(frequency.QuadPart>0)metrics->elapsed_nanoseconds=(uint64_t)(((end.QuadPart-start.QuadPart)*UINT64_C(1000000000))/(uint64_t)frequency.QuadPart);}
    return KQ_STATUS_OK;
}

kq_status kq_layer_prefill_f32(
    const kq_layer_config *c,const kq_layer_weights_f32 *w,const float *h,
    const uint32_t *ids,uint64_t n,const uint8_t *mask,float *o,uint64_t cap,
    kq_layer_state *s,void *scratch,uint64_t scratch_bytes,
    kq_layer_checkpoint_observer observer,void *user,kq_layer_metrics *metrics,
    kq_diagnostic *d){return kq_layer_execute(c,w,h,ids,n,mask,o,cap,s,scratch,scratch_bytes,0,observer,user,metrics,d);}
kq_status kq_layer_decode_f32(
    const kq_layer_config *c,const kq_layer_weights_f32 *w,const float *h,
    uint32_t id,float *o,uint64_t cap,kq_layer_state *s,void *scratch,
    uint64_t scratch_bytes,kq_layer_checkpoint_observer observer,void *user,
    kq_layer_metrics *metrics,kq_diagnostic *d){return kq_layer_execute(c,w,h,&id,1U,NULL,o,cap,s,scratch,scratch_bytes,1,observer,user,metrics,d);}

const char *kq_layer_family_name(kq_layer_family f){switch(f){case KQ_LAYER_FAMILY_GDN:return "GDN";case KQ_LAYER_FAMILY_QSA:return "QSA";case KQ_LAYER_FAMILY_PLE_GDN:return "PLE_GDN";default:return "INVALID";}}
const char *kq_layer_checkpoint_kind_name(kq_layer_checkpoint_kind k){static const char *n[]={"INPUT","PLE_OUTPUT","PLE_ENHANCED_INPUT","ATTN_GR_NORMALIZED","ATTN_GR_READ_GATE","MIXER_INPUT","MIXER_OUTPUT","ATTN_GR_WRITE_GATE","AFTER_MIXER_RESIDUAL","MOE_GR_NORMALIZED","MOE_GR_READ_GATE","MOE_INPUT","MOE_OUTPUT","MOE_GR_WRITE_GATE","OUTPUT"};return (unsigned)k<sizeof(n)/sizeof(n[0])?n[k]:"UNKNOWN";}
