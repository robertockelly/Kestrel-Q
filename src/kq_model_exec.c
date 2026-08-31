#define NOMINMAX
#include <windows.h>

#include "kq_model_exec.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "kq_internal.h"
#include "kq_layer_internal.h"
#include "kq_numeric.h"
#include "kq_weight_provider_internal.h"

#define KQ_MODEL_EXEC_CONFIG_MAGIC UINT32_C(0x4b514d43)
#define KQ_MODEL_EXEC_STATE_MAGIC UINT32_C(0x4b514d53)
#define KQ_MODEL_EXEC_GR_RANK 320U
#define KQ_MODEL_EXEC_FINAL_WIDTH \
    (KQ_MODEL_EXEC_HIDDEN_SIZE * KQ_MODEL_EXEC_BRANCH_COUNT)

struct kq_model_exec_config {
    uint32_t magic;
    const kq_gguf *gguf;
    const kq_model *model;
    const kq_tokenizer *tokenizer;
    kq_weight_provider *provider;
    uint64_t context_capacity;
    kq_layer_config *layers[KQ_MODEL_EXEC_LAYER_COUNT];
    const kq_semantic_tensor *embedding;
    const kq_semantic_tensor *final_norm;
    const kq_semantic_tensor *final_down;
    const kq_semantic_tensor *final_up;
    const kq_semantic_tensor *lm_head;
    uint64_t owned_bytes;
};

struct kq_model_exec_state {
    uint32_t magic;
    const kq_model_exec_config *config;
    kq_layer_state *layers[KQ_MODEL_EXEC_LAYER_COUNT];
    uint64_t position;
    uint64_t owned_bytes;
    int requires_reset;
};

typedef struct kq_model_exec_buffers {
    float *branches_a;
    float *branches_b;
    float *embedding;
    float *norm_delta;
    float *normalized;
    float *low;
    float *low_scaled;
    float *gate_logits;
    float *gate;
    float *gated;
    float *hidden;
    float *logits;
    void *provider_scratch;
    uint64_t provider_scratch_bytes;
    void *layer_scratch;
    uint64_t layer_scratch_bytes;
} kq_model_exec_buffers;

static kq_status model_fail(kq_diagnostic *diagnostic, kq_status status,
                            const char *message) {
    kq_diagnostic_set(diagnostic, status, "%s", message);
    return status;
}

static int add_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (result == NULL || UINT64_MAX - left < right) return 0;
    *result = left + right;
    return 1;
}

static int mul_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (result == NULL || (left != 0U && right > UINT64_MAX / left)) return 0;
    *result = left * right;
    return 1;
}

static int ranges_overlap(const void *left, uint64_t left_bytes,
                          const void *right, uint64_t right_bytes) {
    uintptr_t left_begin;
    uintptr_t right_begin;
    uintptr_t left_end;
    uintptr_t right_end;
    if (left == NULL || right == NULL || left_bytes == 0U || right_bytes == 0U)
        return 0;
    left_begin = (uintptr_t)left;
    right_begin = (uintptr_t)right;
    if (left_bytes > UINTPTR_MAX - left_begin ||
        right_bytes > UINTPTR_MAX - right_begin) return 1;
    left_end = left_begin + (uintptr_t)left_bytes;
    right_end = right_begin + (uintptr_t)right_bytes;
    return left_begin < right_end && right_begin < left_end;
}

static int config_valid(const kq_model_exec_config *config) {
    return config != NULL && config->magic == KQ_MODEL_EXEC_CONFIG_MAGIC &&
           config->gguf != NULL && config->model != NULL &&
           config->tokenizer != NULL && config->provider != NULL;
}

static int state_valid(const kq_model_exec_state *state) {
    return state != NULL && state->magic == KQ_MODEL_EXEC_STATE_MAGIC &&
           config_valid(state->config);
}

static kq_status reset_private_layers(kq_model_exec_state *state,
                                      kq_diagnostic *diagnostic) {
    uint32_t layer;
    kq_status status;
    for (layer = 0U; layer < KQ_MODEL_EXEC_LAYER_COUNT; ++layer) {
        status = kq_layer_state_reset(state->layers[layer], diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    return KQ_STATUS_OK;
}

static int semantic_is_matrix(const kq_semantic_tensor *semantic,
                              kq_semantic_role role, uint64_t rows,
                              uint64_t columns, uint32_t physical_type) {
    return semantic != NULL && semantic->role == role &&
           semantic->runtime_scope == KQ_SCOPE_REQUIRED_INITIAL_TEXT &&
           semantic->canonical_rank == 2U &&
           semantic->canonical_dimensions[0] == rows &&
           semantic->canonical_dimensions[1] == columns &&
           semantic->binding_count == 1U &&
           semantic->bindings[0].part_role == KQ_BINDING_PART_WHOLE &&
           semantic->bindings[0].physical != NULL &&
           semantic->bindings[0].physical->type_id == physical_type;
}

static int semantic_is_vector(const kq_semantic_tensor *semantic,
                              kq_semantic_role role, uint64_t elements,
                              uint32_t physical_type) {
    return semantic != NULL && semantic->role == role &&
           semantic->runtime_scope == KQ_SCOPE_REQUIRED_INITIAL_TEXT &&
           semantic->canonical_rank == 1U &&
           semantic->canonical_dimensions[0] == elements &&
           semantic->binding_count == 1U &&
           semantic->bindings[0].part_role == KQ_BINDING_PART_WHOLE &&
           semantic->bindings[0].physical != NULL &&
           semantic->bindings[0].physical->type_id == physical_type;
}

static void emit_progress(kq_model_exec_progress_observer observer,
                          void *user_data, kq_model_exec_phase phase,
                          uint32_t layer_id, const float *values,
                          uint64_t value_count) {
    kq_model_exec_progress_event event;
    if (observer == NULL) return;
    event.phase = phase;
    event.layer_id = layer_id;
    event.layer_count = KQ_MODEL_EXEC_LAYER_COUNT;
    event.last_token_values = values;
    event.last_token_value_count = value_count;
    observer(&event, user_data);
}

static int qpc_elapsed_ns(LARGE_INTEGER start, LARGE_INTEGER end,
                          LARGE_INTEGER frequency, uint64_t *elapsed) {
    uint64_t ticks;
    uint64_t hz;
    uint64_t whole;
    uint64_t remainder;
    uint64_t whole_ns;
    uint64_t scaled;
    if (elapsed == NULL || frequency.QuadPart <= 0 || end.QuadPart < start.QuadPart)
        return 0;
    ticks = (uint64_t)(end.QuadPart - start.QuadPart);
    hz = (uint64_t)frequency.QuadPart;
    whole = ticks / hz;
    remainder = ticks % hz;
    if (!mul_u64(whole, UINT64_C(1000000000), &whole_ns) ||
        !mul_u64(remainder, UINT64_C(1000000000), &scaled) ||
        !add_u64(whole_ns, scaled / hz, elapsed)) return 0;
    return 1;
}

kq_status kq_model_exec_config_open(
    const kq_gguf *gguf, const kq_model *model,
    const kq_tokenizer *tokenizer, kq_weight_provider *provider,
    uint64_t context_capacity, kq_model_exec_config **out_config,
    kq_diagnostic *diagnostic) {
    kq_model_exec_config *config;
    uint32_t layer;
    kq_status status;

    if (gguf == NULL || model == NULL || tokenizer == NULL ||
        provider == NULL || out_config == NULL || context_capacity < 2U) {
        return model_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                          "valid model-execution owners and context are required");
    }
    *out_config = NULL;
    if (provider->magic != KQ_WEIGHT_PROVIDER_MAGIC ||
        provider->gguf != gguf ||
        provider->model != model || kq_tokenizer_get_metrics(tokenizer) == NULL)
        return model_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MODEL_EXEC,
                          "model-execution owners do not share one verified artifact");
    if (kq_model_hidden_size(model) != KQ_MODEL_EXEC_HIDDEN_SIZE ||
        kq_model_vocabulary_size(model) != KQ_MODEL_EXEC_VOCABULARY_SIZE ||
        kq_model_layer_count(model) != KQ_MODEL_EXEC_LAYER_COUNT ||
        kq_model_gdn_layer_count(model) != 36U ||
        kq_model_qsa_layer_count(model) != 12U ||
        context_capacity > kq_model_context_length(model))
        return model_fail(diagnostic, KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                          "model-execution topology is not the pinned target");

    config = (kq_model_exec_config *)calloc(1U, sizeof(*config));
    if (config == NULL)
        return model_fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                          "could not allocate model-execution config");
    config->magic = KQ_MODEL_EXEC_CONFIG_MAGIC;
    config->gguf = gguf;
    config->model = model;
    config->tokenizer = tokenizer;
    config->provider = provider;
    config->context_capacity = context_capacity;
    config->owned_bytes = sizeof(*config);

    config->embedding = kq_model_find_semantic_tensor(model, "text.token_embedding");
    config->final_norm = kq_model_find_semantic_tensor(model, "text.final_gr.norm");
    config->final_down = kq_model_find_semantic_tensor(model, "text.final_gr.down");
    config->final_up = kq_model_find_semantic_tensor(model, "text.final_gr.up");
    config->lm_head = kq_model_find_semantic_tensor(model, "text.lm_head");
    if (!semantic_is_matrix(config->embedding, KQ_ROLE_TOKEN_EMBEDDING,
                            KQ_MODEL_EXEC_VOCABULARY_SIZE,
                            KQ_MODEL_EXEC_HIDDEN_SIZE, KQ_GGUF_TYPE_Q8_0) ||
        !semantic_is_vector(config->final_norm, KQ_ROLE_HC_NORM,
                            KQ_MODEL_EXEC_FINAL_WIDTH, KQ_GGUF_TYPE_F32) ||
        !semantic_is_matrix(config->final_down, KQ_ROLE_HC_DOWN,
                            KQ_MODEL_EXEC_GR_RANK, KQ_MODEL_EXEC_FINAL_WIDTH,
                            KQ_GGUF_TYPE_Q8_0) ||
        !semantic_is_matrix(config->final_up, KQ_ROLE_HC_UP,
                            KQ_MODEL_EXEC_FINAL_WIDTH, KQ_MODEL_EXEC_GR_RANK,
                            KQ_GGUF_TYPE_Q8_0) ||
        !semantic_is_matrix(config->lm_head, KQ_ROLE_LM_HEAD,
                            KQ_MODEL_EXEC_VOCABULARY_SIZE,
                            KQ_MODEL_EXEC_HIDDEN_SIZE, KQ_GGUF_TYPE_Q8_0)) {
        status = model_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MODEL_EXEC,
                            "entry/final/head semantics do not match the pinned target");
        goto fail;
    }

    for (layer = 0U; layer < KQ_MODEL_EXEC_LAYER_COUNT; ++layer) {
        status = kq_weight_provider_preflight_layer(provider, layer, diagnostic);
        if (status != KQ_STATUS_OK) goto fail;
        status = kq_layer_config_create_reference_f32(
            model, layer, &config->layers[layer], diagnostic);
        if (status != KQ_STATUS_OK) goto fail;
        config->owned_bytes += kq_layer_config_owned_bytes(config->layers[layer]);
        if (kq_layer_config_layer_id(config->layers[layer]) != layer) {
            status = model_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MODEL_EXEC,
                                "layer configuration order is not canonical");
            goto fail;
        }
    }
    *out_config = config;
    return KQ_STATUS_OK;

fail:
    kq_model_exec_config_close(config);
    return status;
}

void kq_model_exec_config_close(kq_model_exec_config *config) {
    uint32_t layer;
    if (config == NULL) return;
    for (layer = 0U; layer < KQ_MODEL_EXEC_LAYER_COUNT; ++layer)
        kq_layer_config_close(config->layers[layer]);
    config->magic = 0U;
    free(config);
}

uint64_t kq_model_exec_context_capacity(const kq_model_exec_config *config) {
    return config_valid(config) ? config->context_capacity : 0U;
}

uint64_t kq_model_exec_config_owned_bytes(const kq_model_exec_config *config) {
    return config_valid(config) ? config->owned_bytes : 0U;
}

kq_status kq_model_exec_state_create(
    const kq_model_exec_config *config, kq_model_exec_state **out_state,
    kq_diagnostic *diagnostic) {
    kq_model_exec_state *state;
    uint32_t layer;
    kq_status status;
    if (!config_valid(config) || out_state == NULL)
        return model_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                          "valid model config and state output are required");
    *out_state = NULL;
    state = (kq_model_exec_state *)calloc(1U, sizeof(*state));
    if (state == NULL)
        return model_fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                          "could not allocate model state");
    state->magic = KQ_MODEL_EXEC_STATE_MAGIC;
    state->config = config;
    state->owned_bytes = sizeof(*state);
    for (layer = 0U; layer < KQ_MODEL_EXEC_LAYER_COUNT; ++layer) {
        status = kq_layer_state_create(config->layers[layer],
                                       config->context_capacity,
                                       &state->layers[layer], diagnostic);
        if (status != KQ_STATUS_OK) {
            kq_model_exec_state_close(state);
            return status;
        }
        if (!add_u64(state->owned_bytes,
                     kq_layer_state_owned_bytes(state->layers[layer]),
                     &state->owned_bytes)) {
            kq_model_exec_state_close(state);
            return model_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                              "model state byte count overflows");
        }
    }
    *out_state = state;
    return KQ_STATUS_OK;
}

void kq_model_exec_state_close(kq_model_exec_state *state) {
    uint32_t layer;
    if (state == NULL) return;
    for (layer = 0U; layer < KQ_MODEL_EXEC_LAYER_COUNT; ++layer)
        kq_layer_state_close(state->layers[layer]);
    state->magic = 0U;
    free(state);
}

kq_status kq_model_exec_state_reset(kq_model_exec_state *state,
                                    kq_diagnostic *diagnostic) {
    kq_status status;
    if (!state_valid(state)) return KQ_STATUS_INVALID_MODEL_STATE;
    status = reset_private_layers(state, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    state->position = 0U;
    state->requires_reset = 0;
    return KQ_STATUS_OK;
}

uint64_t kq_model_exec_state_position(const kq_model_exec_state *state) {
    return state_valid(state) ? state->position : 0U;
}

uint64_t kq_model_exec_state_owned_bytes(const kq_model_exec_state *state) {
    return state_valid(state) ? state->owned_bytes : 0U;
}

static uint64_t hash_u64(uint64_t hash, uint64_t value) {
    uint32_t byte_index;
    for (byte_index = 0U; byte_index < 8U; ++byte_index) {
        hash ^= (value >> (byte_index * 8U)) & UINT64_C(0xff);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

kq_status kq_model_exec_state_get_summary(
    const kq_model_exec_state *state,
    kq_model_exec_state_summary *summary,
    kq_diagnostic *diagnostic) {
    kq_model_exec_state_summary staged;
    uint64_t hash = UINT64_C(14695981039346656037);
    uint32_t layer;
    if (!state_valid(state) || summary == NULL)
        return model_fail(diagnostic, KQ_STATUS_INVALID_MODEL_STATE,
                          "model state summary arguments are invalid");
    memset(&staged, 0, sizeof(staged));
    staged.model_position = state->position;
    staged.layer_position_min = UINT64_MAX;
    staged.qsa_sequence_length_min = UINT64_MAX;
    staged.qsa_complete_blocks_min = UINT64_MAX;
    staged.qsa_incomplete_tail_min = UINT64_MAX;
    staged.gdn_state_hash = UINT64_C(14695981039346656037);
    staged.qsa_state_hash = UINT64_C(14695981039346656037);
    staged.ple_address_state_hash = UINT64_C(14695981039346656037);
    staged.ple_value_state_hash = UINT64_C(14695981039346656037);
    hash = hash_u64(hash, state->position);
    for (layer = 0U; layer < KQ_MODEL_EXEC_LAYER_COUNT; ++layer) {
        kq_layer_state_summary layer_summary;
        kq_status status = kq_layer_state_get_summary(
            state->layers[layer], &layer_summary, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        if (layer_summary.position < staged.layer_position_min)
            staged.layer_position_min = layer_summary.position;
        if (layer_summary.position > staged.layer_position_max)
            staged.layer_position_max = layer_summary.position;
        if (layer_summary.gdn_initialized)
            staged.gdn_initialized_layers += 1U;
        if (layer_summary.family == KQ_LAYER_FAMILY_GDN ||
            layer_summary.family == KQ_LAYER_FAMILY_PLE_GDN)
            staged.gdn_state_hash = hash_u64(
                staged.gdn_state_hash, layer_summary.gdn_state_hash);
        if (layer_summary.family == KQ_LAYER_FAMILY_QSA) {
            uint64_t complete = layer_summary.qsa_length / 4U;
            uint64_t tail = layer_summary.qsa_length % 4U;
            staged.qsa_layers += 1U;
            if (layer_summary.qsa_length < staged.qsa_sequence_length_min)
                staged.qsa_sequence_length_min = layer_summary.qsa_length;
            if (layer_summary.qsa_length > staged.qsa_sequence_length_max)
                staged.qsa_sequence_length_max = layer_summary.qsa_length;
            if (complete < staged.qsa_complete_blocks_min)
                staged.qsa_complete_blocks_min = complete;
            if (complete > staged.qsa_complete_blocks_max)
                staged.qsa_complete_blocks_max = complete;
            if (tail < staged.qsa_incomplete_tail_min)
                staged.qsa_incomplete_tail_min = tail;
            if (tail > staged.qsa_incomplete_tail_max)
                staged.qsa_incomplete_tail_max = tail;
            staged.qsa_state_hash = hash_u64(
                staged.qsa_state_hash, layer_summary.qsa_state_hash);
        }
        if (layer_summary.family == KQ_LAYER_FAMILY_PLE_GDN) {
            staged.ple_address_position = layer_summary.ple_address_position;
            staged.ple_value_position = layer_summary.ple_value_position;
            staged.ple_address_state_hash = hash_u64(
                staged.ple_address_state_hash,
                layer_summary.ple_address_integrity);
            staged.ple_value_state_hash = hash_u64(
                staged.ple_value_state_hash,
                layer_summary.ple_value_state_hash);
        }
        hash = hash_u64(hash, layer);
        hash = hash_u64(hash, (uint64_t)layer_summary.family);
        hash = hash_u64(hash, layer_summary.position);
        hash = hash_u64(hash, layer_summary.qsa_length);
        hash = hash_u64(hash, layer_summary.ple_address_position);
        hash = hash_u64(hash, layer_summary.ple_value_position);
        hash = hash_u64(hash, layer_summary.ple_address_integrity);
        hash = hash_u64(hash, layer_summary.gdn_state_hash);
        hash = hash_u64(hash, layer_summary.qsa_state_hash);
        hash = hash_u64(hash, layer_summary.ple_value_state_hash);
        hash = hash_u64(hash, layer_summary.active_slot);
        hash = hash_u64(hash, (uint64_t)layer_summary.gdn_initialized);
    }
    if (staged.layer_position_min == UINT64_MAX)
        staged.layer_position_min = 0U;
    if (staged.qsa_sequence_length_min == UINT64_MAX) {
        staged.qsa_sequence_length_min = 0U;
        staged.qsa_complete_blocks_min = 0U;
        staged.qsa_incomplete_tail_min = 0U;
    }
    staged.structural_hash = hash;
    *summary = staged;
    return KQ_STATUS_OK;
}

static kq_status required_scratch_internal(
    const kq_model_exec_config *config, const kq_model_exec_state *state,
    uint64_t token_count, uint64_t *total_bytes,
    uint64_t *layer_scratch_bytes, int decode,
    kq_diagnostic *diagnostic) {
    uint64_t branch_elements;
    uint64_t fixed_floats;
    uint64_t bytes;
    uint64_t layer_max = 0U;
    uint64_t candidate;
    uint32_t layer;
    kq_status status;
    const uint64_t provider_bytes =
        ((uint64_t)KQ_MODEL_EXEC_VOCABULARY_SIZE +
         KQ_MODEL_EXEC_HIDDEN_SIZE) * sizeof(float);

    if (!config_valid(config) || !state_valid(state) ||
        state->config != config || token_count == 0U || total_bytes == NULL)
        return model_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                          "model scratch query arguments are invalid");
    if (!decode && token_count >= config->context_capacity)
        return model_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                          "prefill scratch requires room for continuation");
    if ((decode && token_count != 1U))
        return model_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                          "decode scratch requires exactly one token");
    if ((!decode && (state->position != 0U || state->requires_reset)) ||
        (decode && (state->position == 0U || state->requires_reset ||
                    state->position >= config->context_capacity)))
        return model_fail(diagnostic, KQ_STATUS_INVALID_MODEL_STATE,
                          decode ? "decode scratch requires live bounded state" :
                                   "prefill scratch requires reset bounded state");
    for (layer = 0U; layer < KQ_MODEL_EXEC_LAYER_COUNT; ++layer) {
        status = kq_layer_required_quantized_scratch_bytes(
            config->layers[layer], state->layers[layer],
            token_count, &candidate, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        if (candidate > layer_max) layer_max = candidate;
    }
    if (!mul_u64(token_count, KQ_MODEL_EXEC_FINAL_WIDTH, &branch_elements) ||
        !mul_u64(branch_elements, 2U, &fixed_floats) ||
        !add_u64(fixed_floats, KQ_MODEL_EXEC_HIDDEN_SIZE, &fixed_floats) ||
        !add_u64(fixed_floats, KQ_MODEL_EXEC_FINAL_WIDTH * 5U,
                 &fixed_floats) ||
        !add_u64(fixed_floats, KQ_MODEL_EXEC_GR_RANK * 2U,
                 &fixed_floats) ||
        !add_u64(fixed_floats, KQ_MODEL_EXEC_HIDDEN_SIZE, &fixed_floats) ||
        !add_u64(fixed_floats, KQ_MODEL_EXEC_VOCABULARY_SIZE,
                 &fixed_floats) ||
        !mul_u64(fixed_floats, sizeof(float), &bytes) ||
        !add_u64(bytes, provider_bytes, &bytes) ||
        !add_u64(bytes, layer_max, total_bytes))
        return model_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "model scratch byte count overflows");
    if (layer_scratch_bytes != NULL) *layer_scratch_bytes = layer_max;
    return KQ_STATUS_OK;
}

kq_status kq_model_exec_required_scratch_bytes(
    const kq_model_exec_config *config, const kq_model_exec_state *state,
    uint64_t prompt_token_count, uint64_t *scratch_bytes,
    kq_diagnostic *diagnostic) {
    return required_scratch_internal(config, state, prompt_token_count,
                                     scratch_bytes, NULL, 0, diagnostic);
}

kq_status kq_model_exec_required_decode_scratch_bytes(
    const kq_model_exec_config *config, const kq_model_exec_state *state,
    uint64_t *scratch_bytes, kq_diagnostic *diagnostic) {
    return required_scratch_internal(config, state, 1U, scratch_bytes,
                                     NULL, 1, diagnostic);
}

static void partition_scratch(uint64_t token_count, void *scratch,
                              uint64_t layer_bytes,
                              kq_model_exec_buffers *buffers) {
    float *cursor = (float *)scratch;
    const uint64_t branch_elements =
        token_count * KQ_MODEL_EXEC_FINAL_WIDTH;
    const uint64_t provider_floats =
        (uint64_t)KQ_MODEL_EXEC_VOCABULARY_SIZE + KQ_MODEL_EXEC_HIDDEN_SIZE;
    buffers->branches_a = cursor; cursor += branch_elements;
    buffers->branches_b = cursor; cursor += branch_elements;
    buffers->embedding = cursor; cursor += KQ_MODEL_EXEC_HIDDEN_SIZE;
    buffers->norm_delta = cursor; cursor += KQ_MODEL_EXEC_FINAL_WIDTH;
    buffers->normalized = cursor; cursor += KQ_MODEL_EXEC_FINAL_WIDTH;
    buffers->low = cursor; cursor += KQ_MODEL_EXEC_GR_RANK;
    buffers->low_scaled = cursor; cursor += KQ_MODEL_EXEC_GR_RANK;
    buffers->gate_logits = cursor; cursor += KQ_MODEL_EXEC_FINAL_WIDTH;
    buffers->gate = cursor; cursor += KQ_MODEL_EXEC_FINAL_WIDTH;
    buffers->gated = cursor; cursor += KQ_MODEL_EXEC_FINAL_WIDTH;
    buffers->hidden = cursor; cursor += KQ_MODEL_EXEC_HIDDEN_SIZE;
    buffers->logits = cursor; cursor += KQ_MODEL_EXEC_VOCABULARY_SIZE;
    buffers->provider_scratch = cursor;
    buffers->provider_scratch_bytes = provider_floats * sizeof(float);
    cursor += provider_floats;
    buffers->layer_scratch = cursor;
    buffers->layer_scratch_bytes = layer_bytes;
}

kq_status kq_model_exec_greedy_argmax_f32(
    const float *logits, uint64_t logit_count, uint32_t *selected_token_id,
    kq_diagnostic *diagnostic) {
    uint64_t index;
    uint32_t selected = 0U;
    if (logits == NULL || selected_token_id == NULL || logit_count == 0U ||
        logit_count > UINT32_MAX)
        return model_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                          "valid finite logits and selection output are required");
    for (index = 0U; index < logit_count; ++index) {
        if (!isfinite(logits[index]))
            return model_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                              "greedy selection rejects non-finite logits");
        if (index != 0U && logits[index] > logits[selected])
            selected = (uint32_t)index;
    }
    *selected_token_id = selected;
    return KQ_STATUS_OK;
}

static kq_status final_mix(
    const kq_model_exec_config *config, const float *branches,
    kq_model_exec_buffers *buffers, kq_diagnostic *diagnostic) {
    uint32_t branch;
    uint32_t element;
    kq_status status;
    status = kq_weight_provider_vector_f32(
        config->provider, config->final_norm, KQ_MODEL_EXEC_FINAL_WIDTH,
        buffers->norm_delta, KQ_MODEL_EXEC_FINAL_WIDTH,
        buffers->provider_scratch, buffers->provider_scratch_bytes, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    for (branch = 0U; branch < KQ_MODEL_EXEC_BRANCH_COUNT; ++branch) {
        status = kq_f32_rms_norm(
            branches + branch * KQ_MODEL_EXEC_HIDDEN_SIZE,
            buffers->norm_delta + branch * KQ_MODEL_EXEC_HIDDEN_SIZE,
            KQ_MODEL_EXEC_HIDDEN_SIZE, 1.0e-6f,
            buffers->normalized + branch * KQ_MODEL_EXEC_HIDDEN_SIZE,
            diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    status = kq_weight_provider_linear_f32(
        config->provider, config->final_down, KQ_BINDING_PART_WHOLE,
        KQ_WEIGHT_PROVIDER_NO_EXPERT, KQ_MODEL_EXEC_GR_RANK,
        KQ_MODEL_EXEC_FINAL_WIDTH, buffers->normalized, buffers->low,
        KQ_MODEL_EXEC_GR_RANK, buffers->provider_scratch,
        buffers->provider_scratch_bytes, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    status = kq_f32_scale(buffers->low, KQ_MODEL_EXEC_GR_RANK, 0.25f,
                          buffers->low_scaled, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    status = kq_f32_silu(buffers->low_scaled, KQ_MODEL_EXEC_GR_RANK,
                         buffers->low, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    status = kq_weight_provider_linear_f32(
        config->provider, config->final_up, KQ_BINDING_PART_WHOLE,
        KQ_WEIGHT_PROVIDER_NO_EXPERT, KQ_MODEL_EXEC_FINAL_WIDTH,
        KQ_MODEL_EXEC_GR_RANK, buffers->low, buffers->gate_logits,
        KQ_MODEL_EXEC_FINAL_WIDTH, buffers->provider_scratch,
        buffers->provider_scratch_bytes, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    status = kq_f32_sigmoid(buffers->gate_logits,
                            KQ_MODEL_EXEC_FINAL_WIDTH, buffers->gate,
                            diagnostic);
    if (status != KQ_STATUS_OK) return status;
    status = kq_f32_multiply(buffers->normalized, buffers->gate,
                             KQ_MODEL_EXEC_FINAL_WIDTH, buffers->gated,
                             diagnostic);
    if (status != KQ_STATUS_OK) return status;
    for (element = 0U; element < KQ_MODEL_EXEC_HIDDEN_SIZE; ++element) {
        float sum = 0.0f;
        for (branch = 0U; branch < KQ_MODEL_EXEC_BRANCH_COUNT; ++branch)
            sum = sum + buffers->gated[
                branch * KQ_MODEL_EXEC_HIDDEN_SIZE + element];
        buffers->hidden[element] = sum * 0.25f;
        if (!isfinite(buffers->hidden[element]))
            return model_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                              "final hyper-connection mix produced non-finite output");
    }
    return KQ_STATUS_OK;
}

kq_status kq_model_exec_prefill_first_token_f32(
    const kq_model_exec_config *config, kq_model_exec_state *state,
    const uint32_t *token_ids, uint64_t token_count,
    float *logits, uint64_t logits_capacity,
    unsigned char *decoded_utf8, uint64_t decoded_utf8_capacity,
    void *scratch, uint64_t scratch_bytes,
    kq_model_exec_progress_observer observer, void *observer_user_data,
    kq_model_exec_result *result, kq_diagnostic *diagnostic) {
    kq_model_exec_buffers buffers;
    const kq_weight_provider_metrics *before;
    const kq_weight_provider_metrics *after;
    kq_weight_provider_metrics before_copy;
    uint64_t required;
    uint64_t layer_scratch;
    uint64_t branch_elements;
    uint64_t branch_bytes;
    uint64_t logits_bytes;
    uint64_t lm_before;
    uint64_t embedding_before;
    uint64_t embedding_after;
    uint64_t decoded_required = 0U;
    uint64_t token;
    uint32_t layer;
    uint32_t branch;
    uint32_t selected;
    float *current;
    float *next;
    kq_tokenizer_decode_options decode_options;
    kq_model_exec_result staged_result;
    kq_status status;
    LARGE_INTEGER frequency = {0};
    LARGE_INTEGER started = {0};
    LARGE_INTEGER finished = {0};

    kq_diagnostic_clear(diagnostic);
    memset(&staged_result, 0, sizeof(staged_result));
    if (!config_valid(config) || !state_valid(state) ||
        state->config != config || token_ids == NULL || logits == NULL ||
        decoded_utf8 == NULL || scratch == NULL || result == NULL ||
        logits_capacity < KQ_MODEL_EXEC_VOCABULARY_SIZE || token_count == 0U)
        return model_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                          "model first-token arguments are invalid");
    /* A prior failed call may have advanced only private layer staging. Reset
       it before scratch geometry consults QSA lengths/capacities. */
    if (state->requires_reset) {
        status = reset_private_layers(state, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        state->requires_reset = 0;
    }
    status = required_scratch_internal(config, state, token_count, &required,
                                       &layer_scratch, 0, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    if (scratch_bytes < required)
        return model_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                          "model first-token scratch is too small");
    if (!mul_u64(token_count, KQ_MODEL_EXEC_FINAL_WIDTH, &branch_elements) ||
        !mul_u64(branch_elements, sizeof(float), &branch_bytes) ||
        !mul_u64(KQ_MODEL_EXEC_VOCABULARY_SIZE, sizeof(float), &logits_bytes))
        return model_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "model first-token buffer size overflows");
    if (ranges_overlap(logits, logits_bytes, scratch, scratch_bytes) ||
        ranges_overlap(decoded_utf8, decoded_utf8_capacity, scratch, scratch_bytes) ||
        ranges_overlap(logits, logits_bytes, decoded_utf8, decoded_utf8_capacity))
        return model_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                          "model first-token output and scratch buffers overlap");
    for (token = 0U; token < token_count; ++token)
        if (token_ids[token] >= KQ_MODEL_EXEC_CANONICAL_TOKEN_LIMIT)
            return model_fail(diagnostic, KQ_STATUS_INVALID_TOKEN_ID,
                              "prompt contains a padded or invalid canonical token ID");

    before = kq_weight_provider_get_metrics(config->provider);
    if (before == NULL) return KQ_STATUS_INCOMPATIBLE_MODEL_EXEC;
    before_copy = *before;
    if (QueryPerformanceFrequency(&frequency) != 0)
        (void)QueryPerformanceCounter(&started);

    state->requires_reset = 1;
    partition_scratch(token_count, scratch, layer_scratch, &buffers);
    emit_progress(observer, observer_user_data,
                  KQ_MODEL_EXEC_PHASE_TOKENIZED, UINT32_MAX, NULL, 0U);

    embedding_before = before_copy.logical_payload_bytes_touched;
    for (token = 0U; token < token_count; ++token) {
        status = kq_weight_provider_row_f32(
            config->provider, config->embedding, token_ids[token],
            KQ_MODEL_EXEC_HIDDEN_SIZE, buffers.embedding,
            KQ_MODEL_EXEC_HIDDEN_SIZE, buffers.provider_scratch,
            buffers.provider_scratch_bytes, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        for (branch = 0U; branch < KQ_MODEL_EXEC_BRANCH_COUNT; ++branch)
            memcpy(buffers.branches_a +
                       token * KQ_MODEL_EXEC_FINAL_WIDTH +
                       branch * KQ_MODEL_EXEC_HIDDEN_SIZE,
                   buffers.embedding,
                   KQ_MODEL_EXEC_HIDDEN_SIZE * sizeof(float));
    }
    after = kq_weight_provider_get_metrics(config->provider);
    if (after == NULL || after->logical_payload_bytes_touched < embedding_before)
        return KQ_STATUS_INCOMPATIBLE_MODEL_EXEC;
    embedding_after = after->logical_payload_bytes_touched;
    staged_result.metrics.embedding_logical_bytes_touched =
        embedding_after - embedding_before;
    emit_progress(observer, observer_user_data,
                  KQ_MODEL_EXEC_PHASE_EMBEDDING_COMPLETE, UINT32_MAX,
                  buffers.embedding, KQ_MODEL_EXEC_HIDDEN_SIZE);

    current = buffers.branches_a;
    next = buffers.branches_b;
    for (layer = 0U; layer < KQ_MODEL_EXEC_LAYER_COUNT; ++layer) {
        kq_layer_metrics layer_metrics;
        float *swap;
        emit_progress(observer, observer_user_data,
                      KQ_MODEL_EXEC_PHASE_LAYER_BEGIN, layer, NULL, 0U);
        memset(&layer_metrics, 0, sizeof(layer_metrics));
        status = kq_layer_prefill_quantized_f32(
            config->layers[layer], config->provider, current, token_ids,
            token_count, NULL, next, branch_elements,
            state->layers[layer], buffers.layer_scratch,
            buffers.layer_scratch_bytes, NULL, NULL, &layer_metrics,
            diagnostic);
        if (status != KQ_STATUS_OK) return status;
        staged_result.metrics.layers_completed += 1U;
        staged_result.metrics.qsa_selection_events +=
            layer_metrics.qsa_selection_events;
        staged_result.metrics.qsa_candidate_blocks +=
            layer_metrics.qsa_candidate_blocks;
        staged_result.metrics.qsa_selected_blocks +=
            layer_metrics.qsa_selected_blocks;
        staged_result.metrics.qsa_selected_tokens +=
            layer_metrics.qsa_selected_tokens;
        emit_progress(observer, observer_user_data,
                      KQ_MODEL_EXEC_PHASE_LAYER_COMPLETE, layer,
                      next + (token_count - 1U) * KQ_MODEL_EXEC_FINAL_WIDTH,
                      KQ_MODEL_EXEC_FINAL_WIDTH);
        swap = current;
        current = next;
        next = swap;
    }

    status = final_mix(config,
        current + (token_count - 1U) * KQ_MODEL_EXEC_FINAL_WIDTH,
        &buffers, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    emit_progress(observer, observer_user_data,
                  KQ_MODEL_EXEC_PHASE_FINAL_MIX_COMPLETE, UINT32_MAX,
                  buffers.hidden, KQ_MODEL_EXEC_HIDDEN_SIZE);

    before = kq_weight_provider_get_metrics(config->provider);
    if (before == NULL) return KQ_STATUS_INCOMPATIBLE_MODEL_EXEC;
    lm_before = before->logical_payload_bytes_touched;
    status = kq_weight_provider_linear_f32(
        config->provider, config->lm_head, KQ_BINDING_PART_WHOLE,
        KQ_WEIGHT_PROVIDER_NO_EXPERT, KQ_MODEL_EXEC_VOCABULARY_SIZE,
        KQ_MODEL_EXEC_HIDDEN_SIZE, buffers.hidden, buffers.logits,
        KQ_MODEL_EXEC_VOCABULARY_SIZE, buffers.provider_scratch,
        buffers.provider_scratch_bytes, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    after = kq_weight_provider_get_metrics(config->provider);
    if (after == NULL || after->logical_payload_bytes_touched < lm_before)
        return KQ_STATUS_INCOMPATIBLE_MODEL_EXEC;
    staged_result.metrics.lm_head_logical_bytes_touched =
        after->logical_payload_bytes_touched - lm_before;
    emit_progress(observer, observer_user_data,
                  KQ_MODEL_EXEC_PHASE_LOGITS_COMPLETE, UINT32_MAX,
                  buffers.logits, KQ_MODEL_EXEC_VOCABULARY_SIZE);

    status = kq_model_exec_greedy_argmax_f32(
        buffers.logits, KQ_MODEL_EXEC_VOCABULARY_SIZE, &selected, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    if (selected >= KQ_MODEL_EXEC_CANONICAL_TOKEN_LIMIT)
        return model_fail(diagnostic, KQ_STATUS_INVALID_TOKEN_ID,
                          "greedy model output is a padded non-tokenizer ID");
    decode_options.special_policy = KQ_TOKENIZER_DECODE_KEEP_SPECIAL;
    status = kq_tokenizer_decode(config->tokenizer, &selected, 1U,
                                 &decode_options, NULL, 0U,
                                 &decoded_required, diagnostic);
    if (status != KQ_STATUS_BUFFER_TOO_SMALL &&
        !(status == KQ_STATUS_OK && decoded_required == 0U)) return status;
    if (decoded_utf8_capacity < decoded_required)
        return model_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                          "decoded selected token output is too small");
    status = kq_tokenizer_decode(config->tokenizer, &selected, 1U,
                                 &decode_options, decoded_utf8,
                                 decoded_utf8_capacity, &decoded_required,
                                 diagnostic);
    if (status != KQ_STATUS_OK) return status;

    staged_result.selected_token_id = selected;
    staged_result.selected_token_is_eog =
        selected == 248044U || selected == 248046U;
    staged_result.decoded_utf8_bytes = decoded_required;
    staged_result.metrics.prompt_tokens = token_count;
    staged_result.metrics.input_tokens = token_count;
    staged_result.metrics.prompt_prefill_count = 1U;
    staged_result.metrics.persistent_state_bytes = state->owned_bytes;
    staged_result.metrics.peak_scratch_bytes = required;
    staged_result.metrics.logits_bytes = logits_bytes;
    after = kq_weight_provider_get_metrics(config->provider);
    if (after == NULL ||
        after->logical_payload_bytes_touched <
            before_copy.logical_payload_bytes_touched ||
        after->quantized_blocks_touched < before_copy.quantized_blocks_touched ||
        after->selected_expert_requests < before_copy.selected_expert_requests ||
        after->ple_row_requests < before_copy.ple_row_requests)
        return KQ_STATUS_INCOMPATIBLE_MODEL_EXEC;
    staged_result.metrics.logical_payload_bytes_touched =
        after->logical_payload_bytes_touched -
        before_copy.logical_payload_bytes_touched;
    staged_result.metrics.payload_blocks_touched =
        after->quantized_blocks_touched - before_copy.quantized_blocks_touched;
    staged_result.metrics.unique_semantic_tensors_touched =
        after->unique_semantic_tensors_touched;
    staged_result.metrics.routed_expert_member_requests =
        after->selected_expert_requests - before_copy.selected_expert_requests;
    staged_result.metrics.selected_expert_matrix_requests =
        staged_result.metrics.routed_expert_member_requests;
    if (staged_result.metrics.selected_expert_matrix_requests % 3U != 0U)
        return model_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MODEL_EXEC,
                          "routed expert matrix request count is incomplete");
    staged_result.metrics.routed_expert_selections =
        staged_result.metrics.selected_expert_matrix_requests / 3U;
    staged_result.metrics.ple_row_requests =
        after->ple_row_requests - before_copy.ple_row_requests;
    staged_result.metrics.maximum_f32_weight_bytes_materialized =
        after->maximum_f32_weight_bytes_materialized;
    if (frequency.QuadPart > 0 && QueryPerformanceCounter(&finished) != 0)
        (void)qpc_elapsed_ns(started, finished, frequency,
                             &staged_result.metrics.elapsed_nanoseconds);
    memcpy(logits, buffers.logits, (size_t)logits_bytes);
    state->position += token_count;
    state->requires_reset = 0;
    *result = staged_result;
    emit_progress(observer, observer_user_data,
                  KQ_MODEL_EXEC_PHASE_TOKEN_SELECTED, UINT32_MAX, NULL, 0U);
    return KQ_STATUS_OK;
}

int kq_model_exec_token_is_eog(uint32_t token_id) {
    return token_id == 248044U || token_id == 248046U;
}

static kq_status rollback_decode_layers(
    kq_model_exec_state *state, uint32_t committed_layers,
    kq_status original_status, const kq_diagnostic *original_diagnostic,
    kq_diagnostic *diagnostic) {
    while (committed_layers != 0U) {
        kq_diagnostic rollback_diagnostic;
        kq_status rollback_status;
        committed_layers -= 1U;
        kq_diagnostic_clear(&rollback_diagnostic);
        rollback_status = kq_layer_state_rollback_last(
            state->layers[committed_layers], 1U, &rollback_diagnostic);
        if (rollback_status != KQ_STATUS_OK) {
            state->requires_reset = 1;
            return model_fail(diagnostic, KQ_STATUS_INVALID_MODEL_STATE,
                              "model decode rollback could not restore layer state");
        }
    }
    if (diagnostic != NULL && original_diagnostic != NULL)
        *diagnostic = *original_diagnostic;
    return original_status;
}

kq_status kq_model_exec_decode_one_f32(
    const kq_model_exec_config *config, kq_model_exec_state *state,
    uint32_t input_token_id,
    float *logits, uint64_t logits_capacity,
    unsigned char *decoded_utf8, uint64_t decoded_utf8_capacity,
    void *scratch, uint64_t scratch_bytes,
    kq_model_exec_progress_observer observer, void *observer_user_data,
    kq_model_exec_result *result, kq_diagnostic *diagnostic) {
    kq_model_exec_buffers buffers;
    const kq_weight_provider_metrics *before;
    const kq_weight_provider_metrics *after;
    kq_weight_provider_metrics before_copy;
    kq_model_exec_result staged_result;
    unsigned char staged_decoded[1024];
    kq_tokenizer_decode_options decode_options;
    kq_diagnostic original_diagnostic;
    LARGE_INTEGER frequency = {0};
    LARGE_INTEGER started = {0};
    LARGE_INTEGER finished = {0};
    uint64_t required;
    uint64_t layer_scratch;
    uint64_t logits_bytes;
    uint64_t embedding_before;
    uint64_t embedding_after;
    uint64_t lm_before;
    uint64_t decoded_required = 0U;
    uint32_t branch;
    uint32_t layer;
    uint32_t selected;
    uint32_t committed_layers = 0U;
    float *current;
    float *next;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    memset(&staged_result, 0, sizeof(staged_result));
    if (!config_valid(config) || !state_valid(state) ||
        state->config != config || logits == NULL || decoded_utf8 == NULL ||
        scratch == NULL || result == NULL ||
        logits_capacity < KQ_MODEL_EXEC_VOCABULARY_SIZE)
        return model_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                          "model incremental-decode arguments are invalid");
    if (input_token_id >= KQ_MODEL_EXEC_CANONICAL_TOKEN_LIMIT)
        return model_fail(diagnostic, KQ_STATUS_INVALID_TOKEN_ID,
                          "decode input is a padded or invalid canonical token ID");
    if (state->requires_reset || state->position == 0U)
        return model_fail(diagnostic, KQ_STATUS_INVALID_MODEL_STATE,
                          "incremental decode requires successful prefill state");
    if (state->position >= config->context_capacity)
        return model_fail(diagnostic, KQ_STATUS_LIMIT_EXCEEDED,
                          "incremental decode exhausted context capacity");
    status = required_scratch_internal(config, state, 1U, &required,
                                       &layer_scratch, 1, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    if (scratch_bytes < required)
        return model_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                          "model incremental-decode scratch is too small");
    if (!mul_u64(KQ_MODEL_EXEC_VOCABULARY_SIZE, sizeof(float), &logits_bytes))
        return model_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "model incremental-decode buffer size overflows");
    if (ranges_overlap(logits, logits_bytes, scratch, scratch_bytes) ||
        ranges_overlap(decoded_utf8, decoded_utf8_capacity, scratch, scratch_bytes) ||
        ranges_overlap(logits, logits_bytes, decoded_utf8, decoded_utf8_capacity))
        return model_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                          "model incremental outputs and scratch overlap");

    before = kq_weight_provider_get_metrics(config->provider);
    if (before == NULL) return KQ_STATUS_INCOMPATIBLE_MODEL_EXEC;
    before_copy = *before;
    if (QueryPerformanceFrequency(&frequency) != 0)
        (void)QueryPerformanceCounter(&started);
    partition_scratch(1U, scratch, layer_scratch, &buffers);

    embedding_before = before_copy.logical_payload_bytes_touched;
    status = kq_weight_provider_row_f32(
        config->provider, config->embedding, input_token_id,
        KQ_MODEL_EXEC_HIDDEN_SIZE, buffers.embedding,
        KQ_MODEL_EXEC_HIDDEN_SIZE, buffers.provider_scratch,
        buffers.provider_scratch_bytes, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    for (branch = 0U; branch < KQ_MODEL_EXEC_BRANCH_COUNT; ++branch)
        memcpy(buffers.branches_a + branch * KQ_MODEL_EXEC_HIDDEN_SIZE,
               buffers.embedding, KQ_MODEL_EXEC_HIDDEN_SIZE * sizeof(float));
    after = kq_weight_provider_get_metrics(config->provider);
    if (after == NULL || after->logical_payload_bytes_touched < embedding_before)
        return KQ_STATUS_INCOMPATIBLE_MODEL_EXEC;
    embedding_after = after->logical_payload_bytes_touched;
    staged_result.metrics.embedding_logical_bytes_touched =
        embedding_after - embedding_before;
    emit_progress(observer, observer_user_data,
                  KQ_MODEL_EXEC_PHASE_EMBEDDING_COMPLETE, UINT32_MAX,
                  buffers.embedding, KQ_MODEL_EXEC_HIDDEN_SIZE);

    current = buffers.branches_a;
    next = buffers.branches_b;
    for (layer = 0U; layer < KQ_MODEL_EXEC_LAYER_COUNT; ++layer) {
        kq_layer_metrics layer_metrics;
        float *swap;
        emit_progress(observer, observer_user_data,
                      KQ_MODEL_EXEC_PHASE_LAYER_BEGIN, layer, NULL, 0U);
        memset(&layer_metrics, 0, sizeof(layer_metrics));
        status = kq_layer_decode_quantized_f32(
            config->layers[layer], config->provider, current, input_token_id,
            next, KQ_MODEL_EXEC_FINAL_WIDTH, state->layers[layer],
            buffers.layer_scratch, buffers.layer_scratch_bytes,
            NULL, NULL, &layer_metrics, diagnostic);
        if (status != KQ_STATUS_OK) goto rollback;
        committed_layers += 1U;
        staged_result.metrics.layers_completed += 1U;
        staged_result.metrics.qsa_selection_events +=
            layer_metrics.qsa_selection_events;
        staged_result.metrics.qsa_candidate_blocks +=
            layer_metrics.qsa_candidate_blocks;
        staged_result.metrics.qsa_selected_blocks +=
            layer_metrics.qsa_selected_blocks;
        staged_result.metrics.qsa_selected_tokens +=
            layer_metrics.qsa_selected_tokens;
        emit_progress(observer, observer_user_data,
                      KQ_MODEL_EXEC_PHASE_LAYER_COMPLETE, layer, next,
                      KQ_MODEL_EXEC_FINAL_WIDTH);
        swap = current;
        current = next;
        next = swap;
    }

    status = final_mix(config, current, &buffers, diagnostic);
    if (status != KQ_STATUS_OK) goto rollback;
    emit_progress(observer, observer_user_data,
                  KQ_MODEL_EXEC_PHASE_FINAL_MIX_COMPLETE, UINT32_MAX,
                  buffers.hidden, KQ_MODEL_EXEC_HIDDEN_SIZE);

    before = kq_weight_provider_get_metrics(config->provider);
    if (before == NULL) {
        status = KQ_STATUS_INCOMPATIBLE_MODEL_EXEC;
        goto rollback;
    }
    lm_before = before->logical_payload_bytes_touched;
    status = kq_weight_provider_linear_f32(
        config->provider, config->lm_head, KQ_BINDING_PART_WHOLE,
        KQ_WEIGHT_PROVIDER_NO_EXPERT, KQ_MODEL_EXEC_VOCABULARY_SIZE,
        KQ_MODEL_EXEC_HIDDEN_SIZE, buffers.hidden, buffers.logits,
        KQ_MODEL_EXEC_VOCABULARY_SIZE, buffers.provider_scratch,
        buffers.provider_scratch_bytes, diagnostic);
    if (status != KQ_STATUS_OK) goto rollback;
    after = kq_weight_provider_get_metrics(config->provider);
    if (after == NULL || after->logical_payload_bytes_touched < lm_before) {
        status = KQ_STATUS_INCOMPATIBLE_MODEL_EXEC;
        goto rollback;
    }
    staged_result.metrics.lm_head_logical_bytes_touched =
        after->logical_payload_bytes_touched - lm_before;
    emit_progress(observer, observer_user_data,
                  KQ_MODEL_EXEC_PHASE_LOGITS_COMPLETE, UINT32_MAX,
                  buffers.logits, KQ_MODEL_EXEC_VOCABULARY_SIZE);

    status = kq_model_exec_greedy_argmax_f32(
        buffers.logits, KQ_MODEL_EXEC_VOCABULARY_SIZE, &selected, diagnostic);
    if (status != KQ_STATUS_OK) goto rollback;
    if (selected >= KQ_MODEL_EXEC_CANONICAL_TOKEN_LIMIT) {
        status = model_fail(diagnostic, KQ_STATUS_INVALID_TOKEN_ID,
                            "greedy model output is a padded non-tokenizer ID");
        goto rollback;
    }
    decode_options.special_policy = KQ_TOKENIZER_DECODE_KEEP_SPECIAL;
    status = kq_tokenizer_decode(config->tokenizer, &selected, 1U,
                                 &decode_options, NULL, 0U,
                                 &decoded_required, diagnostic);
    if (status != KQ_STATUS_BUFFER_TOO_SMALL &&
        !(status == KQ_STATUS_OK && decoded_required == 0U)) goto rollback;
    if (decoded_utf8_capacity < decoded_required ||
        decoded_required > sizeof(staged_decoded)) {
        status = model_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                            "decoded incremental token output is too small");
        goto rollback;
    }
    status = kq_tokenizer_decode(config->tokenizer, &selected, 1U,
                                 &decode_options, staged_decoded,
                                 sizeof(staged_decoded), &decoded_required,
                                 diagnostic);
    if (status != KQ_STATUS_OK) goto rollback;

    staged_result.selected_token_id = selected;
    staged_result.selected_token_is_eog = kq_model_exec_token_is_eog(selected);
    staged_result.decoded_utf8_bytes = decoded_required;
    staged_result.metrics.input_tokens = 1U;
    staged_result.metrics.incremental_decode_count = 1U;
    staged_result.metrics.persistent_state_bytes = state->owned_bytes;
    staged_result.metrics.peak_scratch_bytes = required;
    staged_result.metrics.logits_bytes = logits_bytes;
    after = kq_weight_provider_get_metrics(config->provider);
    if (after == NULL ||
        after->logical_payload_bytes_touched < before_copy.logical_payload_bytes_touched ||
        after->quantized_blocks_touched < before_copy.quantized_blocks_touched ||
        after->selected_expert_requests < before_copy.selected_expert_requests ||
        after->ple_row_requests < before_copy.ple_row_requests) {
        status = KQ_STATUS_INCOMPATIBLE_MODEL_EXEC;
        goto rollback;
    }
    staged_result.metrics.logical_payload_bytes_touched =
        after->logical_payload_bytes_touched - before_copy.logical_payload_bytes_touched;
    staged_result.metrics.payload_blocks_touched =
        after->quantized_blocks_touched - before_copy.quantized_blocks_touched;
    staged_result.metrics.unique_semantic_tensors_touched =
        after->unique_semantic_tensors_touched;
    staged_result.metrics.routed_expert_member_requests =
        after->selected_expert_requests - before_copy.selected_expert_requests;
    staged_result.metrics.selected_expert_matrix_requests =
        staged_result.metrics.routed_expert_member_requests;
    if (staged_result.metrics.selected_expert_matrix_requests % 3U != 0U) {
        status = model_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MODEL_EXEC,
                            "routed expert matrix request count is incomplete");
        goto rollback;
    }
    staged_result.metrics.routed_expert_selections =
        staged_result.metrics.selected_expert_matrix_requests / 3U;
    staged_result.metrics.ple_row_requests =
        after->ple_row_requests - before_copy.ple_row_requests;
    staged_result.metrics.maximum_f32_weight_bytes_materialized =
        after->maximum_f32_weight_bytes_materialized;
    if (frequency.QuadPart > 0 && QueryPerformanceCounter(&finished) != 0)
        (void)qpc_elapsed_ns(started, finished, frequency,
                             &staged_result.metrics.elapsed_nanoseconds);

    memcpy(logits, buffers.logits, (size_t)logits_bytes);
    if (decoded_required != 0U)
        memcpy(decoded_utf8, staged_decoded, (size_t)decoded_required);
    state->position += 1U;
    *result = staged_result;
    emit_progress(observer, observer_user_data,
                  KQ_MODEL_EXEC_PHASE_TOKEN_SELECTED, UINT32_MAX, NULL, 0U);
    return KQ_STATUS_OK;

rollback:
    if (diagnostic != NULL) original_diagnostic = *diagnostic;
    else kq_diagnostic_clear(&original_diagnostic);
    return rollback_decode_layers(state, committed_layers, status,
                                  &original_diagnostic, diagnostic);
}

kq_status kq_model_exec_generate_first_token_f32(
    const kq_model_exec_config *config, kq_model_exec_state *state,
    const unsigned char *prompt_utf8, uint64_t prompt_utf8_bytes,
    float *logits, uint64_t logits_capacity,
    unsigned char *decoded_utf8, uint64_t decoded_utf8_capacity,
    void *scratch, uint64_t scratch_bytes,
    kq_model_exec_progress_observer observer, void *observer_user_data,
    kq_model_exec_result *result, kq_diagnostic *diagnostic) {
    kq_tokenizer_encode_options options;
    uint32_t *tokens = NULL;
    uint64_t token_count = 0U;
    size_t allocation_bytes;
    kq_status status;
    if (!config_valid(config) || prompt_utf8 == NULL || prompt_utf8_bytes == 0U)
        return model_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                          "non-empty UTF-8 prompt and valid model config are required");
    options.special_policy = KQ_TOKENIZER_SPECIAL_REJECT;
    status = kq_tokenizer_encode(config->tokenizer, prompt_utf8,
                                 prompt_utf8_bytes, &options, NULL, 0U,
                                 &token_count, diagnostic);
    if (status != KQ_STATUS_BUFFER_TOO_SMALL || token_count == 0U) return status;
    if (token_count >= config->context_capacity ||
        token_count > SIZE_MAX / sizeof(*tokens))
        return model_fail(diagnostic, KQ_STATUS_LIMIT_EXCEEDED,
                          "prompt token count exceeds configured M1 context");
    allocation_bytes = (size_t)token_count * sizeof(*tokens);
    tokens = (uint32_t *)malloc(allocation_bytes);
    if (tokens == NULL)
        return model_fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                          "could not allocate bounded prompt token IDs");
    status = kq_tokenizer_encode(config->tokenizer, prompt_utf8,
                                 prompt_utf8_bytes, &options, tokens,
                                 token_count, &token_count, diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_exec_prefill_first_token_f32(
            config, state, tokens, token_count, logits, logits_capacity,
            decoded_utf8, decoded_utf8_capacity, scratch, scratch_bytes,
            observer, observer_user_data, result, diagnostic);
    free(tokens);
    return status;
}
