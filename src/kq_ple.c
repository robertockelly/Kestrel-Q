#include "kq_ple.h"

#include <limits.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include "kq_internal.h"
#include "kq_ple_internal.h"

#define KQ_PLE_HIDDEN_SIZE 2560U
#define KQ_PLE_VOCABULARY_SIZE 248320U
#define KQ_PLE_CONTEXT_LENGTH 262144U
#define KQ_PLE_LAYER_COUNT 48U
#define KQ_PLE_GDN_LAYER_COUNT 36U
#define KQ_PLE_QSA_LAYER_COUNT 12U
#define KQ_PLE_LAYER_ID 1U
#define KQ_PLE_EOS_TOKEN_ID 248044U
#define KQ_PLE_TABLE_WIDTH 160U
#define KQ_PLE_DENSE_SEMANTICS 6U
#define KQ_PLE_METADATA_SEMANTICS 3U
#define KQ_PLE_ACTIVE_ROWS UINT64_C(320001446)
#define KQ_PLE_PADDED_ROWS UINT64_C(320001536)
#define KQ_PLE_STATE_VERSION 1U
#define KQ_PLE_STATE_HASH_OFFSET UINT64_C(1469598103934665603)
#define KQ_PLE_STATE_HASH_PRIME UINT64_C(1099511628211)

static const uint64_t kq_ple_expected_multipliers[KQ_PLE_NGRAM_SIZE] = {
    UINT64_C(23703573157769),
    UINT64_C(20109073645365),
    UINT64_C(8052911324071)
};

static const uint64_t kq_ple_expected_offsets[KQ_PLE_HEAD_COUNT] = {
    UINT64_C(0), UINT64_C(20000003), UINT64_C(40000026),
    UINT64_C(60000059), UINT64_C(80000106), UINT64_C(100000165),
    UINT64_C(120000228), UINT64_C(140000297), UINT64_C(160000374),
    UINT64_C(180000455), UINT64_C(200000548), UINT64_C(220000655),
    UINT64_C(240000802), UINT64_C(260000955), UINT64_C(280001114),
    UINT64_C(300001275)
};

static const uint64_t kq_ple_expected_vocab_sizes[KQ_PLE_HEAD_COUNT] = {
    UINT64_C(20000003), UINT64_C(20000023), UINT64_C(20000033),
    UINT64_C(20000047), UINT64_C(20000059), UINT64_C(20000063),
    UINT64_C(20000069), UINT64_C(20000077), UINT64_C(20000081),
    UINT64_C(20000093), UINT64_C(20000107), UINT64_C(20000147),
    UINT64_C(20000153), UINT64_C(20000159), UINT64_C(20000161),
    UINT64_C(20000171)
};

struct kq_ple_config {
    kq_ple_config_info info;
    kq_ple_config_metrics metrics;
    uint64_t multipliers[KQ_PLE_NGRAM_SIZE];
    uint64_t head_offsets[KQ_PLE_HEAD_COUNT];
    uint64_t head_vocab_sizes[KQ_PLE_HEAD_COUNT];
    uint64_t state_cookie;
};

static kq_status kq_ple_fail(kq_diagnostic *diagnostic,
                             kq_status status,
                             const char *format,
                             ...) {
    va_list arguments;
    char message[KQ_DIAGNOSTIC_CAPACITY];

    message[0] = '\0';
    if (format != NULL) {
        va_start(arguments, format);
        (void)vsnprintf(message, sizeof(message), format, arguments);
        va_end(arguments);
        message[sizeof(message) - 1U] = '\0';
    }
    kq_diagnostic_set(diagnostic, status, "%s", message);
    return status;
}

static uint64_t kq_ple_now_nanoseconds(void) {
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    uint64_t seconds;
    uint64_t remainder;

    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0 ||
        !QueryPerformanceCounter(&counter) || counter.QuadPart < 0) {
        return 0U;
    }
    seconds = (uint64_t)counter.QuadPart / (uint64_t)frequency.QuadPart;
    remainder = (uint64_t)counter.QuadPart % (uint64_t)frequency.QuadPart;
    if (seconds > UINT64_MAX / UINT64_C(1000000000)) {
        return 0U;
    }
    if (remainder > UINT64_MAX / UINT64_C(1000000000)) {
        return 0U;
    }
    return seconds * UINT64_C(1000000000) +
           (remainder * UINT64_C(1000000000)) /
               (uint64_t)frequency.QuadPart;
}

static uint64_t kq_ple_hash_word(uint64_t hash, uint64_t value) {
    unsigned int byte_index;
    for (byte_index = 0U; byte_index < 8U; ++byte_index) {
        hash ^= (value >> (byte_index * 8U)) & UINT64_C(0xff);
        hash *= KQ_PLE_STATE_HASH_PRIME;
    }
    return hash;
}

static uint64_t kq_ple_config_cookie(const kq_ple_config *config) {
    uint64_t hash = KQ_PLE_STATE_HASH_OFFSET;
    uint32_t index;

    hash = kq_ple_hash_word(hash, config->info.model_vocabulary_size);
    hash = kq_ple_hash_word(hash, config->info.model_context_length);
    hash = kq_ple_hash_word(hash, config->info.eos_token_id);
    hash = kq_ple_hash_word(hash, config->info.active_rows);
    for (index = 0U; index < KQ_PLE_NGRAM_SIZE; ++index) {
        hash = kq_ple_hash_word(hash, config->multipliers[index]);
    }
    for (index = 0U; index < KQ_PLE_HEAD_COUNT; ++index) {
        hash = kq_ple_hash_word(hash, config->head_offsets[index]);
        hash = kq_ple_hash_word(hash, config->head_vocab_sizes[index]);
    }
    return hash;
}

static uint64_t kq_ple_state_integrity(const kq_ple_config *config,
                                       const kq_ple_stream_state *state) {
    uint64_t hash = KQ_PLE_STATE_HASH_OFFSET;
    hash = kq_ple_hash_word(hash, config->state_cookie);
    hash = kq_ple_hash_word(hash, state->position);
    hash = kq_ple_hash_word(hash, state->history[0]);
    hash = kq_ple_hash_word(hash, state->history[1]);
    hash = kq_ple_hash_word(hash, state->version);
    hash = kq_ple_hash_word(hash, state->reserved);
    return hash;
}

static int kq_ple_semantic_metadata_array(
    const kq_model *model,
    const char *semantic_id,
    kq_semantic_role expected_role,
    uint64_t expected_count,
    uint64_t *values) {
    const kq_semantic_tensor *semantic =
        kq_model_find_semantic_tensor(model, semantic_id);
    const kq_tensor_binding *binding;
    uint64_t index;

    if (semantic == NULL ||
        semantic->component != KQ_COMPONENT_PLE_ADDRESS_METADATA ||
        semantic->role != expected_role ||
        semantic->relation != KQ_BINDING_METADATA_DERIVED ||
        semantic->layer_id != KQ_PLE_LAYER_ID ||
        semantic->layer_type != KQ_MODEL_LAYER_GDN ||
        semantic->canonical_dtype != KQ_CANONICAL_DTYPE_I64 ||
        semantic->canonical_rank != 1U ||
        semantic->canonical_dimensions[0] != expected_count ||
        semantic->binding_count != 1U) {
        return 0;
    }
    binding = &semantic->bindings[0];
    if (binding->part_role != KQ_BINDING_PART_METADATA ||
        binding->physical != NULL || binding->metadata == NULL ||
        binding->metadata->value_type != KQ_GGUF_VALUE_ARRAY ||
        binding->metadata->array_element_type != KQ_GGUF_VALUE_UINT64 ||
        binding->metadata->array_length != expected_count) {
        return 0;
    }
    for (index = 0U; index < expected_count; ++index) {
        if (!kq_gguf_metadata_array_u64_at(binding->metadata,
                                           index,
                                           &values[(size_t)index])) {
            return 0;
        }
    }
    return 1;
}

static int kq_ple_validate_table_semantics(const kq_model *model) {
    char semantic_id[KQ_SEMANTIC_ID_CAPACITY];
    const kq_semantic_tensor *semantic;
    const kq_tensor_binding *binding;
    uint32_t member;
    int written;

    for (member = 0U; member < KQ_PLE_LOGICAL_MEMBER_COUNT; ++member) {
        written = snprintf(semantic_id,
                           sizeof(semantic_id),
                           "layer.01.ple.table.%03u",
                           (unsigned int)member);
        if (written < 0 || (size_t)written >= sizeof(semantic_id)) {
            return 0;
        }
        semantic = kq_model_find_semantic_tensor(model, semantic_id);
        if (semantic == NULL || semantic->component != KQ_COMPONENT_PLE_TABLE ||
            semantic->role != KQ_ROLE_PLE_TABLE ||
            semantic->relation != KQ_BINDING_MULTIPLE_CANONICAL_TO_ONE_PHYSICAL ||
            semantic->layer_id != KQ_PLE_LAYER_ID ||
            semantic->layer_type != KQ_MODEL_LAYER_GDN ||
            semantic->canonical_rank != 2U ||
            semantic->canonical_dimensions[0] != KQ_PLE_MEMBER_ROWS ||
            semantic->canonical_dimensions[1] != KQ_PLE_TABLE_WIDTH ||
            semantic->binding_count != 1U) {
            return 0;
        }
        binding = &semantic->bindings[0];
        if (binding->physical == NULL || binding->metadata != NULL ||
            binding->part_role != KQ_BINDING_PART_FUSED_MEMBER ||
            binding->fused_member_index != member ||
            binding->fused_member_count != KQ_PLE_LOGICAL_MEMBER_COUNT) {
            return 0;
        }
    }
    return 1;
}

static kq_status kq_ple_descriptor_from_model(
    const kq_model *model,
    kq_ple_compatibility_descriptor *descriptor,
    kq_diagnostic *diagnostic) {
    if (model == NULL || descriptor == NULL) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INVALID_ARGUMENT,
                           "PLE model and descriptor are required");
    }
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->hidden_size = kq_model_hidden_size(model);
    descriptor->vocabulary_size = kq_model_vocabulary_size(model);
    descriptor->context_length = kq_model_context_length(model);
    descriptor->layer_count = kq_model_layer_count(model);
    descriptor->gdn_layer_count = kq_model_gdn_layer_count(model);
    descriptor->qsa_layer_count = kq_model_qsa_layer_count(model);
    descriptor->ple_layer_id = KQ_PLE_LAYER_ID;
    descriptor->ple_layer_type =
        kq_model_layer_type_at(model, KQ_PLE_LAYER_ID);
    descriptor->ngram_size = KQ_PLE_NGRAM_SIZE;
    descriptor->heads_per_order = KQ_PLE_HEADS_PER_ORDER;
    descriptor->eos_token_id = KQ_PLE_EOS_TOKEN_ID;
    descriptor->logical_member_count = KQ_PLE_LOGICAL_MEMBER_COUNT;
    descriptor->member_rows = KQ_PLE_MEMBER_ROWS;
    descriptor->table_width = KQ_PLE_TABLE_WIDTH;
    descriptor->ple_table_semantic_count = kq_model_component_count(
        model, KQ_COMPONENT_PLE_TABLE);
    descriptor->ple_dense_semantic_count = kq_model_component_count(
        model, KQ_COMPONENT_PLE_DENSE);
    descriptor->ple_metadata_semantic_count =
        kq_model_component_count(model, KQ_COMPONENT_PLE_ADDRESS_METADATA);
    descriptor->table_semantics_valid =
        kq_ple_validate_table_semantics(model);
    descriptor->metadata_semantics_valid =
        kq_ple_semantic_metadata_array(
            model,
            "layer.01.ple.address.layer_multipliers",
            KQ_ROLE_PLE_LAYER_MULTIPLIERS,
            KQ_PLE_NGRAM_SIZE,
            descriptor->multipliers) &&
        kq_ple_semantic_metadata_array(
            model,
            "layer.01.ple.address.head_offsets",
            KQ_ROLE_PLE_HEAD_OFFSETS,
            KQ_PLE_HEAD_COUNT,
            descriptor->head_offsets) &&
        kq_ple_semantic_metadata_array(
            model,
            "layer.01.ple.address.head_vocab_sizes",
            KQ_ROLE_PLE_HEAD_VOCAB_SIZES,
            KQ_PLE_HEAD_COUNT,
            descriptor->head_vocab_sizes);
    descriptor->payload_bytes_accessed = 0U;
    return KQ_STATUS_OK;
}

static kq_status kq_ple_validate_descriptor(
    const kq_ple_compatibility_descriptor *descriptor,
    kq_diagnostic *diagnostic) {
    uint64_t cumulative = 0U;
    uint64_t padded_rows;
    uint32_t index;

    if (descriptor == NULL) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INVALID_ARGUMENT,
                           "PLE compatibility descriptor is required");
    }
    if (descriptor->hidden_size != KQ_PLE_HIDDEN_SIZE ||
        descriptor->vocabulary_size != KQ_PLE_VOCABULARY_SIZE ||
        descriptor->context_length != KQ_PLE_CONTEXT_LENGTH ||
        descriptor->layer_count != KQ_PLE_LAYER_COUNT ||
        descriptor->gdn_layer_count != KQ_PLE_GDN_LAYER_COUNT ||
        descriptor->qsa_layer_count != KQ_PLE_QSA_LAYER_COUNT ||
        descriptor->ple_layer_id != KQ_PLE_LAYER_ID ||
        descriptor->ple_layer_type != KQ_MODEL_LAYER_GDN) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INCOMPATIBLE_PLE,
                           "model identity or PLE layer topology is incompatible");
    }
    if (descriptor->ngram_size != KQ_PLE_NGRAM_SIZE ||
        descriptor->heads_per_order != KQ_PLE_HEADS_PER_ORDER ||
        descriptor->eos_token_id != KQ_PLE_EOS_TOKEN_ID ||
        descriptor->logical_member_count != KQ_PLE_LOGICAL_MEMBER_COUNT ||
        descriptor->member_rows != KQ_PLE_MEMBER_ROWS ||
        descriptor->table_width != KQ_PLE_TABLE_WIDTH ||
        descriptor->ple_table_semantic_count !=
            KQ_PLE_LOGICAL_MEMBER_COUNT ||
        descriptor->ple_dense_semantic_count != KQ_PLE_DENSE_SEMANTICS ||
        descriptor->ple_metadata_semantic_count !=
            KQ_PLE_METADATA_SEMANTICS ||
        !descriptor->table_semantics_valid ||
        !descriptor->metadata_semantics_valid) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INCOMPATIBLE_PLE,
                           "PLE order/head/member geometry or semantics are incompatible");
    }
    if (descriptor->payload_bytes_accessed != 0U) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INCOMPATIBLE_PLE,
                           "PLE configuration source reports tensor payload access");
    }
    for (index = 0U; index < KQ_PLE_NGRAM_SIZE; ++index) {
        if (descriptor->multipliers[index] !=
                kq_ple_expected_multipliers[index] ||
            descriptor->multipliers[index] == 0U ||
            (descriptor->multipliers[index] & UINT64_C(1)) == 0U ||
            (uint64_t)(descriptor->vocabulary_size - 1U) >
                (uint64_t)INT64_MAX / descriptor->multipliers[index]) {
            return kq_ple_fail(diagnostic,
                               KQ_STATUS_INCOMPATIBLE_PLE,
                               "PLE multiplier %u is incompatible",
                               (unsigned int)index);
        }
    }
    for (index = 0U; index < KQ_PLE_HEAD_COUNT; ++index) {
        if (descriptor->head_offsets[index] != cumulative ||
            descriptor->head_offsets[index] != kq_ple_expected_offsets[index] ||
            descriptor->head_vocab_sizes[index] !=
                kq_ple_expected_vocab_sizes[index] ||
            descriptor->head_vocab_sizes[index] == 0U ||
            cumulative > UINT64_MAX - descriptor->head_vocab_sizes[index]) {
            return kq_ple_fail(diagnostic,
                               KQ_STATUS_INCOMPATIBLE_PLE,
                               "PLE head %u offset or modulus is incompatible",
                               (unsigned int)index);
        }
        cumulative += descriptor->head_vocab_sizes[index];
    }
    if (descriptor->member_rows >
        UINT64_MAX / descriptor->logical_member_count) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "PLE member geometry overflows total rows");
    }
    padded_rows = descriptor->member_rows *
                  descriptor->logical_member_count;
    if (cumulative != KQ_PLE_ACTIVE_ROWS || padded_rows != KQ_PLE_PADDED_ROWS ||
        cumulative > padded_rows) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INCOMPATIBLE_PLE,
                           "PLE active or padded row geometry is incompatible");
    }
    return KQ_STATUS_OK;
}

static kq_status kq_ple_config_open_from_descriptor(
    const kq_ple_compatibility_descriptor *descriptor,
    kq_ple_config **out_config,
    kq_diagnostic *diagnostic) {
    kq_ple_config *config;
    kq_status status;
    uint64_t started;
    uint64_t finished;

    if (out_config == NULL) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INVALID_ARGUMENT,
                           "PLE output configuration pointer is required");
    }
    *out_config = NULL;
    kq_diagnostic_clear(diagnostic);
    started = kq_ple_now_nanoseconds();
    status = kq_ple_validate_descriptor(descriptor, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    config = (kq_ple_config *)calloc(1U, sizeof(*config));
    if (config == NULL) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_OUT_OF_MEMORY,
                           "could not allocate immutable PLE configuration");
    }
    config->info.model_vocabulary_size = descriptor->vocabulary_size;
    config->info.model_context_length = descriptor->context_length;
    config->info.ple_layer_id = descriptor->ple_layer_id;
    config->info.eos_token_id = descriptor->eos_token_id;
    config->info.ngram_size = descriptor->ngram_size;
    config->info.heads_per_order = descriptor->heads_per_order;
    config->info.head_count = KQ_PLE_HEAD_COUNT;
    config->info.logical_member_count = descriptor->logical_member_count;
    config->info.member_rows = descriptor->member_rows;
    config->info.active_rows = KQ_PLE_ACTIVE_ROWS;
    config->info.padded_rows = KQ_PLE_PADDED_ROWS;
    config->info.addresses_per_token = KQ_PLE_ADDRESSES_PER_TOKEN;
    memcpy(config->multipliers,
           descriptor->multipliers,
           sizeof(config->multipliers));
    memcpy(config->head_offsets,
           descriptor->head_offsets,
           sizeof(config->head_offsets));
    memcpy(config->head_vocab_sizes,
           descriptor->head_vocab_sizes,
           sizeof(config->head_vocab_sizes));
    config->metrics.owned_heap_bytes = sizeof(*config);
    config->metrics.stream_state_bytes = sizeof(kq_ple_stream_state);
    config->state_cookie = kq_ple_config_cookie(config);
    finished = kq_ple_now_nanoseconds();
    if (started != 0U && finished >= started) {
        config->metrics.construction_nanoseconds = finished - started;
    }
    *out_config = config;
    return KQ_STATUS_OK;
}

kq_status kq_ple_config_open_from_model(const kq_model *model,
                                        kq_ple_config **out_config,
                                        kq_diagnostic *diagnostic) {
    kq_ple_compatibility_descriptor descriptor;
    kq_status status;

    if (out_config == NULL) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INVALID_ARGUMENT,
                           "PLE output configuration pointer is required");
    }
    *out_config = NULL;
    status = kq_ple_descriptor_from_model(model, &descriptor, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    return kq_ple_config_open_from_descriptor(&descriptor,
                                              out_config,
                                              diagnostic);
}

kq_status kq_ple_config_open_from_descriptor_for_test(
    const kq_ple_compatibility_descriptor *descriptor,
    kq_ple_config **out_config,
    kq_diagnostic *diagnostic) {
    return kq_ple_config_open_from_descriptor(descriptor,
                                              out_config,
                                              diagnostic);
}

void kq_ple_config_close(kq_ple_config *config) {
    free(config);
}

const kq_ple_config_info *kq_ple_config_get_info(
    const kq_ple_config *config) {
    return config == NULL ? NULL : &config->info;
}

const kq_ple_config_metrics *kq_ple_config_get_metrics(
    const kq_ple_config *config) {
    return config == NULL ? NULL : &config->metrics;
}

kq_status kq_ple_state_reset(const kq_ple_config *config,
                             kq_ple_stream_state *state,
                             kq_diagnostic *diagnostic) {
    if (config == NULL || state == NULL) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INVALID_ARGUMENT,
                           "PLE configuration and stream state are required");
    }
    memset(state, 0, sizeof(*state));
    state->history[0] = config->info.eos_token_id;
    state->history[1] = config->info.eos_token_id;
    state->version = KQ_PLE_STATE_VERSION;
    state->integrity = kq_ple_state_integrity(config, state);
    kq_diagnostic_clear(diagnostic);
    return KQ_STATUS_OK;
}

kq_status kq_ple_state_validate(const kq_ple_config *config,
                                const kq_ple_stream_state *state,
                                kq_diagnostic *diagnostic) {
    if (config == NULL || state == NULL) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INVALID_ARGUMENT,
                           "PLE configuration and stream state are required");
    }
    if (state->version != KQ_PLE_STATE_VERSION || state->reserved != 0U ||
        state->position > config->info.model_context_length ||
        state->history[0] >= config->info.model_vocabulary_size ||
        state->history[1] >= config->info.model_vocabulary_size ||
        (state->position == 0U &&
         (state->history[0] != config->info.eos_token_id ||
          state->history[1] != config->info.eos_token_id)) ||
        (state->position == 1U &&
         state->history[0] != config->info.eos_token_id) ||
        state->integrity != kq_ple_state_integrity(config, state)) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INVALID_PLE_STATE,
                           "PLE stream state failed bounds or integrity validation");
    }
    kq_diagnostic_clear(diagnostic);
    return KQ_STATUS_OK;
}

static kq_status kq_ple_checked_product(uint32_t token_id,
                                        uint64_t multiplier,
                                        uint64_t *product,
                                        kq_diagnostic *diagnostic) {
    if (multiplier != 0U &&
        (uint64_t)token_id > (uint64_t)INT64_MAX / multiplier) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "PLE token/multiplier product exceeds signed int64");
    }
    *product = (uint64_t)token_id * multiplier;
    return KQ_STATUS_OK;
}

static kq_status kq_ple_generate_one(
    const kq_ple_config *config,
    kq_ple_stream_state *state,
    uint32_t token_id,
    kq_ple_address_intent *intents,
    kq_diagnostic *diagnostic) {
    uint32_t shifted[KQ_PLE_NGRAM_SIZE];
    uint64_t products[KQ_PLE_NGRAM_SIZE];
    uint64_t mixed;
    uint64_t head_row;
    uint64_t global_address;
    uint64_t member;
    uint64_t row;
    uint32_t order;
    uint32_t local_head;
    uint32_t global_head;
    kq_status status;

    shifted[0] = token_id;
    shifted[1] = state->history[1] == config->info.eos_token_id
                     ? config->info.eos_token_id
                     : state->history[1];
    shifted[2] = state->history[1] == config->info.eos_token_id ||
                         state->history[0] == config->info.eos_token_id
                     ? config->info.eos_token_id
                     : state->history[0];
    for (order = 0U; order < KQ_PLE_NGRAM_SIZE; ++order) {
        status = kq_ple_checked_product(shifted[order],
                                        config->multipliers[order],
                                        &products[order],
                                        diagnostic);
        if (status != KQ_STATUS_OK) {
            return status;
        }
    }
    for (order = 2U; order <= KQ_PLE_NGRAM_SIZE; ++order) {
        mixed = products[0];
        for (global_head = 1U; global_head < order; ++global_head) {
            mixed ^= products[global_head];
        }
        if (mixed > (uint64_t)INT64_MAX) {
            return kq_ple_fail(diagnostic,
                               KQ_STATUS_ARITHMETIC_OVERFLOW,
                               "PLE XOR result exceeds signed int64");
        }
        for (local_head = 0U; local_head < KQ_PLE_HEADS_PER_ORDER;
             ++local_head) {
            global_head = (order - 2U) * KQ_PLE_HEADS_PER_ORDER +
                          local_head;
            head_row = mixed % config->head_vocab_sizes[global_head];
            if (config->head_offsets[global_head] >
                UINT64_MAX - head_row) {
                return kq_ple_fail(diagnostic,
                                   KQ_STATUS_ARITHMETIC_OVERFLOW,
                                   "PLE global address addition overflowed");
            }
            global_address = config->head_offsets[global_head] + head_row;
            member = global_address / config->info.member_rows;
            row = global_address % config->info.member_rows;
            if (global_address >= config->info.active_rows ||
                member >= config->info.logical_member_count ||
                row >= config->info.member_rows) {
                return kq_ple_fail(diagnostic,
                                   KQ_STATUS_INCOMPATIBLE_PLE,
                                   "PLE address is outside canonical table bounds");
            }
            intents[(size_t)global_head].position = state->position;
            intents[(size_t)global_head].global_address = global_address;
            intents[(size_t)global_head].head_offset =
                config->head_offsets[global_head];
            intents[(size_t)global_head].head_vocab_size =
                config->head_vocab_sizes[global_head];
            intents[(size_t)global_head].member_row = row;
            intents[(size_t)global_head].token_id = token_id;
            intents[(size_t)global_head].ngram_order = order;
            intents[(size_t)global_head].local_head = local_head;
            intents[(size_t)global_head].global_head = global_head;
            intents[(size_t)global_head].logical_member = (uint32_t)member;
        }
    }
    state->history[0] = state->history[1];
    state->history[1] = token_id;
    state->position += 1U;
    state->integrity = kq_ple_state_integrity(config, state);
    return KQ_STATUS_OK;
}

kq_status kq_ple_generate_prefill(
    const kq_ple_config *config,
    kq_ple_stream_state *state,
    const uint32_t *token_ids,
    uint64_t token_count,
    kq_ple_address_intent *intents,
    uint64_t intent_capacity,
    uint64_t *required_intents,
    kq_ple_run_metrics *metrics,
    kq_diagnostic *diagnostic) {
    kq_ple_stream_state working;
    uint64_t required;
    uint64_t token_index;
    uint64_t started;
    uint64_t finished;
    kq_status status;

    if (required_intents == NULL) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INVALID_ARGUMENT,
                           "PLE required-intent output is required");
    }
    *required_intents = 0U;
    if (metrics != NULL) {
        memset(metrics, 0, sizeof(*metrics));
    }
    if (config == NULL || state == NULL ||
        (token_count != 0U && token_ids == NULL)) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INVALID_ARGUMENT,
                           "PLE configuration, state, and non-empty input are required");
    }
    status = kq_ple_state_validate(config, state, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (token_count > UINT64_MAX / KQ_PLE_ADDRESSES_PER_TOKEN) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "PLE intent count multiplication overflowed");
    }
    required = token_count * KQ_PLE_ADDRESSES_PER_TOKEN;
    *required_intents = required;
    if (token_count > config->info.model_context_length - state->position) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_LIMIT_EXCEEDED,
                           "PLE stream would exceed model context length");
    }
    if (required > (uint64_t)(SIZE_MAX / sizeof(kq_ple_address_intent))) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "PLE output pointer arithmetic would overflow");
    }
    for (token_index = 0U; token_index < token_count; ++token_index) {
        if (token_ids[(size_t)token_index] >=
            config->info.model_vocabulary_size) {
            return kq_ple_fail(diagnostic,
                               KQ_STATUS_INVALID_TOKEN_ID,
                               "PLE token ID at index %llu exceeds model vocabulary",
                               (unsigned long long)token_index);
        }
    }
    if (required != 0U &&
        (intents == NULL || intent_capacity < required)) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_BUFFER_TOO_SMALL,
                           "PLE output requires %llu intents",
                           (unsigned long long)required);
    }
    if (required == 0U && intents == NULL && intent_capacity != 0U) {
        return kq_ple_fail(diagnostic,
                           KQ_STATUS_INVALID_ARGUMENT,
                           "nonzero PLE capacity requires an output pointer");
    }

    working = *state;
    started = kq_ple_now_nanoseconds();
    for (token_index = 0U; token_index < token_count; ++token_index) {
        status = kq_ple_generate_one(
            config,
            &working,
            token_ids[(size_t)token_index],
            intents + (size_t)(token_index * KQ_PLE_ADDRESSES_PER_TOKEN),
            diagnostic);
        if (status != KQ_STATUS_OK) {
            return status;
        }
    }
    finished = kq_ple_now_nanoseconds();
    *state = working;
    if (metrics != NULL) {
        metrics->tokens_processed = token_count;
        metrics->addresses_emitted = required;
        if (started != 0U && finished >= started) {
            metrics->elapsed_nanoseconds = finished - started;
        }
    }
    kq_diagnostic_clear(diagnostic);
    return KQ_STATUS_OK;
}

kq_status kq_ple_generate_decode_step(
    const kq_ple_config *config,
    kq_ple_stream_state *state,
    uint32_t token_id,
    kq_ple_address_intent *intents,
    uint64_t intent_capacity,
    uint64_t *required_intents,
    kq_ple_run_metrics *metrics,
    kq_diagnostic *diagnostic) {
    return kq_ple_generate_prefill(config,
                                   state,
                                   &token_id,
                                   1U,
                                   intents,
                                   intent_capacity,
                                   required_intents,
                                   metrics,
                                   diagnostic);
}
