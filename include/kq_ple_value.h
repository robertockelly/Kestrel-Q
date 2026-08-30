#ifndef KQ_PLE_VALUE_H
#define KQ_PLE_VALUE_H

#include <stdint.h>

#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_ple.h"
#include "kq_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KQ_PLE_VALUE_MAX_CHECKPOINT_RANK 3U

typedef struct kq_ple_value_config kq_ple_value_config;
typedef struct kq_ple_value_state kq_ple_value_state;
typedef struct kq_ple_value_gguf_provider kq_ple_value_gguf_provider;

typedef enum kq_ple_value_activation_dtype {
    KQ_PLE_VALUE_ACTIVATION_F32 = 0,
    KQ_PLE_VALUE_ACTIVATION_BF16
} kq_ple_value_activation_dtype;

typedef enum kq_ple_value_checkpoint_kind {
    KQ_PLE_VALUE_CHECKPOINT_RAW_LOOKUPS = 0,
    KQ_PLE_VALUE_CHECKPOINT_EMBEDDING,
    KQ_PLE_VALUE_CHECKPOINT_KEY_PROJECTION,
    KQ_PLE_VALUE_CHECKPOINT_VALUE_PROJECTION,
    KQ_PLE_VALUE_CHECKPOINT_KEY_NORM,
    KQ_PLE_VALUE_CHECKPOINT_QUERY_NORM,
    KQ_PLE_VALUE_CHECKPOINT_GATE_RAW,
    KQ_PLE_VALUE_CHECKPOINT_GATE_TRANSFORMED,
    KQ_PLE_VALUE_CHECKPOINT_GATED_VALUE,
    KQ_PLE_VALUE_CHECKPOINT_CONV_NORM,
    KQ_PLE_VALUE_CHECKPOINT_CONV_PRE_ACTIVATION,
    KQ_PLE_VALUE_CHECKPOINT_CONV_OUTPUT,
    KQ_PLE_VALUE_CHECKPOINT_OPERATOR_OUTPUT
} kq_ple_value_checkpoint_kind;

typedef struct kq_ple_value_checkpoint {
    kq_ple_value_checkpoint_kind kind;
    uint64_t token_index;
    uint32_t rank;
    uint64_t dimensions[KQ_PLE_VALUE_MAX_CHECKPOINT_RANK];
    const float *values;
    uint64_t value_count;
} kq_ple_value_checkpoint;

typedef void (*kq_ple_value_checkpoint_observer)(
    const kq_ple_value_checkpoint *checkpoint, void *user_data);

typedef kq_status (*kq_ple_value_lookup_row_f32)(
    void *user_data, uint32_t logical_member, uint64_t member_row,
    float *output, uint64_t output_capacity, kq_diagnostic *diagnostic);

typedef struct kq_ple_value_lookup_provider {
    void *user_data;
    kq_ple_value_lookup_row_f32 lookup_row;
    uint32_t logical_member_count;
    uint64_t member_rows;
    uint32_t row_width;
} kq_ple_value_lookup_provider;

typedef struct kq_ple_value_weights_f32 {
    const float *key_projection;
    uint64_t key_projection_count;
    const float *value_projection;
    uint64_t value_projection_count;
    const float *norm_key;
    uint64_t norm_key_count;
    const float *norm_query;
    uint64_t norm_query_count;
    const float *norm_conv;
    uint64_t norm_conv_count;
    const float *convolution;
    uint64_t convolution_count;
} kq_ple_value_weights_f32;

typedef struct kq_ple_value_run_metrics {
    uint64_t elapsed_nanoseconds;
    uint64_t tokens_processed;
    uint64_t lookups_performed;
    uint64_t scratch_bytes;
} kq_ple_value_run_metrics;

typedef struct kq_ple_value_gguf_metrics {
    uint64_t logical_payload_bytes_touched;
    uint64_t blocks_touched;
    uint64_t rows_read;
    uint64_t budget_bytes;
} kq_ple_value_gguf_metrics;

kq_status kq_ple_value_config_create(
    const kq_model *model, const kq_ple_config *address_config,
    kq_ple_value_config **out_config, kq_diagnostic *diagnostic);
kq_status kq_ple_value_config_create_reference_f32(
    const kq_model *model, const kq_ple_config *address_config,
    kq_ple_value_config **out_config, kq_diagnostic *diagnostic);
void kq_ple_value_config_close(kq_ple_value_config *config);

uint32_t kq_ple_value_config_layer_id(const kq_ple_value_config *config);
uint32_t kq_ple_value_config_hidden_size(const kq_ple_value_config *config);
uint32_t kq_ple_value_config_residual_branches(const kq_ple_value_config *config);
uint32_t kq_ple_value_config_head_count(const kq_ple_value_config *config);
uint32_t kq_ple_value_config_row_width(const kq_ple_value_config *config);
uint32_t kq_ple_value_config_history_length(const kq_ple_value_config *config);
uint64_t kq_ple_value_config_owned_bytes(const kq_ple_value_config *config);
uint64_t kq_ple_value_config_state_bytes(const kq_ple_value_config *config);
uint64_t kq_ple_value_config_semantic_state_bytes(const kq_ple_value_config *config);
uint64_t kq_ple_value_config_scratch_bytes(const kq_ple_value_config *config);
kq_ple_value_activation_dtype kq_ple_value_config_activation_dtype(
    const kq_ple_value_config *config);

kq_status kq_ple_value_state_create(const kq_ple_value_config *config,
                                    kq_ple_value_state **out_state,
                                    kq_diagnostic *diagnostic);
void kq_ple_value_state_close(kq_ple_value_state *state);
kq_status kq_ple_value_state_reset(const kq_ple_value_config *config,
                                   kq_ple_value_state *state,
                                   kq_diagnostic *diagnostic);
kq_status kq_ple_value_state_import_f32(const kq_ple_value_config *config,
                                        kq_ple_value_state *state,
                                        const float *history,
                                        uint64_t history_count,
                                        uint64_t position,
                                        kq_diagnostic *diagnostic);
kq_status kq_ple_value_state_export_f32(const kq_ple_value_config *config,
                                        const kq_ple_value_state *state,
                                        float *history,
                                        uint64_t history_capacity,
                                        uint64_t *history_count,
                                        uint64_t *position,
                                        kq_diagnostic *diagnostic);

kq_status kq_ple_value_prefill_f32(
    const kq_ple_value_config *config, kq_ple_value_state *state,
    const kq_ple_value_weights_f32 *weights,
    const kq_ple_value_lookup_provider *provider,
    const float *hidden_states, uint64_t token_count,
    const kq_ple_address_intent *intents, uint64_t intent_count,
    float *output, uint64_t output_capacity,
    void *scratch, uint64_t scratch_bytes,
    kq_ple_value_checkpoint_observer observer, void *observer_user_data,
    kq_ple_value_run_metrics *metrics, kq_diagnostic *diagnostic);

kq_status kq_ple_value_decode_f32(
    const kq_ple_value_config *config, kq_ple_value_state *state,
    const kq_ple_value_weights_f32 *weights,
    const kq_ple_value_lookup_provider *provider,
    const float *hidden_token,
    const kq_ple_address_intent *intents, uint64_t intent_count,
    float *output, uint64_t output_capacity,
    void *scratch, uint64_t scratch_bytes,
    kq_ple_value_checkpoint_observer observer, void *observer_user_data,
    kq_ple_value_run_metrics *metrics, kq_diagnostic *diagnostic);

kq_status kq_ple_value_gguf_provider_open(
    const kq_gguf *gguf, const kq_model *model,
    const kq_ple_value_config *config, uint64_t budget_bytes,
    kq_ple_value_gguf_provider **out_provider, kq_diagnostic *diagnostic);
void kq_ple_value_gguf_provider_close(kq_ple_value_gguf_provider *provider);
kq_ple_value_lookup_provider kq_ple_value_gguf_provider_interface(
    kq_ple_value_gguf_provider *provider);
const kq_ple_value_gguf_metrics *kq_ple_value_gguf_provider_metrics(
    const kq_ple_value_gguf_provider *provider);

const char *kq_ple_value_checkpoint_kind_name(kq_ple_value_checkpoint_kind kind);

#ifdef __cplusplus
}
#endif

#endif
