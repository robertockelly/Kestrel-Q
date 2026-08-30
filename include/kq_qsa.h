#ifndef KQ_QSA_H
#define KQ_QSA_H

#include <stdint.h>

#include "kq_model.h"
#include "kq_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KQ_QSA_MAX_CHECKPOINT_RANK 3U

typedef struct kq_qsa_config kq_qsa_config;
typedef struct kq_qsa_state kq_qsa_state;

typedef enum kq_qsa_activation_dtype {
    KQ_QSA_ACTIVATION_F32 = 0,
    KQ_QSA_ACTIVATION_BF16
} kq_qsa_activation_dtype;

typedef enum kq_qsa_checkpoint_kind {
    KQ_QSA_CHECKPOINT_INDEX_QUERY = 0,
    KQ_QSA_CHECKPOINT_RAW_INDEX_KEY,
    KQ_QSA_CHECKPOINT_BLOCK_KEYS,
    KQ_QSA_CHECKPOINT_CANDIDATE_SCORES,
    KQ_QSA_CHECKPOINT_QUERY,
    KQ_QSA_CHECKPOINT_KEY,
    KQ_QSA_CHECKPOINT_VALUE,
    KQ_QSA_CHECKPOINT_ATTENTION_LOGITS,
    KQ_QSA_CHECKPOINT_ATTENTION_PROBABILITIES,
    KQ_QSA_CHECKPOINT_HEAD_CONTEXT,
    KQ_QSA_CHECKPOINT_GATED_CONTEXT,
    KQ_QSA_CHECKPOINT_OPERATOR_OUTPUT
} kq_qsa_checkpoint_kind;

typedef struct kq_qsa_checkpoint {
    kq_qsa_checkpoint_kind kind;
    uint64_t token_index;
    uint32_t rank;
    uint64_t dimensions[KQ_QSA_MAX_CHECKPOINT_RANK];
    const float *values;
    uint64_t value_count;
} kq_qsa_checkpoint;

typedef void (*kq_qsa_checkpoint_observer)(
    const kq_qsa_checkpoint *checkpoint,
    void *user_data);

typedef struct kq_qsa_selection {
    uint64_t token_index;
    uint64_t absolute_position;
    const uint32_t *candidate_block_ids;
    const float *candidate_scores;
    uint64_t candidate_count;
    const uint32_t *selected_block_ids;
    uint64_t selected_block_count;
    const uint32_t *selected_token_positions;
    uint64_t selected_token_count;
    uint64_t tail_count;
} kq_qsa_selection;

typedef void (*kq_qsa_selection_observer)(
    const kq_qsa_selection *selection,
    void *user_data);

typedef struct kq_qsa_weights_f32 {
    const float *query;
    uint64_t query_count;
    const float *key;
    uint64_t key_count;
    const float *value;
    uint64_t value_count;
    const float *output;
    uint64_t output_count;
    const float *query_norm;
    uint64_t query_norm_count;
    const float *key_norm;
    uint64_t key_norm_count;
    const float *index_query;
    uint64_t index_query_count;
    const float *index_key;
    uint64_t index_key_count;
    const float *index_query_norm;
    uint64_t index_query_norm_count;
    const float *index_key_norm;
    uint64_t index_key_norm_count;
} kq_qsa_weights_f32;

kq_status kq_qsa_config_create(const kq_model *model,
                               uint32_t layer_id,
                               kq_qsa_config **out_config,
                               kq_diagnostic *diagnostic);
kq_status kq_qsa_config_create_reference_f32(
    const kq_model *model,
    uint32_t layer_id,
    kq_qsa_config **out_config,
    kq_diagnostic *diagnostic);
void kq_qsa_config_close(kq_qsa_config *config);

uint32_t kq_qsa_config_layer_id(const kq_qsa_config *config);
uint32_t kq_qsa_config_hidden_size(const kq_qsa_config *config);
uint32_t kq_qsa_config_query_head_count(const kq_qsa_config *config);
uint32_t kq_qsa_config_key_value_head_count(const kq_qsa_config *config);
uint32_t kq_qsa_config_head_dimension(const kq_qsa_config *config);
uint32_t kq_qsa_config_index_query_head_count(const kq_qsa_config *config);
uint32_t kq_qsa_config_index_head_dimension(const kq_qsa_config *config);
uint32_t kq_qsa_config_block_size(const kq_qsa_config *config);
uint32_t kq_qsa_config_block_selection_limit(const kq_qsa_config *config);
uint32_t kq_qsa_config_context_limit(const kq_qsa_config *config);
kq_qsa_activation_dtype kq_qsa_config_activation_dtype(
    const kq_qsa_config *config);
uint64_t kq_qsa_config_owned_bytes(const kq_qsa_config *config);
uint64_t kq_qsa_config_semantic_state_bytes_per_token(
    const kq_qsa_config *config);

kq_status kq_qsa_state_create(const kq_qsa_config *config,
                              uint64_t token_capacity,
                              kq_qsa_state **out_state,
                              kq_diagnostic *diagnostic);
void kq_qsa_state_close(kq_qsa_state *state);
kq_status kq_qsa_state_reset(kq_qsa_state *state,
                             kq_diagnostic *diagnostic);
uint64_t kq_qsa_state_length(const kq_qsa_state *state);
uint64_t kq_qsa_state_capacity(const kq_qsa_state *state);
uint64_t kq_qsa_state_owned_bytes(const kq_qsa_state *state);

kq_status kq_qsa_state_export_f32(
    const kq_qsa_state *state,
    float *key_cache,
    uint64_t key_capacity,
    float *value_cache,
    uint64_t value_capacity,
    float *raw_index_key_cache,
    uint64_t raw_index_key_capacity,
    uint64_t *length,
    kq_diagnostic *diagnostic);
kq_status kq_qsa_state_import_f32(
    kq_qsa_state *state,
    const float *key_cache,
    uint64_t key_count,
    const float *value_cache,
    uint64_t value_count,
    const float *raw_index_key_cache,
    uint64_t raw_index_key_count,
    uint64_t length,
    kq_diagnostic *diagnostic);

kq_status kq_qsa_required_scratch_bytes(
    const kq_qsa_config *config,
    const kq_qsa_state *state,
    uint64_t sequence_length,
    uint64_t *scratch_bytes,
    kq_diagnostic *diagnostic);

kq_status kq_qsa_select_blocks_f32(
    const float *candidate_scores,
    uint64_t candidate_count,
    uint64_t selection_limit,
    uint32_t *selected_block_ids,
    uint64_t selected_capacity,
    uint64_t *selected_count,
    kq_diagnostic *diagnostic);

kq_status kq_qsa_prefill_f32(
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

kq_status kq_qsa_decode_f32(
    const kq_qsa_config *config,
    const kq_qsa_weights_f32 *weights,
    const float *hidden_token,
    float *output_token,
    uint64_t output_capacity,
    kq_qsa_state *state,
    void *scratch,
    uint64_t scratch_bytes,
    kq_qsa_selection_observer selection_observer,
    kq_qsa_checkpoint_observer checkpoint_observer,
    void *observer_user_data,
    kq_diagnostic *diagnostic);

const char *kq_qsa_checkpoint_kind_name(kq_qsa_checkpoint_kind kind);

#ifdef __cplusplus
}
#endif

#endif
