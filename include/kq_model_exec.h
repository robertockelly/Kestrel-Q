#ifndef KQ_MODEL_EXEC_H
#define KQ_MODEL_EXEC_H

#include <stdint.h>

#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_status.h"
#include "kq_tokenizer.h"
#include "kq_weight_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KQ_MODEL_EXEC_LAYER_COUNT 48U
#define KQ_MODEL_EXEC_HIDDEN_SIZE 2560U
#define KQ_MODEL_EXEC_BRANCH_COUNT 4U
#define KQ_MODEL_EXEC_VOCABULARY_SIZE 248320U
#define KQ_MODEL_EXEC_CANONICAL_TOKEN_LIMIT 248077U

typedef struct kq_model_exec_config kq_model_exec_config;
typedef struct kq_model_exec_state kq_model_exec_state;

typedef enum kq_model_exec_phase {
    KQ_MODEL_EXEC_PHASE_TOKENIZED = 0,
    KQ_MODEL_EXEC_PHASE_EMBEDDING_COMPLETE,
    KQ_MODEL_EXEC_PHASE_LAYER_BEGIN,
    KQ_MODEL_EXEC_PHASE_LAYER_COMPLETE,
    KQ_MODEL_EXEC_PHASE_FINAL_MIX_COMPLETE,
    KQ_MODEL_EXEC_PHASE_LOGITS_COMPLETE,
    KQ_MODEL_EXEC_PHASE_TOKEN_SELECTED
} kq_model_exec_phase;

typedef struct kq_model_exec_progress_event {
    kq_model_exec_phase phase;
    uint32_t layer_id;
    uint32_t layer_count;
    /* Borrowed last-token checkpoint, valid only during the callback. */
    const float *last_token_values;
    uint64_t last_token_value_count;
} kq_model_exec_progress_event;

typedef void (*kq_model_exec_progress_observer)(
    const kq_model_exec_progress_event *event, void *user_data);

typedef struct kq_model_exec_metrics {
    uint64_t prompt_tokens;
    uint64_t input_tokens;
    uint64_t prompt_prefill_count;
    uint64_t incremental_decode_count;
    uint64_t layers_completed;
    uint64_t logical_payload_bytes_touched;
    uint64_t payload_blocks_touched;
    uint64_t unique_semantic_tensors_touched;
    uint64_t embedding_logical_bytes_touched;
    uint64_t routed_expert_selections;
    uint64_t selected_expert_matrix_requests;
    uint64_t routed_expert_member_requests;
    uint64_t ple_row_requests;
    uint64_t qsa_selection_events;
    uint64_t qsa_candidate_blocks;
    uint64_t qsa_selected_blocks;
    uint64_t qsa_selected_tokens;
    uint64_t lm_head_logical_bytes_touched;
    uint64_t persistent_state_bytes;
    uint64_t peak_scratch_bytes;
    uint64_t logits_bytes;
    uint64_t maximum_f32_weight_bytes_materialized;
    uint64_t elapsed_nanoseconds;
} kq_model_exec_metrics;

typedef struct kq_model_exec_state_summary {
    uint64_t model_position;
    uint64_t layer_position_min;
    uint64_t layer_position_max;
    uint64_t qsa_sequence_length_min;
    uint64_t qsa_sequence_length_max;
    uint64_t qsa_complete_blocks_min;
    uint64_t qsa_complete_blocks_max;
    uint64_t qsa_incomplete_tail_min;
    uint64_t qsa_incomplete_tail_max;
    uint64_t ple_address_position;
    uint64_t ple_value_position;
    uint64_t gdn_state_hash;
    uint64_t qsa_state_hash;
    uint64_t ple_address_state_hash;
    uint64_t ple_value_state_hash;
    uint64_t structural_hash;
    uint32_t gdn_initialized_layers;
    uint32_t qsa_layers;
} kq_model_exec_state_summary;

typedef struct kq_model_exec_result {
    uint32_t selected_token_id;
    int selected_token_is_eog;
    uint64_t decoded_utf8_bytes;
    kq_model_exec_metrics metrics;
} kq_model_exec_result;

/* All supplied GGUF/model/tokenizer/provider owners must outlive the config. */
kq_status kq_model_exec_config_open(
    const kq_gguf *gguf, const kq_model *model,
    const kq_tokenizer *tokenizer, kq_weight_provider *provider,
    uint64_t context_capacity, kq_model_exec_config **out_config,
    kq_diagnostic *diagnostic);
void kq_model_exec_config_close(kq_model_exec_config *config);

uint64_t kq_model_exec_context_capacity(const kq_model_exec_config *config);
uint64_t kq_model_exec_config_owned_bytes(const kq_model_exec_config *config);

kq_status kq_model_exec_state_create(
    const kq_model_exec_config *config, kq_model_exec_state **out_state,
    kq_diagnostic *diagnostic);
void kq_model_exec_state_close(kq_model_exec_state *state);
kq_status kq_model_exec_state_reset(kq_model_exec_state *state,
                                    kq_diagnostic *diagnostic);
uint64_t kq_model_exec_state_position(const kq_model_exec_state *state);
uint64_t kq_model_exec_state_owned_bytes(const kq_model_exec_state *state);
kq_status kq_model_exec_state_get_summary(
    const kq_model_exec_state *state,
    kq_model_exec_state_summary *summary,
    kq_diagnostic *diagnostic);

kq_status kq_model_exec_required_scratch_bytes(
    const kq_model_exec_config *config, const kq_model_exec_state *state,
    uint64_t prompt_token_count, uint64_t *scratch_bytes,
    kq_diagnostic *diagnostic);

kq_status kq_model_exec_required_decode_scratch_bytes(
    const kq_model_exec_config *config, const kq_model_exec_state *state,
    uint64_t *scratch_bytes, kq_diagnostic *diagnostic);

kq_status kq_model_exec_prefill_first_token_f32(
    const kq_model_exec_config *config, kq_model_exec_state *state,
    const uint32_t *token_ids, uint64_t token_count,
    float *logits, uint64_t logits_capacity,
    unsigned char *decoded_utf8, uint64_t decoded_utf8_capacity,
    void *scratch, uint64_t scratch_bytes,
    kq_model_exec_progress_observer observer, void *observer_user_data,
    kq_model_exec_result *result, kq_diagnostic *diagnostic);

kq_status kq_model_exec_generate_first_token_f32(
    const kq_model_exec_config *config, kq_model_exec_state *state,
    const unsigned char *prompt_utf8, uint64_t prompt_utf8_bytes,
    float *logits, uint64_t logits_capacity,
    unsigned char *decoded_utf8, uint64_t decoded_utf8_capacity,
    void *scratch, uint64_t scratch_bytes,
    kq_model_exec_progress_observer observer, void *observer_user_data,
    kq_model_exec_result *result, kq_diagnostic *diagnostic);

/* Consumes exactly one already-selected canonical token and produces the
   following greedy token. State and caller outputs commit atomically. */
kq_status kq_model_exec_decode_one_f32(
    const kq_model_exec_config *config, kq_model_exec_state *state,
    uint32_t input_token_id,
    float *logits, uint64_t logits_capacity,
    unsigned char *decoded_utf8, uint64_t decoded_utf8_capacity,
    void *scratch, uint64_t scratch_bytes,
    kq_model_exec_progress_observer observer, void *observer_user_data,
    kq_model_exec_result *result, kq_diagnostic *diagnostic);

int kq_model_exec_token_is_eog(uint32_t token_id);

kq_status kq_model_exec_greedy_argmax_f32(
    const float *logits, uint64_t logit_count, uint32_t *selected_token_id,
    kq_diagnostic *diagnostic);

#ifdef __cplusplus
}
#endif

#endif
