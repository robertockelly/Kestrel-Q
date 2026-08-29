#ifndef KQ_PLE_H
#define KQ_PLE_H

#include <stdint.h>

#include "kq_model.h"
#include "kq_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KQ_PLE_NGRAM_SIZE 3U
#define KQ_PLE_HEADS_PER_ORDER 8U
#define KQ_PLE_HEAD_COUNT 16U
#define KQ_PLE_LOGICAL_MEMBER_COUNT 128U
#define KQ_PLE_MEMBER_ROWS UINT64_C(2500012)
#define KQ_PLE_ADDRESSES_PER_TOKEN 16U
#define KQ_PLE_HISTORY_LENGTH 2U

typedef struct kq_ple_config kq_ple_config;

typedef struct kq_ple_config_info {
    uint32_t model_vocabulary_size;
    uint32_t model_context_length;
    uint32_t ple_layer_id;
    uint32_t eos_token_id;
    uint32_t ngram_size;
    uint32_t heads_per_order;
    uint32_t head_count;
    uint32_t logical_member_count;
    uint64_t member_rows;
    uint64_t active_rows;
    uint64_t padded_rows;
    uint32_t addresses_per_token;
} kq_ple_config_info;

typedef struct kq_ple_config_metrics {
    uint64_t construction_nanoseconds;
    uint64_t owned_heap_bytes;
    uint64_t stream_state_bytes;
} kq_ple_config_metrics;

typedef struct kq_ple_run_metrics {
    uint64_t elapsed_nanoseconds;
    uint64_t tokens_processed;
    uint64_t addresses_emitted;
} kq_ple_run_metrics;

typedef struct kq_ple_stream_state {
    uint64_t position;
    uint64_t integrity;
    uint32_t history[KQ_PLE_HISTORY_LENGTH];
    uint32_t version;
    uint32_t reserved;
} kq_ple_stream_state;

typedef struct kq_ple_address_intent {
    uint64_t position;
    uint64_t global_address;
    uint64_t head_offset;
    uint64_t head_vocab_size;
    uint64_t member_row;
    uint32_t token_id;
    uint32_t ngram_order;
    uint32_t local_head;
    uint32_t global_head;
    uint32_t logical_member;
} kq_ple_address_intent;

kq_status kq_ple_config_open_from_model(const kq_model *model,
                                        kq_ple_config **out_config,
                                        kq_diagnostic *diagnostic);
void kq_ple_config_close(kq_ple_config *config);

const kq_ple_config_info *kq_ple_config_get_info(
    const kq_ple_config *config);
const kq_ple_config_metrics *kq_ple_config_get_metrics(
    const kq_ple_config *config);

kq_status kq_ple_state_reset(const kq_ple_config *config,
                             kq_ple_stream_state *state,
                             kq_diagnostic *diagnostic);
kq_status kq_ple_state_validate(const kq_ple_config *config,
                                const kq_ple_stream_state *state,
                                kq_diagnostic *diagnostic);

kq_status kq_ple_generate_prefill(
    const kq_ple_config *config,
    kq_ple_stream_state *state,
    const uint32_t *token_ids,
    uint64_t token_count,
    kq_ple_address_intent *intents,
    uint64_t intent_capacity,
    uint64_t *required_intents,
    kq_ple_run_metrics *metrics,
    kq_diagnostic *diagnostic);

kq_status kq_ple_generate_decode_step(
    const kq_ple_config *config,
    kq_ple_stream_state *state,
    uint32_t token_id,
    kq_ple_address_intent *intents,
    uint64_t intent_capacity,
    uint64_t *required_intents,
    kq_ple_run_metrics *metrics,
    kq_diagnostic *diagnostic);

#ifdef __cplusplus
}
#endif

#endif
