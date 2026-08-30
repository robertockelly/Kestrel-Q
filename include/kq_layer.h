#ifndef KQ_LAYER_H
#define KQ_LAYER_H

#include <stdint.h>

#include "kq_gdn.h"
#include "kq_moe.h"
#include "kq_ple.h"
#include "kq_ple_value.h"
#include "kq_qsa.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KQ_LAYER_MAX_CHECKPOINT_RANK 3U

typedef struct kq_layer_config kq_layer_config;
typedef struct kq_layer_state kq_layer_state;

typedef enum kq_layer_family {
    KQ_LAYER_FAMILY_INVALID = 0,
    KQ_LAYER_FAMILY_GDN,
    KQ_LAYER_FAMILY_QSA,
    KQ_LAYER_FAMILY_PLE_GDN
} kq_layer_family;

typedef enum kq_layer_checkpoint_kind {
    KQ_LAYER_CHECKPOINT_INPUT = 0,
    KQ_LAYER_CHECKPOINT_PLE_OUTPUT,
    KQ_LAYER_CHECKPOINT_PLE_ENHANCED_INPUT,
    KQ_LAYER_CHECKPOINT_ATTN_GR_NORMALIZED,
    KQ_LAYER_CHECKPOINT_ATTN_GR_READ_GATE,
    KQ_LAYER_CHECKPOINT_MIXER_INPUT,
    KQ_LAYER_CHECKPOINT_MIXER_OUTPUT,
    KQ_LAYER_CHECKPOINT_ATTN_GR_WRITE_GATE,
    KQ_LAYER_CHECKPOINT_AFTER_MIXER_RESIDUAL,
    KQ_LAYER_CHECKPOINT_MOE_GR_NORMALIZED,
    KQ_LAYER_CHECKPOINT_MOE_GR_READ_GATE,
    KQ_LAYER_CHECKPOINT_MOE_INPUT,
    KQ_LAYER_CHECKPOINT_MOE_OUTPUT,
    KQ_LAYER_CHECKPOINT_MOE_GR_WRITE_GATE,
    KQ_LAYER_CHECKPOINT_OUTPUT
} kq_layer_checkpoint_kind;

typedef struct kq_layer_checkpoint {
    kq_layer_checkpoint_kind kind;
    uint64_t token_index;
    uint32_t rank;
    uint64_t dimensions[KQ_LAYER_MAX_CHECKPOINT_RANK];
    const float *values;
    uint64_t value_count;
} kq_layer_checkpoint;

typedef void (*kq_layer_checkpoint_observer)(
    const kq_layer_checkpoint *checkpoint, void *user_data);

typedef struct kq_layer_gr_weights_f32 {
    const float *norm;
    uint64_t norm_count;
    const float *down;
    uint64_t down_count;
    const float *up;
    uint64_t up_count;
    const float *inject;
    uint64_t inject_count;
} kq_layer_gr_weights_f32;

typedef struct kq_layer_weights_f32 {
    kq_layer_gr_weights_f32 attention_gr;
    kq_layer_gr_weights_f32 moe_gr;
    const kq_gdn_weights_f32 *gdn;
    const kq_qsa_weights_f32 *qsa;
    const kq_moe_weights_f32 *moe;
    const kq_ple_value_weights_f32 *ple_value;
    const kq_ple_value_lookup_provider *ple_provider;
} kq_layer_weights_f32;

typedef struct kq_layer_metrics {
    uint64_t elapsed_nanoseconds;
    uint64_t tokens_processed;
    uint64_t scratch_bytes;
    uint64_t gr_workspace_bytes;
    uint64_t transaction_staging_bytes;
} kq_layer_metrics;

/* The model/GGUF/file owner must outlive the returned immutable config. */
kq_status kq_layer_config_create(const kq_model *model, uint32_t layer_id,
                                 kq_layer_config **out_config,
                                 kq_diagnostic *diagnostic);
kq_status kq_layer_config_create_reference_f32(
    const kq_model *model, uint32_t layer_id, kq_layer_config **out_config,
    kq_diagnostic *diagnostic);
void kq_layer_config_close(kq_layer_config *config);

uint32_t kq_layer_config_layer_id(const kq_layer_config *config);
kq_layer_family kq_layer_config_family(const kq_layer_config *config);
uint32_t kq_layer_config_hidden_size(const kq_layer_config *config);
uint32_t kq_layer_config_branch_count(const kq_layer_config *config);
uint32_t kq_layer_config_gr_rank(const kq_layer_config *config);
uint64_t kq_layer_config_owned_bytes(const kq_layer_config *config);
uint64_t kq_layer_config_gr_workspace_bytes(const kq_layer_config *config);
uint64_t kq_layer_config_persistent_semantic_bytes(
    const kq_layer_config *config, uint64_t qsa_token_capacity);

kq_status kq_layer_state_create(const kq_layer_config *config,
                                uint64_t qsa_token_capacity,
                                kq_layer_state **out_state,
                                kq_diagnostic *diagnostic);
void kq_layer_state_close(kq_layer_state *state);
kq_status kq_layer_state_reset(kq_layer_state *state,
                               kq_diagnostic *diagnostic);
uint64_t kq_layer_state_owned_bytes(const kq_layer_state *state);
uint64_t kq_layer_state_position(const kq_layer_state *state);

kq_status kq_layer_required_scratch_bytes(
    const kq_layer_config *config, const kq_layer_state *state,
    uint64_t token_count, uint64_t *scratch_bytes,
    kq_diagnostic *diagnostic);

kq_status kq_layer_prefill_f32(
    const kq_layer_config *config, const kq_layer_weights_f32 *weights,
    const float *hidden_branches, const uint32_t *token_ids,
    uint64_t token_count, const uint8_t *padding_mask, float *output,
    uint64_t output_capacity, kq_layer_state *state, void *scratch,
    uint64_t scratch_bytes, kq_layer_checkpoint_observer observer,
    void *observer_user_data, kq_layer_metrics *metrics,
    kq_diagnostic *diagnostic);

kq_status kq_layer_decode_f32(
    const kq_layer_config *config, const kq_layer_weights_f32 *weights,
    const float *hidden_branches, uint32_t token_id, float *output,
    uint64_t output_capacity, kq_layer_state *state, void *scratch,
    uint64_t scratch_bytes, kq_layer_checkpoint_observer observer,
    void *observer_user_data, kq_layer_metrics *metrics,
    kq_diagnostic *diagnostic);

const char *kq_layer_family_name(kq_layer_family family);
const char *kq_layer_checkpoint_kind_name(kq_layer_checkpoint_kind kind);

#ifdef __cplusplus
}
#endif

#endif
