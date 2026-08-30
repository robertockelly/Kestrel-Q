#ifndef KQ_GDN_H
#define KQ_GDN_H

#include <stdint.h>

#include "kq_model.h"
#include "kq_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KQ_GDN_MAX_CHECKPOINT_RANK 4U

typedef struct kq_gdn_config kq_gdn_config;
typedef struct kq_gdn_state kq_gdn_state;

typedef enum kq_gdn_activation_dtype {
    KQ_GDN_ACTIVATION_F32 = 0,
    KQ_GDN_ACTIVATION_BF16
} kq_gdn_activation_dtype;

typedef enum kq_gdn_checkpoint_kind {
    KQ_GDN_CHECKPOINT_MASKED_INPUT = 0,
    KQ_GDN_CHECKPOINT_PROJECTED_QKV,
    KQ_GDN_CHECKPOINT_PROJECTED_GATE,
    KQ_GDN_CHECKPOINT_PROJECTED_BETA,
    KQ_GDN_CHECKPOINT_PROJECTED_ALPHA,
    KQ_GDN_CHECKPOINT_CONV_INPUT,
    KQ_GDN_CHECKPOINT_CONV_OUTPUT,
    KQ_GDN_CHECKPOINT_QUERY_BEFORE_NORM,
    KQ_GDN_CHECKPOINT_KEY_BEFORE_NORM,
    KQ_GDN_CHECKPOINT_VALUE,
    KQ_GDN_CHECKPOINT_NORMALIZED_SCALED_QUERY,
    KQ_GDN_CHECKPOINT_NORMALIZED_KEY,
    KQ_GDN_CHECKPOINT_LOG_DECAY,
    KQ_GDN_CHECKPOINT_BETA,
    KQ_GDN_CHECKPOINT_RECURRENT_READ,
    KQ_GDN_CHECKPOINT_RECURRENT_DELTA,
    KQ_GDN_CHECKPOINT_RECURRENT_OUTPUT,
    KQ_GDN_CHECKPOINT_RECURRENT_STATE,
    KQ_GDN_CHECKPOINT_CONV_STATE,
    KQ_GDN_CHECKPOINT_GATED_NORM_OUTPUT,
    KQ_GDN_CHECKPOINT_OPERATOR_OUTPUT
} kq_gdn_checkpoint_kind;

typedef struct kq_gdn_checkpoint {
    kq_gdn_checkpoint_kind kind;
    uint64_t token_index;
    uint32_t rank;
    uint64_t dimensions[KQ_GDN_MAX_CHECKPOINT_RANK];
    const float *values;
    uint64_t value_count;
} kq_gdn_checkpoint;

typedef void (*kq_gdn_checkpoint_observer)(
    const kq_gdn_checkpoint *checkpoint,
    void *user_data);

typedef struct kq_gdn_weights_f32 {
    const float *a_log;
    uint64_t a_log_count;
    const float *conv;
    uint64_t conv_count;
    const float *dt_bias;
    uint64_t dt_bias_count;
    const float *alpha;
    uint64_t alpha_count;
    const float *beta;
    uint64_t beta_count;
    const float *qkv;
    uint64_t qkv_count;
    const float *gate;
    uint64_t gate_count;
    const float *norm;
    uint64_t norm_count;
    const float *output;
    uint64_t output_count;
} kq_gdn_weights_f32;

/* The model and its GGUF/file owner must outlive the returned config. */
kq_status kq_gdn_config_create(const kq_model *model,
                               uint32_t layer_id,
                               kq_gdn_config **out_config,
                               kq_diagnostic *diagnostic);
/* F32 correctness-oracle config over the same validated target semantics. */
kq_status kq_gdn_config_create_reference_f32(
    const kq_model *model,
    uint32_t layer_id,
    kq_gdn_config **out_config,
    kq_diagnostic *diagnostic);
void kq_gdn_config_close(kq_gdn_config *config);

uint32_t kq_gdn_config_layer_id(const kq_gdn_config *config);
uint32_t kq_gdn_config_hidden_size(const kq_gdn_config *config);
uint32_t kq_gdn_config_key_head_count(const kq_gdn_config *config);
uint32_t kq_gdn_config_value_head_count(const kq_gdn_config *config);
uint32_t kq_gdn_config_key_head_dimension(const kq_gdn_config *config);
uint32_t kq_gdn_config_value_head_dimension(const kq_gdn_config *config);
uint32_t kq_gdn_config_conv_channel_count(const kq_gdn_config *config);
uint32_t kq_gdn_config_conv_kernel_size(const kq_gdn_config *config);
kq_gdn_activation_dtype kq_gdn_config_activation_dtype(
    const kq_gdn_config *config);
uint64_t kq_gdn_config_owned_bytes(const kq_gdn_config *config);
uint64_t kq_gdn_config_recurrent_element_count(const kq_gdn_config *config);
uint64_t kq_gdn_config_conv_state_element_count(const kq_gdn_config *config);
uint64_t kq_gdn_config_scratch_bytes(const kq_gdn_config *config);
uint64_t kq_gdn_config_token_scratch_bytes(const kq_gdn_config *config);
uint64_t kq_gdn_config_dequant_scratch_bytes(const kq_gdn_config *config);

kq_status kq_gdn_state_create(const kq_gdn_config *config,
                              kq_gdn_state **out_state,
                              kq_diagnostic *diagnostic);
void kq_gdn_state_close(kq_gdn_state *state);
kq_status kq_gdn_state_reset(kq_gdn_state *state,
                             kq_diagnostic *diagnostic);
uint64_t kq_gdn_state_owned_bytes(const kq_gdn_state *state);

kq_status kq_gdn_state_export_f32(const kq_gdn_state *state,
                                  float *conv_state,
                                  uint64_t conv_capacity,
                                  float *recurrent_state,
                                  uint64_t recurrent_capacity,
                                  int *initialized,
                                  kq_diagnostic *diagnostic);
kq_status kq_gdn_state_import_f32(kq_gdn_state *state,
                                  const float *conv_state,
                                  uint64_t conv_count,
                                  const float *recurrent_state,
                                  uint64_t recurrent_count,
                                  int initialized,
                                  kq_diagnostic *diagnostic);

kq_status kq_gdn_prefill_f32(
    const kq_gdn_config *config,
    const kq_gdn_weights_f32 *weights,
    const float *hidden_states,
    uint64_t sequence_length,
    const uint8_t *padding_mask,
    float *output,
    uint64_t output_capacity,
    kq_gdn_state *state,
    void *scratch,
    uint64_t scratch_bytes,
    kq_gdn_checkpoint_observer observer,
    void *observer_user_data,
    kq_diagnostic *diagnostic);

kq_status kq_gdn_decode_f32(
    const kq_gdn_config *config,
    const kq_gdn_weights_f32 *weights,
    const float *hidden_token,
    float *output_token,
    uint64_t output_capacity,
    kq_gdn_state *state,
    void *scratch,
    uint64_t scratch_bytes,
    kq_gdn_checkpoint_observer observer,
    void *observer_user_data,
    kq_diagnostic *diagnostic);

const char *kq_gdn_checkpoint_kind_name(kq_gdn_checkpoint_kind kind);

#ifdef __cplusplus
}
#endif

#endif
