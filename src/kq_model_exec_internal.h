#ifndef KQ_MODEL_EXEC_INTERNAL_H
#define KQ_MODEL_EXEC_INTERNAL_H

#include "kq_model_exec.h"

/* Test/research-only control: model logits are still produced normally, but
   selection consumes this exact alternate vector. This permits bounded EOG,
   padding and sampler-failure transaction controls without changing the
   production distribution. */
kq_status kq_model_exec_sampled_prefill_with_selection_logits_for_test(
    const kq_model_exec_config *config, kq_model_exec_state *state,
    const kq_sampling_config *sampling_config,
    kq_sampling_rng_state *rng_state,
    const uint32_t *token_ids, uint64_t token_count,
    const float *selection_logits,
    uint64_t selection_logit_count,
    float *model_logits, uint64_t model_logits_capacity,
    unsigned char *decoded_utf8, uint64_t decoded_utf8_capacity,
    void *model_scratch, uint64_t model_scratch_bytes,
    void *sampling_scratch, uint64_t sampling_scratch_bytes,
    kq_model_exec_progress_observer observer, void *observer_user_data,
    kq_model_exec_result *result,
    kq_sampling_result *sampling_result,
    kq_diagnostic *diagnostic);

kq_status kq_model_exec_sampled_decode_one_with_selection_logits_for_test(
    const kq_model_exec_config *config, kq_model_exec_state *state,
    const kq_sampling_config *sampling_config,
    kq_sampling_rng_state *rng_state,
    uint32_t input_token_id,
    const float *selection_logits,
    uint64_t selection_logit_count,
    float *model_logits, uint64_t model_logits_capacity,
    unsigned char *decoded_utf8, uint64_t decoded_utf8_capacity,
    void *model_scratch, uint64_t model_scratch_bytes,
    void *sampling_scratch, uint64_t sampling_scratch_bytes,
    kq_model_exec_progress_observer observer, void *observer_user_data,
    kq_model_exec_result *result,
    kq_sampling_result *sampling_result,
    kq_diagnostic *diagnostic);

#endif
