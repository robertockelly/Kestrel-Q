#ifndef KQ_SAMPLING_INTERNAL_H
#define KQ_SAMPLING_INTERNAL_H

#include "kq_sampling.h"

kq_status kq_sampling_categorical_word_f32_for_test(
    const float *probabilities,
    uint32_t count,
    uint32_t word,
    uint32_t *selected_index,
    kq_diagnostic *diagnostic);

kq_status kq_sampling_copy_work_for_test(
    const kq_sampling_config *config,
    const void *scratch,
    uint64_t scratch_bytes,
    float *scores,
    float *probabilities,
    uint64_t output_capacity,
    kq_diagnostic *diagnostic);

#endif
