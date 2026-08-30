#ifndef KQ_NUMERIC_H
#define KQ_NUMERIC_H

#include <stdint.h>

#include "kq_status.h"
#include "kq_tensor_view.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KQ_NUMERIC_MAX_BLOCK_ELEMENTS 256U

typedef enum kq_numeric_view_order {
    KQ_NUMERIC_PHYSICAL_ORDER = 0,
    KQ_NUMERIC_REQUIRE_CANONICAL_ORDER
} kq_numeric_view_order;

kq_status kq_dequantize_blocks_f32(uint32_t type_id,
                                    const void *packed,
                                    uint64_t packed_bytes,
                                    float *output,
                                    uint64_t output_capacity,
                                    uint64_t *output_count,
                                    kq_diagnostic *diagnostic);

kq_status kq_dequantize_view_blocks_f32(
    const kq_tensor_view *view,
    uint64_t relative_first_block,
    uint64_t block_count,
    kq_numeric_view_order order,
    float *output,
    uint64_t output_capacity,
    uint64_t *output_count,
    kq_diagnostic *diagnostic);

kq_status kq_quantized_row_dot_f32(uint32_t type_id,
                                    const void *packed,
                                    uint64_t packed_bytes,
                                    const float *activation,
                                    uint64_t activation_count,
                                    float *output,
                                    kq_diagnostic *diagnostic);

kq_status kq_f32_add(const float *left,
                     const float *right,
                     uint64_t count,
                     float *output,
                     kq_diagnostic *diagnostic);
kq_status kq_f32_multiply(const float *left,
                          const float *right,
                          uint64_t count,
                          float *output,
                          kq_diagnostic *diagnostic);
kq_status kq_f32_scale(const float *input,
                       uint64_t count,
                       float scale,
                       float *output,
                       kq_diagnostic *diagnostic);
kq_status kq_f32_dot(const float *left,
                     const float *right,
                     uint64_t count,
                     float *output,
                     kq_diagnostic *diagnostic);
kq_status kq_f32_sigmoid(const float *input,
                         uint64_t count,
                         float *output,
                         kq_diagnostic *diagnostic);
kq_status kq_f32_silu(const float *input,
                      uint64_t count,
                      float *output,
                      kq_diagnostic *diagnostic);
kq_status kq_f32_swiglu(const float *gate,
                        const float *up,
                        uint64_t count,
                        float *output,
                        kq_diagnostic *diagnostic);
kq_status kq_f32_rms_norm(const float *input,
                          const float *weight_delta,
                          uint64_t count,
                          float epsilon,
                          float *output,
                          kq_diagnostic *diagnostic);
kq_status kq_f32_softmax(const float *input,
                         uint64_t count,
                         float *output,
                         kq_diagnostic *diagnostic);
kq_status kq_f32_stable_top_k(const float *scores,
                              uint64_t count,
                              uint32_t k,
                              uint32_t *indices,
                              float *values,
                              uint64_t output_capacity,
                              kq_diagnostic *diagnostic);
kq_status kq_f32_renormalize(const float *input,
                             uint64_t count,
                             float *output,
                             kq_diagnostic *diagnostic);

#ifdef __cplusplus
}
#endif

#endif
