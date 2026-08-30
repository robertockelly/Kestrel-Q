#include "kq_numeric.h"

#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "kq_internal.h"

static const int8_t kq_iq4_nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

static kq_status kq_numeric_fail(kq_diagnostic *diagnostic,
                                 kq_status status,
                                 const char *message) {
    kq_diagnostic_set(diagnostic, status, "%s", message);
    return status;
}

static int kq_u64_add(uint64_t left, uint64_t right, uint64_t *output) {
    if (UINT64_MAX - left < right) {
        return 0;
    }
    *output = left + right;
    return 1;
}

static int kq_u64_mul(uint64_t left, uint64_t right, uint64_t *output) {
    if (left != 0U && right > UINT64_MAX / left) {
        return 0;
    }
    *output = left * right;
    return 1;
}

static uint16_t kq_read_u16_le(const unsigned char *bytes) {
    return (uint16_t)((uint16_t)bytes[0] |
                      ((uint16_t)bytes[1] << 8U));
}

static uint32_t kq_read_u32_le(const unsigned char *bytes) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8U) |
           ((uint32_t)bytes[2] << 16U) |
           ((uint32_t)bytes[3] << 24U);
}

static float kq_f32_from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float kq_f16_to_f32(uint16_t value) {
    uint32_t sign = ((uint32_t)value & UINT32_C(0x8000)) << 16U;
    uint32_t exponent = ((uint32_t)value >> 10U) & UINT32_C(0x1f);
    uint32_t fraction = (uint32_t)value & UINT32_C(0x03ff);
    uint32_t bits;

    if (exponent == 0U) {
        if (fraction == 0U) {
            bits = sign;
        } else {
            uint32_t adjusted_exponent = 113U;
            while ((fraction & UINT32_C(0x0400)) == 0U) {
                fraction <<= 1U;
                adjusted_exponent -= 1U;
            }
            fraction &= UINT32_C(0x03ff);
            bits = sign | (adjusted_exponent << 23U) | (fraction << 13U);
        }
    } else if (exponent == 31U) {
        bits = sign | UINT32_C(0x7f800000) | (fraction << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (fraction << 13U);
    }
    return kq_f32_from_bits(bits);
}

static int kq_numeric_rounding_ok(void) {
    return fegetround() == FE_TONEAREST;
}

static int kq_range_bytes(uint64_t count, uint64_t element_bytes,
                          uint64_t *bytes) {
    return kq_u64_mul(count, element_bytes, bytes) &&
           *bytes <= (uint64_t)SIZE_MAX;
}

static int kq_ranges_overlap(const void *left, uint64_t left_bytes,
                             const void *right, uint64_t right_bytes) {
    uintptr_t left_start;
    uintptr_t right_start;
    uintptr_t left_end;
    uintptr_t right_end;

    if (left == NULL || right == NULL || left_bytes == 0U ||
        right_bytes == 0U || left_bytes > (uint64_t)UINTPTR_MAX ||
        right_bytes > (uint64_t)UINTPTR_MAX) {
        return 0;
    }
    left_start = (uintptr_t)left;
    right_start = (uintptr_t)right;
    if (left_start > UINTPTR_MAX - (uintptr_t)left_bytes ||
        right_start > UINTPTR_MAX - (uintptr_t)right_bytes) {
        return 1;
    }
    left_end = left_start + (uintptr_t)left_bytes;
    right_end = right_start + (uintptr_t)right_bytes;
    return left_start < right_end && right_start < left_end;
}

static kq_status kq_validate_vector(const float *input,
                                    uint64_t count,
                                    kq_diagnostic *diagnostic) {
    uint64_t index;
    if (input == NULL || count == 0U) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                               "non-empty F32 input is required");
    }
    if (count > (uint64_t)(SIZE_MAX / sizeof(float))) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                               "F32 vector byte count overflows size_t");
    }
    if (!kq_numeric_rounding_ok()) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                               "round-to-nearest floating mode is required");
    }
    for (index = 0U; index < count; ++index) {
        if (!isfinite(input[index])) {
            return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                   "numeric primitive input must be finite");
        }
    }
    return KQ_STATUS_OK;
}

static kq_status kq_validate_output_no_alias(const void *input_a,
                                              uint64_t input_a_bytes,
                                              const void *input_b,
                                              uint64_t input_b_bytes,
                                              float *output,
                                              uint64_t output_bytes,
                                              kq_diagnostic *diagnostic) {
    if (output == NULL) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                               "numeric output is required");
    }
    if (kq_ranges_overlap(input_a, input_a_bytes, output, output_bytes) ||
        kq_ranges_overlap(input_b, input_b_bytes, output, output_bytes)) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                               "numeric input and output buffers overlap");
    }
    return KQ_STATUS_OK;
}

static void kq_unpack_k_scale(uint32_t index,
                              const unsigned char *scales,
                              uint8_t *scale,
                              uint8_t *minimum) {
    if (index < 4U) {
        *scale = (uint8_t)(scales[index] & 63U);
        *minimum = (uint8_t)(scales[index + 4U] & 63U);
    } else {
        *scale = (uint8_t)((scales[index + 4U] & 15U) |
                           ((scales[index - 4U] >> 6U) << 4U));
        *minimum = (uint8_t)((scales[index + 4U] >> 4U) |
                             ((scales[index] >> 6U) << 4U));
    }
}

static kq_status kq_decode_one_block(uint32_t type_id,
                                      const unsigned char *packed,
                                      float *output,
                                      kq_diagnostic *diagnostic) {
    uint32_t index;

    if (type_id == KQ_GGUF_TYPE_F32) {
        output[0] = kq_f32_from_bits(kq_read_u32_le(packed));
        return KQ_STATUS_OK;
    }
    if (type_id == KQ_GGUF_TYPE_BF16) {
        output[0] = kq_f32_from_bits((uint32_t)kq_read_u16_le(packed) << 16U);
        return KQ_STATUS_OK;
    }
    if (type_id == KQ_GGUF_TYPE_Q5_1) {
        float d = kq_f16_to_f32(kq_read_u16_le(packed));
        float minimum = kq_f16_to_f32(kq_read_u16_le(packed + 2U));
        uint32_t high = kq_read_u32_le(packed + 4U);
        if (!isfinite(d) || !isfinite(minimum)) {
            return kq_numeric_fail(diagnostic,
                                   KQ_STATUS_MALFORMED_QUANTIZED_DATA,
                                   "Q5_1 scale/min is non-finite");
        }
        for (index = 0U; index < 16U; ++index) {
            uint8_t byte = packed[8U + index];
            uint32_t low_code = (uint32_t)(byte & 15U) |
                                (((high >> index) & 1U) << 4U);
            uint32_t high_code = (uint32_t)(byte >> 4U) |
                                 (((high >> (index + 16U)) & 1U) << 4U);
            output[index] = (float)low_code * d + minimum;
            output[index + 16U] = (float)high_code * d + minimum;
        }
        return KQ_STATUS_OK;
    }
    if (type_id == KQ_GGUF_TYPE_Q8_0) {
        float d = kq_f16_to_f32(kq_read_u16_le(packed));
        if (!isfinite(d)) {
            return kq_numeric_fail(diagnostic,
                                   KQ_STATUS_MALFORMED_QUANTIZED_DATA,
                                   "Q8_0 scale is non-finite");
        }
        for (index = 0U; index < 32U; ++index) {
            int8_t code = (int8_t)packed[2U + index];
            output[index] = (float)code * d;
        }
        return KQ_STATUS_OK;
    }
    if (type_id == KQ_GGUF_TYPE_Q4_K ||
        type_id == KQ_GGUF_TYPE_Q5_K) {
        float d = kq_f16_to_f32(kq_read_u16_le(packed));
        float dmin = kq_f16_to_f32(kq_read_u16_le(packed + 2U));
        const unsigned char *scales = packed + 4U;
        const unsigned char *high = type_id == KQ_GGUF_TYPE_Q5_K
                                        ? packed + 16U
                                        : NULL;
        const unsigned char *quants = type_id == KQ_GGUF_TYPE_Q5_K
                                          ? packed + 48U
                                          : packed + 16U;
        uint32_t group;
        if (!isfinite(d) || !isfinite(dmin)) {
            return kq_numeric_fail(diagnostic,
                                   KQ_STATUS_MALFORMED_QUANTIZED_DATA,
                                   "K-quant scale/min is non-finite");
        }
        for (group = 0U; group < 4U; ++group) {
            uint8_t scale_low;
            uint8_t min_low;
            uint8_t scale_high;
            uint8_t min_high;
            float d_low;
            float m_low;
            float d_high;
            float m_high;
            uint8_t high_low_mask = (uint8_t)(1U << (2U * group));
            uint8_t high_high_mask = (uint8_t)(2U << (2U * group));
            kq_unpack_k_scale(2U * group, scales, &scale_low, &min_low);
            kq_unpack_k_scale(2U * group + 1U, scales,
                              &scale_high, &min_high);
            d_low = d * (float)scale_low;
            m_low = dmin * (float)min_low;
            d_high = d * (float)scale_high;
            m_high = dmin * (float)min_high;
            for (index = 0U; index < 32U; ++index) {
                uint8_t byte = quants[group * 32U + index];
                uint32_t low_code = (uint32_t)(byte & 15U);
                uint32_t high_code = (uint32_t)(byte >> 4U);
                if (high != NULL) {
                    if ((high[index] & high_low_mask) != 0U) {
                        low_code += 16U;
                    }
                    if ((high[index] & high_high_mask) != 0U) {
                        high_code += 16U;
                    }
                }
                output[group * 64U + index] =
                    d_low * (float)low_code - m_low;
                output[group * 64U + 32U + index] =
                    d_high * (float)high_code - m_high;
            }
        }
        return KQ_STATUS_OK;
    }
    if (type_id == KQ_GGUF_TYPE_IQ4_NL) {
        float d = kq_f16_to_f32(kq_read_u16_le(packed));
        if (!isfinite(d)) {
            return kq_numeric_fail(diagnostic,
                                   KQ_STATUS_MALFORMED_QUANTIZED_DATA,
                                   "IQ4_NL scale is non-finite");
        }
        for (index = 0U; index < 16U; ++index) {
            uint8_t byte = packed[2U + index];
            output[index] = d * (float)kq_iq4_nl_values[byte & 15U];
            output[index + 16U] =
                d * (float)kq_iq4_nl_values[byte >> 4U];
        }
        return KQ_STATUS_OK;
    }
    return kq_numeric_fail(diagnostic, KQ_STATUS_UNSUPPORTED_TENSOR_TYPE,
                           "numeric decoder does not support this GGUF type");
}

kq_status kq_dequantize_blocks_f32(uint32_t type_id,
                                    const void *packed,
                                    uint64_t packed_bytes,
                                    float *output,
                                    uint64_t output_capacity,
                                    uint64_t *output_count,
                                    kq_diagnostic *diagnostic) {
    kq_quant_geometry geometry;
    uint64_t block_count;
    uint64_t required_count;
    uint64_t output_bytes;
    uint64_t block_index;
    const unsigned char *source = (const unsigned char *)packed;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    if (output_count == NULL || packed == NULL || packed_bytes == 0U) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                               "packed input and output count are required");
    }
    *output_count = 0U;
    status = kq_quant_geometry_for_type(type_id, &geometry, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if ((packed_bytes % geometry.bytes_per_block) != 0U ||
        packed_bytes > (uint64_t)SIZE_MAX) {
        return kq_numeric_fail(diagnostic,
                               KQ_STATUS_MALFORMED_QUANTIZED_DATA,
                               "packed bytes are not a bounded whole-block span");
    }
    block_count = packed_bytes / geometry.bytes_per_block;
    if (!kq_u64_mul(block_count, geometry.block_elements,
                    &required_count) ||
        !kq_range_bytes(required_count, sizeof(float), &output_bytes)) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                               "decoded output count overflows");
    }
    *output_count = required_count;
    if (output == NULL || output_capacity < required_count) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                               "decoded output capacity is too small");
    }
    if (kq_ranges_overlap(packed, packed_bytes, output, output_bytes)) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                               "packed input and decoded output overlap");
    }
    if (!kq_numeric_rounding_ok()) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                               "round-to-nearest floating mode is required");
    }
    for (block_index = 0U; block_index < block_count; ++block_index) {
        status = kq_decode_one_block(
            type_id,
            source + (size_t)(block_index * geometry.bytes_per_block),
            output + (size_t)(block_index * geometry.block_elements),
            diagnostic);
        if (status != KQ_STATUS_OK) {
            return status;
        }
    }
    return KQ_STATUS_OK;
}

kq_status kq_dequantize_view_blocks_f32(
    const kq_tensor_view *view,
    uint64_t relative_first_block,
    uint64_t block_count,
    kq_numeric_view_order order,
    float *output,
    uint64_t output_capacity,
    uint64_t *output_count,
    kq_diagnostic *diagnostic) {
    const kq_tensor_view_info *info;
    const unsigned char *data;
    uint64_t end_block;
    uint64_t byte_offset;
    uint64_t byte_count;

    kq_diagnostic_clear(diagnostic);
    if (view == NULL || output_count == NULL || block_count == 0U ||
        (order != KQ_NUMERIC_PHYSICAL_ORDER &&
         order != KQ_NUMERIC_REQUIRE_CANONICAL_ORDER)) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                               "valid tensor view, block range and order are required");
    }
    *output_count = 0U;
    info = kq_tensor_view_get_info(view);
    data = kq_tensor_view_physical_data(view);
    if (info == NULL || data == NULL || info->bytes_per_block == 0U ||
        info->block_elements == 0U) {
        return kq_numeric_fail(diagnostic,
                               KQ_STATUS_TENSOR_OWNERSHIP_MISMATCH,
                               "tensor view is not live or structurally valid");
    }
    if (order == KQ_NUMERIC_REQUIRE_CANONICAL_ORDER &&
        (info->canonical_contiguous == 0U ||
         info->layout == KQ_TENSOR_LAYOUT_TRANSFORMED_PHYSICAL ||
         info->layout == KQ_TENSOR_LAYOUT_SPLIT_SEGMENT)) {
        return kq_numeric_fail(diagnostic,
                               KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                               "view does not provide canonical-contiguous order");
    }
    if (!kq_u64_add(relative_first_block, block_count, &end_block) ||
        end_block > info->block_count ||
        !kq_u64_mul(relative_first_block, info->bytes_per_block,
                    &byte_offset) ||
        !kq_u64_mul(block_count, info->bytes_per_block, &byte_count) ||
        byte_offset > info->mapped_logical_length ||
        byte_count > info->mapped_logical_length - byte_offset ||
        byte_offset > (uint64_t)SIZE_MAX) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_SPAN_OUT_OF_RANGE,
                               "requested decode blocks are outside the view");
    }
    return kq_dequantize_blocks_f32(info->type_id,
                                    data + (size_t)byte_offset,
                                    byte_count,
                                    output,
                                    output_capacity,
                                    output_count,
                                    diagnostic);
}

kq_status kq_quantized_row_dot_f32(uint32_t type_id,
                                    const void *packed,
                                    uint64_t packed_bytes,
                                    const float *activation,
                                    uint64_t activation_count,
                                    float *output,
                                    kq_diagnostic *diagnostic) {
    kq_quant_geometry geometry;
    float scratch[KQ_NUMERIC_MAX_BLOCK_ELEMENTS];
    float accumulator = 0.0f;
    uint64_t block_count;
    uint64_t expected_count;
    uint64_t block;
    uint64_t index;
    uint64_t required = 0U;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    if (packed == NULL || activation == NULL || output == NULL ||
        packed_bytes == 0U || activation_count == 0U) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                               "row-dot inputs and scalar output are required");
    }
    status = kq_quant_geometry_for_type(type_id, &geometry, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (geometry.block_elements > KQ_NUMERIC_MAX_BLOCK_ELEMENTS ||
        packed_bytes > (uint64_t)SIZE_MAX ||
        (packed_bytes % geometry.bytes_per_block) != 0U) {
        return kq_numeric_fail(diagnostic,
                               KQ_STATUS_MALFORMED_QUANTIZED_DATA,
                               "row is not a bounded whole-block span");
    }
    block_count = packed_bytes / geometry.bytes_per_block;
    if (!kq_u64_mul(block_count, geometry.block_elements, &expected_count)) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                               "row element count overflows");
    }
    if (expected_count != activation_count) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_DIMENSION_MISMATCH,
                               "quantized row and activation lengths differ");
    }
    status = kq_validate_vector(activation, activation_count, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (kq_ranges_overlap(packed, packed_bytes, output, sizeof(*output)) ||
        kq_ranges_overlap(activation,
                          activation_count * sizeof(*activation),
                          output, sizeof(*output))) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                               "row-dot scalar output overlaps an input");
    }
    for (block = 0U; block < block_count; ++block) {
        status = kq_dequantize_blocks_f32(
            type_id,
            (const unsigned char *)packed +
                (size_t)(block * geometry.bytes_per_block),
            geometry.bytes_per_block,
            scratch,
            KQ_NUMERIC_MAX_BLOCK_ELEMENTS,
            &required,
            diagnostic);
        if (status != KQ_STATUS_OK || required != geometry.block_elements) {
            return status == KQ_STATUS_OK
                       ? kq_numeric_fail(diagnostic,
                                         KQ_STATUS_MALFORMED_QUANTIZED_DATA,
                                         "decoded row block size is inconsistent")
                       : status;
        }
        for (index = 0U; index < geometry.block_elements; ++index) {
            float product = scratch[index] *
                activation[block * geometry.block_elements + index];
            accumulator = accumulator + product;
        }
    }
    if (!isfinite(accumulator)) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                               "row-dot produced a non-finite result");
    }
    *output = accumulator;
    return KQ_STATUS_OK;
}

typedef enum kq_binary_operation {
    KQ_BINARY_ADD = 0,
    KQ_BINARY_MULTIPLY
} kq_binary_operation;

static kq_status kq_binary_vector(const float *left,
                                  const float *right,
                                  uint64_t count,
                                  float *output,
                                  kq_binary_operation operation,
                                  kq_diagnostic *diagnostic) {
    uint64_t bytes;
    uint64_t index;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    status = kq_validate_vector(left, count, diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_validate_vector(right, count, diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        return status;
    }
    (void)kq_range_bytes(count, sizeof(float), &bytes);
    status = kq_validate_output_no_alias(left, bytes, right, bytes,
                                         output, bytes, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    for (index = 0U; index < count; ++index) {
        float value = operation == KQ_BINARY_ADD
                          ? left[index] + right[index]
                          : left[index] * right[index];
        if (!isfinite(value)) {
            return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                   "binary primitive produced a non-finite result");
        }
        output[index] = value;
    }
    return KQ_STATUS_OK;
}

kq_status kq_f32_add(const float *left, const float *right, uint64_t count,
                     float *output, kq_diagnostic *diagnostic) {
    return kq_binary_vector(left, right, count, output,
                            KQ_BINARY_ADD, diagnostic);
}

kq_status kq_f32_multiply(const float *left, const float *right,
                          uint64_t count, float *output,
                          kq_diagnostic *diagnostic) {
    return kq_binary_vector(left, right, count, output,
                            KQ_BINARY_MULTIPLY, diagnostic);
}

kq_status kq_f32_scale(const float *input, uint64_t count, float scale,
                       float *output, kq_diagnostic *diagnostic) {
    uint64_t bytes;
    uint64_t index;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    if (!isfinite(scale)) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                               "scale must be finite");
    }
    status = kq_validate_vector(input, count, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    (void)kq_range_bytes(count, sizeof(float), &bytes);
    status = kq_validate_output_no_alias(input, bytes, NULL, 0U,
                                         output, bytes, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    for (index = 0U; index < count; ++index) {
        output[index] = input[index] * scale;
        if (!isfinite(output[index])) {
            return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                   "scale primitive produced a non-finite result");
        }
    }
    return KQ_STATUS_OK;
}

kq_status kq_f32_dot(const float *left, const float *right, uint64_t count,
                     float *output, kq_diagnostic *diagnostic) {
    float accumulator = 0.0f;
    uint64_t index;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    if (output == NULL) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                               "dot scalar output is required");
    }
    status = kq_validate_vector(left, count, diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_validate_vector(right, count, diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (kq_ranges_overlap(left, count * sizeof(*left),
                          output, sizeof(*output)) ||
        kq_ranges_overlap(right, count * sizeof(*right),
                          output, sizeof(*output))) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                               "dot scalar output overlaps an input");
    }
    for (index = 0U; index < count; ++index) {
        float product = left[index] * right[index];
        accumulator = accumulator + product;
    }
    if (!isfinite(accumulator)) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                               "dot primitive produced a non-finite result");
    }
    *output = accumulator;
    return KQ_STATUS_OK;
}

static float kq_sigmoid_scalar(float input) {
    float result;
    if (input >= 0.0f) {
        float exponential = expf(-input);
        result = 1.0f / (1.0f + exponential);
    } else {
        float exponential = expf(input);
        result = exponential / (1.0f + exponential);
    }
    return result;
}

typedef enum kq_unary_operation {
    KQ_UNARY_SIGMOID = 0,
    KQ_UNARY_SILU
} kq_unary_operation;

static kq_status kq_unary_vector(const float *input,
                                 uint64_t count,
                                 float *output,
                                 kq_unary_operation operation,
                                 kq_diagnostic *diagnostic) {
    uint64_t bytes;
    uint64_t index;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    status = kq_validate_vector(input, count, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    (void)kq_range_bytes(count, sizeof(float), &bytes);
    status = kq_validate_output_no_alias(input, bytes, NULL, 0U,
                                         output, bytes, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    for (index = 0U; index < count; ++index) {
        float sigmoid = kq_sigmoid_scalar(input[index]);
        output[index] = operation == KQ_UNARY_SIGMOID
                            ? sigmoid
                            : input[index] * sigmoid;
        if (!isfinite(output[index])) {
            return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                   "activation produced a non-finite result");
        }
    }
    return KQ_STATUS_OK;
}

kq_status kq_f32_sigmoid(const float *input, uint64_t count, float *output,
                         kq_diagnostic *diagnostic) {
    return kq_unary_vector(input, count, output,
                           KQ_UNARY_SIGMOID, diagnostic);
}

kq_status kq_f32_silu(const float *input, uint64_t count, float *output,
                      kq_diagnostic *diagnostic) {
    return kq_unary_vector(input, count, output,
                           KQ_UNARY_SILU, diagnostic);
}

kq_status kq_f32_swiglu(const float *gate, const float *up, uint64_t count,
                        float *output, kq_diagnostic *diagnostic) {
    uint64_t bytes;
    uint64_t index;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    status = kq_validate_vector(gate, count, diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_validate_vector(up, count, diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        return status;
    }
    (void)kq_range_bytes(count, sizeof(float), &bytes);
    status = kq_validate_output_no_alias(gate, bytes, up, bytes,
                                         output, bytes, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    for (index = 0U; index < count; ++index) {
        float sigmoid = kq_sigmoid_scalar(gate[index]);
        float silu = gate[index] * sigmoid;
        output[index] = silu * up[index];
        if (!isfinite(output[index])) {
            return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                   "SwiGLU produced a non-finite result");
        }
    }
    return KQ_STATUS_OK;
}

kq_status kq_f32_rms_norm(const float *input,
                          const float *weight_delta,
                          uint64_t count,
                          float epsilon,
                          float *output,
                          kq_diagnostic *diagnostic) {
    float sum = 0.0f;
    float mean;
    float inverse_root;
    uint64_t bytes;
    uint64_t index;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    if (!isfinite(epsilon) || epsilon <= 0.0f) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                               "RMSNorm epsilon must be positive finite");
    }
    status = kq_validate_vector(input, count, diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_validate_vector(weight_delta, count, diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        return status;
    }
    (void)kq_range_bytes(count, sizeof(float), &bytes);
    status = kq_validate_output_no_alias(input, bytes,
                                         weight_delta, bytes,
                                         output, bytes, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    for (index = 0U; index < count; ++index) {
        float square = input[index] * input[index];
        sum = sum + square;
    }
    mean = sum / (float)count;
    inverse_root = 1.0f / sqrtf(mean + epsilon);
    if (!isfinite(inverse_root)) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                               "RMSNorm inverse root is non-finite");
    }
    for (index = 0U; index < count; ++index) {
        float normalized = input[index] * inverse_root;
        float weight = 1.0f + weight_delta[index];
        output[index] = normalized * weight;
        if (!isfinite(output[index])) {
            return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                   "RMSNorm produced a non-finite result");
        }
    }
    return KQ_STATUS_OK;
}

kq_status kq_f32_softmax(const float *input, uint64_t count, float *output,
                         kq_diagnostic *diagnostic) {
    float maximum;
    float sum = 0.0f;
    uint64_t bytes;
    uint64_t index;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    status = kq_validate_vector(input, count, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    (void)kq_range_bytes(count, sizeof(float), &bytes);
    status = kq_validate_output_no_alias(input, bytes, NULL, 0U,
                                         output, bytes, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    maximum = input[0];
    for (index = 1U; index < count; ++index) {
        if (input[index] > maximum) {
            maximum = input[index];
        }
    }
    for (index = 0U; index < count; ++index) {
        output[index] = expf(input[index] - maximum);
        sum = sum + output[index];
    }
    if (!isfinite(sum) || sum <= 0.0f) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                               "softmax normalization sum is invalid");
    }
    for (index = 0U; index < count; ++index) {
        output[index] = output[index] / sum;
    }
    return KQ_STATUS_OK;
}

kq_status kq_f32_stable_top_k(const float *scores,
                              uint64_t count,
                              uint32_t k,
                              uint32_t *indices,
                              float *values,
                              uint64_t output_capacity,
                              kq_diagnostic *diagnostic) {
    uint64_t score_bytes;
    uint64_t value_bytes;
    uint64_t index_bytes;
    uint32_t selected;
    uint64_t candidate;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    if (k == 0U || (uint64_t)k > count || count > UINT32_MAX ||
        indices == NULL || values == NULL) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                               "top-k request or output buffers are invalid");
    }
    if (output_capacity < (uint64_t)k) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                               "top-k output capacity is too small");
    }
    status = kq_validate_vector(scores, count, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    (void)kq_range_bytes(count, sizeof(float), &score_bytes);
    (void)kq_range_bytes(k, sizeof(float), &value_bytes);
    (void)kq_range_bytes(k, sizeof(uint32_t), &index_bytes);
    if (kq_ranges_overlap(scores, score_bytes, values, value_bytes) ||
        kq_ranges_overlap(scores, score_bytes, indices, index_bytes) ||
        kq_ranges_overlap(values, value_bytes, indices, index_bytes)) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                               "top-k input and outputs overlap");
    }
    for (selected = 0U; selected < k; ++selected) {
        uint32_t best_index = UINT32_MAX;
        float best_value = 0.0f;
        for (candidate = 0U; candidate < count; ++candidate) {
            uint32_t prior;
            int already_selected = 0;
            for (prior = 0U; prior < selected; ++prior) {
                if (indices[prior] == (uint32_t)candidate) {
                    already_selected = 1;
                    break;
                }
            }
            if (!already_selected &&
                (best_index == UINT32_MAX || scores[candidate] > best_value ||
                 (scores[candidate] == best_value &&
                  candidate < (uint64_t)best_index))) {
                best_index = (uint32_t)candidate;
                best_value = scores[candidate];
            }
        }
        indices[selected] = best_index;
        values[selected] = best_value;
    }
    return KQ_STATUS_OK;
}

kq_status kq_f32_renormalize(const float *input, uint64_t count,
                             float *output, kq_diagnostic *diagnostic) {
    float sum = 0.0f;
    uint64_t bytes;
    uint64_t index;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    status = kq_validate_vector(input, count, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    (void)kq_range_bytes(count, sizeof(float), &bytes);
    status = kq_validate_output_no_alias(input, bytes, NULL, 0U,
                                         output, bytes, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    for (index = 0U; index < count; ++index) {
        if (input[index] < 0.0f) {
            return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                   "renormalization weights must be non-negative");
        }
        sum = sum + input[index];
    }
    if (!isfinite(sum) || sum <= 0.0f) {
        return kq_numeric_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                               "renormalization sum is invalid");
    }
    for (index = 0U; index < count; ++index) {
        output[index] = input[index] / sum;
    }
    return KQ_STATUS_OK;
}
