#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "kq_numeric.h"

static int failures = 0;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures += 1;
    }
}

static uint32_t f32_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void put_u16(unsigned char *bytes, uint16_t value) {
    bytes[0] = (unsigned char)(value & 0xffU);
    bytes[1] = (unsigned char)(value >> 8U);
}

static void test_exact_storage(void) {
    unsigned char f32[] = {0x00U, 0x00U, 0x80U, 0x3fU};
    unsigned char bf16[] = {0x80U, 0x3fU};
    unsigned char q5_1[24] = {0};
    unsigned char q8_0[34] = {0};
    unsigned char q4_k[144] = {0};
    unsigned char q5_k[176] = {0};
    unsigned char iq4_nl[18] = {0};
    float output[256];
    uint64_t count = 0U;
    kq_diagnostic diagnostic;
    kq_status status;
    uint32_t index;

    status = kq_dequantize_blocks_f32(KQ_GGUF_TYPE_F32, f32, sizeof(f32),
                                      output, 256U, &count, &diagnostic);
    check(status == KQ_STATUS_OK && count == 1U &&
              f32_bits(output[0]) == UINT32_C(0x3f800000),
          "F32 storage decode is bit exact");
    status = kq_dequantize_blocks_f32(KQ_GGUF_TYPE_BF16, bf16, sizeof(bf16),
                                      output, 256U, &count, &diagnostic);
    check(status == KQ_STATUS_OK && count == 1U &&
              f32_bits(output[0]) == UINT32_C(0x3f800000),
          "BF16 storage decode is exact");
    bf16[0] = 1U; bf16[1] = 0U;
    status = kq_dequantize_blocks_f32(KQ_GGUF_TYPE_BF16, bf16, sizeof(bf16),
                                      output, 256U, &count, &diagnostic);
    check(status == KQ_STATUS_OK &&
              f32_bits(output[0]) == UINT32_C(0x00010000),
          "BF16 subnormal payload is preserved");

    put_u16(q5_1, UINT16_C(0x3c00));
    put_u16(q5_1 + 2U, UINT16_C(0xbc00));
    q5_1[4] = 1U;
    q5_1[8] = 0x21U;
    status = kq_dequantize_blocks_f32(KQ_GGUF_TYPE_Q5_1, q5_1,
                                      sizeof(q5_1), output, 256U,
                                      &count, &diagnostic);
    check(status == KQ_STATUS_OK && count == 32U &&
              output[0] == 16.0f && output[16] == 1.0f,
          "Q5_1 nibble/high-bit/min layout decodes");

    put_u16(q8_0, UINT16_C(0x3c00));
    for (index = 0U; index < 32U; ++index)
        q8_0[index + 2U] = (unsigned char)((int)index - 16);
    status = kq_dequantize_blocks_f32(KQ_GGUF_TYPE_Q8_0, q8_0,
                                      sizeof(q8_0), output, 256U,
                                      &count, &diagnostic);
    check(status == KQ_STATUS_OK && count == 32U &&
              output[0] == -16.0f && output[31] == 15.0f,
          "Q8_0 signed codes decode");

    put_u16(q4_k, UINT16_C(0x3c00));
    q4_k[4] = 1U;
    q4_k[16] = 0x21U;
    status = kq_dequantize_blocks_f32(KQ_GGUF_TYPE_Q4_K, q4_k,
                                      sizeof(q4_k), output, 256U,
                                      &count, &diagnostic);
    check(status == KQ_STATUS_OK && count == 256U &&
              output[0] == 1.0f && output[32] == 0.0f,
          "Q4_K scale and nibble layout decodes");

    put_u16(q5_k, UINT16_C(0x3c00));
    q5_k[4] = 1U;
    q5_k[16] = 1U;
    q5_k[48] = 0x21U;
    status = kq_dequantize_blocks_f32(KQ_GGUF_TYPE_Q5_K, q5_k,
                                      sizeof(q5_k), output, 256U,
                                      &count, &diagnostic);
    check(status == KQ_STATUS_OK && count == 256U &&
              output[0] == 17.0f,
          "Q5_K high-bit layout decodes");

    put_u16(iq4_nl, UINT16_C(0x3c00));
    iq4_nl[2] = 0xf0U;
    status = kq_dequantize_blocks_f32(KQ_GGUF_TYPE_IQ4_NL, iq4_nl,
                                      sizeof(iq4_nl), output, 256U,
                                      &count, &diagnostic);
    check(status == KQ_STATUS_OK && count == 32U &&
              output[0] == -127.0f && output[16] == 113.0f,
          "IQ4_NL low/high codebook layout decodes");
}

static void test_row_dot_and_primitives(void) {
    unsigned char q8_0[34] = {0};
    float activation[32];
    float left[] = {1.0f, -2.0f, 3.0f, 4.0f};
    float right[] = {2.0f, 3.0f, -1.0f, 0.5f};
    float output[4];
    float scalar = 0.0f;
    float weights[] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t indices[3];
    kq_diagnostic diagnostic;
    kq_status status;
    uint32_t index;

    put_u16(q8_0, UINT16_C(0x3c00));
    for (index = 0U; index < 32U; ++index) {
        q8_0[index + 2U] = 1U;
        activation[index] = 1.0f;
    }
    status = kq_quantized_row_dot_f32(KQ_GGUF_TYPE_Q8_0, q8_0,
                                      sizeof(q8_0), activation, 32U,
                                      &scalar, &diagnostic);
    check(status == KQ_STATUS_OK && scalar == 32.0f,
          "quantized row-dot uses bounded block scratch");
    check(kq_f32_add(left, right, 4U, output, &diagnostic) == KQ_STATUS_OK &&
              output[0] == 3.0f && output[1] == 1.0f,
          "F32 add works");
    check(kq_f32_multiply(left, right, 4U, output, &diagnostic) == KQ_STATUS_OK &&
              output[2] == -3.0f,
          "F32 multiply works");
    check(kq_f32_scale(left, 4U, 2.0f, output, &diagnostic) == KQ_STATUS_OK &&
              output[3] == 8.0f,
          "F32 scale works");
    check(kq_f32_dot(left, right, 4U, &scalar, &diagnostic) == KQ_STATUS_OK &&
              scalar == -5.0f,
          "F32 dot uses specified order");
    check(kq_f32_sigmoid(left, 4U, output, &diagnostic) == KQ_STATUS_OK &&
              output[0] > 0.7f && output[1] < 0.2f,
          "sigmoid works");
    check(kq_f32_silu(left, 4U, output, &diagnostic) == KQ_STATUS_OK &&
              output[0] > 0.7f,
          "SiLU works");
    check(kq_f32_swiglu(left, right, 4U, output, &diagnostic) == KQ_STATUS_OK &&
              output[0] > 1.4f,
          "SwiGLU combine works");
    check(kq_f32_rms_norm(left, weights, 4U, 1.0e-6f,
                          output, &diagnostic) == KQ_STATUS_OK,
          "RMSNorm works");
    check(kq_f32_softmax(left, 4U, output, &diagnostic) == KQ_STATUS_OK &&
              output[3] > output[2],
          "softmax works");
    {
        float ties[] = {3.0f, 1.0f, 3.0f, 2.0f};
        check(kq_f32_stable_top_k(ties, 4U, 3U, indices, output, 3U,
                                  &diagnostic) == KQ_STATUS_OK &&
                  indices[0] == 0U && indices[1] == 2U && indices[2] == 3U,
              "top-k ties choose the lower index");
    }
    {
        float probabilities[] = {0.2f, 0.3f, 0.5f};
        check(kq_f32_renormalize(probabilities, 3U, output,
                                 &diagnostic) == KQ_STATUS_OK &&
                  output[2] == 0.5f,
              "top-k weights renormalize");
    }
}

static void test_fail_closed(void) {
    unsigned char packed[176] = {0};
    float input[32] = {0};
    float output[256];
    uint64_t count = 0U;
    uint32_t indices[2];
    kq_diagnostic diagnostic;
    int prior_rounding = fegetround();

    check(kq_dequantize_blocks_f32(999U, packed, 4U, output, 256U,
                                   &count, &diagnostic) ==
              KQ_STATUS_UNSUPPORTED_TENSOR_TYPE,
          "unsupported decoder type fails closed");
    check(kq_dequantize_blocks_f32(KQ_GGUF_TYPE_F32, NULL, 4U,
                                   output, 256U, &count, &diagnostic) ==
              KQ_STATUS_INVALID_ARGUMENT,
          "null packed input fails closed");
    check(kq_dequantize_blocks_f32(KQ_GGUF_TYPE_Q8_0, packed,
                                   UINT64_MAX - (UINT64_MAX % 34U),
                                   output, 256U,
                                   &count, &diagnostic) ==
              KQ_STATUS_ARITHMETIC_OVERFLOW,
          "decoded output byte overflow fails closed");
    check(kq_dequantize_blocks_f32(KQ_GGUF_TYPE_Q8_0, packed, 33U,
                                   output, 256U, &count, &diagnostic) ==
              KQ_STATUS_MALFORMED_QUANTIZED_DATA,
          "partial packed block fails closed");
    check(kq_dequantize_blocks_f32(KQ_GGUF_TYPE_Q8_0, packed, 34U,
                                   output, 31U, &count, &diagnostic) ==
              KQ_STATUS_BUFFER_TOO_SMALL && count == 32U,
          "decode reports required capacity without truncation");
    check(kq_dequantize_blocks_f32(KQ_GGUF_TYPE_Q8_0, packed, 34U,
                                   (float *)packed, 32U, &count,
                                   &diagnostic) ==
              KQ_STATUS_ALIASING_VIOLATION,
          "packed/output alias fails closed");
    check(kq_quantized_row_dot_f32(KQ_GGUF_TYPE_Q8_0, packed, 34U,
                                   input, 31U, output, &diagnostic) ==
              KQ_STATUS_DIMENSION_MISMATCH,
          "row-dot dimension mismatch fails closed");
    check(kq_quantized_row_dot_f32(KQ_GGUF_TYPE_Q8_0, packed, 34U,
                                   input, 32U, input, &diagnostic) ==
              KQ_STATUS_ALIASING_VIOLATION,
          "row-dot scalar alias fails closed");
    input[0] = NAN;
    check(kq_f32_softmax(input, 32U, output, &diagnostic) ==
              KQ_STATUS_NUMERIC_DOMAIN,
          "non-finite primitive input fails closed");
    input[0] = 0.0f;
    check(kq_f32_add(input, input, 32U, input, &diagnostic) ==
              KQ_STATUS_ALIASING_VIOLATION,
          "forbidden primitive alias fails closed");
    check(kq_f32_stable_top_k(input, 32U, 0U, indices, output, 2U,
                              &diagnostic) == KQ_STATUS_INVALID_ARGUMENT,
          "invalid top-k fails closed");
    check(kq_f32_stable_top_k(input, 32U, 2U, indices, output, 1U,
                              &diagnostic) == KQ_STATUS_BUFFER_TOO_SMALL,
          "top-k capacity fails without truncation");
    check(kq_f32_dot(input, input, 32U, input, &diagnostic) ==
              KQ_STATUS_ALIASING_VIOLATION,
          "dot scalar alias fails closed");
    if (fesetround(FE_DOWNWARD) == 0) {
        check(kq_f32_dot(input, input, 32U, output, &diagnostic) ==
                  KQ_STATUS_NUMERIC_DOMAIN,
              "non-canonical rounding mode fails closed");
        (void)fesetround(prior_rounding);
    }
    put_u16(packed, UINT16_C(0x7c00));
    check(kq_dequantize_blocks_f32(KQ_GGUF_TYPE_Q8_0, packed, 34U,
                                   output, 256U, &count, &diagnostic) ==
              KQ_STATUS_MALFORMED_QUANTIZED_DATA,
          "non-finite quantized scale fails closed");
}

int main(void) {
    test_exact_storage();
    test_row_dot_and_primitives();
    test_fail_closed();
    if (failures != 0) {
        fprintf(stderr, "%d numeric test(s) failed\n", failures);
        return 1;
    }
    printf("numeric synthetic: PASS; types=7, scratch_bytes=%u\n",
           (unsigned int)(KQ_NUMERIC_MAX_BLOCK_ELEMENTS * sizeof(float)));
    return 0;
}
