#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_numeric.h"

#define KQ_PROBE_MAX_VALUES 4096U
#define KQ_PROBE_MAX_PACKED 4096U

static int hex_digit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static int parse_type(const char *text, uint32_t *type_id) {
    if (strcmp(text, "F32") == 0) *type_id = KQ_GGUF_TYPE_F32;
    else if (strcmp(text, "BF16") == 0) *type_id = KQ_GGUF_TYPE_BF16;
    else if (strcmp(text, "Q5_1") == 0) *type_id = KQ_GGUF_TYPE_Q5_1;
    else if (strcmp(text, "Q8_0") == 0) *type_id = KQ_GGUF_TYPE_Q8_0;
    else if (strcmp(text, "Q4_K") == 0) *type_id = KQ_GGUF_TYPE_Q4_K;
    else if (strcmp(text, "Q5_K") == 0) *type_id = KQ_GGUF_TYPE_Q5_K;
    else if (strcmp(text, "IQ4_NL") == 0) *type_id = KQ_GGUF_TYPE_IQ4_NL;
    else return 0;
    return 1;
}

static int parse_hex(const char *text, unsigned char *bytes,
                     uint64_t capacity, uint64_t *count) {
    size_t length = strlen(text);
    uint64_t index;
    if (length == 0U || (length % 2U) != 0U ||
        (uint64_t)(length / 2U) > capacity) return 0;
    *count = (uint64_t)(length / 2U);
    for (index = 0U; index < *count; ++index) {
        int high = hex_digit(text[(size_t)index * 2U]);
        int low = hex_digit(text[(size_t)index * 2U + 1U]);
        if (high < 0 || low < 0) return 0;
        bytes[index] = (unsigned char)((high << 4) | low);
    }
    return 1;
}

static float f32_from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t f32_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int parse_f32_csv(const char *text, float *values,
                         uint64_t capacity, uint64_t *count) {
    const char *cursor = text;
    *count = 0U;
    while (*cursor != '\0') {
        char *end = NULL;
        unsigned long bits;
        if (*count >= capacity) return 0;
        bits = strtoul(cursor, &end, 16);
        if (end == cursor || (size_t)(end - cursor) != 8U ||
            bits > UINT32_MAX) return 0;
        values[*count] = f32_from_bits((uint32_t)bits);
        *count += 1U;
        if (*end == '\0') return 1;
        if (*end != ',') return 0;
        cursor = end + 1;
    }
    return 0;
}

static void print_bits(const float *values, uint64_t count) {
    uint64_t index;
    printf("bits=");
    for (index = 0U; index < count; ++index) {
        if (index != 0U) putchar(',');
        printf("%08x", (unsigned int)f32_bits(values[index]));
    }
    putchar('\n');
}

static int report_status(kq_status status, const kq_diagnostic *diagnostic) {
    if (status == KQ_STATUS_OK) return 0;
    fprintf(stderr, "%s: %s\n", kq_status_string(status),
            diagnostic->message);
    return 1;
}

int main(int argc, char **argv) {
    unsigned char packed[KQ_PROBE_MAX_PACKED];
    float left[KQ_PROBE_MAX_VALUES];
    float right[KQ_PROBE_MAX_VALUES];
    float output[KQ_PROBE_MAX_VALUES];
    uint32_t indices[KQ_PROBE_MAX_VALUES];
    uint64_t packed_count = 0U;
    uint64_t left_count = 0U;
    uint64_t right_count = 0U;
    uint64_t output_count = 0U;
    uint32_t type_id = 0U;
    kq_diagnostic diagnostic;
    kq_status status;

    if (argc >= 2 && strcmp(argv[1], "dequant") == 0 && argc == 4 &&
        parse_type(argv[2], &type_id) &&
        parse_hex(argv[3], packed, KQ_PROBE_MAX_PACKED, &packed_count)) {
        status = kq_dequantize_blocks_f32(type_id, packed, packed_count,
                                          output, KQ_PROBE_MAX_VALUES,
                                          &output_count, &diagnostic);
        if (report_status(status, &diagnostic) != 0) return 1;
        print_bits(output, output_count);
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "rowdot") == 0 && argc == 5 &&
        parse_type(argv[2], &type_id) &&
        parse_hex(argv[3], packed, KQ_PROBE_MAX_PACKED, &packed_count) &&
        parse_f32_csv(argv[4], left, KQ_PROBE_MAX_VALUES, &left_count)) {
        status = kq_quantized_row_dot_f32(type_id, packed, packed_count,
                                          left, left_count, output,
                                          &diagnostic);
        if (report_status(status, &diagnostic) != 0) return 1;
        print_bits(output, 1U);
        return 0;
    }
    if (argc >= 2 &&
        (strcmp(argv[1], "add") == 0 ||
         strcmp(argv[1], "multiply") == 0 ||
         strcmp(argv[1], "swiglu") == 0) && argc == 4 &&
        parse_f32_csv(argv[2], left, KQ_PROBE_MAX_VALUES, &left_count) &&
        parse_f32_csv(argv[3], right, KQ_PROBE_MAX_VALUES, &right_count) &&
        left_count == right_count) {
        if (strcmp(argv[1], "add") == 0)
            status = kq_f32_add(left, right, left_count, output, &diagnostic);
        else if (strcmp(argv[1], "multiply") == 0)
            status = kq_f32_multiply(left, right, left_count, output, &diagnostic);
        else
            status = kq_f32_swiglu(left, right, left_count, output, &diagnostic);
        if (report_status(status, &diagnostic) != 0) return 1;
        print_bits(output, left_count);
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "scale") == 0 && argc == 4 &&
        parse_f32_csv(argv[2], left, KQ_PROBE_MAX_VALUES, &left_count) &&
        parse_f32_csv(argv[3], right, 1U, &right_count) && right_count == 1U) {
        status = kq_f32_scale(left, left_count, right[0], output, &diagnostic);
        if (report_status(status, &diagnostic) != 0) return 1;
        print_bits(output, left_count);
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "dot") == 0 && argc == 4 &&
        parse_f32_csv(argv[2], left, KQ_PROBE_MAX_VALUES, &left_count) &&
        parse_f32_csv(argv[3], right, KQ_PROBE_MAX_VALUES, &right_count) &&
        left_count == right_count) {
        status = kq_f32_dot(left, right, left_count, output, &diagnostic);
        if (report_status(status, &diagnostic) != 0) return 1;
        print_bits(output, 1U);
        return 0;
    }
    if (argc >= 2 &&
        (strcmp(argv[1], "sigmoid") == 0 ||
         strcmp(argv[1], "silu") == 0 ||
         strcmp(argv[1], "softmax") == 0 ||
         strcmp(argv[1], "renormalize") == 0) && argc == 3 &&
        parse_f32_csv(argv[2], left, KQ_PROBE_MAX_VALUES, &left_count)) {
        if (strcmp(argv[1], "sigmoid") == 0)
            status = kq_f32_sigmoid(left, left_count, output, &diagnostic);
        else if (strcmp(argv[1], "silu") == 0)
            status = kq_f32_silu(left, left_count, output, &diagnostic);
        else if (strcmp(argv[1], "softmax") == 0)
            status = kq_f32_softmax(left, left_count, output, &diagnostic);
        else
            status = kq_f32_renormalize(left, left_count, output, &diagnostic);
        if (report_status(status, &diagnostic) != 0) return 1;
        print_bits(output, left_count);
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "rmsnorm") == 0 && argc == 5 &&
        parse_f32_csv(argv[2], left, KQ_PROBE_MAX_VALUES, &left_count) &&
        parse_f32_csv(argv[3], right, KQ_PROBE_MAX_VALUES, &right_count) &&
        parse_f32_csv(argv[4], output, 1U, &output_count) &&
        left_count == right_count && output_count == 1U) {
        float epsilon = output[0];
        status = kq_f32_rms_norm(left, right, left_count, epsilon,
                                 output, &diagnostic);
        if (report_status(status, &diagnostic) != 0) return 1;
        print_bits(output, left_count);
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "topk") == 0 && argc == 4 &&
        parse_f32_csv(argv[2], left, KQ_PROBE_MAX_VALUES, &left_count)) {
        char *end = NULL;
        unsigned long requested = strtoul(argv[3], &end, 10);
        uint32_t index;
        if (end == argv[3] || *end != '\0' || requested > UINT32_MAX)
            return 2;
        status = kq_f32_stable_top_k(left, left_count, (uint32_t)requested,
                                     indices, output, KQ_PROBE_MAX_VALUES,
                                     &diagnostic);
        if (report_status(status, &diagnostic) != 0) return 1;
        printf("indices=");
        for (index = 0U; index < (uint32_t)requested; ++index) {
            if (index != 0U) putchar(',');
            printf("%u", indices[index]);
        }
        putchar(';');
        print_bits(output, requested);
        return 0;
    }
    fprintf(stderr, "invalid numeric probe command\n");
    return 2;
}
