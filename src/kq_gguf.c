#include "kq_gguf.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "kq_internal.h"

#define KQ_GGUF_FIXED_HEADER_BYTES 24U
#define KQ_GGUF_MAX_HEADER_DIRECTORY_BYTES (512ULL * 1024ULL * 1024ULL)
#define KQ_GGUF_MAX_STRING_BYTES (128ULL * 1024ULL * 1024ULL)
#define KQ_GGUF_MAX_ARRAY_LENGTH 2000000ULL
#define KQ_GGUF_MAX_METADATA_COUNT 1024ULL
#define KQ_GGUF_MAX_TENSOR_COUNT 8192ULL
#define KQ_GGUF_DEFAULT_ALIGNMENT 32ULL
#define KQ_GGUF_MAX_ALIGNMENT 4096ULL

typedef struct kq_reader {
    const unsigned char *data;
    uint64_t length;
    uint64_t file_size;
    uint64_t offset;
    uint64_t max_consumed;
    kq_diagnostic *diagnostic;
} kq_reader;

typedef struct kq_tensor_order {
    uint64_t offset;
    uint64_t packed_bytes;
    uint64_t index;
} kq_tensor_order;

struct kq_gguf {
    kq_file *file;
    kq_file_view *directory_view;
    kq_gguf_metadata *metadata;
    kq_gguf_tensor *tensors;
    uint32_t version;
    uint64_t metadata_count;
    uint64_t tensor_count;
    uint64_t alignment;
    uint64_t directory_end_offset;
    uint64_t data_section_offset;
    uint64_t packed_tensor_bytes;
    uint64_t format_overhead_bytes;
    uint64_t directory_bytes_parsed;
    uint64_t payload_bytes_accessed;
    kq_string_view architecture;
};

/* Format facts verified by Task 1.3 against its pinned MIT-licensed evidence;
 * this table is independent Kestrel-Q code and no GGML implementation is linked.
 */
static const kq_gguf_type_info kq_supported_types[] = {
    {KQ_GGUF_TYPE_F32, "F32", 1U, 4U},
    {KQ_GGUF_TYPE_Q5_1, "Q5_1", 32U, 24U},
    {KQ_GGUF_TYPE_Q8_0, "Q8_0", 32U, 34U},
    {KQ_GGUF_TYPE_Q4_K, "Q4_K", 256U, 144U},
    {KQ_GGUF_TYPE_Q5_K, "Q5_K", 256U, 176U},
    {KQ_GGUF_TYPE_IQ4_NL, "IQ4_NL", 32U, 18U},
    {KQ_GGUF_TYPE_BF16, "BF16", 1U, 2U}
};

static int kq_u64_add_checked(uint64_t left,
                              uint64_t right,
                              uint64_t *result) {
    if (UINT64_MAX - left < right) {
        return 0;
    }
    *result = left + right;
    return 1;
}

static int kq_u64_mul_checked(uint64_t left,
                              uint64_t right,
                              uint64_t *result) {
    if (left != 0U && right > UINT64_MAX / left) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static int kq_size_allocation_checked(uint64_t count, size_t item_size) {
    return item_size != 0U && count <= (uint64_t)(SIZE_MAX / item_size);
}

static kq_status kq_align_up_checked(uint64_t value,
                                     uint64_t alignment,
                                     uint64_t *result,
                                     kq_diagnostic *diagnostic) {
    uint64_t added;

    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U ||
        alignment > KQ_GGUF_MAX_ALIGNMENT) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INVALID_ALIGNMENT,
                          "GGUF alignment %llu is not a supported power of two",
                          (unsigned long long)alignment);
        return KQ_STATUS_INVALID_ALIGNMENT;
    }
    if (!kq_u64_add_checked(value, alignment - 1U, &added)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "alignment calculation overflows uint64");
        return KQ_STATUS_ARITHMETIC_OVERFLOW;
    }
    *result = added & ~(alignment - 1U);
    return KQ_STATUS_OK;
}

static kq_status kq_reader_take(kq_reader *reader,
                                uint64_t length,
                                const unsigned char **data) {
    uint64_t end;

    if (!kq_u64_add_checked(reader->offset, length, &end)) {
        kq_diagnostic_set(reader->diagnostic,
                          KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "GGUF cursor offset plus length overflows uint64");
        return KQ_STATUS_ARITHMETIC_OVERFLOW;
    }
    if (end > reader->length) {
        if (reader->length < reader->file_size) {
            kq_diagnostic_set(reader->diagnostic,
                              KQ_STATUS_LIMIT_EXCEEDED,
                              "GGUF header/directory exceeds %llu-byte limit",
                              (unsigned long long)reader->length);
            return KQ_STATUS_LIMIT_EXCEEDED;
        }
        kq_diagnostic_set(reader->diagnostic,
                          KQ_STATUS_TRUNCATED,
                          "GGUF structure ends at %llu but file size is %llu",
                          (unsigned long long)end,
                          (unsigned long long)reader->length);
        return KQ_STATUS_TRUNCATED;
    }

    if (data != NULL) {
        *data = reader->data + (size_t)reader->offset;
    }
    reader->offset = end;
    if (end > reader->max_consumed) {
        reader->max_consumed = end;
    }
    return KQ_STATUS_OK;
}

static kq_status kq_reader_u8(kq_reader *reader, uint8_t *value) {
    const unsigned char *data;
    kq_status status = kq_reader_take(reader, 1U, &data);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    *value = data[0];
    return KQ_STATUS_OK;
}

static kq_status kq_reader_u16(kq_reader *reader, uint16_t *value) {
    const unsigned char *data;
    kq_status status = kq_reader_take(reader, 2U, &data);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    *value = (uint16_t)((uint16_t)data[0] |
                        ((uint16_t)data[1] << 8U));
    return KQ_STATUS_OK;
}

static kq_status kq_reader_u32(kq_reader *reader, uint32_t *value) {
    const unsigned char *data;
    kq_status status = kq_reader_take(reader, 4U, &data);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    *value = (uint32_t)data[0] |
             ((uint32_t)data[1] << 8U) |
             ((uint32_t)data[2] << 16U) |
             ((uint32_t)data[3] << 24U);
    return KQ_STATUS_OK;
}

static kq_status kq_reader_u64(kq_reader *reader, uint64_t *value) {
    const unsigned char *data;
    kq_status status = kq_reader_take(reader, 8U, &data);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    *value = (uint64_t)data[0] |
             ((uint64_t)data[1] << 8U) |
             ((uint64_t)data[2] << 16U) |
             ((uint64_t)data[3] << 24U) |
             ((uint64_t)data[4] << 32U) |
             ((uint64_t)data[5] << 40U) |
             ((uint64_t)data[6] << 48U) |
             ((uint64_t)data[7] << 56U);
    return KQ_STATUS_OK;
}

static int kq_utf8_is_continuation(unsigned char value) {
    return (value & 0xc0U) == 0x80U;
}

static int kq_utf8_is_valid(const unsigned char *data, uint64_t length) {
    uint64_t index = 0U;
    unsigned char first;
    unsigned char second;

    while (index < length) {
        first = data[(size_t)index];
        if (first <= 0x7fU) {
            index += 1U;
        } else if (first >= 0xc2U && first <= 0xdfU) {
            if (length - index < 2U ||
                !kq_utf8_is_continuation(data[(size_t)(index + 1U)])) {
                return 0;
            }
            index += 2U;
        } else if (first >= 0xe0U && first <= 0xefU) {
            if (length - index < 3U) {
                return 0;
            }
            second = data[(size_t)(index + 1U)];
            if (!kq_utf8_is_continuation(data[(size_t)(index + 2U)]) ||
                (first == 0xe0U && (second < 0xa0U || second > 0xbfU)) ||
                (first == 0xedU && (second < 0x80U || second > 0x9fU)) ||
                ((first != 0xe0U && first != 0xedU) &&
                 !kq_utf8_is_continuation(second))) {
                return 0;
            }
            index += 3U;
        } else if (first >= 0xf0U && first <= 0xf4U) {
            if (length - index < 4U) {
                return 0;
            }
            second = data[(size_t)(index + 1U)];
            if (!kq_utf8_is_continuation(data[(size_t)(index + 2U)]) ||
                !kq_utf8_is_continuation(data[(size_t)(index + 3U)]) ||
                (first == 0xf0U && (second < 0x90U || second > 0xbfU)) ||
                (first == 0xf4U && (second < 0x80U || second > 0x8fU)) ||
                ((first != 0xf0U && first != 0xf4U) &&
                 !kq_utf8_is_continuation(second))) {
                return 0;
            }
            index += 4U;
        } else {
            return 0;
        }
    }
    return 1;
}

static kq_status kq_reader_string(kq_reader *reader,
                                  kq_string_view *view) {
    uint64_t length;
    const unsigned char *data;
    kq_status status = kq_reader_u64(reader, &length);

    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (length > KQ_GGUF_MAX_STRING_BYTES) {
        kq_diagnostic_set(reader->diagnostic,
                          KQ_STATUS_LIMIT_EXCEEDED,
                          "GGUF string length %llu exceeds defensive limit",
                          (unsigned long long)length);
        return KQ_STATUS_LIMIT_EXCEEDED;
    }
    status = kq_reader_take(reader, length, &data);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (!kq_utf8_is_valid(data, length)) {
        kq_diagnostic_set(reader->diagnostic,
                          KQ_STATUS_MALFORMED_METADATA,
                          "GGUF string is not valid UTF-8");
        return KQ_STATUS_MALFORMED_METADATA;
    }
    view->data = data;
    view->length = length;
    return KQ_STATUS_OK;
}

int kq_string_view_equal(const kq_string_view *left,
                         const kq_string_view *right) {
    if (left == NULL || right == NULL || left->length != right->length) {
        return 0;
    }
    if (left->length == 0U) {
        return 1;
    }
    return memcmp(left->data, right->data, (size_t)left->length) == 0;
}

int kq_string_view_equal_cstr(const kq_string_view *view,
                              const char *text) {
    size_t length;

    if (view == NULL || text == NULL) {
        return 0;
    }
    length = strlen(text);
    if ((uint64_t)length != view->length) {
        return 0;
    }
    return length == 0U || memcmp(view->data, text, length) == 0;
}

const kq_gguf_type_info *kq_gguf_type_info_for(uint32_t type_id) {
    size_t index;

    for (index = 0U;
         index < sizeof(kq_supported_types) / sizeof(kq_supported_types[0]);
         ++index) {
        if (kq_supported_types[index].type_id == type_id) {
            return &kq_supported_types[index];
        }
    }
    return NULL;
}

static int kq_metadata_key_exists(const kq_gguf *gguf,
                                  uint64_t before_index,
                                  const kq_string_view *key) {
    uint64_t index;
    for (index = 0U; index < before_index; ++index) {
        if (kq_string_view_equal(&gguf->metadata[index].key, key)) {
            return 1;
        }
    }
    return 0;
}

static int kq_tensor_name_exists(const kq_gguf *gguf,
                                 uint64_t before_index,
                                 const kq_string_view *name) {
    uint64_t index;
    for (index = 0U; index < before_index; ++index) {
        if (kq_string_view_equal(&gguf->tensors[index].name, name)) {
            return 1;
        }
    }
    return 0;
}

static kq_status kq_read_metadata_scalar(kq_reader *reader,
                                         uint32_t value_type,
                                         uint64_t *value) {
    uint8_t value_u8;
    uint16_t value_u16;
    uint32_t value_u32;
    kq_status status;

    switch (value_type) {
        case KQ_GGUF_VALUE_UINT16:
            status = kq_reader_u16(reader, &value_u16);
            if (status == KQ_STATUS_OK) {
                *value = value_u16;
            }
            return status;
        case KQ_GGUF_VALUE_UINT32:
        case KQ_GGUF_VALUE_INT32:
        case KQ_GGUF_VALUE_FLOAT32:
            status = kq_reader_u32(reader, &value_u32);
            if (status == KQ_STATUS_OK) {
                *value = value_u32;
            }
            return status;
        case KQ_GGUF_VALUE_BOOL:
            status = kq_reader_u8(reader, &value_u8);
            if (status != KQ_STATUS_OK) {
                return status;
            }
            if (value_u8 > 1U) {
                kq_diagnostic_set(reader->diagnostic,
                                  KQ_STATUS_MALFORMED_METADATA,
                                  "GGUF bool value %u is not 0 or 1",
                                  (unsigned int)value_u8);
                return KQ_STATUS_MALFORMED_METADATA;
            }
            *value = value_u8;
            return KQ_STATUS_OK;
        default:
            kq_diagnostic_set(reader->diagnostic,
                              KQ_STATUS_UNSUPPORTED_METADATA,
                              "unsupported scalar GGUF metadata type %u",
                              (unsigned int)value_type);
            return KQ_STATUS_UNSUPPORTED_METADATA;
    }
}

static kq_status kq_read_metadata_array(kq_reader *reader,
                                        kq_gguf_metadata *metadata) {
    const unsigned char *array_data = NULL;
    uint32_t element_type;
    uint64_t length;
    uint64_t bytes;
    uint64_t index;
    kq_string_view ignored_string;
    kq_status status;

    status = kq_reader_u32(reader, &element_type);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    status = kq_reader_u64(reader, &length);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (length > KQ_GGUF_MAX_ARRAY_LENGTH) {
        kq_diagnostic_set(reader->diagnostic,
                          KQ_STATUS_LIMIT_EXCEEDED,
                          "GGUF array length %llu exceeds defensive limit",
                          (unsigned long long)length);
        return KQ_STATUS_LIMIT_EXCEEDED;
    }

    metadata->array_element_type = element_type;
    metadata->array_length = length;
    switch (element_type) {
        case KQ_GGUF_VALUE_INT32:
            if (!kq_u64_mul_checked(length, 4U, &bytes)) {
                kq_diagnostic_set(reader->diagnostic,
                                  KQ_STATUS_ARITHMETIC_OVERFLOW,
                                  "GGUF int32 array byte count overflows");
                return KQ_STATUS_ARITHMETIC_OVERFLOW;
            }
            status = kq_reader_take(reader, bytes, &array_data);
            if (status == KQ_STATUS_OK) {
                metadata->array_data.data = array_data;
                metadata->array_data.length = bytes;
            }
            return status;
        case KQ_GGUF_VALUE_UINT64:
            if (!kq_u64_mul_checked(length, 8U, &bytes)) {
                kq_diagnostic_set(reader->diagnostic,
                                  KQ_STATUS_ARITHMETIC_OVERFLOW,
                                  "GGUF uint64 array byte count overflows");
                return KQ_STATUS_ARITHMETIC_OVERFLOW;
            }
            status = kq_reader_take(reader, bytes, &array_data);
            if (status == KQ_STATUS_OK) {
                metadata->array_data.data = array_data;
                metadata->array_data.length = bytes;
            }
            return status;
        case KQ_GGUF_VALUE_STRING:
            for (index = 0U; index < length; ++index) {
                status = kq_reader_string(reader, &ignored_string);
                if (status != KQ_STATUS_OK) {
                    return status;
                }
            }
            return KQ_STATUS_OK;
        default:
            kq_diagnostic_set(reader->diagnostic,
                              KQ_STATUS_UNSUPPORTED_METADATA,
                              "unsupported GGUF array element type %u",
                              (unsigned int)element_type);
            return KQ_STATUS_UNSUPPORTED_METADATA;
    }
}

static kq_status kq_parse_metadata(kq_reader *reader, kq_gguf *gguf) {
    uint64_t index;
    uint64_t scalar_value = 0U;
    uint32_t value_type;
    kq_status status;

    gguf->alignment = KQ_GGUF_DEFAULT_ALIGNMENT;
    for (index = 0U; index < gguf->metadata_count; ++index) {
        kq_gguf_metadata *metadata = &gguf->metadata[index];

        status = kq_reader_string(reader, &metadata->key);
        if (status != KQ_STATUS_OK) {
            return status;
        }
        if (kq_metadata_key_exists(gguf, index, &metadata->key)) {
            kq_diagnostic_set(reader->diagnostic,
                              KQ_STATUS_DUPLICATE_METADATA,
                              "duplicate GGUF metadata key at index %llu",
                              (unsigned long long)index);
            return KQ_STATUS_DUPLICATE_METADATA;
        }
        status = kq_reader_u32(reader, &value_type);
        if (status != KQ_STATUS_OK) {
            return status;
        }
        metadata->value_type = value_type;

        if (value_type == KQ_GGUF_VALUE_STRING) {
            status = kq_reader_string(reader, &metadata->string_value);
        } else if (value_type == KQ_GGUF_VALUE_ARRAY) {
            status = kq_read_metadata_array(reader, metadata);
        } else {
            status = kq_read_metadata_scalar(reader,
                                             value_type,
                                             &scalar_value);
            if (status == KQ_STATUS_OK) {
                metadata->scalar_value = scalar_value;
            }
        }
        if (status != KQ_STATUS_OK) {
            return status;
        }

        if (kq_string_view_equal_cstr(&metadata->key,
                                      "general.architecture")) {
            if (value_type != KQ_GGUF_VALUE_STRING) {
                kq_diagnostic_set(reader->diagnostic,
                                  KQ_STATUS_MALFORMED_METADATA,
                                  "general.architecture must be a string");
                return KQ_STATUS_MALFORMED_METADATA;
            }
            gguf->architecture = metadata->string_value;
        } else if (kq_string_view_equal_cstr(&metadata->key,
                                             "general.alignment")) {
            if (value_type != KQ_GGUF_VALUE_UINT32) {
                kq_diagnostic_set(reader->diagnostic,
                                  KQ_STATUS_MALFORMED_METADATA,
                                  "general.alignment must be uint32");
                return KQ_STATUS_MALFORMED_METADATA;
            }
            gguf->alignment = scalar_value;
        }
    }

    if (gguf->architecture.data == NULL) {
        kq_diagnostic_set(reader->diagnostic,
                          KQ_STATUS_MALFORMED_METADATA,
                          "required general.architecture metadata is absent");
        return KQ_STATUS_MALFORMED_METADATA;
    }
    if (gguf->alignment == 0U ||
        gguf->alignment > KQ_GGUF_MAX_ALIGNMENT ||
        (gguf->alignment & (gguf->alignment - 1U)) != 0U) {
        kq_diagnostic_set(reader->diagnostic,
                          KQ_STATUS_INVALID_ALIGNMENT,
                          "GGUF alignment %llu is not a supported power of two",
                          (unsigned long long)gguf->alignment);
        return KQ_STATUS_INVALID_ALIGNMENT;
    }
    return KQ_STATUS_OK;
}

static kq_status kq_parse_tensors(kq_reader *reader, kq_gguf *gguf) {
    uint64_t index;
    uint64_t dimension_index;
    uint64_t element_count;
    uint64_t packed_blocks;
    uint32_t rank;
    uint32_t type_id;
    const kq_gguf_type_info *type_info;
    kq_status status;

    for (index = 0U; index < gguf->tensor_count; ++index) {
        kq_gguf_tensor *tensor = &gguf->tensors[index];

        status = kq_reader_string(reader, &tensor->name);
        if (status != KQ_STATUS_OK) {
            return status;
        }
        if (kq_tensor_name_exists(gguf, index, &tensor->name)) {
            kq_diagnostic_set(reader->diagnostic,
                              KQ_STATUS_DUPLICATE_TENSOR,
                              "duplicate GGUF tensor name at index %llu",
                              (unsigned long long)index);
            return KQ_STATUS_DUPLICATE_TENSOR;
        }
        status = kq_reader_u32(reader, &rank);
        if (status != KQ_STATUS_OK) {
            return status;
        }
        if (rank == 0U || rank > KQ_GGUF_MAX_DIMS) {
            kq_diagnostic_set(reader->diagnostic,
                              KQ_STATUS_MALFORMED_TENSOR,
                              "tensor %llu has invalid rank %u",
                              (unsigned long long)index,
                              (unsigned int)rank);
            return KQ_STATUS_MALFORMED_TENSOR;
        }
        tensor->rank = rank;
        element_count = 1U;
        for (dimension_index = 0U;
             dimension_index < (uint64_t)rank;
             ++dimension_index) {
            status = kq_reader_u64(
                reader,
                &tensor->dimensions[(size_t)dimension_index]);
            if (status != KQ_STATUS_OK) {
                return status;
            }
            if (tensor->dimensions[(size_t)dimension_index] == 0U) {
                kq_diagnostic_set(reader->diagnostic,
                                  KQ_STATUS_MALFORMED_TENSOR,
                                  "tensor %llu has a zero dimension",
                                  (unsigned long long)index);
                return KQ_STATUS_MALFORMED_TENSOR;
            }
            if (!kq_u64_mul_checked(
                    element_count,
                    tensor->dimensions[(size_t)dimension_index],
                    &element_count)) {
                kq_diagnostic_set(reader->diagnostic,
                                  KQ_STATUS_ARITHMETIC_OVERFLOW,
                                  "tensor %llu dimension product overflows",
                                  (unsigned long long)index);
                return KQ_STATUS_ARITHMETIC_OVERFLOW;
            }
        }
        status = kq_reader_u32(reader, &type_id);
        if (status != KQ_STATUS_OK) {
            return status;
        }
        type_info = kq_gguf_type_info_for(type_id);
        if (type_info == NULL) {
            kq_diagnostic_set(reader->diagnostic,
                              KQ_STATUS_UNSUPPORTED_TENSOR_TYPE,
                              "tensor %llu uses unsupported GGML type %u",
                              (unsigned long long)index,
                              (unsigned int)type_id);
            return KQ_STATUS_UNSUPPORTED_TENSOR_TYPE;
        }
        status = kq_reader_u64(reader, &tensor->relative_offset);
        if (status != KQ_STATUS_OK) {
            return status;
        }
        if ((tensor->relative_offset % gguf->alignment) != 0U) {
            kq_diagnostic_set(reader->diagnostic,
                              KQ_STATUS_INVALID_ALIGNMENT,
                              "tensor %llu offset %llu is not aligned to %llu",
                              (unsigned long long)index,
                              (unsigned long long)tensor->relative_offset,
                              (unsigned long long)gguf->alignment);
            return KQ_STATUS_INVALID_ALIGNMENT;
        }
        if ((tensor->dimensions[0] % type_info->block_elements) != 0U) {
            kq_diagnostic_set(reader->diagnostic,
                              KQ_STATUS_MALFORMED_TENSOR,
                              "tensor %llu first dimension %llu is not divisible by block %llu",
                              (unsigned long long)index,
                              (unsigned long long)tensor->dimensions[0],
                              (unsigned long long)type_info->block_elements);
            return KQ_STATUS_MALFORMED_TENSOR;
        }

        packed_blocks = element_count / type_info->block_elements;
        if (!kq_u64_mul_checked(packed_blocks,
                                type_info->bytes_per_block,
                                &tensor->packed_bytes)) {
            kq_diagnostic_set(reader->diagnostic,
                              KQ_STATUS_ARITHMETIC_OVERFLOW,
                              "tensor %llu packed byte count overflows",
                              (unsigned long long)index);
            return KQ_STATUS_ARITHMETIC_OVERFLOW;
        }
        tensor->type_id = type_id;
        tensor->block_elements = type_info->block_elements;
        tensor->bytes_per_block = type_info->bytes_per_block;
        tensor->element_count = element_count;
    }
    return KQ_STATUS_OK;
}

static int kq_tensor_order_compare(const void *left, const void *right) {
    const kq_tensor_order *left_order = (const kq_tensor_order *)left;
    const kq_tensor_order *right_order = (const kq_tensor_order *)right;

    if (left_order->offset < right_order->offset) {
        return -1;
    }
    if (left_order->offset > right_order->offset) {
        return 1;
    }
    if (left_order->index < right_order->index) {
        return -1;
    }
    return left_order->index > right_order->index ? 1 : 0;
}

static kq_status kq_validate_spans(kq_gguf *gguf,
                                   kq_diagnostic *diagnostic) {
    kq_tensor_order *order;
    uint64_t index;
    uint64_t relative_end;
    uint64_t absolute_end;
    uint64_t next_offset;
    uint64_t total_packed = 0U;
    uint64_t tensor_index;
    kq_gguf_tensor *tensor;

    if (!kq_size_allocation_checked(gguf->tensor_count, sizeof(*order))) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "tensor-order allocation size overflows SIZE_T");
        return KQ_STATUS_ARITHMETIC_OVERFLOW;
    }
    order = (kq_tensor_order *)calloc((size_t)gguf->tensor_count,
                                      sizeof(*order));
    if (order == NULL) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_OUT_OF_MEMORY,
                          "could not allocate tensor span order");
        return KQ_STATUS_OUT_OF_MEMORY;
    }

    for (index = 0U; index < gguf->tensor_count; ++index) {
        order[(size_t)index].offset =
            gguf->tensors[(size_t)index].relative_offset;
        order[(size_t)index].packed_bytes =
            gguf->tensors[(size_t)index].packed_bytes;
        order[(size_t)index].index = index;
    }
    qsort(order,
          (size_t)gguf->tensor_count,
          sizeof(*order),
          kq_tensor_order_compare);

    if (order[0].offset != 0U) {
        free(order);
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INCONSISTENT_DATA_SECTION,
                          "first tensor relative offset is %llu instead of zero",
                          (unsigned long long)order[0].offset);
        return KQ_STATUS_INCONSISTENT_DATA_SECTION;
    }

    for (index = 0U; index < gguf->tensor_count; ++index) {
        tensor_index = order[(size_t)index].index;
        tensor = &gguf->tensors[(size_t)tensor_index];
        if (!kq_u64_add_checked(tensor->relative_offset,
                                tensor->packed_bytes,
                                &relative_end) ||
            !kq_u64_add_checked(gguf->data_section_offset,
                                tensor->relative_offset,
                                &tensor->data_offset) ||
            !kq_u64_add_checked(tensor->data_offset,
                                tensor->packed_bytes,
                                &absolute_end)) {
            free(order);
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_ARITHMETIC_OVERFLOW,
                              "tensor %llu span arithmetic overflows",
                              (unsigned long long)tensor_index);
            return KQ_STATUS_ARITHMETIC_OVERFLOW;
        }
        if (absolute_end > kq_file_size(gguf->file)) {
            free(order);
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_SPAN_OUT_OF_RANGE,
                              "tensor %llu span ends at %llu beyond file size %llu",
                              (unsigned long long)tensor_index,
                              (unsigned long long)absolute_end,
                              (unsigned long long)kq_file_size(gguf->file));
            return KQ_STATUS_SPAN_OUT_OF_RANGE;
        }
        if (index + 1U < gguf->tensor_count) {
            next_offset = order[(size_t)(index + 1U)].offset;
            if (relative_end > next_offset) {
                free(order);
                kq_diagnostic_set(diagnostic,
                                  KQ_STATUS_INCONSISTENT_DATA_SECTION,
                                  "tensor %llu overlaps the next tensor",
                                  (unsigned long long)tensor_index);
                return KQ_STATUS_INCONSISTENT_DATA_SECTION;
            }
        }
        if (!kq_u64_add_checked(total_packed,
                                tensor->packed_bytes,
                                &total_packed)) {
            free(order);
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_ARITHMETIC_OVERFLOW,
                              "aggregate packed tensor bytes overflow");
            return KQ_STATUS_ARITHMETIC_OVERFLOW;
        }
    }
    free(order);

    if (total_packed > kq_file_size(gguf->file)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INCONSISTENT_DATA_SECTION,
                          "packed tensor bytes exceed file size");
        return KQ_STATUS_INCONSISTENT_DATA_SECTION;
    }
    gguf->packed_tensor_bytes = total_packed;
    gguf->format_overhead_bytes = kq_file_size(gguf->file) - total_packed;
    return KQ_STATUS_OK;
}

kq_status kq_gguf_open(kq_file *file,
                       kq_gguf **out_gguf,
                       kq_diagnostic *diagnostic) {
    uint64_t file_size;
    uint64_t view_length;
    uint64_t tensor_count = 0U;
    uint64_t metadata_count = 0U;
    uint32_t magic;
    uint32_t version;
    kq_reader reader;
    kq_gguf *gguf = NULL;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    if (file == NULL || out_gguf == NULL) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INVALID_ARGUMENT,
                          "file and output GGUF are required");
        return KQ_STATUS_INVALID_ARGUMENT;
    }
    *out_gguf = NULL;
    file_size = kq_file_size(file);
    if (file_size < KQ_GGUF_FIXED_HEADER_BYTES) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_TRUNCATED,
                          "file size %llu is smaller than GGUF fixed header",
                          (unsigned long long)file_size);
        return KQ_STATUS_TRUNCATED;
    }

    gguf = (kq_gguf *)calloc(1U, sizeof(*gguf));
    if (gguf == NULL) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_OUT_OF_MEMORY,
                          "could not allocate GGUF state");
        return KQ_STATUS_OUT_OF_MEMORY;
    }
    gguf->file = file;

    view_length = file_size < KQ_GGUF_MAX_HEADER_DIRECTORY_BYTES
                      ? file_size
                      : KQ_GGUF_MAX_HEADER_DIRECTORY_BYTES;
    status = kq_file_view_open(file,
                               0U,
                               view_length,
                               &gguf->directory_view,
                               diagnostic);
    if (status != KQ_STATUS_OK) {
        kq_gguf_close(gguf);
        return status;
    }

    reader.data = kq_file_view_data(gguf->directory_view);
    reader.length = kq_file_view_length(gguf->directory_view);
    reader.file_size = file_size;
    reader.offset = 0U;
    reader.max_consumed = 0U;
    reader.diagnostic = diagnostic;

    status = kq_reader_u32(&reader, &magic);
    if (status != KQ_STATUS_OK) {
        kq_gguf_close(gguf);
        return status;
    }
    if (magic != UINT32_C(0x46554747)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_BAD_MAGIC,
                          "expected GGUF magic bytes");
        kq_gguf_close(gguf);
        return KQ_STATUS_BAD_MAGIC;
    }
    status = kq_reader_u32(&reader, &version);
    if (status != KQ_STATUS_OK) {
        kq_gguf_close(gguf);
        return status;
    }
    if (version != 3U) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_UNSUPPORTED_VERSION,
                          "GGUF version %u is unsupported; expected version 3",
                          (unsigned int)version);
        kq_gguf_close(gguf);
        return KQ_STATUS_UNSUPPORTED_VERSION;
    }
    status = kq_reader_u64(&reader, &tensor_count);
    if (status == KQ_STATUS_OK) {
        status = kq_reader_u64(&reader, &metadata_count);
    }
    if (status != KQ_STATUS_OK) {
        kq_gguf_close(gguf);
        return status;
    }
    if (tensor_count == 0U || metadata_count == 0U) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_MALFORMED_METADATA,
                          "GGUF requires non-zero metadata and tensor counts");
        kq_gguf_close(gguf);
        return KQ_STATUS_MALFORMED_METADATA;
    }
    if (!kq_size_allocation_checked(metadata_count,
                                    sizeof(*gguf->metadata)) ||
        !kq_size_allocation_checked(tensor_count,
                                    sizeof(*gguf->tensors))) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "GGUF directory allocation size overflows SIZE_T");
        kq_gguf_close(gguf);
        return KQ_STATUS_ARITHMETIC_OVERFLOW;
    }
    if (metadata_count > KQ_GGUF_MAX_METADATA_COUNT ||
        tensor_count > KQ_GGUF_MAX_TENSOR_COUNT) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_LIMIT_EXCEEDED,
                          "GGUF directory count exceeds defensive limit");
        kq_gguf_close(gguf);
        return KQ_STATUS_LIMIT_EXCEEDED;
    }

    gguf->version = version;
    gguf->metadata_count = metadata_count;
    gguf->tensor_count = tensor_count;
    gguf->metadata = (kq_gguf_metadata *)calloc(
        (size_t)metadata_count,
        sizeof(*gguf->metadata));
    gguf->tensors = (kq_gguf_tensor *)calloc((size_t)tensor_count,
                                             sizeof(*gguf->tensors));
    if (gguf->metadata == NULL || gguf->tensors == NULL) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_OUT_OF_MEMORY,
                          "could not allocate GGUF directory descriptors");
        kq_gguf_close(gguf);
        return KQ_STATUS_OUT_OF_MEMORY;
    }

    status = kq_parse_metadata(&reader, gguf);
    if (status == KQ_STATUS_OK) {
        status = kq_parse_tensors(&reader, gguf);
    }
    if (status != KQ_STATUS_OK) {
        kq_gguf_close(gguf);
        return status;
    }

    gguf->directory_end_offset = reader.offset;
    gguf->directory_bytes_parsed = reader.max_consumed;
    status = kq_align_up_checked(gguf->directory_end_offset,
                                 gguf->alignment,
                                 &gguf->data_section_offset,
                                 diagnostic);
    if (status != KQ_STATUS_OK) {
        kq_gguf_close(gguf);
        return status;
    }
    if (gguf->data_section_offset > file_size) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INCONSISTENT_DATA_SECTION,
                          "GGUF data section starts beyond end of file");
        kq_gguf_close(gguf);
        return KQ_STATUS_INCONSISTENT_DATA_SECTION;
    }

    status = kq_validate_spans(gguf, diagnostic);
    if (status != KQ_STATUS_OK) {
        kq_gguf_close(gguf);
        return status;
    }
    gguf->payload_bytes_accessed = 0U;
    *out_gguf = gguf;
    return KQ_STATUS_OK;
}

void kq_gguf_close(kq_gguf *gguf) {
    if (gguf == NULL) {
        return;
    }
    free(gguf->tensors);
    free(gguf->metadata);
    kq_file_view_close(gguf->directory_view);
    free(gguf);
}

uint32_t kq_gguf_version(const kq_gguf *gguf) {
    return gguf == NULL ? 0U : gguf->version;
}

uint64_t kq_gguf_metadata_count(const kq_gguf *gguf) {
    return gguf == NULL ? 0U : gguf->metadata_count;
}

uint64_t kq_gguf_tensor_count(const kq_gguf *gguf) {
    return gguf == NULL ? 0U : gguf->tensor_count;
}

uint64_t kq_gguf_alignment(const kq_gguf *gguf) {
    return gguf == NULL ? 0U : gguf->alignment;
}

uint64_t kq_gguf_directory_end_offset(const kq_gguf *gguf) {
    return gguf == NULL ? 0U : gguf->directory_end_offset;
}

uint64_t kq_gguf_data_section_offset(const kq_gguf *gguf) {
    return gguf == NULL ? 0U : gguf->data_section_offset;
}

uint64_t kq_gguf_packed_tensor_bytes(const kq_gguf *gguf) {
    return gguf == NULL ? 0U : gguf->packed_tensor_bytes;
}

uint64_t kq_gguf_format_overhead_bytes(const kq_gguf *gguf) {
    return gguf == NULL ? 0U : gguf->format_overhead_bytes;
}

uint64_t kq_gguf_directory_bytes_parsed(const kq_gguf *gguf) {
    return gguf == NULL ? 0U : gguf->directory_bytes_parsed;
}

uint64_t kq_gguf_payload_bytes_accessed(const kq_gguf *gguf) {
    return gguf == NULL ? 0U : gguf->payload_bytes_accessed;
}

kq_string_view kq_gguf_architecture(const kq_gguf *gguf) {
    kq_string_view empty = {NULL, 0U};
    return gguf == NULL ? empty : gguf->architecture;
}

const kq_gguf_metadata *kq_gguf_metadata_at(const kq_gguf *gguf,
                                            uint64_t index) {
    if (gguf == NULL || index >= gguf->metadata_count) {
        return NULL;
    }
    return &gguf->metadata[(size_t)index];
}

const kq_gguf_metadata *kq_gguf_find_metadata(const kq_gguf *gguf,
                                              const char *key) {
    uint64_t index;
    if (gguf == NULL || key == NULL) {
        return NULL;
    }
    for (index = 0U; index < gguf->metadata_count; ++index) {
        if (kq_string_view_equal_cstr(&gguf->metadata[(size_t)index].key,
                                      key)) {
            return &gguf->metadata[(size_t)index];
        }
    }
    return NULL;
}

const kq_gguf_tensor *kq_gguf_tensor_at(const kq_gguf *gguf,
                                        uint64_t index) {
    if (gguf == NULL || index >= gguf->tensor_count) {
        return NULL;
    }
    return &gguf->tensors[(size_t)index];
}

const kq_gguf_tensor *kq_gguf_find_tensor(const kq_gguf *gguf,
                                          const char *name) {
    uint64_t index;
    if (gguf == NULL || name == NULL) {
        return NULL;
    }
    for (index = 0U; index < gguf->tensor_count; ++index) {
        if (kq_string_view_equal_cstr(&gguf->tensors[(size_t)index].name,
                                      name)) {
            return &gguf->tensors[(size_t)index];
        }
    }
    return NULL;
}

int kq_gguf_metadata_u16(const kq_gguf_metadata *metadata, uint16_t *value) {
    if (metadata == NULL || value == NULL ||
        metadata->value_type != KQ_GGUF_VALUE_UINT16) {
        return 0;
    }
    *value = (uint16_t)metadata->scalar_value;
    return 1;
}

int kq_gguf_metadata_u32(const kq_gguf_metadata *metadata, uint32_t *value) {
    if (metadata == NULL || value == NULL ||
        metadata->value_type != KQ_GGUF_VALUE_UINT32) {
        return 0;
    }
    *value = (uint32_t)metadata->scalar_value;
    return 1;
}

int kq_gguf_metadata_i32(const kq_gguf_metadata *metadata, int32_t *value) {
    uint32_t bits;
    if (metadata == NULL || value == NULL ||
        metadata->value_type != KQ_GGUF_VALUE_INT32) {
        return 0;
    }
    bits = (uint32_t)metadata->scalar_value;
    memcpy(value, &bits, sizeof(bits));
    return 1;
}

int kq_gguf_metadata_array_i32_at(const kq_gguf_metadata *metadata,
                                  uint64_t index,
                                  int32_t *value) {
    const unsigned char *data;
    uint32_t bits;
    if (metadata == NULL || value == NULL ||
        metadata->value_type != KQ_GGUF_VALUE_ARRAY ||
        metadata->array_element_type != KQ_GGUF_VALUE_INT32 ||
        index >= metadata->array_length || metadata->array_data.data == NULL) {
        return 0;
    }
    data = metadata->array_data.data + (size_t)(index * 4U);
    bits = (uint32_t)data[0] |
           ((uint32_t)data[1] << 8U) |
           ((uint32_t)data[2] << 16U) |
           ((uint32_t)data[3] << 24U);
    memcpy(value, &bits, sizeof(bits));
    return 1;
}

int kq_gguf_metadata_array_u64_at(const kq_gguf_metadata *metadata,
                                  uint64_t index,
                                  uint64_t *value) {
    const unsigned char *data;
    uint64_t result = 0U;
    unsigned int byte_index;
    if (metadata == NULL || value == NULL ||
        metadata->value_type != KQ_GGUF_VALUE_ARRAY ||
        metadata->array_element_type != KQ_GGUF_VALUE_UINT64 ||
        index >= metadata->array_length || metadata->array_data.data == NULL) {
        return 0;
    }
    data = metadata->array_data.data + (size_t)(index * 8U);
    for (byte_index = 0U; byte_index < 8U; ++byte_index) {
        result |= (uint64_t)data[byte_index] << (byte_index * 8U);
    }
    *value = result;
    return 1;
}
