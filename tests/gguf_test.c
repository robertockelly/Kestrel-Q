#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_status.h"

#define TEST_BUFFER_CAPACITY 8192U
#define TEST_TENSOR_COUNT 7U
#define TEST_METADATA_COUNT 4U
#define TEST_ALIGNMENT 32U
#define TEST_PACKED_BYTES 420U

typedef struct test_fixture {
    unsigned char bytes[TEST_BUFFER_CAPACITY];
    size_t size;
    size_t first_key_length_offset;
    size_t alignment_value_offset;
    size_t array_key_offset[2];
    size_t array_element_type_offset;
    size_t array_length_offset;
    size_t tensor_name_offset[TEST_TENSOR_COUNT];
    size_t tensor_rank_offset[TEST_TENSOR_COUNT];
    size_t tensor_dimension_offset[TEST_TENSOR_COUNT];
    size_t tensor_type_offset[TEST_TENSOR_COUNT];
    size_t tensor_offset_offset[TEST_TENSOR_COUNT];
    size_t directory_end;
    size_t data_section_offset;
} test_fixture;

static int test_failures = 0;

static void test_check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        test_failures += 1;
    }
}

static int fixture_reserve(test_fixture *fixture, size_t length) {
    return fixture->size <= TEST_BUFFER_CAPACITY &&
           length <= TEST_BUFFER_CAPACITY - fixture->size;
}

static int fixture_append_bytes(test_fixture *fixture,
                                const void *bytes,
                                size_t length) {
    if (!fixture_reserve(fixture, length)) {
        return 0;
    }
    if (length != 0U) {
        memcpy(fixture->bytes + fixture->size, bytes, length);
    }
    fixture->size += length;
    return 1;
}

static int fixture_append_u8(test_fixture *fixture, uint8_t value) {
    return fixture_append_bytes(fixture, &value, 1U);
}

static int fixture_append_u32(test_fixture *fixture, uint32_t value) {
    unsigned char bytes[4];
    bytes[0] = (unsigned char)(value & UINT32_C(0xff));
    bytes[1] = (unsigned char)((value >> 8U) & UINT32_C(0xff));
    bytes[2] = (unsigned char)((value >> 16U) & UINT32_C(0xff));
    bytes[3] = (unsigned char)((value >> 24U) & UINT32_C(0xff));
    return fixture_append_bytes(fixture, bytes, sizeof(bytes));
}

static int fixture_append_u64(test_fixture *fixture, uint64_t value) {
    unsigned char bytes[8];
    unsigned int index;
    for (index = 0U; index < 8U; ++index) {
        bytes[index] = (unsigned char)((value >> (index * 8U)) &
                                       UINT64_C(0xff));
    }
    return fixture_append_bytes(fixture, bytes, sizeof(bytes));
}

static int fixture_append_string(test_fixture *fixture, const char *text) {
    size_t length = strlen(text);
    return fixture_append_u64(fixture, (uint64_t)length) &&
           fixture_append_bytes(fixture, text, length);
}

static int fixture_pad_to(test_fixture *fixture, size_t alignment) {
    while ((fixture->size % alignment) != 0U) {
        if (!fixture_append_u8(fixture, 0U)) {
            return 0;
        }
    }
    return 1;
}

static size_t align_size(size_t value, size_t alignment) {
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static void patch_u32(test_fixture *fixture, size_t offset, uint32_t value) {
    unsigned int index;
    for (index = 0U; index < 4U; ++index) {
        fixture->bytes[offset + index] =
            (unsigned char)((value >> (index * 8U)) & UINT32_C(0xff));
    }
}

static void patch_u64(test_fixture *fixture, size_t offset, uint64_t value) {
    unsigned int index;
    for (index = 0U; index < 8U; ++index) {
        fixture->bytes[offset + index] =
            (unsigned char)((value >> (index * 8U)) & UINT64_C(0xff));
    }
}

static int build_valid_fixture(test_fixture *fixture) {
    static const uint32_t type_ids[TEST_TENSOR_COUNT] = {
        KQ_GGUF_TYPE_F32,
        KQ_GGUF_TYPE_BF16,
        KQ_GGUF_TYPE_Q5_1,
        KQ_GGUF_TYPE_Q8_0,
        KQ_GGUF_TYPE_Q4_K,
        KQ_GGUF_TYPE_Q5_K,
        KQ_GGUF_TYPE_IQ4_NL
    };
    static const uint64_t dimensions[TEST_TENSOR_COUNT] = {
        4U, 4U, 32U, 32U, 256U, 256U, 32U
    };
    static const uint64_t packed_bytes[TEST_TENSOR_COUNT] = {
        16U, 8U, 24U, 34U, 144U, 176U, 18U
    };
    uint64_t relative_offsets[TEST_TENSOR_COUNT];
    uint64_t relative_cursor = 0U;
    unsigned int index;
    char tensor_name[3] = {'t', '0', '\0'};

    memset(fixture, 0, sizeof(*fixture));
    for (index = 0U; index < TEST_TENSOR_COUNT; ++index) {
        relative_offsets[index] = relative_cursor;
        relative_cursor = (uint64_t)align_size(
            (size_t)(relative_cursor + packed_bytes[index]),
            TEST_ALIGNMENT);
    }

    if (!fixture_append_bytes(fixture, "GGUF", 4U) ||
        !fixture_append_u32(fixture, 3U) ||
        !fixture_append_u64(fixture, TEST_TENSOR_COUNT) ||
        !fixture_append_u64(fixture, TEST_METADATA_COUNT)) {
        return 0;
    }

    fixture->first_key_length_offset = fixture->size;
    if (!fixture_append_string(fixture, "general.architecture") ||
        !fixture_append_u32(fixture, 8U) ||
        !fixture_append_string(fixture, "testarch") ||
        !fixture_append_string(fixture, "general.alignment") ||
        !fixture_append_u32(fixture, 4U)) {
        return 0;
    }
    fixture->alignment_value_offset = fixture->size;
    if (!fixture_append_u32(fixture, TEST_ALIGNMENT)) {
        return 0;
    }
    fixture->array_key_offset[0] = fixture->size + 8U;
    if (!fixture_append_string(fixture, "test.array0") ||
        !fixture_append_u32(fixture, 9U)) {
        return 0;
    }
    fixture->array_element_type_offset = fixture->size;
    if (!fixture_append_u32(fixture, 5U)) {
        return 0;
    }
    fixture->array_length_offset = fixture->size;
    if (!fixture_append_u64(fixture, 3U) ||
        !fixture_append_u32(fixture, 1U) ||
        !fixture_append_u32(fixture, 2U) ||
        !fixture_append_u32(fixture, 3U)) {
        return 0;
    }
    fixture->array_key_offset[1] = fixture->size + 8U;
    if (!fixture_append_string(fixture, "test.array1") ||
        !fixture_append_u32(fixture, 9U) ||
        !fixture_append_u32(fixture, 8U) ||
        !fixture_append_u64(fixture, 2U) ||
        !fixture_append_string(fixture, "one") ||
        !fixture_append_string(fixture, "two")) {
        return 0;
    }

    for (index = 0U; index < TEST_TENSOR_COUNT; ++index) {
        tensor_name[1] = (char)('0' + index);
        if (!fixture_append_u64(fixture, 2U)) {
            return 0;
        }
        fixture->tensor_name_offset[index] = fixture->size;
        if (!fixture_append_bytes(fixture, tensor_name, 2U)) {
            return 0;
        }
        fixture->tensor_rank_offset[index] = fixture->size;
        if (!fixture_append_u32(fixture, 1U)) {
            return 0;
        }
        fixture->tensor_dimension_offset[index] = fixture->size;
        if (!fixture_append_u64(fixture, dimensions[index])) {
            return 0;
        }
        fixture->tensor_type_offset[index] = fixture->size;
        if (!fixture_append_u32(fixture, type_ids[index])) {
            return 0;
        }
        fixture->tensor_offset_offset[index] = fixture->size;
        if (!fixture_append_u64(fixture, relative_offsets[index])) {
            return 0;
        }
    }

    fixture->directory_end = fixture->size;
    if (!fixture_pad_to(fixture, TEST_ALIGNMENT)) {
        return 0;
    }
    fixture->data_section_offset = fixture->size;
    if (!fixture_reserve(fixture,
                         (size_t)(relative_offsets[TEST_TENSOR_COUNT - 1U] +
                                  packed_bytes[TEST_TENSOR_COUNT - 1U]))) {
        return 0;
    }
    fixture->size +=
        (size_t)(relative_offsets[TEST_TENSOR_COUNT - 1U] +
                 packed_bytes[TEST_TENSOR_COUNT - 1U]);
    memset(fixture->bytes + fixture->data_section_offset,
           0xa5,
           fixture->size - fixture->data_section_offset);
    return 1;
}

static int write_temp_fixture(const test_fixture *fixture,
                              size_t file_size,
                              wchar_t path[MAX_PATH]) {
    wchar_t temp_directory[MAX_PATH];
    DWORD directory_length;
    FILE *stream = NULL;
    size_t written;

    if (file_size > fixture->size) {
        return 0;
    }
    directory_length = GetTempPathW(MAX_PATH, temp_directory);
    if (directory_length == 0U || directory_length >= MAX_PATH) {
        return 0;
    }
    if (GetTempFileNameW(temp_directory, L"kqg", 0U, path) == 0U) {
        return 0;
    }
    if (_wfopen_s(&stream, path, L"wb") != 0 || stream == NULL) {
        DeleteFileW(path);
        return 0;
    }
    written = fwrite(fixture->bytes, 1U, file_size, stream);
    if (written != file_size || fclose(stream) != 0) {
        if (written != file_size) {
            (void)fclose(stream);
        }
        DeleteFileW(path);
        return 0;
    }
    return 1;
}

static void test_failure_case(const char *name,
                              const test_fixture *fixture,
                              size_t file_size,
                              kq_status expected_status) {
    wchar_t path[MAX_PATH];
    kq_diagnostic diagnostic;
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_status status;

    if (!write_temp_fixture(fixture, file_size, path)) {
        fprintf(stderr, "FAIL: %s could not write fixture\n", name);
        test_failures += 1;
        return;
    }
    status = kq_file_open_readonly(path, &file, &diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_gguf_open(file, &gguf, &diagnostic);
    }
    if (status != expected_status) {
        fprintf(stderr,
                "FAIL: %s expected %s but received %s (%s)\n",
                name,
                kq_status_string(expected_status),
                kq_status_string(status),
                diagnostic.message);
        test_failures += 1;
    }
    test_check(gguf == NULL, "failed parse must not return a GGUF object");
    kq_gguf_close(gguf);
    kq_file_close(file);
    test_check(DeleteFileW(path) != 0,
               "failed parse must release all views and file handles");
}

static void test_valid_fixture(const test_fixture *fixture) {
    wchar_t path[MAX_PATH];
    kq_diagnostic diagnostic;
    kq_file *file = NULL;
    kq_file_view *view = NULL;
    kq_gguf *gguf = NULL;
    const kq_gguf_tensor *tensor;
    const kq_gguf_metadata *metadata;
    kq_string_view metadata_string;
    const kq_gguf_type_info *type_info;
    kq_string_view architecture;
    kq_status status;
    unsigned int index;
    uint32_t metadata_u32 = 0U;
    int32_t metadata_i32 = 0;

    if (!write_temp_fixture(fixture, fixture->size, path)) {
        fprintf(stderr, "FAIL: valid fixture could not be written\n");
        test_failures += 1;
        return;
    }
    status = kq_file_open_readonly(path, &file, &diagnostic);
    test_check(status == KQ_STATUS_OK, "valid fixture file must open");
    if (status != KQ_STATUS_OK) {
        DeleteFileW(path);
        return;
    }

    status = kq_file_view_open(file, 3U, 13U, &view, &diagnostic);
    test_check(status == KQ_STATUS_OK,
               "unaligned logical view must be mapped successfully");
    if (status == KQ_STATUS_OK) {
        test_check(kq_file_view_offset(view) == 3U,
                   "logical view offset must be preserved");
        test_check(kq_file_view_length(view) == 13U,
                   "logical view length must be preserved");
        test_check(memcmp(kq_file_view_data(view),
                          fixture->bytes + 3U,
                          13U) == 0,
                   "logical view data must begin at requested offset");
    }
    kq_file_view_close(view);
    view = NULL;
    status = kq_file_view_open(file, 0U, 0U, &view, &diagnostic);
    test_check(status == KQ_STATUS_INVALID_ARGUMENT,
               "zero-length logical views must fail closed");
    status = kq_file_view_open(file, UINT64_MAX, 2U, &view, &diagnostic);
    test_check(status == KQ_STATUS_ARITHMETIC_OVERFLOW,
               "logical view offset overflow must fail closed");
    status = kq_file_view_open(file,
                               (uint64_t)fixture->size,
                               1U,
                               &view,
                               &diagnostic);
    test_check(status == KQ_STATUS_SPAN_OUT_OF_RANGE,
               "logical view past EOF must fail closed");

    status = kq_gguf_open(file, &gguf, &diagnostic);
    if (status != KQ_STATUS_OK) {
        fprintf(stderr,
                "FAIL: valid GGUF fixture did not parse: %s (%s)\n",
                kq_status_string(status),
                diagnostic.message);
        test_failures += 1;
        kq_file_close(file);
        DeleteFileW(path);
        return;
    }

    test_check(kq_gguf_version(gguf) == 3U, "GGUF version must be 3");
    test_check(kq_gguf_metadata_count(gguf) == TEST_METADATA_COUNT,
               "metadata count must match fixture");
    test_check(kq_gguf_tensor_count(gguf) == TEST_TENSOR_COUNT,
               "tensor count must match fixture");
    test_check(kq_gguf_alignment(gguf) == TEST_ALIGNMENT,
               "alignment must match fixture");
    test_check(kq_gguf_directory_end_offset(gguf) == fixture->directory_end,
               "directory end must match fixture");
    test_check(kq_gguf_data_section_offset(gguf) ==
                   fixture->data_section_offset,
               "data section must match fixture");
    test_check(kq_gguf_packed_tensor_bytes(gguf) == TEST_PACKED_BYTES,
               "aggregate packed bytes must match fixture");
    test_check(kq_gguf_format_overhead_bytes(gguf) ==
                   (uint64_t)fixture->size - TEST_PACKED_BYTES,
               "format overhead must exclude packed tensor bytes");
    test_check(kq_gguf_directory_bytes_parsed(gguf) == fixture->directory_end,
               "parser must consume exactly the directory bytes");
    test_check(kq_gguf_directory_bytes_parsed(gguf) <
                   kq_gguf_data_section_offset(gguf),
               "directory parse must stop before tensor payload");
    test_check(kq_gguf_payload_bytes_accessed(gguf) == 0U,
               "normal parse must not access tensor payload");

    architecture = kq_gguf_architecture(gguf);
    test_check(kq_string_view_equal_cstr(&architecture, "testarch"),
               "architecture must be a bounded string view");
    test_check(kq_gguf_metadata_at(gguf, TEST_METADATA_COUNT) == NULL,
               "out-of-range metadata lookup must return NULL");
    test_check(kq_gguf_tensor_at(gguf, TEST_TENSOR_COUNT) == NULL,
               "out-of-range tensor lookup must return NULL");
    test_check(kq_gguf_find_tensor(gguf, "missing") == NULL,
               "unknown tensor lookup must return NULL");
    metadata = kq_gguf_find_metadata(gguf, "general.alignment");
    test_check(kq_gguf_metadata_u32(metadata, &metadata_u32) &&
                   metadata_u32 == TEST_ALIGNMENT,
               "typed scalar metadata lookup must preserve uint32 values");
    metadata = kq_gguf_find_metadata(gguf, "test.array0");
    test_check(kq_gguf_metadata_array_i32_at(metadata, 2U, &metadata_i32) &&
                   metadata_i32 == 3,
               "typed array metadata lookup must preserve bounded values");
    test_check(!kq_gguf_metadata_array_i32_at(metadata, 3U, &metadata_i32),
               "metadata array lookup must reject out-of-range indices");
    metadata = kq_gguf_find_metadata(gguf, "test.array1");
    test_check(kq_gguf_metadata_array_string_at(metadata,
                                                0U,
                                                &metadata_string) &&
                   kq_string_view_equal_cstr(&metadata_string, "one"),
               "string-array lookup must preserve its first bounded value");
    test_check(kq_gguf_metadata_array_string_at(metadata,
                                                1U,
                                                &metadata_string) &&
                   kq_string_view_equal_cstr(&metadata_string, "two"),
               "string-array lookup must preserve its last bounded value");
    test_check(!kq_gguf_metadata_array_string_at(metadata,
                                                 2U,
                                                 &metadata_string),
               "string-array lookup must reject out-of-range indices");
    test_check(kq_gguf_find_metadata(gguf, "missing") == NULL,
               "unknown metadata lookup must return NULL");

    tensor = kq_gguf_find_tensor(gguf, "t3");
    test_check(tensor != NULL && tensor->type_id == KQ_GGUF_TYPE_Q8_0,
               "tensor lookup must return immutable physical descriptor");
    for (index = 0U; index < TEST_TENSOR_COUNT; ++index) {
        tensor = kq_gguf_tensor_at(gguf, index);
        test_check(tensor != NULL, "every fixture tensor must be addressable");
        if (tensor != NULL) {
            type_info = kq_gguf_type_info_for(tensor->type_id);
            test_check(type_info != NULL,
                       "every fixture tensor type must have block geometry");
            test_check(tensor->data_offset >=
                           kq_gguf_data_section_offset(gguf),
                       "tensor descriptor must use an absolute data offset");
        }
    }

    kq_gguf_close(gguf);
    kq_file_close(file);
    test_check(DeleteFileW(path) != 0,
               "valid parse cleanup must release views and file handles");
}

int main(void) {
    test_fixture valid;
    test_fixture malformed;

    if (!build_valid_fixture(&valid)) {
        fprintf(stderr, "could not construct deterministic GGUF fixture\n");
        return 1;
    }
    test_valid_fixture(&valid);

    malformed = valid;
    malformed.bytes[0] = (unsigned char)'B';
    test_failure_case("bad magic",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_BAD_MAGIC);

    malformed = valid;
    patch_u32(&malformed, 4U, 2U);
    test_failure_case("bad version",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_UNSUPPORTED_VERSION);

    test_failure_case("truncated fixed header",
                      &valid,
                      20U,
                      KQ_STATUS_TRUNCATED);

    malformed = valid;
    patch_u64(&malformed,
              malformed.first_key_length_offset,
              UINT64_C(128) * 1024U * 1024U + 1U);
    test_failure_case("absurd string length",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_LIMIT_EXCEEDED);

    malformed = valid;
    malformed.bytes[malformed.first_key_length_offset + 8U] = 0xc0U;
    test_failure_case("invalid UTF-8 string",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_MALFORMED_METADATA);

    malformed = valid;
    patch_u32(&malformed, malformed.array_element_type_offset, 9U);
    test_failure_case("nested metadata array",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_UNSUPPORTED_METADATA);

    malformed = valid;
    patch_u64(&malformed, malformed.array_length_offset, UINT64_C(2000001));
    test_failure_case("absurd metadata array",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_LIMIT_EXCEEDED);

    malformed = valid;
    patch_u64(&malformed, 8U, UINT64_MAX);
    test_failure_case("directory allocation overflow",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_ARITHMETIC_OVERFLOW);

    malformed = valid;
    patch_u64(&malformed, 8U, UINT64_C(8193));
    test_failure_case("tensor count defensive limit",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_LIMIT_EXCEEDED);

    malformed = valid;
    patch_u32(&malformed, malformed.tensor_rank_offset[0], 0U);
    test_failure_case("zero tensor rank",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_MALFORMED_TENSOR);

    malformed = valid;
    patch_u32(&malformed, malformed.tensor_rank_offset[0], 9U);
    test_failure_case("excessive tensor rank",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_MALFORMED_TENSOR);

    malformed = valid;
    patch_u64(&malformed, malformed.tensor_dimension_offset[0], 0U);
    test_failure_case("zero tensor dimension",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_MALFORMED_TENSOR);

    malformed = valid;
    patch_u64(&malformed,
              malformed.tensor_dimension_offset[0],
              UINT64_MAX);
    test_failure_case("packed tensor byte overflow",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_ARITHMETIC_OVERFLOW);

    malformed = valid;
    patch_u32(&malformed, malformed.tensor_type_offset[0], 1U);
    test_failure_case("unsupported tensor type",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_UNSUPPORTED_TENSOR_TYPE);

    malformed = valid;
    memcpy(malformed.bytes + malformed.array_key_offset[1],
           malformed.bytes + malformed.array_key_offset[0],
           strlen("test.array0"));
    test_failure_case("duplicate metadata name",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_DUPLICATE_METADATA);

    malformed = valid;
    memcpy(malformed.bytes + malformed.tensor_name_offset[1], "t0", 2U);
    test_failure_case("duplicate tensor name",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_DUPLICATE_TENSOR);

    malformed = valid;
    patch_u32(&malformed, malformed.alignment_value_offset, 3U);
    test_failure_case("invalid GGUF alignment",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_INVALID_ALIGNMENT);

    malformed = valid;
    patch_u64(&malformed, malformed.tensor_offset_offset[1], 1U);
    test_failure_case("unaligned tensor offset",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_INVALID_ALIGNMENT);

    malformed = valid;
    patch_u64(&malformed, malformed.tensor_offset_offset[0], TEST_ALIGNMENT);
    test_failure_case("missing zero data-section origin",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_INCONSISTENT_DATA_SECTION);

    test_failure_case("tensor span past EOF",
                      &valid,
                      valid.size - 1U,
                      KQ_STATUS_SPAN_OUT_OF_RANGE);

    malformed = valid;
    patch_u64(&malformed, malformed.tensor_dimension_offset[3], 31U);
    test_failure_case("invalid quantized block geometry",
                      &malformed,
                      malformed.size,
                      KQ_STATUS_MALFORMED_TENSOR);

    if (test_failures != 0) {
        fprintf(stderr, "%d GGUF test assertion(s) failed\n", test_failures);
        return 1;
    }
    printf("GGUF synthetic/malformed tests: PASS\n");
    return 0;
}
