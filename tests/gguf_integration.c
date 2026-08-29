#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_status.h"

#define KQ_TEST_SKIP 77
#define EXPECTED_FILE_SIZE UINT64_C(111334654400)
#define EXPECTED_METADATA_COUNT UINT64_C(67)
#define EXPECTED_TENSOR_COUNT UINT64_C(1224)
#define EXPECTED_ALIGNMENT UINT64_C(32)
#define EXPECTED_DIRECTORY_END UINT64_C(11024307)
#define EXPECTED_DATA_SECTION UINT64_C(11024320)
#define EXPECTED_PACKED_BYTES UINT64_C(111323630080)
#define EXPECTED_OVERHEAD_BYTES UINT64_C(11024320)

typedef struct expected_type_count {
    uint32_t type_id;
    uint64_t expected_count;
} expected_type_count;

static int file_identity_equal(const WIN32_FILE_ATTRIBUTE_DATA *left,
                               const WIN32_FILE_ATTRIBUTE_DATA *right) {
    return left->nFileSizeHigh == right->nFileSizeHigh &&
           left->nFileSizeLow == right->nFileSizeLow &&
           left->ftLastWriteTime.dwHighDateTime ==
               right->ftLastWriteTime.dwHighDateTime &&
           left->ftLastWriteTime.dwLowDateTime ==
               right->ftLastWriteTime.dwLowDateTime;
}

static void print_error(kq_status status, const kq_diagnostic *diagnostic) {
    fprintf(stderr, "real GGUF integration failed: %s", kq_status_string(status));
    if (diagnostic != NULL && diagnostic->message[0] != '\0') {
        fprintf(stderr, ": %s", diagnostic->message);
    }
    fputc('\n', stderr);
}

int main(void) {
    static const expected_type_count expected_types[] = {
        {KQ_GGUF_TYPE_BF16, 24U},
        {KQ_GGUF_TYPE_F32, 557U},
        {KQ_GGUF_TYPE_IQ4_NL, 1U},
        {KQ_GGUF_TYPE_Q4_K, 94U},
        {KQ_GGUF_TYPE_Q5_1, 43U},
        {KQ_GGUF_TYPE_Q5_K, 2U},
        {KQ_GGUF_TYPE_Q8_0, 503U}
    };
    wchar_t *path = NULL;
    DWORD required;
    DWORD copied;
    DWORD error_code;
    WIN32_FILE_ATTRIBUTE_DATA before_attributes;
    WIN32_FILE_ATTRIBUTE_DATA after_attributes;
    kq_diagnostic diagnostic;
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    const kq_gguf_tensor *tensor;
    kq_string_view architecture;
    uint64_t observed_counts[sizeof(expected_types) /
                             sizeof(expected_types[0])] = {0};
    uint64_t tensor_index;
    size_t type_index;
    kq_status status;
    int have_before_attributes = 0;
    int result = 1;

    SetLastError(ERROR_SUCCESS);
    required = GetEnvironmentVariableW(L"KQ_GGUF_PATH", NULL, 0U);
    if (required == 0U) {
        error_code = GetLastError();
        if (error_code == ERROR_ENVVAR_NOT_FOUND || error_code == ERROR_SUCCESS) {
            printf("KQ_GGUF_PATH is unavailable; real GGUF integration skipped\n");
            return KQ_TEST_SKIP;
        }
        fprintf(stderr,
                "GetEnvironmentVariableW size query failed (Win32 error %lu)\n",
                (unsigned long)error_code);
        return 1;
    }
    if ((uint64_t)required > (uint64_t)(SIZE_MAX / sizeof(*path))) {
        fprintf(stderr, "KQ_GGUF_PATH length overflows allocation size\n");
        return 1;
    }
    path = (wchar_t *)calloc((size_t)required, sizeof(*path));
    if (path == NULL) {
        fprintf(stderr, "could not allocate KQ_GGUF_PATH buffer\n");
        return 1;
    }
    copied = GetEnvironmentVariableW(L"KQ_GGUF_PATH", path, required);
    if (copied == 0U || copied >= required) {
        fprintf(stderr, "KQ_GGUF_PATH changed or could not be read\n");
        goto cleanup;
    }
    if (!GetFileAttributesExW(path,
                              GetFileExInfoStandard,
                              &before_attributes)) {
        fprintf(stderr, "KQ_GGUF_PATH does not identify a readable file\n");
        goto cleanup;
    }
    have_before_attributes = 1;

    status = kq_file_open_readonly(path, &file, &diagnostic);
    if (status != KQ_STATUS_OK) {
        print_error(status, &diagnostic);
        goto cleanup;
    }
    if (kq_file_size(file) != EXPECTED_FILE_SIZE) {
        fprintf(stderr,
                "real GGUF size mismatch: expected %llu, received %llu\n",
                (unsigned long long)EXPECTED_FILE_SIZE,
                (unsigned long long)kq_file_size(file));
        goto cleanup;
    }
    status = kq_gguf_open(file, &gguf, &diagnostic);
    if (status != KQ_STATUS_OK) {
        print_error(status, &diagnostic);
        goto cleanup;
    }

    architecture = kq_gguf_architecture(gguf);
    if (kq_gguf_version(gguf) != 3U ||
        !kq_string_view_equal_cstr(&architecture, "qwen4exp") ||
        kq_gguf_metadata_count(gguf) != EXPECTED_METADATA_COUNT ||
        kq_gguf_tensor_count(gguf) != EXPECTED_TENSOR_COUNT ||
        kq_gguf_alignment(gguf) != EXPECTED_ALIGNMENT ||
        kq_gguf_directory_end_offset(gguf) != EXPECTED_DIRECTORY_END ||
        kq_gguf_data_section_offset(gguf) != EXPECTED_DATA_SECTION ||
        kq_gguf_packed_tensor_bytes(gguf) != EXPECTED_PACKED_BYTES ||
        kq_gguf_format_overhead_bytes(gguf) != EXPECTED_OVERHEAD_BYTES ||
        kq_gguf_payload_bytes_accessed(gguf) != 0U ||
        kq_gguf_directory_bytes_parsed(gguf) >=
            kq_gguf_data_section_offset(gguf)) {
        fprintf(stderr, "real GGUF structural oracle mismatch\n");
        goto cleanup;
    }

    for (tensor_index = 0U;
         tensor_index < kq_gguf_tensor_count(gguf);
         ++tensor_index) {
        tensor = kq_gguf_tensor_at(gguf, tensor_index);
        if (tensor == NULL) {
            fprintf(stderr, "real GGUF tensor descriptor lookup failed\n");
            goto cleanup;
        }
        for (type_index = 0U;
             type_index < sizeof(expected_types) / sizeof(expected_types[0]);
             ++type_index) {
            if (tensor->type_id == expected_types[type_index].type_id) {
                observed_counts[type_index] += 1U;
                break;
            }
        }
        if (type_index == sizeof(expected_types) / sizeof(expected_types[0])) {
            fprintf(stderr, "real GGUF contains an unexpected tensor type\n");
            goto cleanup;
        }
    }
    for (type_index = 0U;
         type_index < sizeof(expected_types) / sizeof(expected_types[0]);
         ++type_index) {
        if (observed_counts[type_index] !=
            expected_types[type_index].expected_count) {
            fprintf(stderr,
                    "real GGUF type %u count mismatch: expected %llu, received %llu\n",
                    (unsigned int)expected_types[type_index].type_id,
                    (unsigned long long)expected_types[type_index].expected_count,
                    (unsigned long long)observed_counts[type_index]);
            goto cleanup;
        }
    }

    printf("real GGUF oracle: PASS, payload_bytes_accessed=0\n");
    result = 0;

cleanup:
    kq_gguf_close(gguf);
    kq_file_close(file);
    if (path != NULL && have_before_attributes &&
        GetFileAttributesExW(path,
                             GetFileExInfoStandard,
                             &after_attributes)) {
        if (!file_identity_equal(&before_attributes, &after_attributes)) {
            fprintf(stderr, "real GGUF size or last-write time changed\n");
            result = 1;
        }
    } else if (path != NULL && have_before_attributes) {
        fprintf(stderr, "real GGUF disappeared during read-only inspection\n");
        result = 1;
    }
    free(path);
    return result;
}
