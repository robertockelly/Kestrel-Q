#include <stdio.h>
#include <wchar.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_status.h"

typedef struct kq_type_count {
    uint32_t type_id;
    uint64_t count;
} kq_type_count;

static void kq_print_usage(void) {
    fprintf(stderr, "usage: kq-inspect <model.gguf>\n");
}

static void kq_print_bounded_text(kq_string_view value) {
    uint64_t index;
    unsigned int byte_value;

    for (index = 0U; index < value.length; ++index) {
        byte_value = value.data[(size_t)index];
        if (byte_value >= 0x20U && byte_value <= 0x7eU &&
            byte_value != (unsigned int)'\\') {
            fputc((int)byte_value, stdout);
        } else {
            fprintf(stdout, "\\x%02x", byte_value);
        }
    }
}

static void kq_print_diagnostic(kq_status status,
                                const kq_diagnostic *diagnostic) {
    fprintf(stderr, "kq-inspect: %s", kq_status_string(status));
    if (diagnostic != NULL && diagnostic->message[0] != '\0') {
        fprintf(stderr, ": %s", diagnostic->message);
    }
    fputc('\n', stderr);
}

int wmain(int argc, wchar_t **argv) {
    static const uint32_t type_ids[] = {
        KQ_GGUF_TYPE_BF16,
        KQ_GGUF_TYPE_F32,
        KQ_GGUF_TYPE_IQ4_NL,
        KQ_GGUF_TYPE_Q4_K,
        KQ_GGUF_TYPE_Q5_1,
        KQ_GGUF_TYPE_Q5_K,
        KQ_GGUF_TYPE_Q8_0
    };
    kq_type_count counts[sizeof(type_ids) / sizeof(type_ids[0])] = {0};
    kq_diagnostic diagnostic;
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_status status;
    kq_string_view architecture;
    const kq_gguf_tensor *tensor;
    const kq_gguf_type_info *type_info;
    uint64_t tensor_index;
    size_t type_index;
    int result = 1;

    if (argc == 2 && wcscmp(argv[1], L"--help") == 0) {
        kq_print_usage();
        return 0;
    }
    if (argc != 2) {
        kq_print_usage();
        return 2;
    }

    status = kq_file_open_readonly(argv[1], &file, &diagnostic);
    if (status != KQ_STATUS_OK) {
        kq_print_diagnostic(status, &diagnostic);
        goto cleanup;
    }
    status = kq_gguf_open(file, &gguf, &diagnostic);
    if (status != KQ_STATUS_OK) {
        kq_print_diagnostic(status, &diagnostic);
        goto cleanup;
    }

    for (type_index = 0U;
         type_index < sizeof(type_ids) / sizeof(type_ids[0]);
         ++type_index) {
        counts[type_index].type_id = type_ids[type_index];
    }
    for (tensor_index = 0U;
         tensor_index < kq_gguf_tensor_count(gguf);
         ++tensor_index) {
        tensor = kq_gguf_tensor_at(gguf, tensor_index);
        if (tensor == NULL) {
            fprintf(stderr, "kq-inspect: internal tensor lookup failure\n");
            goto cleanup;
        }
        for (type_index = 0U;
             type_index < sizeof(type_ids) / sizeof(type_ids[0]);
             ++type_index) {
            if (counts[type_index].type_id == tensor->type_id) {
                counts[type_index].count += 1U;
                break;
            }
        }
        if (type_index == sizeof(type_ids) / sizeof(type_ids[0])) {
            fprintf(stderr,
                    "kq-inspect: parser returned unsupported tensor type\n");
            goto cleanup;
        }
    }

    architecture = kq_gguf_architecture(gguf);
    printf("file_size_bytes=%llu\n",
           (unsigned long long)kq_file_size(file));
    printf("gguf_version=%u\n", (unsigned int)kq_gguf_version(gguf));
    fputs("architecture=", stdout);
    kq_print_bounded_text(architecture);
    fputc('\n', stdout);
    printf("metadata_count=%llu\n",
           (unsigned long long)kq_gguf_metadata_count(gguf));
    printf("tensor_count=%llu\n",
           (unsigned long long)kq_gguf_tensor_count(gguf));
    printf("alignment_bytes=%llu\n",
           (unsigned long long)kq_gguf_alignment(gguf));
    printf("directory_end_offset=%llu\n",
           (unsigned long long)kq_gguf_directory_end_offset(gguf));
    printf("data_section_offset=%llu\n",
           (unsigned long long)kq_gguf_data_section_offset(gguf));
    printf("packed_tensor_bytes=%llu\n",
           (unsigned long long)kq_gguf_packed_tensor_bytes(gguf));
    printf("format_overhead_bytes=%llu\n",
           (unsigned long long)kq_gguf_format_overhead_bytes(gguf));
    printf("directory_bytes_parsed=%llu\n",
           (unsigned long long)kq_gguf_directory_bytes_parsed(gguf));
    printf("payload_bytes_accessed=%llu\n",
           (unsigned long long)kq_gguf_payload_bytes_accessed(gguf));
    for (type_index = 0U;
         type_index < sizeof(type_ids) / sizeof(type_ids[0]);
         ++type_index) {
        type_info = kq_gguf_type_info_for(counts[type_index].type_id);
        if (type_info == NULL) {
            fprintf(stderr, "kq-inspect: internal type lookup failure\n");
            goto cleanup;
        }
        printf("type.%s=%llu\n",
               type_info->name,
               (unsigned long long)counts[type_index].count);
    }
    result = 0;

cleanup:
    kq_gguf_close(gguf);
    kq_file_close(file);
    return result;
}
