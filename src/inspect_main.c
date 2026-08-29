#include <stdio.h>
#include <wchar.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_status.h"
#include "kq_tensor_view.h"

typedef struct kq_type_count {
    uint32_t type_id;
    uint64_t count;
} kq_type_count;

static void kq_print_usage(void) {
    fprintf(stderr,
            "usage: kq-inspect <model.gguf>\n"
            "       kq-inspect --semantic-summary <model.gguf>\n"
            "       kq-inspect --semantic <stable-id> <model.gguf>\n"
            "       kq-inspect --semantic-dump <model.gguf>\n"
            "       kq-inspect --view-geometry-dump <model.gguf>\n");
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

static void kq_print_semantic_summary(const kq_model *model,
                                      const kq_gguf *gguf) {
    kq_binding_relation relation;
    kq_placement_hint placement;

    printf("model=qwen3.8-flash-next\n");
    printf("hidden_size=%u\n", (unsigned int)kq_model_hidden_size(model));
    printf("vocabulary_size=%u\n",
           (unsigned int)kq_model_vocabulary_size(model));
    printf("context_length=%u\n",
           (unsigned int)kq_model_context_length(model));
    printf("semantic_entries=%llu\n",
           (unsigned long long)kq_model_semantic_tensor_count(model));
    printf("metadata_derived=%llu\n",
           (unsigned long long)kq_model_metadata_derived_count(model));
    printf("physical_tensors=%llu\n",
           (unsigned long long)kq_model_physical_tensor_count(model));
    printf("physical_coverage=%llu\n",
           (unsigned long long)kq_model_physical_coverage_count(model));
    printf("unknown_physical=%llu\n",
           (unsigned long long)kq_model_unknown_physical_count(model));
    printf("unbound_required=%llu\n",
           (unsigned long long)kq_model_unbound_required_count(model));
    printf("layers=%u\n", (unsigned int)kq_model_layer_count(model));
    printf("layers.GDN=%u\n",
           (unsigned int)kq_model_gdn_layer_count(model));
    printf("layers.QSA=%u\n",
           (unsigned int)kq_model_qsa_layer_count(model));
    printf("experts=%u\n", (unsigned int)kq_model_expert_count(model));
    printf("expert_top_k=%u\n", (unsigned int)kq_model_expert_top_k(model));
    for (relation = KQ_BINDING_DIRECT_ONE_TO_ONE;
         relation < KQ_BINDING_RELATION_COUNT;
         relation = (kq_binding_relation)(relation + 1)) {
        printf("relation.%s=%llu\n",
               kq_binding_relation_name(relation),
               (unsigned long long)kq_model_relation_count(model, relation));
    }
    for (placement = KQ_PLACEMENT_ALWAYS_NEEDED_CANDIDATE;
         placement < KQ_PLACEMENT_HINT_COUNT;
         placement = (kq_placement_hint)(placement + 1)) {
        printf("placement.%s=%llu\n",
               kq_placement_hint_name(placement),
               (unsigned long long)kq_model_placement_count(model, placement));
    }
    printf("payload_bytes_accessed=%llu\n",
           (unsigned long long)kq_gguf_payload_bytes_accessed(gguf));
}

static void kq_print_binding_names(const kq_semantic_tensor *semantic) {
    uint32_t index;
    for (index = 0U; index < semantic->binding_count; ++index) {
        if (index != 0U) {
            fputc(';', stdout);
        }
        if (semantic->bindings[index].physical != NULL) {
            kq_print_bounded_text(semantic->bindings[index].physical->name);
        }
    }
}

static void kq_print_metadata_key(const kq_semantic_tensor *semantic) {
    if (semantic->binding_count == 1U &&
        semantic->bindings[0].metadata != NULL) {
        kq_print_bounded_text(semantic->bindings[0].metadata->key);
    }
}

static void kq_print_semantic(const kq_semantic_tensor *semantic) {
    uint32_t index;
    printf("semantic_id=%s\n", semantic->semantic_id);
    printf("canonical_name=%s\n", semantic->canonical_name);
    printf("component=%s\n", kq_semantic_component_name(semantic->component));
    printf("role=%s\n", kq_semantic_role_name(semantic->role));
    printf("relation=%s\n", kq_binding_relation_name(semantic->relation));
    printf("runtime_scope=%s\n",
           kq_runtime_scope_name(semantic->runtime_scope));
    printf("placement=%s\n",
           kq_placement_hint_name(semantic->placement_hint));
    if (semantic->layer_id == KQ_MODEL_NO_LAYER) {
        printf("layer_id=NONE\n");
        printf("layer_type=NONE\n");
    } else {
        printf("layer_id=%u\n", (unsigned int)semantic->layer_id);
        printf("layer_type=%s\n",
               kq_model_layer_type_name(semantic->layer_type));
    }
    printf("canonical_shape=");
    for (index = 0U; index < semantic->canonical_rank; ++index) {
        if (index != 0U) {
            fputc('x', stdout);
        }
        printf("%llu",
               (unsigned long long)semantic->canonical_dimensions[index]);
    }
    fputc('\n', stdout);
    printf("binding_count=%u\n", (unsigned int)semantic->binding_count);
    fputs("physical_names=", stdout);
    kq_print_binding_names(semantic);
    fputc('\n', stdout);
    fputs("metadata_key=", stdout);
    kq_print_metadata_key(semantic);
    fputc('\n', stdout);
}

static void kq_print_semantic_dump(const kq_model *model) {
    const kq_semantic_tensor *semantic;
    uint64_t index;

    fputs("semantic_id\tcanonical_name\tcomponent\tlayer_id\trole\trelation\t"
          "placement\tphysical_names\tmetadata_key\n",
          stdout);
    for (index = 0U; index < kq_model_semantic_tensor_count(model); ++index) {
        semantic = kq_model_semantic_tensor_at(model, index);
        printf("%s\t%s\t%s\t",
               semantic->semantic_id,
               semantic->canonical_name,
               kq_semantic_component_name(semantic->component));
        if (semantic->layer_id == KQ_MODEL_NO_LAYER) {
            fputs("\t", stdout);
        } else {
            printf("%u\t", (unsigned int)semantic->layer_id);
        }
        printf("%s\t%s\t%s\t",
               kq_semantic_role_name(semantic->role),
               kq_binding_relation_name(semantic->relation),
               kq_placement_hint_name(semantic->placement_hint));
        kq_print_binding_names(semantic);
        fputc('\t', stdout);
        kq_print_metadata_key(semantic);
        fputc('\n', stdout);
    }
}

static int kq_print_view_geometry_dump(const kq_gguf *gguf) {
    const kq_gguf_tensor *tensor;
    const kq_gguf_type_info *type_info;
    kq_diagnostic diagnostic;
    uint64_t derived_packed_bytes;
    uint64_t tensor_index;
    uint32_t dimension;
    kq_status status;

    fputs("physical_name\trank\tdimensions\ttype_id\ttype_name\t"
          "block_elements\tbytes_per_block\telement_count\t"
          "relative_offset\tdata_offset\tpacked_bytes\n",
          stdout);
    for (tensor_index = 0U;
         tensor_index < kq_gguf_tensor_count(gguf);
         ++tensor_index) {
        tensor = kq_gguf_tensor_at(gguf, tensor_index);
        if (tensor == NULL) {
            fputs("kq-inspect: physical tensor lookup failed\n", stderr);
            return 0;
        }
        type_info = kq_gguf_type_info_for(tensor->type_id);
        status = kq_quant_packed_size(tensor->type_id,
                                      tensor->element_count,
                                      &derived_packed_bytes,
                                      &diagnostic);
        if (type_info == NULL || status != KQ_STATUS_OK ||
            tensor->block_elements != type_info->block_elements ||
            tensor->bytes_per_block != type_info->bytes_per_block ||
            tensor->packed_bytes != derived_packed_bytes) {
            kq_print_diagnostic(status == KQ_STATUS_OK
                                    ? KQ_STATUS_INVALID_QUANTIZED_GEOMETRY
                                    : status,
                                &diagnostic);
            return 0;
        }
        kq_print_bounded_text(tensor->name);
        printf("\t%u\t", (unsigned int)tensor->rank);
        for (dimension = 0U; dimension < tensor->rank; ++dimension) {
            if (dimension != 0U) {
                fputc('x', stdout);
            }
            printf("%llu",
                   (unsigned long long)tensor->dimensions[dimension]);
        }
        printf("\t%u\t%s\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\n",
               (unsigned int)tensor->type_id,
               type_info->name,
               (unsigned long long)tensor->block_elements,
               (unsigned long long)tensor->bytes_per_block,
               (unsigned long long)tensor->element_count,
               (unsigned long long)tensor->relative_offset,
               (unsigned long long)tensor->data_offset,
               (unsigned long long)tensor->packed_bytes);
    }
    return 1;
}

static int kq_copy_semantic_id(const wchar_t *wide,
                               char id[KQ_SEMANTIC_ID_CAPACITY]) {
    size_t index;
    size_t length = wcslen(wide);
    if (length == 0U || length >= KQ_SEMANTIC_ID_CAPACITY) {
        return 0;
    }
    for (index = 0U; index < length; ++index) {
        if ((unsigned int)wide[index] > 0x7fU) {
            return 0;
        }
        id[index] = (char)wide[index];
    }
    id[length] = '\0';
    return 1;
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
    kq_model *model = NULL;
    kq_status status;
    kq_string_view architecture;
    const kq_gguf_tensor *tensor;
    const kq_gguf_type_info *type_info;
    uint64_t tensor_index;
    size_t type_index;
    int result = 1;
    int semantic_summary = 0;
    int semantic_dump = 0;
    int semantic_query = 0;
    int view_geometry_dump = 0;
    wchar_t *path_argument = NULL;
    char semantic_id[KQ_SEMANTIC_ID_CAPACITY] = {0};

    if (argc == 2 && wcscmp(argv[1], L"--help") == 0) {
        kq_print_usage();
        return 0;
    }
    if (argc == 2) {
        path_argument = argv[1];
    } else if (argc == 3 && wcscmp(argv[1], L"--semantic-summary") == 0) {
        semantic_summary = 1;
        path_argument = argv[2];
    } else if (argc == 3 && wcscmp(argv[1], L"--semantic-dump") == 0) {
        semantic_dump = 1;
        path_argument = argv[2];
    } else if (argc == 3 &&
               wcscmp(argv[1], L"--view-geometry-dump") == 0) {
        view_geometry_dump = 1;
        path_argument = argv[2];
    } else if (argc == 4 && wcscmp(argv[1], L"--semantic") == 0 &&
               kq_copy_semantic_id(argv[2], semantic_id)) {
        semantic_query = 1;
        path_argument = argv[3];
    } else {
        kq_print_usage();
        return 2;
    }

    status = kq_file_open_readonly(path_argument, &file, &diagnostic);
    if (status != KQ_STATUS_OK) {
        kq_print_diagnostic(status, &diagnostic);
        goto cleanup;
    }

    status = kq_gguf_open(file, &gguf, &diagnostic);
    if (status != KQ_STATUS_OK) {
        kq_print_diagnostic(status, &diagnostic);
        goto cleanup;
    }

    if (semantic_summary || semantic_dump || semantic_query) {
        const kq_semantic_tensor *semantic = NULL;
        status = kq_model_open_from_gguf(gguf, &model, &diagnostic);
        if (status != KQ_STATUS_OK) {
            kq_print_diagnostic(status, &diagnostic);
            goto cleanup;
        }
        if (semantic_summary) {
            kq_print_semantic_summary(model, gguf);
        } else if (semantic_dump) {
            kq_print_semantic_dump(model);
        } else {
            semantic = kq_model_find_semantic_tensor(model, semantic_id);
            if (semantic == NULL) {
                fprintf(stderr,
                        "kq-inspect: unknown semantic identifier %s\n",
                        semantic_id);
                goto cleanup;
            }
            kq_print_semantic(semantic);
        }
        result = 0;
        goto cleanup;
    }

    if (view_geometry_dump) {
        if (!kq_print_view_geometry_dump(gguf)) {
            goto cleanup;
        }
        result = 0;
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
    kq_model_close(model);
    kq_gguf_close(gguf);
    kq_file_close(file);
    return result;
}
