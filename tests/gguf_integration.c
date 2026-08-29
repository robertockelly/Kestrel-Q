#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_status.h"
#include "kq_tensor_view.h"

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

static int validate_opened_view(const kq_tensor_view *view,
                                const kq_tensor_binding *binding,
                                kq_tensor_view_kind expected_kind) {
    const kq_tensor_view_info *info = kq_tensor_view_get_info(view);

    return info != NULL && binding != NULL && binding->physical != NULL &&
           info->kind == expected_kind &&
           info->type_id == binding->physical->type_id &&
           info->file_byte_offset >= binding->physical->data_offset &&
           info->mapped_logical_offset == info->file_byte_offset &&
           info->mapped_logical_length > 0U &&
           info->mapped_logical_length <= binding->physical->packed_bytes;
}

static int open_binding_sample(const kq_gguf *gguf,
                               const kq_semantic_tensor *semantic,
                               uint32_t binding_index,
                               kq_tensor_view_kind expected_kind) {
    kq_diagnostic diagnostic;
    kq_tensor_view *view = NULL;
    kq_status status = kq_tensor_view_open_binding(
        gguf, semantic, binding_index, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    int valid = status == KQ_STATUS_OK &&
                validate_opened_view(view,
                                     &semantic->bindings[binding_index],
                                     expected_kind);
    if (!valid) {
        print_error(status, &diagnostic);
    }
    kq_tensor_view_close(view);
    return valid;
}

static int open_expert_sample(const kq_gguf *gguf,
                              const kq_semantic_tensor *semantic,
                              uint32_t binding_index,
                              uint32_t expert_id) {
    const kq_tensor_binding *binding = &semantic->bindings[binding_index];
    kq_diagnostic diagnostic;
    kq_tensor_view *view = NULL;
    const kq_tensor_view_info *info;
    uint64_t expected_bytes;
    uint64_t expected_offset;
    kq_status status;
    int valid = 0;

    if (binding->physical == NULL || semantic->expert_count == 0U ||
        binding->physical->packed_bytes % semantic->expert_count != 0U) {
        return 0;
    }
    expected_bytes = binding->physical->packed_bytes / semantic->expert_count;
    expected_offset = expected_bytes * expert_id;
    status = kq_tensor_view_open_expert_member(
        gguf, semantic, binding_index, expert_id,
        KQ_TENSOR_VIEW_PHYSICAL_LAYOUT, &view, &diagnostic);
    info = kq_tensor_view_get_info(view);
    valid = status == KQ_STATUS_OK &&
            validate_opened_view(view, binding,
                                 KQ_TENSOR_VIEW_EXPERT_MEMBER) &&
            info->expert_id == expert_id &&
            info->tensor_byte_offset == expected_offset &&
            info->mapped_logical_length == expected_bytes &&
            info->leading_elements == 0U && info->trailing_elements == 0U;
    if (!valid) {
        print_error(status, &diagnostic);
    }
    kq_tensor_view_close(view);
    return valid;
}

static int open_ple_sample(const kq_gguf *gguf,
                           const kq_semantic_tensor *semantic) {
    const kq_tensor_binding *binding = &semantic->bindings[0];
    kq_diagnostic diagnostic;
    kq_tensor_view *view = NULL;
    const kq_tensor_view_info *info;
    uint64_t expected_bytes;
    uint64_t expected_offset;
    kq_status status;
    int valid = 0;

    if (binding->physical == NULL || binding->fused_member_count == 0U ||
        binding->physical->packed_bytes % binding->fused_member_count != 0U) {
        return 0;
    }
    expected_bytes = binding->physical->packed_bytes /
                     binding->fused_member_count;
    expected_offset = expected_bytes * binding->fused_member_index;
    status = kq_tensor_view_open_ple_member(
        gguf, semantic, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    info = kq_tensor_view_get_info(view);
    valid = status == KQ_STATUS_OK &&
            validate_opened_view(view, binding,
                                 KQ_TENSOR_VIEW_PLE_FUSED_MEMBER) &&
            info->fused_member_index == binding->fused_member_index &&
            info->tensor_byte_offset == expected_offset &&
            info->mapped_logical_length == expected_bytes &&
            info->mapped_logical_length < binding->physical->packed_bytes &&
            info->leading_elements == 0U && info->trailing_elements == 0U;
    if (!valid) {
        print_error(status, &diagnostic);
    }
    kq_tensor_view_close(view);
    return valid;
}

static int validate_real_view_samples(const kq_gguf *gguf,
                                      const kq_model *model) {
    static const uint32_t type_ids[] = {
        KQ_GGUF_TYPE_BF16, KQ_GGUF_TYPE_F32, KQ_GGUF_TYPE_IQ4_NL,
        KQ_GGUF_TYPE_Q4_K, KQ_GGUF_TYPE_Q5_1, KQ_GGUF_TYPE_Q5_K,
        KQ_GGUF_TYPE_Q8_0
    };
    const kq_semantic_tensor *candidates[
        sizeof(type_ids) / sizeof(type_ids[0])] = {0};
    uint32_t candidate_bindings[
        sizeof(type_ids) / sizeof(type_ids[0])] = {0};
    uint64_t candidate_sizes[
        sizeof(type_ids) / sizeof(type_ids[0])];
    const kq_semantic_tensor *semantic;
    const kq_tensor_binding *binding;
    kq_diagnostic diagnostic;
    kq_tensor_view *view = NULL;
    kq_status status;
    uint64_t semantic_index;
    uint32_t binding_index;
    size_t type_index;

    for (type_index = 0U;
         type_index < sizeof(type_ids) / sizeof(type_ids[0]);
         ++type_index) {
        candidate_sizes[type_index] = UINT64_MAX;
    }
    for (semantic_index = 0U;
         semantic_index < kq_model_semantic_tensor_count(model);
         ++semantic_index) {
        semantic = kq_model_semantic_tensor_at(model, semantic_index);
        for (binding_index = 0U;
             semantic != NULL && binding_index < semantic->binding_count;
             ++binding_index) {
            binding = &semantic->bindings[binding_index];
            if (binding->physical == NULL) {
                continue;
            }
            for (type_index = 0U;
                 type_index < sizeof(type_ids) / sizeof(type_ids[0]);
                 ++type_index) {
                if (binding->physical->type_id == type_ids[type_index] &&
                    binding->physical->packed_bytes <
                        candidate_sizes[type_index]) {
                    candidates[type_index] = semantic;
                    candidate_bindings[type_index] = binding_index;
                    candidate_sizes[type_index] =
                        binding->physical->packed_bytes;
                }
            }
        }
    }
    for (type_index = 0U;
         type_index < sizeof(type_ids) / sizeof(type_ids[0]);
         ++type_index) {
        semantic = candidates[type_index];
        if (semantic == NULL) {
            return 0;
        }
        binding_index = candidate_bindings[type_index];
        if (semantic->component == KQ_COMPONENT_PLE_TABLE) {
            if (!open_ple_sample(gguf, semantic)) {
                return 0;
            }
        } else if (semantic->component ==
                   KQ_COMPONENT_ROUTED_EXPERT_STACK) {
            if (!open_expert_sample(gguf, semantic, binding_index, 0U)) {
                return 0;
            }
        } else if (!open_binding_sample(
                       gguf, semantic, binding_index,
                       semantic->relation ==
                               KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL
                           ? KQ_TENSOR_VIEW_SPLIT_PART
                           : KQ_TENSOR_VIEW_WHOLE_PHYSICAL)) {
            return 0;
        }
    }

    semantic = kq_model_find_semantic_tensor(model,
                                             "text.token_embedding");
    if (semantic == NULL ||
        !open_binding_sample(gguf, semantic, 0U,
                             KQ_TENSOR_VIEW_WHOLE_PHYSICAL)) {
        return 0;
    }
    semantic = kq_model_find_semantic_tensor(
        model, "layer.03.qsa.indexer.qk");
    if (semantic == NULL ||
        !open_binding_sample(gguf, semantic, 0U,
                             KQ_TENSOR_VIEW_SPLIT_PART) ||
        !open_binding_sample(gguf, semantic, 1U,
                             KQ_TENSOR_VIEW_SPLIT_PART)) {
        return 0;
    }
    semantic = kq_model_find_semantic_tensor(
        model, "layer.02.moe.routed.gate_up");
    if (semantic == NULL ||
        !open_expert_sample(gguf, semantic, 0U, 0U) ||
        !open_expert_sample(gguf, semantic, 1U, 511U)) {
        return 0;
    }
    semantic = kq_model_find_semantic_tensor(
        model, "layer.02.moe.routed.down");
    if (semantic == NULL ||
        !open_expert_sample(gguf, semantic, 0U, 17U)) {
        return 0;
    }
    semantic = kq_model_find_semantic_tensor(model,
                                             "layer.01.ple.table.000");
    if (semantic == NULL || !open_ple_sample(gguf, semantic)) {
        return 0;
    }
    semantic = kq_model_find_semantic_tensor(model,
                                             "layer.01.ple.table.064");
    if (semantic == NULL || !open_ple_sample(gguf, semantic)) {
        return 0;
    }
    semantic = kq_model_find_semantic_tensor(model,
                                             "layer.01.ple.table.127");
    if (semantic == NULL || !open_ple_sample(gguf, semantic)) {
        return 0;
    }
    semantic = kq_model_find_semantic_tensor(
        model, "layer.01.ple.address.head_offsets");
    status = semantic == NULL
                 ? KQ_STATUS_INVALID_ARGUMENT
                 : kq_tensor_view_open_binding(
                       gguf, semantic, 0U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
                       &view, &diagnostic);
    kq_tensor_view_close(view);
    view = NULL;
    if (status != KQ_STATUS_NO_TENSOR_PAYLOAD) {
        return 0;
    }
    semantic = kq_model_find_semantic_tensor(model, "layer.00.gdn.qkv");
    status = semantic == NULL
                 ? KQ_STATUS_INVALID_ARGUMENT
                 : kq_tensor_view_open_binding(
                       gguf, semantic, 0U,
                       KQ_TENSOR_VIEW_REQUIRE_CANONICAL_CONTIGUOUS,
                       &view, &diagnostic);
    kq_tensor_view_close(view);
    return status == KQ_STATUS_TENSOR_LAYOUT_MISMATCH;
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
    kq_model *model = NULL;
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

    status = kq_model_open_from_gguf(gguf, &model, &diagnostic);
    if (status != KQ_STATUS_OK) {
        print_error(status, &diagnostic);
        goto cleanup;
    }
    if (kq_model_hidden_size(model) != 2560U ||
        kq_model_vocabulary_size(model) != 248320U ||
        kq_model_context_length(model) != 262144U ||
        kq_model_semantic_tensor_count(model) != 1294U ||
        kq_model_metadata_derived_count(model) != 3U ||
        kq_model_physical_tensor_count(model) != EXPECTED_TENSOR_COUNT ||
        kq_model_physical_coverage_count(model) != EXPECTED_TENSOR_COUNT ||
        kq_model_unknown_physical_count(model) != 0U ||
        kq_model_unbound_required_count(model) != 0U ||
        kq_model_layer_count(model) != 48U ||
        kq_model_gdn_layer_count(model) != 36U ||
        kq_model_qsa_layer_count(model) != 12U ||
        kq_model_expert_count(model) != 512U ||
        kq_model_expert_top_k(model) != 10U ||
        kq_model_relation_count(model,
                                KQ_BINDING_RENAMED_ONE_TO_ONE) != 847U ||
        kq_model_relation_count(model,
                                KQ_BINDING_TRANSFORMED_LAYOUT) != 256U ||
        kq_model_relation_count(
            model,
            KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL) != 60U ||
        kq_model_relation_count(
            model,
            KQ_BINDING_MULTIPLE_CANONICAL_TO_ONE_PHYSICAL) != 128U ||
        kq_model_relation_count(model,
                                KQ_BINDING_METADATA_DERIVED) != 3U ||
        kq_model_placement_count(
            model, KQ_PLACEMENT_ALWAYS_NEEDED_CANDIDATE) != 1061U ||
        kq_model_placement_count(
            model, KQ_PLACEMENT_ROUTED_EXPERT_CACHE_CANDIDATE) != 96U ||
        kq_model_placement_count(
            model, KQ_PLACEMENT_PLE_DISK_BACKED_CANDIDATE) != 137U ||
        kq_model_find_semantic_tensor(
            model, "layer.03.qsa.indexer.qk") == NULL ||
        kq_model_find_semantic_tensor(
            model, "layer.01.ple.table.127") == NULL ||
        kq_model_find_semantic_tensor(
            model, "layer.01.ple.address.head_offsets") == NULL ||
        kq_gguf_payload_bytes_accessed(gguf) != 0U) {
        fprintf(stderr, "real GGUF semantic registry oracle mismatch\n");
        goto cleanup;
    }

    if (!validate_real_view_samples(gguf, model) ||
        kq_gguf_payload_bytes_accessed(gguf) != 0U) {
        fprintf(stderr, "real GGUF bounded-view geometry mismatch\n");
        goto cleanup;
    }

    printf("real GGUF physical/semantic/view oracle: PASS, "
           "semantics=1294, payload_bytes_accessed=0, "
           "payload_bytes_touched_by_test=0\n");
    result = 0;

cleanup:
    kq_model_close(model);
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
