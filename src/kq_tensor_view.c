#include "kq_tensor_view.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_gguf_internal.h"
#include "kq_internal.h"

struct kq_tensor_view {
    kq_file_view *file_view;
    kq_tensor_view_info info;
};

static int kq_view_u64_add(uint64_t left,
                           uint64_t right,
                           uint64_t *result) {
    if (UINT64_MAX - left < right) {
        return 0;
    }
    *result = left + right;
    return 1;
}

static int kq_view_u64_mul(uint64_t left,
                           uint64_t right,
                           uint64_t *result) {
    if (left != 0U && right > UINT64_MAX / left) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static kq_status kq_view_fail(kq_diagnostic *diagnostic,
                              kq_status status,
                              const char *message) {
    kq_diagnostic_set(diagnostic, status, "%s", message);
    return status;
}

kq_status kq_quant_geometry_for_type(uint32_t type_id,
                                     kq_quant_geometry *out_geometry,
                                     kq_diagnostic *diagnostic) {
    const kq_gguf_type_info *type_info;

    kq_diagnostic_clear(diagnostic);
    if (out_geometry == NULL) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_ARGUMENT,
                            "output quantized geometry is required");
    }
    memset(out_geometry, 0, sizeof(*out_geometry));
    type_info = kq_gguf_type_info_for(type_id);
    if (type_info == NULL) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_UNSUPPORTED_TENSOR_TYPE,
                            "GGUF tensor type is unsupported");
    }
    if (type_info->block_elements == 0U ||
        type_info->bytes_per_block == 0U) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                            "quantized type has zero block geometry");
    }
    out_geometry->type_id = type_info->type_id;
    out_geometry->block_elements = type_info->block_elements;
    out_geometry->bytes_per_block = type_info->bytes_per_block;
    return KQ_STATUS_OK;
}

kq_status kq_quant_packed_size(uint32_t type_id,
                               uint64_t element_count,
                               uint64_t *out_packed_bytes,
                               kq_diagnostic *diagnostic) {
    kq_quant_geometry geometry;
    uint64_t block_count;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    if (out_packed_bytes == NULL) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_ARGUMENT,
                            "output packed byte count is required");
    }
    *out_packed_bytes = 0U;
    status = kq_quant_geometry_for_type(type_id, &geometry, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (element_count == 0U ||
        (element_count % geometry.block_elements) != 0U) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                            "element count is not a non-zero whole number of quant blocks");
    }
    block_count = element_count / geometry.block_elements;
    if (!kq_view_u64_mul(block_count,
                         geometry.bytes_per_block,
                         out_packed_bytes)) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_ARITHMETIC_OVERFLOW,
                            "packed byte count overflows uint64");
    }
    return KQ_STATUS_OK;
}

kq_status kq_quant_element_to_block(uint32_t type_id,
                                    uint64_t total_elements,
                                    uint64_t element_index,
                                    uint64_t *out_block_index,
                                    uint64_t *out_element_in_block,
                                    kq_diagnostic *diagnostic) {
    kq_quant_geometry geometry;
    uint64_t ignored_packed_bytes;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    if (out_block_index == NULL || out_element_in_block == NULL) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_ARGUMENT,
                            "block index and intra-block outputs are required");
    }
    *out_block_index = 0U;
    *out_element_in_block = 0U;
    status = kq_quant_geometry_for_type(type_id, &geometry, diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_quant_packed_size(type_id,
                                      total_elements,
                                      &ignored_packed_bytes,
                                      diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (element_index >= total_elements) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_SPAN_OUT_OF_RANGE,
                            "logical element index is outside the tensor");
    }
    *out_block_index = element_index / geometry.block_elements;
    *out_element_in_block = element_index % geometry.block_elements;
    return KQ_STATUS_OK;
}

kq_status kq_quant_range_to_blocks(uint32_t type_id,
                                   uint64_t total_elements,
                                   uint64_t element_offset,
                                   uint64_t element_count,
                                   kq_quant_block_span *out_span,
                                   kq_diagnostic *diagnostic) {
    kq_quant_geometry geometry;
    uint64_t ignored_packed_bytes;
    uint64_t element_end;
    uint64_t last_block_exclusive;
    uint64_t block_element_offset;
    uint64_t block_element_end;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    if (out_span == NULL || element_count == 0U) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_ARGUMENT,
                            "non-zero logical range and output span are required");
    }
    memset(out_span, 0, sizeof(*out_span));
    status = kq_quant_geometry_for_type(type_id, &geometry, diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_quant_packed_size(type_id,
                                      total_elements,
                                      &ignored_packed_bytes,
                                      diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (!kq_view_u64_add(element_offset, element_count, &element_end)) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_ARITHMETIC_OVERFLOW,
                            "logical element range overflows uint64");
    }
    if (element_offset >= total_elements || element_end > total_elements) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_SPAN_OUT_OF_RANGE,
                            "logical element range is outside the tensor");
    }

    out_span->requested_element_offset = element_offset;
    out_span->requested_element_count = element_count;
    out_span->first_block = element_offset / geometry.block_elements;
    last_block_exclusive = ((element_end - 1U) /
                            geometry.block_elements) + 1U;
    out_span->block_count = last_block_exclusive - out_span->first_block;
    if (!kq_view_u64_mul(out_span->first_block,
                         geometry.bytes_per_block,
                         &out_span->physical_byte_offset) ||
        !kq_view_u64_mul(out_span->block_count,
                         geometry.bytes_per_block,
                         &out_span->physical_byte_length) ||
        !kq_view_u64_mul(out_span->first_block,
                         geometry.block_elements,
                         &block_element_offset) ||
        !kq_view_u64_mul(last_block_exclusive,
                         geometry.block_elements,
                         &block_element_end)) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_ARITHMETIC_OVERFLOW,
                            "quantized block span arithmetic overflows uint64");
    }
    out_span->leading_elements = element_offset - block_element_offset;
    out_span->trailing_elements = block_element_end - element_end;
    return KQ_STATUS_OK;
}

static kq_status kq_validate_physical_tensor(
    const kq_gguf_tensor *tensor,
    kq_diagnostic *diagnostic) {
    const kq_gguf_type_info *type_info;
    uint64_t elements = 1U;
    uint64_t packed_bytes = 0U;
    uint32_t dimension;
    kq_status status;

    if (tensor == NULL || tensor->rank == 0U ||
        tensor->rank > KQ_GGUF_MAX_DIMS) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                            "physical tensor rank is invalid");
    }
    type_info = kq_gguf_type_info_for(tensor->type_id);
    if (type_info == NULL) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_UNSUPPORTED_TENSOR_TYPE,
                            "physical tensor type is unsupported");
    }
    if (tensor->block_elements != type_info->block_elements ||
        tensor->bytes_per_block != type_info->bytes_per_block) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                            "physical tensor block geometry disagrees with its type");
    }
    for (dimension = 0U; dimension < tensor->rank; ++dimension) {
        if (tensor->dimensions[dimension] == 0U) {
            return kq_view_fail(diagnostic,
                                KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                                "physical tensor has a zero dimension");
        }
        if (!kq_view_u64_mul(elements,
                             tensor->dimensions[dimension],
                             &elements)) {
            return kq_view_fail(diagnostic,
                                KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "physical tensor dimension product overflows uint64");
        }
    }
    if (elements != tensor->element_count ||
        (tensor->dimensions[0] % type_info->block_elements) != 0U) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                            "physical tensor element or first-dimension geometry is inconsistent");
    }
    status = kq_quant_packed_size(tensor->type_id,
                                  tensor->element_count,
                                  &packed_bytes,
                                  diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (packed_bytes != tensor->packed_bytes) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                            "physical tensor packed bytes disagree with block geometry");
    }
    return KQ_STATUS_OK;
}

static kq_status kq_semantic_element_count(
    const kq_semantic_tensor *semantic,
    uint64_t *out_elements,
    kq_diagnostic *diagnostic) {
    uint64_t elements = 1U;
    uint32_t dimension;

    for (dimension = 0U; dimension < semantic->canonical_rank; ++dimension) {
        if (semantic->canonical_dimensions[dimension] == 0U) {
            return kq_view_fail(diagnostic,
                                KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                                "canonical semantic has a zero dimension");
        }
        if (!kq_view_u64_mul(elements,
                             semantic->canonical_dimensions[dimension],
                             &elements)) {
            return kq_view_fail(diagnostic,
                                KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "canonical semantic dimension product overflows uint64");
        }
    }
    *out_elements = elements;
    return KQ_STATUS_OK;
}

static kq_status kq_validate_split(const kq_semantic_tensor *semantic,
                                   kq_diagnostic *diagnostic) {
    const kq_tensor_binding *first;
    const kq_tensor_binding *second;
    int moe_roles;
    int qsa_roles;
    uint64_t canonical_elements = 0U;
    uint64_t physical_elements = 0U;
    kq_status status;

    if (semantic->relation !=
        KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL) {
        return KQ_STATUS_OK;
    }
    if (semantic->binding_count != 2U) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                            "split semantic must have exactly two physical parts");
    }
    first = &semantic->bindings[0];
    second = &semantic->bindings[1];
    if (first->physical == NULL || second->physical == NULL ||
        first->part_count != 2U || second->part_count != 2U ||
        first->part_index != 0U || second->part_index != 1U ||
        first->physical == second->physical) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                            "split semantic part count, order or uniqueness is invalid");
    }
    moe_roles = first->part_role == KQ_BINDING_PART_GATE &&
                second->part_role == KQ_BINDING_PART_UP;
    qsa_roles = first->part_role == KQ_BINDING_PART_INDEX_QUERY &&
                second->part_role == KQ_BINDING_PART_INDEX_KEY;
    if (!moe_roles && !qsa_roles) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                            "split semantic part roles are invalid");
    }
    status = kq_validate_physical_tensor(first->physical, diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_validate_physical_tensor(second->physical, diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_semantic_element_count(semantic,
                                           &canonical_elements,
                                           diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (!kq_view_u64_add(first->physical->element_count,
                         second->physical->element_count,
                         &physical_elements)) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_ARITHMETIC_OVERFLOW,
                            "split physical element count overflows uint64");
    }
    if (canonical_elements != physical_elements) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                            "split physical parts do not cover the canonical element count");
    }
    return KQ_STATUS_OK;
}

static void kq_copy_view_shapes(kq_tensor_view_info *info,
                                const kq_semantic_tensor *semantic,
                                const kq_gguf_tensor *physical) {
    uint32_t dimension;

    info->canonical_rank = semantic->canonical_rank;
    for (dimension = 0U; dimension < semantic->canonical_rank; ++dimension) {
        info->canonical_dimensions[dimension] =
            semantic->canonical_dimensions[dimension];
    }
    info->physical_rank = physical->rank;
    for (dimension = 0U; dimension < physical->rank; ++dimension) {
        info->physical_dimensions[dimension] =
            physical->dimensions[dimension];
    }
    info->physical_element_count = physical->element_count;
}

static kq_status kq_open_span(const kq_gguf *gguf,
                              const kq_semantic_tensor *semantic,
                              uint32_t binding_index,
                              kq_tensor_view_kind kind,
                              kq_tensor_view_layout layout,
                              uint32_t canonical_contiguous,
                              uint32_t expert_id,
                              uint32_t fused_member_index,
                              uint32_t logical_rank,
                              const uint64_t *logical_dimensions,
                              uint64_t element_offset,
                              uint64_t element_count,
                              kq_tensor_view **out_view,
                              kq_diagnostic *diagnostic) {
    const kq_tensor_binding *binding =
        &semantic->bindings[binding_index];
    const kq_gguf_tensor *physical = binding->physical;
    kq_quant_block_span span;
    kq_tensor_view *view = NULL;
    kq_file_view *file_view = NULL;
    uint64_t file_offset;
    uint64_t logical_element_bytes;
    uint64_t logical_unpacked_bytes;
    uint32_t dimension;
    kq_status status;

    status = kq_validate_physical_tensor(physical, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    logical_element_bytes = semantic->canonical_dtype ==
                                    KQ_CANONICAL_DTYPE_I64
                                ? 8U
                                : 2U;
    if (!kq_view_u64_mul(element_count,
                         logical_element_bytes,
                         &logical_unpacked_bytes)) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_ARITHMETIC_OVERFLOW,
                            "logical unpacked byte count overflows uint64");
    }
    status = kq_quant_range_to_blocks(physical->type_id,
                                      physical->element_count,
                                      element_offset,
                                      element_count,
                                      &span,
                                      diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    status = kq_gguf_open_tensor_span(gguf,
                                      physical,
                                      span.physical_byte_offset,
                                      span.physical_byte_length,
                                      &file_view,
                                      diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    view = (kq_tensor_view *)calloc(1U, sizeof(*view));
    if (view == NULL) {
        kq_file_view_close(file_view);
        return kq_view_fail(diagnostic,
                            KQ_STATUS_OUT_OF_MEMORY,
                            "could not allocate tensor view state");
    }
    if (!kq_view_u64_add(physical->data_offset,
                         span.physical_byte_offset,
                         &file_offset)) {
        kq_file_view_close(file_view);
        free(view);
        return kq_view_fail(diagnostic,
                            KQ_STATUS_ARITHMETIC_OVERFLOW,
                            "tensor view file offset overflows uint64");
    }

    view->file_view = file_view;
    (void)snprintf(view->info.semantic_id,
                   sizeof(view->info.semantic_id),
                   "%s",
                   semantic->semantic_id);
    view->info.kind = kind;
    view->info.layout = layout;
    view->info.relation = semantic->relation;
    view->info.part_role = binding->part_role;
    view->info.binding_index = binding_index;
    view->info.type_id = physical->type_id;
    view->info.block_elements = physical->block_elements;
    view->info.bytes_per_block = physical->bytes_per_block;
    view->info.logical_rank = logical_rank;
    for (dimension = 0U; dimension < logical_rank; ++dimension) {
        view->info.logical_dimensions[dimension] = logical_dimensions[dimension];
    }
    kq_copy_view_shapes(&view->info, semantic, physical);
    view->info.requested_element_offset = span.requested_element_offset;
    view->info.requested_element_count = span.requested_element_count;
    view->info.logical_unpacked_bytes = logical_unpacked_bytes;
    view->info.first_block = span.first_block;
    view->info.block_count = span.block_count;
    view->info.tensor_byte_offset = span.physical_byte_offset;
    view->info.file_byte_offset = file_offset;
    view->info.mapped_logical_offset = kq_file_view_offset(file_view);
    view->info.mapped_logical_length = kq_file_view_length(file_view);
    view->info.leading_elements = span.leading_elements;
    view->info.trailing_elements = span.trailing_elements;
    view->info.canonical_contiguous = canonical_contiguous;
    view->info.expert_id = expert_id;
    view->info.fused_member_index = fused_member_index;
    *out_view = view;
    return KQ_STATUS_OK;
}

static kq_status kq_validate_open_arguments(
    const kq_gguf *gguf,
    const kq_semantic_tensor *semantic,
    uint32_t binding_index,
    kq_tensor_view_access access,
    kq_tensor_view **out_view,
    kq_diagnostic *diagnostic) {
    const kq_tensor_binding *binding;

    kq_diagnostic_clear(diagnostic);
    if (gguf == NULL || semantic == NULL || out_view == NULL ||
        (access != KQ_TENSOR_VIEW_PHYSICAL_LAYOUT &&
         access != KQ_TENSOR_VIEW_REQUIRE_CANONICAL_CONTIGUOUS)) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_ARGUMENT,
                            "GGUF, semantic, valid access and output view are required");
    }
    *out_view = NULL;
    if (semantic->relation == KQ_BINDING_METADATA_DERIVED) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_NO_TENSOR_PAYLOAD,
                            "metadata-derived semantic has no tensor payload");
    }
    if (semantic->canonical_rank == 0U ||
        semantic->canonical_rank > KQ_SEMANTIC_MAX_DIMS ||
        (semantic->canonical_dtype != KQ_CANONICAL_DTYPE_BF16 &&
         semantic->canonical_dtype != KQ_CANONICAL_DTYPE_I64) ||
        semantic->binding_count == 0U ||
        semantic->binding_count > KQ_SEMANTIC_MAX_BINDINGS) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                            "semantic rank or binding cardinality is invalid");
    }
    if (binding_index >= semantic->binding_count) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_SPAN_OUT_OF_RANGE,
                            "semantic binding index is out of range");
    }
    binding = &semantic->bindings[binding_index];
    if (binding->physical == NULL) {
        if (binding->metadata != NULL ||
            semantic->relation == KQ_BINDING_METADATA_DERIVED) {
            return kq_view_fail(diagnostic,
                                KQ_STATUS_NO_TENSOR_PAYLOAD,
                                "metadata-derived semantic has no tensor payload");
        }
        return kq_view_fail(diagnostic,
                            KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                            "semantic binding has no physical tensor");
    }
    return KQ_STATUS_OK;
}

kq_status kq_tensor_view_open_binding(
    const kq_gguf *gguf,
    const kq_semantic_tensor *semantic,
    uint32_t binding_index,
    kq_tensor_view_access access,
    kq_tensor_view **out_view,
    kq_diagnostic *diagnostic) {
    const kq_gguf_tensor *physical;
    kq_tensor_view_layout layout;
    uint64_t logical_dimensions[KQ_SEMANTIC_MAX_DIMS] = {0U};
    uint32_t logical_rank;
    uint32_t dimension;
    uint32_t canonical_contiguous;
    uint64_t canonical_elements = 0U;
    kq_status status;

    status = kq_validate_open_arguments(gguf, semantic, binding_index,
                                        access, out_view, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (semantic->relation ==
        KQ_BINDING_MULTIPLE_CANONICAL_TO_ONE_PHYSICAL) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                            "fused PLE semantics require the bounded member API");
    }
    if (semantic->relation != KQ_BINDING_DIRECT_ONE_TO_ONE &&
        semantic->relation != KQ_BINDING_RENAMED_ONE_TO_ONE &&
        semantic->relation != KQ_BINDING_TRANSFORMED_LAYOUT &&
        semantic->relation !=
            KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                            "semantic relation has no supported physical view");
    }
    status = kq_validate_split(semantic, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    physical = semantic->bindings[binding_index].physical;
    status = kq_validate_physical_tensor(physical, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (semantic->relation !=
            KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL &&
        semantic->binding_count != 1U) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                            "non-split semantic must have exactly one physical binding");
    }
    if (semantic->relation !=
        KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL) {
        status = kq_semantic_element_count(semantic,
                                           &canonical_elements,
                                           diagnostic);
        if (status != KQ_STATUS_OK) {
            return status;
        }
        if (canonical_elements != physical->element_count) {
            return kq_view_fail(diagnostic,
                                KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                                "physical tensor does not cover the canonical element count");
        }
    }
    if (semantic->relation == KQ_BINDING_TRANSFORMED_LAYOUT) {
        layout = KQ_TENSOR_LAYOUT_TRANSFORMED_PHYSICAL;
        canonical_contiguous = 0U;
    } else if (semantic->relation ==
               KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL) {
        layout = KQ_TENSOR_LAYOUT_SPLIT_SEGMENT;
        canonical_contiguous = 0U;
    } else {
        layout = KQ_TENSOR_LAYOUT_CANONICAL_CONTIGUOUS;
        canonical_contiguous = 1U;
    }
    if (access == KQ_TENSOR_VIEW_REQUIRE_CANONICAL_CONTIGUOUS &&
        canonical_contiguous == 0U) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                            "physical bytes are transformed or only one split segment");
    }

    if (canonical_contiguous != 0U) {
        logical_rank = semantic->canonical_rank;
        for (dimension = 0U; dimension < logical_rank; ++dimension) {
            logical_dimensions[dimension] =
                semantic->canonical_dimensions[dimension];
        }
    } else {
        if (physical->rank > KQ_SEMANTIC_MAX_DIMS) {
            return kq_view_fail(diagnostic,
                                KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                                "physical rank cannot be represented by the logical view descriptor");
        }
        logical_rank = physical->rank;
        for (dimension = 0U; dimension < logical_rank; ++dimension) {
            logical_dimensions[dimension] =
                physical->dimensions[logical_rank - dimension - 1U];
        }
    }
    return kq_open_span(
        gguf, semantic, binding_index,
        semantic->relation == KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL
            ? KQ_TENSOR_VIEW_SPLIT_PART
            : KQ_TENSOR_VIEW_WHOLE_PHYSICAL,
        layout, canonical_contiguous, UINT32_MAX, UINT32_MAX,
        logical_rank, logical_dimensions, 0U, physical->element_count,
        out_view, diagnostic);
}

kq_status kq_tensor_view_open_expert_member(
    const kq_gguf *gguf,
    const kq_semantic_tensor *semantic,
    uint32_t binding_index,
    uint32_t expert_id,
    kq_tensor_view_access access,
    kq_tensor_view **out_view,
    kq_diagnostic *diagnostic) {
    const kq_tensor_binding *binding;
    const kq_gguf_tensor *physical;
    uint64_t logical_dimensions[KQ_SEMANTIC_MAX_DIMS] = {0U};
    uint64_t member_elements = 1U;
    uint64_t element_offset;
    uint32_t axis;
    uint32_t logical_rank;
    uint32_t dimension;
    uint32_t canonical_contiguous;
    kq_tensor_view_layout layout;
    kq_quant_block_span span;
    uint64_t canonical_elements = 0U;
    kq_status status;

    status = kq_validate_open_arguments(gguf, semantic, binding_index,
                                        access, out_view, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (semantic->component != KQ_COMPONENT_ROUTED_EXPERT_STACK ||
        semantic->expert_count == 0U ||
        semantic->canonical_expert_axis >= semantic->canonical_rank ||
        semantic->canonical_dimensions[semantic->canonical_expert_axis] !=
            semantic->expert_count) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                            "semantic is not a valid routed expert stack");
    }
    if (expert_id >= semantic->expert_count) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_SPAN_OUT_OF_RANGE,
                            "expert identifier is out of range");
    }
    status = kq_validate_split(semantic, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    binding = &semantic->bindings[binding_index];
    physical = binding->physical;
    status = kq_validate_physical_tensor(physical, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (semantic->relation != KQ_BINDING_DIRECT_ONE_TO_ONE &&
        semantic->relation != KQ_BINDING_RENAMED_ONE_TO_ONE &&
        semantic->relation !=
            KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                            "routed expert relation is not a supported contiguous representation");
    }
    if (semantic->relation !=
        KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL) {
        status = kq_semantic_element_count(semantic,
                                           &canonical_elements,
                                           diagnostic);
        if (status != KQ_STATUS_OK) {
            return status;
        }
        if (canonical_elements != physical->element_count) {
            return kq_view_fail(diagnostic,
                                KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                                "expert stack does not cover the canonical element count");
        }
    }
    axis = binding->physical_expert_axis;
    if (axis >= physical->rank ||
        physical->dimensions[axis] != semantic->expert_count) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                            "physical expert axis is invalid");
    }
    if (axis + 1U != physical->rank) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_NONCONTIGUOUS_TENSOR_VIEW,
                            "expert axis is not the slowest physical axis");
    }
    if (axis == 0U || axis > KQ_SEMANTIC_MAX_DIMS) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                            "expert member rank is unsupported");
    }
    for (dimension = 0U; dimension < axis; ++dimension) {
        if (!kq_view_u64_mul(member_elements,
                             physical->dimensions[dimension],
                             &member_elements)) {
            return kq_view_fail(diagnostic,
                                KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "expert member element count overflows uint64");
        }
    }
    if (!kq_view_u64_mul((uint64_t)expert_id,
                         member_elements,
                         &element_offset)) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_ARITHMETIC_OVERFLOW,
                            "expert member offset overflows uint64");
    }
    status = kq_quant_range_to_blocks(physical->type_id,
                                      physical->element_count,
                                      element_offset,
                                      member_elements,
                                      &span,
                                      diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (span.leading_elements != 0U || span.trailing_elements != 0U) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_NONCONTIGUOUS_TENSOR_VIEW,
                            "expert member does not start and end on quant block boundaries");
    }
    canonical_contiguous =
        semantic->relation == KQ_BINDING_DIRECT_ONE_TO_ONE ||
        semantic->relation == KQ_BINDING_RENAMED_ONE_TO_ONE;
    layout = canonical_contiguous != 0U
                 ? KQ_TENSOR_LAYOUT_CANONICAL_CONTIGUOUS
                 : KQ_TENSOR_LAYOUT_SPLIT_SEGMENT;
    if (access == KQ_TENSOR_VIEW_REQUIRE_CANONICAL_CONTIGUOUS &&
        canonical_contiguous == 0U) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                            "expert view is one split physical segment, not the complete canonical projection");
    }
    logical_rank = axis;
    for (dimension = 0U; dimension < logical_rank; ++dimension) {
        logical_dimensions[dimension] =
            physical->dimensions[logical_rank - dimension - 1U];
    }
    return kq_open_span(gguf, semantic, binding_index,
                        KQ_TENSOR_VIEW_EXPERT_MEMBER, layout,
                        canonical_contiguous, expert_id, UINT32_MAX,
                        logical_rank, logical_dimensions,
                        element_offset, member_elements,
                        out_view, diagnostic);
}

kq_status kq_tensor_view_open_ple_member(
    const kq_gguf *gguf,
    const kq_semantic_tensor *semantic,
    kq_tensor_view_access access,
    kq_tensor_view **out_view,
    kq_diagnostic *diagnostic) {
    const kq_tensor_binding *binding;
    const kq_gguf_tensor *physical;
    uint64_t member_elements;
    uint64_t element_offset;
    uint64_t member_rows;
    kq_quant_block_span span;
    kq_status status;

    status = kq_validate_open_arguments(gguf, semantic, 0U,
                                        access, out_view, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (semantic->component != KQ_COMPONENT_PLE_TABLE ||
        semantic->relation !=
            KQ_BINDING_MULTIPLE_CANONICAL_TO_ONE_PHYSICAL ||
        semantic->binding_count != 1U || semantic->canonical_rank != 2U) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                            "semantic is not a valid fused PLE table member");
    }
    binding = &semantic->bindings[0];
    physical = binding->physical;
    status = kq_validate_physical_tensor(physical, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (binding->part_role != KQ_BINDING_PART_FUSED_MEMBER ||
        binding->fused_member_count == 0U ||
        binding->fused_member_index >= binding->fused_member_count ||
        physical->rank != 2U ||
        physical->dimensions[1] % binding->fused_member_count != 0U) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                            "PLE fused member metadata or physical rank is invalid");
    }
    member_rows = physical->dimensions[1] /
                  binding->fused_member_count;
    if (physical->dimensions[0] != semantic->canonical_dimensions[1] ||
        member_rows != semantic->canonical_dimensions[0] ||
        physical->element_count % binding->fused_member_count != 0U) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                            "PLE fused dimensions do not match the canonical member");
    }
    member_elements = physical->element_count /
                      binding->fused_member_count;
    if (!kq_view_u64_mul(binding->fused_member_index,
                         member_elements,
                         &element_offset)) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_ARITHMETIC_OVERFLOW,
                            "PLE member offset overflows uint64");
    }
    status = kq_quant_range_to_blocks(physical->type_id,
                                      physical->element_count,
                                      element_offset,
                                      member_elements,
                                      &span,
                                      diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (span.leading_elements != 0U || span.trailing_elements != 0U ||
        span.physical_byte_length >= physical->packed_bytes) {
        return kq_view_fail(diagnostic,
                            KQ_STATUS_NONCONTIGUOUS_TENSOR_VIEW,
                            "PLE member is not an independently bounded quant-block span");
    }
    return kq_open_span(gguf, semantic, 0U,
                        KQ_TENSOR_VIEW_PLE_FUSED_MEMBER,
                        KQ_TENSOR_LAYOUT_FUSED_MEMBER, 1U,
                        UINT32_MAX, binding->fused_member_index,
                        semantic->canonical_rank,
                        semantic->canonical_dimensions,
                        element_offset, member_elements,
                        out_view, diagnostic);
}

const kq_tensor_view_info *kq_tensor_view_get_info(
    const kq_tensor_view *view) {
    return view == NULL ? NULL : &view->info;
}

const unsigned char *kq_tensor_view_physical_data(
    const kq_tensor_view *view) {
    return view == NULL ? NULL : kq_file_view_data(view->file_view);
}

void kq_tensor_view_close(kq_tensor_view *view) {
    if (view == NULL) {
        return;
    }
    kq_file_view_close(view->file_view);
    free(view);
}

const char *kq_tensor_view_kind_name(kq_tensor_view_kind kind) {
    switch (kind) {
        case KQ_TENSOR_VIEW_WHOLE_PHYSICAL:
            return "WHOLE_PHYSICAL";
        case KQ_TENSOR_VIEW_SPLIT_PART:
            return "SPLIT_PART";
        case KQ_TENSOR_VIEW_EXPERT_MEMBER:
            return "EXPERT_MEMBER";
        case KQ_TENSOR_VIEW_PLE_FUSED_MEMBER:
            return "PLE_FUSED_MEMBER";
        default:
            return "INVALID";
    }
}

const char *kq_tensor_view_layout_name(kq_tensor_view_layout layout) {
    switch (layout) {
        case KQ_TENSOR_LAYOUT_CANONICAL_CONTIGUOUS:
            return "CANONICAL_CONTIGUOUS";
        case KQ_TENSOR_LAYOUT_TRANSFORMED_PHYSICAL:
            return "TRANSFORMED_PHYSICAL";
        case KQ_TENSOR_LAYOUT_SPLIT_SEGMENT:
            return "SPLIT_SEGMENT";
        case KQ_TENSOR_LAYOUT_FUSED_MEMBER:
            return "FUSED_MEMBER";
        default:
            return "INVALID";
    }
}
