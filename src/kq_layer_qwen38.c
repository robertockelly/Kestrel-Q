#include "kq_layer_internal.h"

#include <math.h>
#include <stddef.h>

#include "kq_numeric.h"
#include "kq_internal.h"

static kq_status kq_layer_gr_validate(
    const kq_layer_config *config, const kq_layer_gr_weights_f32 *weights,
    kq_diagnostic *diagnostic) {
    uint64_t width;
    uint64_t rank;
    if (config == NULL || config->magic != KQ_LAYER_CONFIG_MAGIC ||
        weights == NULL) {
        kq_diagnostic_set(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                          "GR arguments are null or invalid");
        return KQ_STATUS_INVALID_ARGUMENT;
    }
    width = (uint64_t)config->dimensions.hidden_size *
            config->dimensions.branch_count;
    rank = config->dimensions.gr_rank;
    if (weights->norm == NULL || weights->down == NULL ||
        weights->up == NULL || weights->inject == NULL ||
        weights->norm_count != width ||
        weights->down_count != rank * width ||
        weights->up_count != width * rank ||
        weights->inject_count !=
            (uint64_t)config->dimensions.branch_count * width) {
        kq_diagnostic_set(diagnostic, KQ_STATUS_INCOMPATIBLE_LAYER,
                          "GR weights do not match branch/rank geometry");
        return KQ_STATUS_INCOMPATIBLE_LAYER;
    }
    return KQ_STATUS_OK;
}

static kq_status kq_layer_gr_read_common(
    const kq_layer_config *config, const kq_layer_gr_weights_f32 *weights,
    kq_weight_provider *provider, uint32_t binding_base,
    const float *branches, float *normalized, float *read_gate,
    float *mixed, float *rank_workspace, float *write_gate,
    void *weight_scratch, uint64_t weight_scratch_bytes,
    kq_diagnostic *diagnostic) {
    uint32_t branch;
    uint32_t branches_count;
    uint32_t hidden;
    uint32_t rank;
    uint64_t width;
    uint64_t index;
    const float *norm;
    float *norm_storage = (float *)weight_scratch;
    float *temporary;
    void *linear_scratch;
    uint64_t linear_scratch_bytes;
    kq_status status;
    if (branches == NULL || normalized == NULL || read_gate == NULL ||
        mixed == NULL || rank_workspace == NULL || write_gate == NULL) {
        kq_diagnostic_set(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                          "GR buffers must be non-null");
        return KQ_STATUS_INVALID_ARGUMENT;
    }
    branches_count = config->dimensions.branch_count;
    hidden = config->dimensions.hidden_size;
    rank = config->dimensions.gr_rank;
    width = (uint64_t)branches_count * hidden;
    if (provider == NULL) {
        status = kq_layer_gr_validate(config, weights, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        norm = weights->norm;
        linear_scratch = weight_scratch;
        linear_scratch_bytes = weight_scratch_bytes;
    } else {
        uint64_t vector_bytes = width * sizeof(float);
        if (binding_base + 3U >= KQ_LAYER_GR_BINDING_COUNT ||
            weight_scratch == NULL ||
            weight_scratch_bytes < vector_bytes * 2U + 65536U)
            return KQ_STATUS_BUFFER_TOO_SMALL;
        temporary = norm_storage + width;
        status = kq_weight_provider_vector_f32(
            provider, config->gr_bindings[binding_base + 1U], width,
            norm_storage, width, temporary, vector_bytes, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        norm = norm_storage;
        linear_scratch = (unsigned char *)weight_scratch + vector_bytes * 2U;
        linear_scratch_bytes = weight_scratch_bytes - vector_bytes * 2U;
    }
    for (branch = 0U; branch < branches_count; ++branch) {
        status = kq_f32_rms_norm(
            branches + (uint64_t)branch * hidden,
            norm + (uint64_t)branch * hidden, hidden,
            config->dimensions.rms_epsilon,
            normalized + (uint64_t)branch * hidden, diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    if (provider != NULL) status = kq_weight_provider_linear_f32(
        provider, config->gr_bindings[binding_base + 2U],
        KQ_BINDING_PART_WHOLE, KQ_WEIGHT_PROVIDER_NO_EXPERT, rank, width,
        normalized, rank_workspace, rank, linear_scratch,
        linear_scratch_bytes, diagnostic);
    else {
        status = KQ_STATUS_OK;
        for (index = 0U; index < rank; ++index) {
            float sum = 0.0f; uint64_t column;
            for (column = 0U; column < width; ++column)
                sum += weights->down[index * width + column] * normalized[column];
            rank_workspace[index] = sum;
        }
    }
    if (status != KQ_STATUS_OK) return status;
    for (index = 0U; index < rank; ++index)
        rank_workspace[index] /= (float)branches_count;
    for (index = 0U; index < rank; ++index) {
        float sigmoid = rank_workspace[index] >= 0.0f
            ? 1.0f / (1.0f + expf(-rank_workspace[index]))
            : expf(rank_workspace[index]) /
                  (1.0f + expf(rank_workspace[index]));
        rank_workspace[index] *= sigmoid;
    }
    if (provider != NULL) status = kq_weight_provider_linear_f32(
        provider, config->gr_bindings[binding_base + 3U],
        KQ_BINDING_PART_WHOLE, KQ_WEIGHT_PROVIDER_NO_EXPERT, width, rank,
        rank_workspace, read_gate, width, linear_scratch,
        linear_scratch_bytes, diagnostic);
    else {
        status = KQ_STATUS_OK;
        for (index = 0U; index < width; ++index) {
            float sum = 0.0f; uint64_t column;
            for (column = 0U; column < rank; ++column)
                sum += weights->up[index * rank + column] * rank_workspace[column];
            read_gate[index] = sum;
        }
    }
    if (status != KQ_STATUS_OK) return status;
    for (index = 0U; index < width; ++index) {
        float value = read_gate[index];
        read_gate[index] = value >= 0.0f
            ? 1.0f / (1.0f + expf(-value))
            : expf(value) / (1.0f + expf(value));
    }
    for (index = 0U; index < hidden; ++index) {
        float sum = 0.0f;
        for (branch = 0U; branch < branches_count; ++branch) {
            uint64_t offset = (uint64_t)branch * hidden + index;
            sum += read_gate[offset] * normalized[offset];
        }
        mixed[index] = sum / (float)branches_count;
    }
    if (provider != NULL) status = kq_weight_provider_linear_f32(
        provider, config->gr_bindings[binding_base], KQ_BINDING_PART_WHOLE,
        KQ_WEIGHT_PROVIDER_NO_EXPERT, branches_count, width, normalized,
        write_gate, branches_count, linear_scratch, linear_scratch_bytes,
        diagnostic);
    else {
        status = KQ_STATUS_OK;
        for (branch = 0U; branch < branches_count; ++branch) {
            float sum = 0.0f;
            for (index = 0U; index < width; ++index)
                sum += weights->inject[(uint64_t)branch * width + index] * normalized[index];
            write_gate[branch] = sum;
        }
    }
    if (status != KQ_STATUS_OK) return status;
    for (branch = 0U; branch < branches_count; ++branch)
        write_gate[branch] /= (float)branches_count;
    for (branch = 0U; branch < branches_count; ++branch) {
        float value = write_gate[branch];
        float sigmoid = value >= 0.0f
            ? 1.0f / (1.0f + expf(-value))
            : expf(value) / (1.0f + expf(value));
        write_gate[branch] = 2.0f * sigmoid;
    }
    return KQ_STATUS_OK;
}

kq_status kq_layer_gr_read_f32(
    const kq_layer_config *config, const kq_layer_gr_weights_f32 *weights,
    const float *branches, float *normalized, float *read_gate,
    float *mixed, float *rank_workspace, float *write_gate,
    kq_diagnostic *diagnostic) {
    return kq_layer_gr_read_common(config, weights, NULL, 0U, branches,
        normalized, read_gate, mixed, rank_workspace, write_gate,
        NULL, 0U, diagnostic);
}

kq_status kq_layer_gr_read_quantized_f32(
    const kq_layer_config *config, kq_weight_provider *provider,
    uint32_t binding_base, const float *branches, float *normalized,
    float *read_gate, float *mixed, float *rank_workspace,
    float *write_gate, void *weight_scratch, uint64_t weight_scratch_bytes,
    kq_diagnostic *diagnostic) {
    return kq_layer_gr_read_common(config, NULL, provider, binding_base,
        branches, normalized, read_gate, mixed, rank_workspace, write_gate,
        weight_scratch, weight_scratch_bytes, diagnostic);
}

void kq_layer_gr_write_f32(const kq_layer_config *config,
                           const float *branches, const float *block_output,
                           const float *write_gate, float *output) {
    uint32_t branch;
    uint32_t hidden = config->dimensions.hidden_size;
    for (branch = 0U; branch < config->dimensions.branch_count; ++branch) {
        uint32_t index;
        for (index = 0U; index < hidden; ++index) {
            uint64_t offset = (uint64_t)branch * hidden + index;
            output[offset] = branches[offset] +
                             block_output[index] * write_gate[branch];
        }
    }
}
