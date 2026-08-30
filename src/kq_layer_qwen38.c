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

kq_status kq_layer_gr_read_f32(
    const kq_layer_config *config, const kq_layer_gr_weights_f32 *weights,
    const float *branches, float *normalized, float *read_gate,
    float *mixed, float *rank_workspace, float *write_gate,
    kq_diagnostic *diagnostic) {
    uint32_t branch;
    uint32_t branches_count;
    uint32_t hidden;
    uint32_t rank;
    uint64_t width;
    uint64_t index;
    kq_status status;
    if (branches == NULL || normalized == NULL || read_gate == NULL ||
        mixed == NULL || rank_workspace == NULL || write_gate == NULL) {
        kq_diagnostic_set(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                          "GR buffers must be non-null");
        return KQ_STATUS_INVALID_ARGUMENT;
    }
    status = kq_layer_gr_validate(config, weights, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    branches_count = config->dimensions.branch_count;
    hidden = config->dimensions.hidden_size;
    rank = config->dimensions.gr_rank;
    width = (uint64_t)branches_count * hidden;
    for (branch = 0U; branch < branches_count; ++branch) {
        status = kq_f32_rms_norm(
            branches + (uint64_t)branch * hidden,
            weights->norm + (uint64_t)branch * hidden, hidden,
            config->dimensions.rms_epsilon,
            normalized + (uint64_t)branch * hidden, diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    for (index = 0U; index < rank; ++index) {
        float sum = 0.0f;
        uint64_t column;
        for (column = 0U; column < width; ++column)
            sum += weights->down[index * width + column] * normalized[column];
        rank_workspace[index] = sum / (float)branches_count;
    }
    for (index = 0U; index < rank; ++index) {
        float sigmoid = rank_workspace[index] >= 0.0f
            ? 1.0f / (1.0f + expf(-rank_workspace[index]))
            : expf(rank_workspace[index]) /
                  (1.0f + expf(rank_workspace[index]));
        rank_workspace[index] *= sigmoid;
    }
    for (index = 0U; index < width; ++index) {
        float sum = 0.0f;
        uint64_t column;
        for (column = 0U; column < rank; ++column)
            sum += weights->up[index * rank + column] * rank_workspace[column];
        read_gate[index] = sum;
    }
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
    for (branch = 0U; branch < branches_count; ++branch) {
        float sum = 0.0f;
        for (index = 0U; index < width; ++index)
            sum += weights->inject[(uint64_t)branch * width + index] *
                   normalized[index];
        write_gate[branch] = sum / (float)branches_count;
    }
    for (branch = 0U; branch < branches_count; ++branch) {
        float value = write_gate[branch];
        float sigmoid = value >= 0.0f
            ? 1.0f / (1.0f + expf(-value))
            : expf(value) / (1.0f + expf(value));
        write_gate[branch] = 2.0f * sigmoid;
    }
    return KQ_STATUS_OK;
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
