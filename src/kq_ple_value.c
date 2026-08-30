#include "kq_ple_value_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_internal.h"
#include "kq_numeric.h"
#include "kq_tensor_view.h"

typedef struct kq_ple_value_dense_spec {
    kq_semantic_role role;
    const char *id;
    kq_binding_relation relation;
    uint32_t rank;
    uint64_t dimensions[3];
    uint32_t physical_type;
    uint32_t physical_rank;
    uint64_t physical_dimensions[3];
} kq_ple_value_dense_spec;

static const kq_ple_value_dense_spec kq_ple_value_dense_specs[] = {
    {KQ_ROLE_PLE_KEY, "layer.01.ple.key", KQ_BINDING_RENAMED_ONE_TO_ONE,
     2U, {10240U,2560U,0U}, KQ_GGUF_TYPE_Q8_0, 2U, {2560U,10240U,0U}},
    {KQ_ROLE_PLE_VALUE, "layer.01.ple.value", KQ_BINDING_RENAMED_ONE_TO_ONE,
     2U, {2560U,2560U,0U}, KQ_GGUF_TYPE_Q8_0, 2U, {2560U,2560U,0U}},
    {KQ_ROLE_PLE_NORM_KEY, "layer.01.ple.norm_key", KQ_BINDING_TRANSFORMED_LAYOUT,
     1U, {10240U,0U,0U}, KQ_GGUF_TYPE_F32, 1U, {10240U,0U,0U}},
    {KQ_ROLE_PLE_NORM_QUERY, "layer.01.ple.norm_query", KQ_BINDING_TRANSFORMED_LAYOUT,
     1U, {10240U,0U,0U}, KQ_GGUF_TYPE_F32, 1U, {10240U,0U,0U}},
    {KQ_ROLE_PLE_NORM_CONV, "layer.01.ple.norm_conv", KQ_BINDING_TRANSFORMED_LAYOUT,
     1U, {10240U,0U,0U}, KQ_GGUF_TYPE_F32, 1U, {10240U,0U,0U}},
    {KQ_ROLE_PLE_CONV, "layer.01.ple.conv", KQ_BINDING_TRANSFORMED_LAYOUT,
     3U, {10240U,1U,4U}, KQ_GGUF_TYPE_F32, 2U, {4U,10240U,0U}},
};

struct kq_ple_value_gguf_provider {
    const kq_gguf *gguf;
    const kq_model *model;
    const kq_ple_value_config *config;
    kq_ple_value_gguf_metrics metrics;
};

static kq_status fail(kq_diagnostic *diagnostic, kq_status status,
                      const char *message) {
    kq_diagnostic_set(diagnostic, status, "%s", message);
    return status;
}

int kq_ple_value_u64_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == NULL || UINT64_MAX - a < b) return 0;
    *out = a + b; return 1;
}

int kq_ple_value_u64_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == NULL || (a != 0U && b > UINT64_MAX / a)) return 0;
    *out = a * b; return 1;
}

int kq_ple_value_config_valid(const kq_ple_value_config *config) {
    return config != NULL && config->magic == KQ_PLE_VALUE_CONFIG_MAGIC;
}

static kq_status compute_sizes(kq_ple_value_config *config,
                               kq_diagnostic *diagnostic) {
    const kq_ple_value_dimensions *d = &config->dimensions;
    uint64_t floats = 0U, temporary = 0U;
    if (d->hidden_size == 0U || d->residual_branches == 0U ||
        d->heads_per_order == 0U || d->head_count != d->heads_per_order * 2U ||
        d->row_width == 0U || d->logical_member_count == 0U ||
        d->member_rows == 0U || d->convolution_kernel != 4U ||
        d->convolution_dilation != 3U || d->history_length != 9U ||
        !isfinite(d->rms_epsilon) || d->rms_epsilon <= 0.0f) {
        return fail(diagnostic, KQ_STATUS_INCOMPATIBLE_PLE_VALUE,
                    "invalid PLE value dimensions or convolution geometry");
    }
    if (!kq_ple_value_u64_mul(d->head_count, d->row_width,
                              &config->embedding_width) ||
        !kq_ple_value_u64_mul(d->hidden_size, d->residual_branches,
                              &config->branch_width) ||
        !kq_ple_value_u64_mul(config->branch_width, d->history_length,
                              &config->state_elements) ||
        !kq_ple_value_u64_mul(config->state_elements, sizeof(float),
                              &config->state_bytes)) {
        return fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                    "PLE value geometry overflows");
    }
    if (!kq_ple_value_u64_mul(config->state_elements, sizeof(uint16_t),
                              &config->semantic_state_bytes)) {
        return fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                    "PLE semantic state size overflows");
    }
    /* staged state + embedding + row + value + seven branch-width buffers */
    if (!kq_ple_value_u64_add(config->state_elements,
                              config->embedding_width, &floats) ||
        !kq_ple_value_u64_add(floats, d->row_width, &floats) ||
        !kq_ple_value_u64_add(floats, d->hidden_size, &floats) ||
        !kq_ple_value_u64_mul(config->branch_width, 7U, &temporary) ||
        !kq_ple_value_u64_add(floats, temporary, &floats) ||
        !kq_ple_value_u64_mul(floats, sizeof(float), &config->scratch_bytes)) {
        return fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                    "PLE value scratch size overflows");
    }
    return KQ_STATUS_OK;
}

kq_status kq_ple_value_test_config_create(
    const kq_ple_value_dimensions *dimensions,
    kq_ple_value_config **out_config, kq_diagnostic *diagnostic) {
    kq_ple_value_config *config;
    kq_status status;
    kq_diagnostic_clear(diagnostic);
    if (dimensions == NULL || out_config == NULL) {
        return fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                    "PLE value dimensions and output are required");
    }
    *out_config = NULL;
    config = (kq_ple_value_config *)calloc(1U, sizeof(*config));
    if (config == NULL) return fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                                    "could not allocate PLE value config");
    config->dimensions = *dimensions;
    config->layer_id = 1U;
    status = compute_sizes(config, diagnostic);
    if (status != KQ_STATUS_OK) { free(config); return status; }
    config->magic = KQ_PLE_VALUE_CONFIG_MAGIC;
    *out_config = config;
    return KQ_STATUS_OK;
}

static kq_status validate_dense(const kq_semantic_tensor *semantic,
                                const kq_ple_value_dense_spec *spec,
                                kq_diagnostic *diagnostic) {
    uint32_t i;
    const kq_gguf_tensor *physical;
    if (semantic == NULL || strcmp(semantic->semantic_id, spec->id) != 0 ||
        semantic->component != KQ_COMPONENT_PLE_DENSE ||
        semantic->role != spec->role || semantic->relation != spec->relation ||
        semantic->layer_id != 1U || semantic->layer_type != KQ_MODEL_LAYER_GDN ||
        semantic->canonical_dtype != KQ_CANONICAL_DTYPE_BF16 ||
        semantic->runtime_scope != KQ_SCOPE_REQUIRED_INITIAL_TEXT ||
        semantic->canonical_rank != spec->rank || semantic->binding_count != 1U ||
        semantic->bindings[0].part_role != KQ_BINDING_PART_WHOLE ||
        semantic->bindings[0].part_index != 0U ||
        semantic->bindings[0].part_count != 1U) {
        return fail(diagnostic, KQ_STATUS_INCOMPATIBLE_PLE_VALUE,
                    "PLE dense semantic identity, relation, or rank mismatch");
    }
    for (i = 0U; i < spec->rank; ++i) {
        if (semantic->canonical_dimensions[i] != spec->dimensions[i])
            return fail(diagnostic, KQ_STATUS_INCOMPATIBLE_PLE_VALUE,
                        "PLE dense canonical shape mismatch");
    }
    physical = semantic->bindings[0].physical;
    if (physical == NULL || physical->type_id != spec->physical_type ||
        physical->rank != spec->physical_rank)
        return fail(diagnostic, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                    "PLE dense physical type/rank mismatch");
    for (i = 0U; i < spec->physical_rank; ++i) {
        if (physical->dimensions[i] != spec->physical_dimensions[i])
            return fail(diagnostic, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                        "PLE dense physical shape mismatch");
    }
    return KQ_STATUS_OK;
}

static kq_status create_target(const kq_model *model,
                               const kq_ple_config *address_config,
                               kq_ple_value_activation_dtype dtype,
                               kq_ple_value_config **out_config,
                               kq_diagnostic *diagnostic) {
    kq_ple_value_semantic_source source;
    const kq_ple_config_info *address_info;
    uint32_t index;
    if (model == NULL || address_config == NULL || out_config == NULL)
        return fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                    "model, address config, and output are required");
    *out_config = NULL;
    address_info = kq_ple_config_get_info(address_config);
    if (address_info == NULL)
        return fail(diagnostic, KQ_STATUS_INCOMPATIBLE_PLE_VALUE,
                    "address configuration has no model information");
    memset(&source, 0, sizeof(source));
    source.model = model;
    source.address_config = address_config;
    source.hidden_size = kq_model_hidden_size(model);
    source.layer_type = kq_model_layer_type_at(model, 1U);
    source.address_info = *address_info;
    for (index = 0U; index < KQ_PLE_VALUE_DENSE_ROLE_COUNT; ++index)
        source.dense[index] = kq_model_find_semantic_tensor(
            model, kq_ple_value_dense_specs[index].id);
    for (index = 0U; index < KQ_PLE_LOGICAL_MEMBER_COUNT; ++index) {
        char id[KQ_SEMANTIC_ID_CAPACITY];
        (void)snprintf(id, sizeof(id), "layer.01.ple.table.%03u", index);
        source.tables[index] = kq_model_find_semantic_tensor(model, id);
    }
    return kq_ple_value_config_create_from_source(
        &source, dtype, out_config, diagnostic);
}

kq_status kq_ple_value_config_create_from_source(
    const kq_ple_value_semantic_source *source,
    kq_ple_value_activation_dtype dtype,
    kq_ple_value_config **out_config, kq_diagnostic *diagnostic) {
    kq_ple_value_dimensions d = {2560U,4U,8U,16U,160U,128U,UINT64_C(2500012),
                                 4U,3U,9U,1.0e-6f,dtype};
    kq_ple_value_config *config = NULL;
    const kq_gguf_tensor *fused_physical = NULL;
    uint32_t i;
    kq_status status;
    if (source == NULL || out_config == NULL)
        return fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                    "PLE semantic source and output are required");
    *out_config = NULL;
    if (dtype != KQ_PLE_VALUE_ACTIVATION_F32 &&
        dtype != KQ_PLE_VALUE_ACTIVATION_BF16)
        return fail(diagnostic, KQ_STATUS_INCOMPATIBLE_PLE_VALUE,
                    "unsupported PLE value activation dtype");
    if (source->address_info.ple_layer_id != 1U ||
        source->address_info.head_count != 16U ||
        source->address_info.logical_member_count != 128U ||
        source->address_info.member_rows != UINT64_C(2500012) ||
        source->hidden_size != 2560U || source->layer_type != KQ_MODEL_LAYER_GDN)
        return fail(diagnostic, KQ_STATUS_INCOMPATIBLE_PLE_VALUE,
                    "address config and model are not the canonical PLE target");
    status = kq_ple_value_test_config_create(&d, &config, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    config->model = source->model;
    config->address_config = source->address_config;
    for (i = 0U; i < KQ_PLE_VALUE_DENSE_ROLE_COUNT; ++i) {
        config->dense[i] = source->dense[i];
        status = validate_dense(config->dense[i], &kq_ple_value_dense_specs[i], diagnostic);
        if (status != KQ_STATUS_OK) { kq_ple_value_config_close(config); return status; }
    }
    for (i = 0U; i < 128U; ++i) {
        const kq_semantic_tensor *semantic = source->tables[i];
        const kq_gguf_tensor *physical;
        if (semantic == NULL || semantic->component != KQ_COMPONENT_PLE_TABLE ||
            semantic->role != KQ_ROLE_PLE_TABLE ||
            semantic->relation != KQ_BINDING_MULTIPLE_CANONICAL_TO_ONE_PHYSICAL ||
            semantic->runtime_scope != KQ_SCOPE_REQUIRED_INITIAL_TEXT ||
            semantic->placement_hint != KQ_PLACEMENT_PLE_DISK_BACKED_CANDIDATE ||
            semantic->layer_id != 1U || semantic->layer_type != KQ_MODEL_LAYER_GDN ||
            semantic->canonical_dtype != KQ_CANONICAL_DTYPE_BF16 ||
            semantic->canonical_rank != 2U ||
            semantic->canonical_dimensions[0] != UINT64_C(2500012) ||
            semantic->canonical_dimensions[1] != 160U ||
            semantic->binding_count != 1U ||
            semantic->bindings[0].fused_member_index != i ||
            semantic->bindings[0].fused_member_count != 128U ||
            semantic->bindings[0].part_role != KQ_BINDING_PART_FUSED_MEMBER ||
            semantic->bindings[0].part_index != 0U ||
            semantic->bindings[0].part_count != 1U) {
            kq_ple_value_config_close(config);
            return fail(diagnostic, KQ_STATUS_INCOMPATIBLE_PLE_VALUE,
                        "PLE table member semantic geometry mismatch");
        }
        physical = semantic->bindings[0].physical;
        if (physical == NULL || physical->type_id != KQ_GGUF_TYPE_IQ4_NL ||
            physical->rank != 2U || physical->dimensions[0] != 160U ||
            physical->dimensions[1] != UINT64_C(320001536) ||
            (fused_physical != NULL && physical != fused_physical)) {
            kq_ple_value_config_close(config);
            return fail(diagnostic, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                        "PLE fused IQ4_NL physical geometry mismatch");
        }
        if (fused_physical == NULL) fused_physical = physical;
        config->tables[i] = semantic;
    }
    *out_config = config;
    return KQ_STATUS_OK;
}

kq_status kq_ple_value_config_create(const kq_model *model,
    const kq_ple_config *address_config, kq_ple_value_config **out_config,
    kq_diagnostic *diagnostic) {
    kq_diagnostic_clear(diagnostic);
    return create_target(model, address_config, KQ_PLE_VALUE_ACTIVATION_BF16,
                         out_config, diagnostic);
}

kq_status kq_ple_value_config_create_reference_f32(const kq_model *model,
    const kq_ple_config *address_config, kq_ple_value_config **out_config,
    kq_diagnostic *diagnostic) {
    kq_diagnostic_clear(diagnostic);
    return create_target(model, address_config, KQ_PLE_VALUE_ACTIVATION_F32,
                         out_config, diagnostic);
}

void kq_ple_value_config_close(kq_ple_value_config *config) {
    if (config == NULL) return;
    config->magic = 0U; free(config);
}

#define CFG_GETTER(name, field, type, fallback) \
type name(const kq_ple_value_config *c) { return kq_ple_value_config_valid(c) ? (field) : (fallback); }
CFG_GETTER(kq_ple_value_config_layer_id, c->layer_id, uint32_t, 0U)
CFG_GETTER(kq_ple_value_config_hidden_size, c->dimensions.hidden_size, uint32_t, 0U)
CFG_GETTER(kq_ple_value_config_residual_branches, c->dimensions.residual_branches, uint32_t, 0U)
CFG_GETTER(kq_ple_value_config_head_count, c->dimensions.head_count, uint32_t, 0U)
CFG_GETTER(kq_ple_value_config_row_width, c->dimensions.row_width, uint32_t, 0U)
CFG_GETTER(kq_ple_value_config_history_length, c->dimensions.history_length, uint32_t, 0U)
CFG_GETTER(kq_ple_value_config_state_bytes, c->state_bytes, uint64_t, 0U)
CFG_GETTER(kq_ple_value_config_semantic_state_bytes, c->semantic_state_bytes, uint64_t, 0U)
CFG_GETTER(kq_ple_value_config_scratch_bytes, c->scratch_bytes, uint64_t, 0U)
CFG_GETTER(kq_ple_value_config_activation_dtype, c->dimensions.activation_dtype,
           kq_ple_value_activation_dtype, KQ_PLE_VALUE_ACTIVATION_F32)
uint64_t kq_ple_value_config_owned_bytes(const kq_ple_value_config *c) {
    return kq_ple_value_config_valid(c) ? sizeof(*c) : 0U;
}

static int state_valid(const kq_ple_value_config *config,
                       const kq_ple_value_state *state) {
    return kq_ple_value_config_valid(config) && state != NULL &&
           state->magic == KQ_PLE_VALUE_STATE_MAGIC && state->config == config &&
           state->history != NULL && state->history_count == config->state_elements;
}

kq_status kq_ple_value_state_create(const kq_ple_value_config *config,
                                    kq_ple_value_state **out_state,
                                    kq_diagnostic *diagnostic) {
    kq_ple_value_state *state;
    kq_diagnostic_clear(diagnostic);
    if (!kq_ple_value_config_valid(config) || out_state == NULL)
        return fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                    "valid PLE value config and state output are required");
    *out_state = NULL;
    if (config->dimensions.activation_dtype != KQ_PLE_VALUE_ACTIVATION_F32)
        return fail(diagnostic, KQ_STATUS_INCOMPATIBLE_PLE_VALUE,
                    "scalar PLE value state requires the F32 reference config");
    state = (kq_ple_value_state *)calloc(1U, sizeof(*state));
    if (state == NULL) return fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                                   "could not allocate PLE value state");
    state->history = (float *)calloc((size_t)config->state_elements, sizeof(float));
    if (state->history == NULL) { free(state); return fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                                                           "could not allocate PLE history"); }
    state->magic = KQ_PLE_VALUE_STATE_MAGIC; state->config = config;
    state->history_count = config->state_elements;
    *out_state = state; return KQ_STATUS_OK;
}

void kq_ple_value_state_close(kq_ple_value_state *state) {
    if (state == NULL) return;
    free(state->history); state->history = NULL; state->magic = 0U; free(state);
}

kq_status kq_ple_value_state_reset(const kq_ple_value_config *config,
                                   kq_ple_value_state *state,
                                   kq_diagnostic *diagnostic) {
    kq_diagnostic_clear(diagnostic);
    if (!state_valid(config, state)) return fail(diagnostic, KQ_STATUS_INVALID_PLE_VALUE_STATE,
                                                  "PLE value state/config mismatch");
    memset(state->history, 0, (size_t)config->state_bytes); state->position = 0U;
    return KQ_STATUS_OK;
}

kq_status kq_ple_value_state_import_f32(const kq_ple_value_config *config,
    kq_ple_value_state *state, const float *history, uint64_t history_count,
    uint64_t position, kq_diagnostic *diagnostic) {
    uint64_t i;
    kq_diagnostic_clear(diagnostic);
    if (!state_valid(config,state) || history == NULL || history_count != config->state_elements)
        return fail(diagnostic, KQ_STATUS_INVALID_PLE_VALUE_STATE,
                    "invalid PLE value state import");
    for (i=0U;i<history_count;++i) if (!isfinite(history[i]))
        return fail(diagnostic,KQ_STATUS_NUMERIC_DOMAIN,"non-finite PLE value history");
    memcpy(state->history,history,(size_t)config->state_bytes); state->position=position;
    return KQ_STATUS_OK;
}

kq_status kq_ple_value_state_export_f32(const kq_ple_value_config *config,
    const kq_ple_value_state *state, float *history, uint64_t history_capacity,
    uint64_t *history_count, uint64_t *position, kq_diagnostic *diagnostic) {
    kq_diagnostic_clear(diagnostic);
    if (history_count != NULL) *history_count = kq_ple_value_config_valid(config) ? config->state_elements : 0U;
    if (!state_valid(config,state) || history_count == NULL || position == NULL)
        return fail(diagnostic,KQ_STATUS_INVALID_PLE_VALUE_STATE,"invalid PLE value state export");
    if (history == NULL || history_capacity < config->state_elements)
        return fail(diagnostic,KQ_STATUS_BUFFER_TOO_SMALL,"PLE value history output is too small");
    memcpy(history,state->history,(size_t)config->state_bytes); *position=state->position;
    return KQ_STATUS_OK;
}

static kq_status gguf_lookup(void *user_data, uint32_t member, uint64_t row,
                             float *output, uint64_t capacity,
                             kq_diagnostic *diagnostic) {
    kq_ple_value_gguf_provider *provider = (kq_ple_value_gguf_provider *)user_data;
    kq_tensor_view *view = NULL;
    uint64_t first_block, block_count, bytes, next;
    uint64_t output_count = 0U;
    kq_status status;
    if (provider == NULL || !kq_ple_value_config_valid(provider->config) ||
        output == NULL || capacity < provider->config->dimensions.row_width ||
        member >= provider->config->dimensions.logical_member_count ||
        row >= provider->config->dimensions.member_rows)
        return fail(diagnostic,KQ_STATUS_INVALID_ARGUMENT,"invalid bounded GGUF PLE row request");
    block_count = 5U;
    if (!kq_ple_value_u64_mul(row, block_count, &first_block) ||
        !kq_ple_value_u64_mul(block_count, 18U, &bytes))
        return fail(diagnostic,KQ_STATUS_ARITHMETIC_OVERFLOW,"PLE row block geometry overflows");
    if (!kq_ple_value_u64_add(provider->metrics.logical_payload_bytes_touched, bytes, &next) ||
        next > provider->metrics.budget_bytes)
        return fail(diagnostic,KQ_STATUS_LIMIT_EXCEEDED,"PLE real payload sample budget exceeded");
    status = kq_tensor_view_open_ple_member(provider->gguf, provider->config->tables[member],
                                             KQ_TENSOR_VIEW_PHYSICAL_LAYOUT, &view, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    status = kq_dequantize_view_blocks_f32(view,first_block,block_count,
                                            KQ_NUMERIC_PHYSICAL_ORDER,output,capacity,
                                            &output_count,diagnostic);
    kq_tensor_view_close(view);
    if (status != KQ_STATUS_OK || output_count != provider->config->dimensions.row_width)
        return status != KQ_STATUS_OK ? status : fail(diagnostic,KQ_STATUS_PLE_LOOKUP_FAILED,
                                                       "decoded PLE row width mismatch");
    provider->metrics.logical_payload_bytes_touched = next;
    provider->metrics.blocks_touched += block_count; provider->metrics.rows_read += 1U;
    return KQ_STATUS_OK;
}

kq_status kq_ple_value_gguf_provider_open(const kq_gguf *gguf,
    const kq_model *model, const kq_ple_value_config *config, uint64_t budget_bytes,
    kq_ple_value_gguf_provider **out_provider, kq_diagnostic *diagnostic) {
    kq_ple_value_gguf_provider *provider;
    kq_diagnostic_clear(diagnostic);
    if (gguf==NULL || model==NULL || !kq_ple_value_config_valid(config) ||
        config->model != model || budget_bytes==0U || out_provider==NULL)
        return fail(diagnostic,KQ_STATUS_INVALID_ARGUMENT,"invalid GGUF PLE provider arguments");
    *out_provider=NULL; provider=(kq_ple_value_gguf_provider *)calloc(1U,sizeof(*provider));
    if(provider==NULL) return fail(diagnostic,KQ_STATUS_OUT_OF_MEMORY,"could not allocate GGUF PLE provider");
    provider->gguf=gguf;provider->model=model;provider->config=config;
    provider->metrics.budget_bytes=budget_bytes;*out_provider=provider;return KQ_STATUS_OK;
}

void kq_ple_value_gguf_provider_close(kq_ple_value_gguf_provider *provider){free(provider);}
kq_ple_value_lookup_provider kq_ple_value_gguf_provider_interface(kq_ple_value_gguf_provider *p){
    kq_ple_value_lookup_provider out={0}; if(p!=NULL){out.user_data=p;out.lookup_row=gguf_lookup;
    out.logical_member_count=p->config->dimensions.logical_member_count;
    out.member_rows=p->config->dimensions.member_rows;out.row_width=p->config->dimensions.row_width;}return out;}
const kq_ple_value_gguf_metrics *kq_ple_value_gguf_provider_metrics(const kq_ple_value_gguf_provider *p){return p!=NULL?&p->metrics:NULL;}

const char *kq_ple_value_checkpoint_kind_name(kq_ple_value_checkpoint_kind kind) {
    static const char *names[]={"RAW_LOOKUPS","EMBEDDING","KEY_PROJECTION","VALUE_PROJECTION",
        "KEY_NORM","QUERY_NORM","GATE_RAW","GATE_TRANSFORMED","GATED_VALUE","CONV_NORM",
        "CONV_PRE_ACTIVATION","CONV_OUTPUT","OPERATOR_OUTPUT"};
    return (unsigned)kind < sizeof(names)/sizeof(names[0]) ? names[kind] : "UNKNOWN";
}
