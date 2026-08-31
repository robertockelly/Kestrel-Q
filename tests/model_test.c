#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_model_internal.h"
#include "kq_status.h"

#define FIXTURE_TENSOR_CAPACITY 1226U
#define FIXTURE_METADATA_CAPACITY 48U
#define FIXTURE_NAME_CAPACITY 96U
#define FIXTURE_ARRAY_BYTES 1024U

typedef struct model_fixture {
    kq_model_source source;
    kq_gguf_tensor tensors[FIXTURE_TENSOR_CAPACITY];
    char tensor_names[FIXTURE_TENSOR_CAPACITY][FIXTURE_NAME_CAPACITY];
    kq_gguf_metadata metadata[FIXTURE_METADATA_CAPACITY];
    char metadata_keys[FIXTURE_METADATA_CAPACITY][FIXTURE_NAME_CAPACITY];
    unsigned char array_data[FIXTURE_ARRAY_BYTES];
    size_t array_size;
    uint64_t tensor_count;
    uint64_t metadata_count;
} model_fixture;

static int failures = 0;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures += 1;
    }
}

static const kq_gguf_metadata *fixture_find_metadata(void *context,
                                                     const char *key) {
    model_fixture *fixture = (model_fixture *)context;
    uint64_t index;
    for (index = 0U; index < fixture->metadata_count; ++index) {
        if (kq_string_view_equal_cstr(&fixture->metadata[index].key, key)) {
            return &fixture->metadata[index];
        }
    }
    return NULL;
}

static const kq_gguf_tensor *fixture_tensor_at(void *context, uint64_t index) {
    model_fixture *fixture = (model_fixture *)context;
    return index < fixture->tensor_count ? &fixture->tensors[index] : NULL;
}

static int fixture_copy_name(char destination[FIXTURE_NAME_CAPACITY],
                             kq_string_view *view,
                             const char *name) {
    size_t length = strlen(name);
    if (length >= FIXTURE_NAME_CAPACITY) {
        return 0;
    }
    memcpy(destination, name, length + 1U);
    view->data = (const unsigned char *)destination;
    view->length = (uint64_t)length;
    return 1;
}

static kq_gguf_metadata *fixture_add_metadata(model_fixture *fixture,
                                              const char *key,
                                              uint32_t value_type) {
    kq_gguf_metadata *metadata;
    uint64_t index = fixture->metadata_count;
    if (index >= FIXTURE_METADATA_CAPACITY) {
        return NULL;
    }
    metadata = &fixture->metadata[index];
    if (!fixture_copy_name(fixture->metadata_keys[index],
                           &metadata->key,
                           key)) {
        return NULL;
    }
    metadata->value_type = value_type;
    fixture->metadata_count += 1U;
    return metadata;
}

static int fixture_add_u32(model_fixture *fixture,
                           const char *key,
                           uint32_t value) {
    kq_gguf_metadata *metadata = fixture_add_metadata(
        fixture, key, KQ_GGUF_VALUE_UINT32);
    if (metadata == NULL) {
        return 0;
    }
    metadata->scalar_value = value;
    return 1;
}

static int fixture_append_array_bytes(model_fixture *fixture,
                                      const unsigned char *bytes,
                                      size_t length,
                                      kq_string_view *view) {
    if (length > FIXTURE_ARRAY_BYTES - fixture->array_size) {
        return 0;
    }
    memcpy(fixture->array_data + fixture->array_size, bytes, length);
    view->data = fixture->array_data + fixture->array_size;
    view->length = (uint64_t)length;
    fixture->array_size += length;
    return 1;
}

static int fixture_add_i32_array(model_fixture *fixture,
                                 const char *key,
                                 const int32_t *values,
                                 uint64_t count) {
    unsigned char bytes[48U * 4U];
    kq_gguf_metadata *metadata;
    uint32_t bits;
    uint64_t index;
    if (count > 48U) {
        return 0;
    }
    for (index = 0U; index < count; ++index) {
        memcpy(&bits, &values[index], sizeof(bits));
        bytes[(size_t)(index * 4U)] = (unsigned char)(bits & 0xffU);
        bytes[(size_t)(index * 4U + 1U)] =
            (unsigned char)((bits >> 8U) & 0xffU);
        bytes[(size_t)(index * 4U + 2U)] =
            (unsigned char)((bits >> 16U) & 0xffU);
        bytes[(size_t)(index * 4U + 3U)] =
            (unsigned char)((bits >> 24U) & 0xffU);
    }
    metadata = fixture_add_metadata(fixture, key, KQ_GGUF_VALUE_ARRAY);
    if (metadata == NULL) {
        return 0;
    }
    metadata->array_element_type = KQ_GGUF_VALUE_INT32;
    metadata->array_length = count;
    return fixture_append_array_bytes(fixture,
                                      bytes,
                                      (size_t)(count * 4U),
                                      &metadata->array_data);
}

static int fixture_add_u64_array(model_fixture *fixture,
                                 const char *key,
                                 const uint64_t *values,
                                 uint64_t count) {
    unsigned char bytes[16U * 8U];
    kq_gguf_metadata *metadata;
    uint64_t index;
    unsigned int byte_index;
    if (count > 16U) {
        return 0;
    }
    for (index = 0U; index < count; ++index) {
        for (byte_index = 0U; byte_index < 8U; ++byte_index) {
            bytes[(size_t)(index * 8U + byte_index)] =
                (unsigned char)((values[index] >> (byte_index * 8U)) & 0xffU);
        }
    }
    metadata = fixture_add_metadata(fixture, key, KQ_GGUF_VALUE_ARRAY);
    if (metadata == NULL) {
        return 0;
    }
    metadata->array_element_type = KQ_GGUF_VALUE_UINT64;
    metadata->array_length = count;
    return fixture_append_array_bytes(fixture,
                                      bytes,
                                      (size_t)(count * 8U),
                                      &metadata->array_data);
}

static int fixture_add_string_array_shape(model_fixture *fixture,
                                          const char *key,
                                          uint64_t count) {
    kq_gguf_metadata *metadata = fixture_add_metadata(
        fixture, key, KQ_GGUF_VALUE_ARRAY);
    if (metadata == NULL) {
        return 0;
    }
    metadata->array_element_type = KQ_GGUF_VALUE_STRING;
    metadata->array_length = count;
    return 1;
}

static kq_gguf_tensor *fixture_add_tensor(model_fixture *fixture,
                                          const char *name,
                                          uint32_t type_id,
                                          uint32_t rank,
                                          const uint64_t *dimensions) {
    kq_gguf_tensor *tensor;
    uint64_t index = fixture->tensor_count;
    uint32_t dimension;
    if (index >= FIXTURE_TENSOR_CAPACITY || rank > KQ_GGUF_MAX_DIMS) {
        return NULL;
    }
    tensor = &fixture->tensors[index];
    if (!fixture_copy_name(fixture->tensor_names[index],
                           &tensor->name,
                           name)) {
        return NULL;
    }
    tensor->rank = rank;
    tensor->type_id = type_id;
    for (dimension = 0U; dimension < rank; ++dimension) {
        tensor->dimensions[dimension] = dimensions[dimension];
    }
    fixture->tensor_count += 1U;
    return tensor;
}

static int fixture_add_tensor_1d(model_fixture *fixture,
                                 const char *name,
                                 uint32_t type_id,
                                 uint64_t d0) {
    const uint64_t dimensions[] = {d0};
    return fixture_add_tensor(fixture, name, type_id, 1U, dimensions) != NULL;
}

static int fixture_add_tensor_2d(model_fixture *fixture,
                                 const char *name,
                                 uint32_t type_id,
                                 uint64_t d0,
                                 uint64_t d1) {
    const uint64_t dimensions[] = {d0, d1};
    return fixture_add_tensor(fixture, name, type_id, 2U, dimensions) != NULL;
}

static int fixture_add_tensor_3d(model_fixture *fixture,
                                 const char *name,
                                 uint32_t type_id,
                                 uint64_t d0,
                                 uint64_t d1,
                                 uint64_t d2) {
    const uint64_t dimensions[] = {d0, d1, d2};
    return fixture_add_tensor(fixture, name, type_id, 3U, dimensions) != NULL;
}

static int fixture_name(char output[FIXTURE_NAME_CAPACITY],
                        uint32_t layer,
                        const char *leaf) {
    int written = snprintf(output,
                           FIXTURE_NAME_CAPACITY,
                           "blk.%u.%s",
                           (unsigned int)layer,
                           leaf);
    return written >= 0 && (size_t)written < FIXTURE_NAME_CAPACITY;
}

static int fixture_add_layer_1d(model_fixture *fixture,
                                uint32_t layer,
                                const char *leaf,
                                uint32_t type_id,
                                uint64_t d0) {
    char name[FIXTURE_NAME_CAPACITY];
    return fixture_name(name, layer, leaf) &&
           fixture_add_tensor_1d(fixture, name, type_id, d0);
}

static int fixture_add_layer_2d(model_fixture *fixture,
                                uint32_t layer,
                                const char *leaf,
                                uint32_t type_id,
                                uint64_t d0,
                                uint64_t d1) {
    char name[FIXTURE_NAME_CAPACITY];
    return fixture_name(name, layer, leaf) &&
           fixture_add_tensor_2d(fixture, name, type_id, d0, d1);
}

static int fixture_add_layer_3d(model_fixture *fixture,
                                uint32_t layer,
                                const char *leaf,
                                uint32_t type_id,
                                uint64_t d0,
                                uint64_t d1,
                                uint64_t d2) {
    char name[FIXTURE_NAME_CAPACITY];
    return fixture_name(name, layer, leaf) &&
           fixture_add_tensor_3d(fixture, name, type_id, d0, d1, d2);
}

static int fixture_build_metadata(model_fixture *fixture) {
    static const uint64_t multipliers[] = {
        UINT64_C(23703573157769), UINT64_C(20109073645365),
        UINT64_C(8052911324071)
    };
    static const uint64_t offsets[] = {
        0U, 20000003U, 40000026U, 60000059U,
        80000106U, 100000165U, 120000228U, 140000297U,
        160000374U, 180000455U, 200000548U, 220000655U,
        240000802U, 260000955U, 280001114U, 300001275U
    };
    static const uint64_t vocab_sizes[] = {
        20000003U, 20000023U, 20000033U, 20000047U,
        20000059U, 20000063U, 20000069U, 20000077U,
        20000081U, 20000093U, 20000107U, 20000147U,
        20000153U, 20000159U, 20000161U, 20000171U
    };
    int32_t layer_pattern[48];
    const int32_t ple_layers[] = {1};
    uint32_t layer;

    for (layer = 0U; layer < 48U; ++layer) {
        layer_pattern[layer] = ((layer + 1U) % 4U) == 0U ? 4 : 0;
    }
    return
        fixture_add_u32(fixture, "qwen4exp.block_count", 48U) &&
        fixture_add_u32(fixture, "qwen4exp.context_length", 262144U) &&
        fixture_add_u32(fixture, "qwen4exp.embedding_length", 2560U) &&
        fixture_add_u32(fixture, "qwen4exp.attention.head_count", 24U) &&
        fixture_add_u32(fixture, "qwen4exp.attention.head_count_kv", 2U) &&
        fixture_add_u32(fixture, "qwen4exp.attention.key_length", 256U) &&
        fixture_add_u32(fixture, "qwen4exp.attention.value_length", 256U) &&
        fixture_add_u32(fixture, "qwen4exp.expert_count", 512U) &&
        fixture_add_u32(fixture, "qwen4exp.expert_used_count", 10U) &&
        fixture_add_u32(fixture, "qwen4exp.expert_feed_forward_length", 640U) &&
        fixture_add_u32(fixture,
                        "qwen4exp.expert_shared_feed_forward_length", 640U) &&
        fixture_add_u32(fixture, "qwen4exp.ssm.conv_kernel", 4U) &&
        fixture_add_u32(fixture, "qwen4exp.ssm.state_size", 128U) &&
        fixture_add_u32(fixture, "qwen4exp.ssm.group_count", 16U) &&
        fixture_add_u32(fixture, "qwen4exp.ssm.time_step_rank", 48U) &&
        fixture_add_u32(fixture, "qwen4exp.ssm.inner_size", 6144U) &&
        fixture_add_u32(fixture, "qwen4exp.full_attention_interval", 4U) &&
        fixture_add_u32(fixture, "qwen4exp.rope.dimension_count", 64U) &&
        fixture_add_u32(fixture, "qwen4exp.hyper_connection.count", 4U) &&
        fixture_add_u32(fixture, "qwen4exp.hyper_connection.low_rank", 320U) &&
        fixture_add_u32(fixture,
                        "qwen4exp.attention.indexer.head_count", 4U) &&
        fixture_add_u32(fixture,
                        "qwen4exp.attention.indexer.key_length", 128U) &&
        fixture_add_u32(fixture, "qwen4exp.attention.indexer.top_k", 2048U) &&
        fixture_add_u32(fixture, "qwen4exp.ple.ngram_size", 3U) &&
        fixture_add_u32(fixture, "qwen4exp.ple.heads_per_ngram", 8U) &&
        fixture_add_u32(fixture, "qwen4exp.ple.conv_kernel", 4U) &&
        fixture_add_u32(fixture, "qwen4exp.ple.eos_token_id", 248044U) &&
        fixture_add_u32(fixture,
                        "qwen4exp.embedding_length_per_layer_input", 160U) &&
        fixture_add_string_array_shape(fixture,
                                       "tokenizer.ggml.tokens", 248320U) &&
        fixture_add_i32_array(fixture,
                              "qwen4exp.attention.compress_ratios",
                              layer_pattern, 48U) &&
        fixture_add_i32_array(fixture, "qwen4exp.ple.layers", ple_layers, 1U) &&
        fixture_add_u64_array(fixture,
                              "qwen4exp.ple.layer_multipliers",
                              multipliers, 3U) &&
        fixture_add_u64_array(fixture,
                              "qwen4exp.ple.head_offsets", offsets, 16U) &&
        fixture_add_u64_array(fixture,
                              "qwen4exp.ple.head_vocab_sizes",
                              vocab_sizes, 16U);
}

static int fixture_build_tensors(model_fixture *fixture) {
    uint32_t layer;

    if (!fixture_add_tensor_2d(fixture, "token_embd.weight",
                               KQ_GGUF_TYPE_Q8_0, 2560U, 248320U) ||
        !fixture_add_tensor_2d(fixture, "output.weight",
                               KQ_GGUF_TYPE_Q8_0, 2560U, 248320U) ||
        !fixture_add_tensor_1d(fixture, "output_hc_norm.weight",
                               KQ_GGUF_TYPE_F32, 10240U) ||
        !fixture_add_tensor_2d(fixture, "output_hc_down.weight",
                               KQ_GGUF_TYPE_Q8_0, 10240U, 320U) ||
        !fixture_add_tensor_2d(fixture, "output_hc_up.weight",
                               KQ_GGUF_TYPE_Q8_0, 320U, 10240U)) {
        return 0;
    }

    for (layer = 0U; layer < 48U; ++layer) {
        if (!fixture_add_layer_2d(fixture, layer, "hc_attn_inject.weight",
                                  KQ_GGUF_TYPE_F32, 10240U, 4U) ||
            !fixture_add_layer_1d(fixture, layer, "hc_attn_norm.weight",
                                  KQ_GGUF_TYPE_F32, 10240U) ||
            !fixture_add_layer_2d(fixture, layer, "hc_attn_down.weight",
                                  KQ_GGUF_TYPE_Q8_0, 10240U, 320U) ||
            !fixture_add_layer_2d(fixture, layer, "hc_attn_up.weight",
                                  KQ_GGUF_TYPE_Q8_0, 320U, 10240U) ||
            !fixture_add_layer_2d(fixture, layer, "hc_ffn_inject.weight",
                                  KQ_GGUF_TYPE_F32, 10240U, 4U) ||
            !fixture_add_layer_1d(fixture, layer, "hc_ffn_norm.weight",
                                  KQ_GGUF_TYPE_F32, 10240U) ||
            !fixture_add_layer_2d(fixture, layer, "hc_ffn_down.weight",
                                  KQ_GGUF_TYPE_Q8_0, 10240U, 320U) ||
            !fixture_add_layer_2d(fixture, layer, "hc_ffn_up.weight",
                                  KQ_GGUF_TYPE_Q8_0, 320U, 10240U)) {
            return 0;
        }

        if (((layer + 1U) % 4U) != 0U) {
            if (!fixture_add_layer_1d(fixture, layer, "ssm_a",
                                      KQ_GGUF_TYPE_F32, 48U) ||
                !fixture_add_layer_2d(fixture, layer, "ssm_conv1d.weight",
                                      KQ_GGUF_TYPE_F32, 4U, 10240U) ||
                !fixture_add_layer_1d(fixture, layer, "ssm_dt.bias",
                                      KQ_GGUF_TYPE_F32, 48U) ||
                !fixture_add_layer_2d(fixture, layer, "ssm_alpha.weight",
                                      KQ_GGUF_TYPE_F32, 2560U, 48U) ||
                !fixture_add_layer_2d(fixture, layer, "ssm_beta.weight",
                                      KQ_GGUF_TYPE_F32, 2560U, 48U) ||
                !fixture_add_layer_2d(fixture, layer, "attn_qkv.weight",
                                      KQ_GGUF_TYPE_Q8_0, 2560U, 10240U) ||
                !fixture_add_layer_2d(fixture, layer, "attn_gate.weight",
                                      KQ_GGUF_TYPE_Q8_0, 2560U, 6144U) ||
                !fixture_add_layer_1d(fixture, layer, "ssm_norm.weight",
                                      KQ_GGUF_TYPE_F32, 128U) ||
                !fixture_add_layer_2d(fixture, layer, "ssm_out.weight",
                                      KQ_GGUF_TYPE_Q8_0, 6144U, 2560U)) {
                return 0;
            }
        } else {
            if (!fixture_add_layer_1d(fixture, layer, "attn_k_norm.weight",
                                      KQ_GGUF_TYPE_F32, 256U) ||
                !fixture_add_layer_2d(fixture, layer, "attn_k.weight",
                                      KQ_GGUF_TYPE_Q8_0, 2560U, 512U) ||
                !fixture_add_layer_2d(fixture, layer, "attn_output.weight",
                                      KQ_GGUF_TYPE_Q8_0, 6144U, 2560U) ||
                !fixture_add_layer_1d(fixture, layer, "attn_q_norm.weight",
                                      KQ_GGUF_TYPE_F32, 256U) ||
                !fixture_add_layer_2d(fixture, layer, "attn_q.weight",
                                      KQ_GGUF_TYPE_Q8_0, 2560U, 12288U) ||
                !fixture_add_layer_2d(fixture, layer, "attn_v.weight",
                                      KQ_GGUF_TYPE_Q8_0, 2560U, 512U) ||
                !fixture_add_layer_1d(fixture, layer, "indexer.k_norm.weight",
                                      KQ_GGUF_TYPE_F32, 128U) ||
                !fixture_add_layer_1d(fixture, layer, "indexer.q_norm.weight",
                                      KQ_GGUF_TYPE_F32, 128U) ||
                !fixture_add_layer_2d(fixture, layer, "indexer.q_proj.weight",
                                      KQ_GGUF_TYPE_BF16, 2560U, 512U) ||
                !fixture_add_layer_2d(fixture, layer, "indexer.k_proj.weight",
                                      KQ_GGUF_TYPE_BF16, 2560U, 128U)) {
                return 0;
            }
        }

        if (!fixture_add_layer_2d(fixture, layer, "ffn_gate_inp.weight",
                                  KQ_GGUF_TYPE_F32, 2560U, 512U) ||
            !fixture_add_layer_3d(fixture, layer, "ffn_down_exps.weight",
                                  KQ_GGUF_TYPE_Q5_1, 640U, 2560U, 512U) ||
            !fixture_add_layer_3d(fixture, layer, "ffn_gate_exps.weight",
                                  KQ_GGUF_TYPE_Q4_K, 2560U, 640U, 512U) ||
            !fixture_add_layer_3d(fixture, layer, "ffn_up_exps.weight",
                                  KQ_GGUF_TYPE_Q4_K, 2560U, 640U, 512U) ||
            !fixture_add_layer_2d(fixture, layer, "ffn_down_shexp.weight",
                                  KQ_GGUF_TYPE_Q8_0, 640U, 2560U) ||
            !fixture_add_layer_2d(fixture, layer, "ffn_gate_shexp.weight",
                                  KQ_GGUF_TYPE_Q8_0, 2560U, 640U) ||
            !fixture_add_layer_2d(fixture, layer, "ffn_up_shexp.weight",
                                  KQ_GGUF_TYPE_Q8_0, 2560U, 640U) ||
            !fixture_add_layer_1d(fixture, layer, "ffn_gate_inp_shexp.weight",
                                  KQ_GGUF_TYPE_F32, 2560U)) {
            return 0;
        }
    }

    return fixture_add_tensor_2d(fixture, "per_layer_token_embd.weight",
                                 KQ_GGUF_TYPE_IQ4_NL, 160U, 320001536U) &&
           fixture_add_layer_2d(fixture, 1U, "ple_conv1d.weight",
                                KQ_GGUF_TYPE_F32, 4U, 10240U) &&
           fixture_add_layer_2d(fixture, 1U, "ple_key.weight",
                                KQ_GGUF_TYPE_Q8_0, 2560U, 10240U) &&
           fixture_add_layer_1d(fixture, 1U, "ple_norm_conv.weight",
                                KQ_GGUF_TYPE_F32, 10240U) &&
           fixture_add_layer_1d(fixture, 1U, "ple_norm_key.weight",
                                KQ_GGUF_TYPE_F32, 10240U) &&
           fixture_add_layer_1d(fixture, 1U, "ple_norm_query.weight",
                                KQ_GGUF_TYPE_F32, 10240U) &&
           fixture_add_layer_2d(fixture, 1U, "ple_value.weight",
                                KQ_GGUF_TYPE_Q8_0, 2560U, 2560U);
}

static int fixture_build(model_fixture *fixture) {
    static const unsigned char architecture[] = "qwen4exp";
    memset(fixture, 0, sizeof(*fixture));
    if (!fixture_build_metadata(fixture) || !fixture_build_tensors(fixture) ||
        fixture->tensor_count != 1224U) {
        return 0;
    }
    fixture->source.context = fixture;
    fixture->source.architecture.data = architecture;
    fixture->source.architecture.length = sizeof(architecture) - 1U;
    fixture->source.metadata_count = fixture->metadata_count;
    fixture->source.find_metadata = fixture_find_metadata;
    fixture->source.tensor_count = fixture->tensor_count;
    fixture->source.tensor_at = fixture_tensor_at;
    fixture->source.payload_bytes_accessed = 0U;
    return 1;
}

static kq_gguf_tensor *fixture_find_tensor(model_fixture *fixture,
                                           const char *name) {
    uint64_t index;
    for (index = 0U; index < fixture->tensor_count; ++index) {
        if (kq_string_view_equal_cstr(&fixture->tensors[index].name, name)) {
            return &fixture->tensors[index];
        }
    }
    return NULL;
}

static int fixture_rename_tensor(model_fixture *fixture,
                                 const char *old_name,
                                 const char *new_name) {
    kq_gguf_tensor *tensor = fixture_find_tensor(fixture, old_name);
    uint64_t index;
    if (tensor == NULL) {
        return 0;
    }
    index = (uint64_t)(tensor - fixture->tensors);
    return fixture_copy_name(fixture->tensor_names[index],
                             &tensor->name,
                             new_name);
}

static int fixture_rename_metadata(model_fixture *fixture,
                                   const char *old_key,
                                   const char *new_key) {
    const kq_gguf_metadata *found = fixture_find_metadata(fixture, old_key);
    kq_gguf_metadata *metadata = (kq_gguf_metadata *)found;
    uint64_t index;
    if (metadata == NULL) {
        return 0;
    }
    index = (uint64_t)(metadata - fixture->metadata);
    return fixture_copy_name(fixture->metadata_keys[index],
                             &metadata->key,
                             new_key);
}

static void expect_failure(const char *name,
                           model_fixture *fixture,
                           kq_status expected_status) {
    kq_model *model = NULL;
    kq_diagnostic diagnostic;
    kq_status status = kq_model_open_from_source(&fixture->source,
                                                 &model,
                                                 &diagnostic);
    if (status != expected_status) {
        fprintf(stderr,
                "FAIL: %s expected %s, received %s (%s)\n",
                name,
                kq_status_string(expected_status),
                kq_status_string(status),
                diagnostic.message);
        failures += 1;
    }
    check(model == NULL, "failed registry construction must return NULL");
    kq_model_close(model);
}

static int rebuild(model_fixture *fixture, const char *name) {
    if (!fixture_build(fixture)) {
        fprintf(stderr, "FAIL: could not rebuild fixture for %s\n", name);
        failures += 1;
        return 0;
    }
    return 1;
}

static void test_valid_registry(model_fixture *fixture) {
    const kq_semantic_tensor *semantic;
    const kq_semantic_tensor *other;
    kq_diagnostic diagnostic;
    kq_model *model = NULL;
    kq_status status = kq_model_open_from_source(&fixture->source,
                                                 &model,
                                                 &diagnostic);
    uint64_t left;
    uint64_t right;
    if (status != KQ_STATUS_OK) {
        fprintf(stderr,
                "FAIL: valid semantic fixture failed: %s (%s)\n",
                kq_status_string(status),
                diagnostic.message);
        failures += 1;
        return;
    }
    check(kq_model_semantic_tensor_count(model) == 1294U,
          "valid registry must contain 1294 semantic entries");
    check(kq_model_physical_tensor_count(model) == 1224U &&
              kq_model_physical_coverage_count(model) == 1224U,
          "valid registry must cover 1224 physical tensors");
    check(kq_model_metadata_derived_count(model) == 3U,
          "valid registry must contain three metadata-derived semantics");
    check(kq_model_unknown_physical_count(model) == 0U &&
              kq_model_unbound_required_count(model) == 0U,
          "valid registry must have no unknown or unbound entries");
    check(kq_model_layer_count(model) == 48U &&
              kq_model_gdn_layer_count(model) == 36U &&
              kq_model_qsa_layer_count(model) == 12U,
          "valid registry must expose 48/36/12 topology");
    check(kq_model_expert_count(model) == 512U &&
              kq_model_expert_top_k(model) == 10U,
          "valid registry must expose expert count and top-k");
    check(kq_model_relation_count(model,
                                  KQ_BINDING_RENAMED_ONE_TO_ONE) == 847U &&
              kq_model_relation_count(model,
                                      KQ_BINDING_TRANSFORMED_LAYOUT) == 256U &&
              kq_model_relation_count(
                  model,
                  KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL) == 60U &&
              kq_model_relation_count(
                  model,
                  KQ_BINDING_MULTIPLE_CANONICAL_TO_ONE_PHYSICAL) == 128U &&
              kq_model_relation_count(model,
                                      KQ_BINDING_METADATA_DERIVED) == 3U,
          "binding relation counts must match the canonical reconciliation");
    check(kq_model_placement_count(
              model, KQ_PLACEMENT_ALWAYS_NEEDED_CANDIDATE) == 1061U &&
              kq_model_placement_count(
                  model, KQ_PLACEMENT_ROUTED_EXPERT_CACHE_CANDIDATE) == 96U &&
              kq_model_placement_count(
                  model, KQ_PLACEMENT_PLE_DISK_BACKED_CANDIDATE) == 137U,
          "placement annotations must match semantic families");
    check(kq_model_layer_type_at(model, 3U) == KQ_MODEL_LAYER_QSA &&
              kq_model_layer_type_at(model, 48U) == KQ_MODEL_LAYER_INVALID,
          "layer topology lookup must reject impossible IDs");
    check(strcmp(kq_binding_relation_name(KQ_BINDING_DIRECT_ONE_TO_ONE),
                 "DIRECT_ONE_TO_ONE") == 0,
          "direct one-to-one relation must remain an explicit API relation");
    for (left = 0U; left < kq_model_semantic_tensor_count(model); ++left) {
        semantic = kq_model_semantic_tensor_at(model, left);
        for (right = left + 1U;
             right < kq_model_semantic_tensor_count(model);
             ++right) {
            other = kq_model_semantic_tensor_at(model, right);
            if (strcmp(semantic->semantic_id, other->semantic_id) == 0) {
                check(0, "semantic identifiers must be globally unique");
                left = kq_model_semantic_tensor_count(model);
                break;
            }
        }
    }

    semantic = kq_model_find_semantic_tensor(
        model, "layer.03.qsa.indexer.qk");
    check(semantic != NULL && semantic->binding_count == 2U &&
              semantic->bindings[0].part_role ==
                  KQ_BINDING_PART_INDEX_QUERY &&
              semantic->bindings[1].part_role == KQ_BINDING_PART_INDEX_KEY,
          "QSA index q/k split must preserve deterministic part order");
    semantic = kq_model_find_semantic_tensor(
        model, "layer.00.moe.routed.gate_up");
    check(semantic != NULL && semantic->canonical_expert_axis == 0U &&
              semantic->expert_count == 512U &&
              semantic->bindings[0].physical_expert_axis == 2U &&
              semantic->bindings[1].physical_expert_axis == 2U,
          "routed expert stacks must preserve canonical and physical axes");
    semantic = kq_model_find_semantic_tensor(
        model, "layer.01.ple.table.127");
    check(semantic != NULL && semantic->binding_count == 1U &&
              semantic->bindings[0].fused_member_index == 127U &&
              semantic->bindings[0].fused_member_count == 128U,
          "PLE fusion must preserve all logical member identities");
    semantic = kq_model_find_semantic_tensor(
        model, "layer.01.ple.address.head_offsets");
    check(semantic != NULL && semantic->bindings[0].physical == NULL &&
              semantic->bindings[0].metadata != NULL,
          "PLE addresses must bind to metadata without a payload tensor");
    check(kq_model_find_semantic_tensor(model, "unknown") == NULL,
          "unknown semantic lookup must return NULL");
    kq_model_close(model);
}

int main(void) {
    static const unsigned char bad_architecture[] = "qwen4exq";
    model_fixture fixture;
    kq_gguf_metadata *metadata;
    kq_gguf_tensor *tensor;
    int32_t zero = 0;

    if (!fixture_build(&fixture)) {
        fprintf(stderr, "could not construct complete semantic fixture\n");
        return 1;
    }
    test_valid_registry(&fixture);

    if (rebuild(&fixture, "wrong architecture")) {
        fixture.source.architecture.data = bad_architecture;
        fixture.source.architecture.length = sizeof(bad_architecture) - 1U;
        expect_failure("wrong architecture", &fixture,
                       KQ_STATUS_UNSUPPORTED_MODEL);
    }
    if (rebuild(&fixture, "wrong hidden size")) {
        metadata = (kq_gguf_metadata *)fixture_find_metadata(
            &fixture, "qwen4exp.embedding_length");
        metadata->scalar_value = 2559U;
        expect_failure("wrong hidden size", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "wrong vocabulary")) {
        metadata = (kq_gguf_metadata *)fixture_find_metadata(
            &fixture, "tokenizer.ggml.tokens");
        metadata->array_length = 248319U;
        expect_failure("wrong vocabulary", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "wrong layer count")) {
        metadata = (kq_gguf_metadata *)fixture_find_metadata(
            &fixture, "qwen4exp.block_count");
        metadata->scalar_value = 47U;
        expect_failure("wrong layer count", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "wrong expert count")) {
        metadata = (kq_gguf_metadata *)fixture_find_metadata(
            &fixture, "qwen4exp.expert_count");
        metadata->scalar_value = 511U;
        expect_failure("wrong expert count", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "wrong top-k")) {
        metadata = (kq_gguf_metadata *)fixture_find_metadata(
            &fixture, "qwen4exp.expert_used_count");
        metadata->scalar_value = 9U;
        expect_failure("wrong expert top-k", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "wrong layer topology")) {
        metadata = (kq_gguf_metadata *)fixture_find_metadata(
            &fixture, "qwen4exp.attention.compress_ratios");
        memcpy((unsigned char *)metadata->array_data.data + 3U * 4U,
               &zero,
               sizeof(zero));
        expect_failure("wrong layer topology", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "missing required tensor") &&
        fixture_rename_tensor(&fixture, "token_embd.weight", "missing.weight")) {
        expect_failure("missing required tensor", &fixture,
                       KQ_STATUS_SEMANTIC_MAPPING_FAILED);
    }
    if (rebuild(&fixture, "missing LM head") &&
        fixture_rename_tensor(&fixture, "output.weight", "missing_output.weight")) {
        expect_failure("missing LM head", &fixture,
                       KQ_STATUS_SEMANTIC_MAPPING_FAILED);
    }
    if (rebuild(&fixture, "missing final norm") &&
        fixture_rename_tensor(&fixture, "output_hc_norm.weight",
                              "missing_output_hc_norm.weight")) {
        expect_failure("missing final norm", &fixture,
                       KQ_STATUS_SEMANTIC_MAPPING_FAILED);
    }
    if (rebuild(&fixture, "missing layer semantic") &&
        fixture_rename_tensor(&fixture, "blk.17.hc_attn_norm.weight",
                              "blk.17.missing_hc_attn_norm.weight")) {
        expect_failure("missing layer semantic", &fixture,
                       KQ_STATUS_SEMANTIC_MAPPING_FAILED);
    }
    if (rebuild(&fixture, "unknown physical tensor") &&
        fixture_add_tensor_1d(&fixture, "unknown.weight",
                              KQ_GGUF_TYPE_F32, 1U)) {
        fixture.source.tensor_count = fixture.tensor_count;
        expect_failure("unknown physical tensor", &fixture,
                       KQ_STATUS_UNKNOWN_PHYSICAL_TENSOR);
    }
    if (rebuild(&fixture, "ambiguous physical tensor") &&
        fixture_rename_tensor(&fixture, "output.weight", "token_embd.weight")) {
        expect_failure("ambiguous physical tensor", &fixture,
                       KQ_STATUS_SEMANTIC_MAPPING_FAILED);
    }
    if (rebuild(&fixture, "wrong rank")) {
        tensor = fixture_find_tensor(&fixture, "token_embd.weight");
        tensor->rank = 1U;
        expect_failure("wrong rank", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "wrong LM head shape")) {
        tensor = fixture_find_tensor(&fixture, "output.weight");
        tensor->dimensions[1] = 248319U;
        expect_failure("wrong LM head shape", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "wrong embedding type")) {
        tensor = fixture_find_tensor(&fixture, "token_embd.weight");
        tensor->type_id = KQ_GGUF_TYPE_F32;
        expect_failure("wrong embedding type", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "wrong expert axis")) {
        tensor = fixture_find_tensor(&fixture, "blk.0.ffn_down_exps.weight");
        tensor->dimensions[2] = 511U;
        expect_failure("wrong expert axis", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "missing split part") &&
        fixture_rename_tensor(&fixture, "blk.0.ffn_up_exps.weight",
                              "blk.0.ffn_up_missing.weight")) {
        expect_failure("missing split part", &fixture,
                       KQ_STATUS_SEMANTIC_MAPPING_FAILED);
    }
    if (rebuild(&fixture, "duplicate split part") &&
        fixture_rename_tensor(&fixture, "blk.0.ffn_up_exps.weight",
                              "blk.0.ffn_gate_exps.weight")) {
        expect_failure("duplicate split part", &fixture,
                       KQ_STATUS_SEMANTIC_MAPPING_FAILED);
    }
    if (rebuild(&fixture, "invalid split geometry")) {
        tensor = fixture_find_tensor(&fixture, "blk.0.ffn_up_exps.weight");
        tensor->dimensions[1] = 641U;
        expect_failure("invalid split geometry", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "invalid PLE fusion")) {
        tensor = fixture_find_tensor(&fixture, "per_layer_token_embd.weight");
        tensor->dimensions[1] -= 1U;
        expect_failure("invalid PLE fusion", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "missing PLE metadata") &&
        fixture_rename_metadata(&fixture, "qwen4exp.ple.head_offsets",
                                "qwen4exp.ple.missing_offsets")) {
        expect_failure("missing PLE metadata", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "impossible layer ID") &&
        fixture_rename_tensor(&fixture, "blk.47.attn_q.weight",
                              "blk.48.attn_q.weight")) {
        expect_failure("impossible layer ID", &fixture,
                       KQ_STATUS_SEMANTIC_MAPPING_FAILED);
    }
    if (rebuild(&fixture, "incompatible physical type")) {
        tensor = fixture_find_tensor(&fixture, "output_hc_norm.weight");
        tensor->type_id = KQ_GGUF_TYPE_Q8_0;
        expect_failure("incompatible physical type", &fixture,
                       KQ_STATUS_MODEL_TOPOLOGY_MISMATCH);
    }
    if (rebuild(&fixture, "payload boundary")) {
        fixture.source.payload_bytes_accessed = 1U;
        expect_failure("payload boundary", &fixture,
                       KQ_STATUS_SEMANTIC_MAPPING_FAILED);
    }

    if (failures != 0) {
        fprintf(stderr, "%d semantic registry assertion(s) failed\n", failures);
        return 1;
    }
    printf("semantic registry synthetic/fail-closed tests: PASS\n");
    return 0;
}
