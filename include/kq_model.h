#ifndef KQ_MODEL_H
#define KQ_MODEL_H

#include <stdint.h>

#include "kq_gguf.h"
#include "kq_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KQ_SEMANTIC_ID_CAPACITY 96U
#define KQ_CANONICAL_NAME_CAPACITY 160U
#define KQ_SEMANTIC_MAX_DIMS 4U
#define KQ_SEMANTIC_MAX_BINDINGS 2U
#define KQ_MODEL_NO_LAYER UINT32_MAX
#define KQ_MODEL_NO_AXIS UINT32_MAX

typedef enum kq_model_layer_type {
    KQ_MODEL_LAYER_INVALID = 0,
    KQ_MODEL_LAYER_GDN,
    KQ_MODEL_LAYER_QSA
} kq_model_layer_type;

typedef enum kq_semantic_component {
    KQ_COMPONENT_TOKEN_EMBEDDING = 0,
    KQ_COMPONENT_LM_HEAD,
    KQ_COMPONENT_FINAL_GATED_RESIDUAL,
    KQ_COMPONENT_GATED_RESIDUAL,
    KQ_COMPONENT_GDN,
    KQ_COMPONENT_QSA_ATTENTION,
    KQ_COMPONENT_QSA_INDEXER,
    KQ_COMPONENT_MOE_ROUTER,
    KQ_COMPONENT_ROUTED_EXPERT_STACK,
    KQ_COMPONENT_SHARED_EXPERT,
    KQ_COMPONENT_PLE_TABLE,
    KQ_COMPONENT_PLE_DENSE,
    KQ_COMPONENT_PLE_ADDRESS_METADATA,
    KQ_COMPONENT_COUNT
} kq_semantic_component;

typedef enum kq_semantic_role {
    KQ_ROLE_TOKEN_EMBEDDING = 0,
    KQ_ROLE_LM_HEAD,
    KQ_ROLE_HC_NORM,
    KQ_ROLE_HC_DOWN,
    KQ_ROLE_HC_UP,
    KQ_ROLE_HC_INJECT,
    KQ_ROLE_GDN_A_LOG,
    KQ_ROLE_GDN_CONV,
    KQ_ROLE_GDN_DT_BIAS,
    KQ_ROLE_GDN_ALPHA,
    KQ_ROLE_GDN_BETA,
    KQ_ROLE_GDN_QKV,
    KQ_ROLE_GDN_GATE,
    KQ_ROLE_GDN_NORM,
    KQ_ROLE_GDN_OUT,
    KQ_ROLE_QSA_K_NORM,
    KQ_ROLE_QSA_K,
    KQ_ROLE_QSA_OUTPUT,
    KQ_ROLE_QSA_Q_NORM,
    KQ_ROLE_QSA_Q,
    KQ_ROLE_QSA_V,
    KQ_ROLE_QSA_INDEX_QK,
    KQ_ROLE_QSA_INDEX_K_NORM,
    KQ_ROLE_QSA_INDEX_Q_NORM,
    KQ_ROLE_MOE_ROUTER,
    KQ_ROLE_ROUTED_DOWN,
    KQ_ROLE_ROUTED_GATE_UP,
    KQ_ROLE_SHARED_DOWN,
    KQ_ROLE_SHARED_GATE,
    KQ_ROLE_SHARED_UP,
    KQ_ROLE_SHARED_GATE_WEIGHT,
    KQ_ROLE_PLE_TABLE,
    KQ_ROLE_PLE_CONV,
    KQ_ROLE_PLE_KEY,
    KQ_ROLE_PLE_NORM_CONV,
    KQ_ROLE_PLE_NORM_KEY,
    KQ_ROLE_PLE_NORM_QUERY,
    KQ_ROLE_PLE_VALUE,
    KQ_ROLE_PLE_LAYER_MULTIPLIERS,
    KQ_ROLE_PLE_HEAD_OFFSETS,
    KQ_ROLE_PLE_HEAD_VOCAB_SIZES
} kq_semantic_role;

typedef enum kq_binding_relation {
    KQ_BINDING_DIRECT_ONE_TO_ONE = 0,
    KQ_BINDING_RENAMED_ONE_TO_ONE,
    KQ_BINDING_TRANSFORMED_LAYOUT,
    KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL,
    KQ_BINDING_MULTIPLE_CANONICAL_TO_ONE_PHYSICAL,
    KQ_BINDING_METADATA_DERIVED,
    KQ_BINDING_ABSENT_INITIAL_SCOPE,
    KQ_BINDING_RELATION_COUNT
} kq_binding_relation;

typedef enum kq_binding_part_role {
    KQ_BINDING_PART_WHOLE = 0,
    KQ_BINDING_PART_GATE,
    KQ_BINDING_PART_UP,
    KQ_BINDING_PART_INDEX_QUERY,
    KQ_BINDING_PART_INDEX_KEY,
    KQ_BINDING_PART_FUSED_MEMBER,
    KQ_BINDING_PART_METADATA
} kq_binding_part_role;

typedef enum kq_runtime_scope {
    KQ_SCOPE_REQUIRED_INITIAL_TEXT = 0,
    KQ_SCOPE_EXCLUDED_INITIAL_VISION,
    KQ_SCOPE_EXCLUDED_INITIAL_MTP
} kq_runtime_scope;

typedef enum kq_placement_hint {
    KQ_PLACEMENT_ALWAYS_NEEDED_CANDIDATE = 0,
    KQ_PLACEMENT_ROUTED_EXPERT_CACHE_CANDIDATE,
    KQ_PLACEMENT_PLE_DISK_BACKED_CANDIDATE,
    KQ_PLACEMENT_EXCLUDED_INITIAL_SCOPE,
    KQ_PLACEMENT_NEUTRAL,
    KQ_PLACEMENT_HINT_COUNT
} kq_placement_hint;

typedef enum kq_canonical_dtype {
    KQ_CANONICAL_DTYPE_BF16 = 0,
    KQ_CANONICAL_DTYPE_I64
} kq_canonical_dtype;

typedef struct kq_tensor_binding {
    const kq_gguf_tensor *physical;
    const kq_gguf_metadata *metadata;
    uint64_t physical_index;
    kq_binding_part_role part_role;
    uint32_t part_index;
    uint32_t part_count;
    uint32_t fused_member_index;
    uint32_t fused_member_count;
    uint32_t physical_expert_axis;
} kq_tensor_binding;

typedef struct kq_semantic_tensor {
    char semantic_id[KQ_SEMANTIC_ID_CAPACITY];
    char canonical_name[KQ_CANONICAL_NAME_CAPACITY];
    kq_semantic_component component;
    kq_semantic_role role;
    kq_binding_relation relation;
    kq_runtime_scope runtime_scope;
    kq_placement_hint placement_hint;
    kq_model_layer_type layer_type;
    uint32_t layer_id;
    kq_canonical_dtype canonical_dtype;
    uint32_t canonical_rank;
    uint64_t canonical_dimensions[KQ_SEMANTIC_MAX_DIMS];
    uint32_t canonical_expert_axis;
    uint32_t expert_count;
    uint32_t binding_count;
    kq_tensor_binding bindings[KQ_SEMANTIC_MAX_BINDINGS];
} kq_semantic_tensor;

typedef struct kq_model kq_model;

kq_status kq_model_open_from_gguf(const kq_gguf *gguf,
                                  kq_model **out_model,
                                  kq_diagnostic *diagnostic);
void kq_model_close(kq_model *model);

uint32_t kq_model_hidden_size(const kq_model *model);
uint32_t kq_model_vocabulary_size(const kq_model *model);
uint32_t kq_model_context_length(const kq_model *model);
uint32_t kq_model_layer_count(const kq_model *model);
uint32_t kq_model_gdn_layer_count(const kq_model *model);
uint32_t kq_model_qsa_layer_count(const kq_model *model);
uint32_t kq_model_expert_count(const kq_model *model);
uint32_t kq_model_expert_top_k(const kq_model *model);
kq_model_layer_type kq_model_layer_type_at(const kq_model *model,
                                           uint32_t layer_id);

uint64_t kq_model_semantic_tensor_count(const kq_model *model);
uint64_t kq_model_physical_tensor_count(const kq_model *model);
uint64_t kq_model_physical_coverage_count(const kq_model *model);
uint64_t kq_model_metadata_derived_count(const kq_model *model);
uint64_t kq_model_unknown_physical_count(const kq_model *model);
uint64_t kq_model_unbound_required_count(const kq_model *model);
uint64_t kq_model_relation_count(const kq_model *model,
                                 kq_binding_relation relation);
uint64_t kq_model_component_count(const kq_model *model,
                                  kq_semantic_component component);
uint64_t kq_model_placement_count(const kq_model *model,
                                  kq_placement_hint placement);

const kq_semantic_tensor *kq_model_semantic_tensor_at(const kq_model *model,
                                                      uint64_t index);
const kq_semantic_tensor *kq_model_find_semantic_tensor(
    const kq_model *model,
    const char *semantic_id);

const char *kq_model_layer_type_name(kq_model_layer_type type);
const char *kq_semantic_component_name(kq_semantic_component component);
const char *kq_semantic_role_name(kq_semantic_role role);
const char *kq_binding_relation_name(kq_binding_relation relation);
const char *kq_binding_part_role_name(kq_binding_part_role role);
const char *kq_runtime_scope_name(kq_runtime_scope scope);
const char *kq_placement_hint_name(kq_placement_hint placement);

#ifdef __cplusplus
}
#endif

#endif
