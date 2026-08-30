#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_ple_value.h"
#include "kq_ple_value_internal.h"

typedef struct synthetic_provider {
    uint64_t call_count;
    uint64_t fail_at;
    uint32_t row_width;
} synthetic_provider;

static int failures = 0;
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); ++failures; } } while (0)

static kq_status lookup(void *user, uint32_t member, uint64_t row,
                        float *output, uint64_t capacity,
                        kq_diagnostic *diagnostic) {
    synthetic_provider *provider = (synthetic_provider *)user;
    uint32_t i;
    if (provider == NULL || output == NULL || capacity < provider->row_width)
        return KQ_STATUS_INVALID_ARGUMENT;
    if (provider->call_count == provider->fail_at) {
        (void)diagnostic;
        return KQ_STATUS_PLE_LOOKUP_FAILED;
    }
    for (i=0U;i<provider->row_width;++i)
        output[i]=(float)((member+1U)*3U+(uint32_t)(row%17U)+i)*0.03125f;
    provider->call_count+=1U;return KQ_STATUS_OK;
}

static void fill_intents(kq_ple_address_intent *intents,uint64_t tokens,
                         uint32_t heads_per,uint32_t members,uint64_t rows) {
    uint64_t t;uint32_t h;for(t=0U;t<tokens;++t)for(h=0U;h<heads_per*2U;++h){
        kq_ple_address_intent *in=&intents[t*heads_per*2U+h];memset(in,0,sizeof(*in));
        in->position=t;in->ngram_order=2U+h/heads_per;in->local_head=h%heads_per;
        in->global_head=h;in->logical_member=h%members;in->member_row=(t*13U+h)%rows;
    }
}

static kq_ple_value_config *make_config(void) {
    kq_ple_value_dimensions d={8U,2U,2U,4U,2U,3U,64U,4U,3U,9U,1.0e-6f,KQ_PLE_VALUE_ACTIVATION_F32};
    kq_ple_value_config *config=NULL;kq_diagnostic diagnostic;
    CHECK(kq_ple_value_test_config_create(&d,&config,&diagnostic)==KQ_STATUS_OK);
    return config;
}

static void fill_weights(kq_ple_value_weights_f32 *w,float **owned) {
    uint64_t counts[6]={16U*8U,8U*8U,16U,16U,16U,16U*4U};uint32_t i;uint64_t j;
    for(i=0U;i<6U;++i){owned[i]=(float *)calloc((size_t)counts[i],sizeof(float));CHECK(owned[i]!=NULL);}
    for(j=0U;j<8U;++j){owned[1][j*8U+j]=1.0f;}
    for(j=0U;j<16U;++j)owned[5][j*4U+3U]=0.25f;
    w->key_projection=owned[0];w->key_projection_count=counts[0];
    w->value_projection=owned[1];w->value_projection_count=counts[1];
    w->norm_key=owned[2];w->norm_key_count=counts[2];
    w->norm_query=owned[3];w->norm_query_count=counts[3];
    w->norm_conv=owned[4];w->norm_conv_count=counts[4];
    w->convolution=owned[5];w->convolution_count=counts[5];
}

static void fill_target_source(kq_ple_value_semantic_source *source,
                               kq_semantic_tensor dense[6],
                               kq_gguf_tensor dense_physical[6],
                               kq_semantic_tensor tables[128],
                               kq_gguf_tensor *table_physical) {
    static const char *ids[6]={"layer.01.ple.key","layer.01.ple.value",
        "layer.01.ple.norm_key","layer.01.ple.norm_query",
        "layer.01.ple.norm_conv","layer.01.ple.conv"};
    static const kq_semantic_role roles[6]={KQ_ROLE_PLE_KEY,KQ_ROLE_PLE_VALUE,
        KQ_ROLE_PLE_NORM_KEY,KQ_ROLE_PLE_NORM_QUERY,KQ_ROLE_PLE_NORM_CONV,KQ_ROLE_PLE_CONV};
    static const uint32_t ranks[6]={2U,2U,1U,1U,1U,3U};
    static const uint64_t canonical[6][3]={{10240U,2560U,0U},{2560U,2560U,0U},
        {10240U,0U,0U},{10240U,0U,0U},{10240U,0U,0U},{10240U,1U,4U}};
    static const uint32_t physical_ranks[6]={2U,2U,1U,1U,1U,2U};
    static const uint64_t physical_dims[6][3]={{2560U,10240U,0U},{2560U,2560U,0U},
        {10240U,0U,0U},{10240U,0U,0U},{10240U,0U,0U},{4U,10240U,0U}};
    uint32_t i,j;
    memset(source,0,sizeof(*source));memset(dense,0,6U*sizeof(*dense));
    memset(dense_physical,0,6U*sizeof(*dense_physical));memset(tables,0,128U*sizeof(*tables));
    memset(table_physical,0,sizeof(*table_physical));
    source->model=(const kq_model *)(uintptr_t)1U;source->address_config=(const kq_ple_config *)(uintptr_t)1U;
    source->hidden_size=2560U;source->layer_type=KQ_MODEL_LAYER_GDN;
    source->address_info.ple_layer_id=1U;source->address_info.head_count=16U;
    source->address_info.logical_member_count=128U;source->address_info.member_rows=UINT64_C(2500012);
    for(i=0U;i<6U;++i){
        (void)snprintf(dense[i].semantic_id,sizeof(dense[i].semantic_id),"%s",ids[i]);
        dense[i].component=KQ_COMPONENT_PLE_DENSE;dense[i].role=roles[i];
        dense[i].relation=i<2U?KQ_BINDING_RENAMED_ONE_TO_ONE:KQ_BINDING_TRANSFORMED_LAYOUT;
        dense[i].runtime_scope=KQ_SCOPE_REQUIRED_INITIAL_TEXT;dense[i].layer_type=KQ_MODEL_LAYER_GDN;
        dense[i].layer_id=1U;dense[i].canonical_dtype=KQ_CANONICAL_DTYPE_BF16;
        dense[i].canonical_rank=ranks[i];dense[i].binding_count=1U;
        for(j=0U;j<ranks[i];++j)dense[i].canonical_dimensions[j]=canonical[i][j];
        dense_physical[i].type_id=i<2U?KQ_GGUF_TYPE_Q8_0:KQ_GGUF_TYPE_F32;
        dense_physical[i].rank=physical_ranks[i];
        for(j=0U;j<physical_ranks[i];++j)dense_physical[i].dimensions[j]=physical_dims[i][j];
        dense[i].bindings[0].physical=&dense_physical[i];dense[i].bindings[0].part_role=KQ_BINDING_PART_WHOLE;
        dense[i].bindings[0].part_count=1U;source->dense[i]=&dense[i];
    }
    table_physical->type_id=KQ_GGUF_TYPE_IQ4_NL;table_physical->rank=2U;
    table_physical->dimensions[0]=160U;table_physical->dimensions[1]=UINT64_C(320001536);
    for(i=0U;i<128U;++i){
        (void)snprintf(tables[i].semantic_id,sizeof(tables[i].semantic_id),"layer.01.ple.table.%03u",i);
        tables[i].component=KQ_COMPONENT_PLE_TABLE;tables[i].role=KQ_ROLE_PLE_TABLE;
        tables[i].relation=KQ_BINDING_MULTIPLE_CANONICAL_TO_ONE_PHYSICAL;
        tables[i].runtime_scope=KQ_SCOPE_REQUIRED_INITIAL_TEXT;
        tables[i].placement_hint=KQ_PLACEMENT_PLE_DISK_BACKED_CANDIDATE;
        tables[i].layer_type=KQ_MODEL_LAYER_GDN;tables[i].layer_id=1U;
        tables[i].canonical_dtype=KQ_CANONICAL_DTYPE_BF16;tables[i].canonical_rank=2U;
        tables[i].canonical_dimensions[0]=UINT64_C(2500012);tables[i].canonical_dimensions[1]=160U;
        tables[i].binding_count=1U;tables[i].bindings[0].physical=table_physical;
        tables[i].bindings[0].part_role=KQ_BINDING_PART_FUSED_MEMBER;
        tables[i].bindings[0].part_count=1U;tables[i].bindings[0].fused_member_index=i;
        tables[i].bindings[0].fused_member_count=128U;source->tables[i]=&tables[i];
    }
}

int main(void) {
    kq_ple_value_config *config=make_config();kq_ple_value_state *state=NULL,*split_state=NULL;
    kq_ple_value_config *target=NULL;kq_ple_value_semantic_source target_source;
    kq_semantic_tensor target_dense[6],target_tables[128];kq_gguf_tensor target_dense_physical[6],target_table_physical,other_table_physical;
    kq_ple_value_weights_f32 weights={0};float *owned[6]={0};
    synthetic_provider synthetic={0,UINT64_MAX,2U};
    kq_ple_value_lookup_provider provider={&synthetic,lookup,3U,64U,2U};
    kq_ple_address_intent intents[12];float hidden[48],output[48],split_output[48],before[144],after[144],split_after[144];
    uint64_t count=0U,position=0U;void *scratch=NULL;kq_diagnostic diagnostic;kq_ple_value_run_metrics metrics,decode_metrics;uint32_t i;
    CHECK(config!=NULL);if(config==NULL)return 1;fill_weights(&weights,owned);
    CHECK(kq_ple_value_config_state_bytes(config)==144U*sizeof(float));
    CHECK(kq_ple_value_config_semantic_state_bytes(config)==144U*sizeof(uint16_t));
    CHECK(kq_ple_value_state_create(config,&state,&diagnostic)==KQ_STATUS_OK);
    CHECK(kq_ple_value_state_create(config,&split_state,&diagnostic)==KQ_STATUS_OK);
    scratch=calloc(1U,(size_t)kq_ple_value_config_scratch_bytes(config));CHECK(scratch!=NULL);
    for(i=0U;i<48U;++i)hidden[i]=(float)((int)(i%11U)-5)*0.0625f;
    fill_intents(intents,3U,2U,3U,64U);
    CHECK(kq_ple_value_prefill_f32(config,state,&weights,&provider,hidden,3U,intents,12U,
          output,48U,scratch,kq_ple_value_config_scratch_bytes(config),NULL,NULL,&metrics,&diagnostic)==KQ_STATUS_OK);
    CHECK(metrics.tokens_processed==3U&&metrics.lookups_performed==12U&&synthetic.call_count==12U);
    CHECK(kq_ple_value_state_export_f32(config,state,after,144U,&count,&position,&diagnostic)==KQ_STATUS_OK);
    CHECK(count==144U&&position==3U);CHECK(isfinite(output[0]));
    synthetic.call_count=0U;
    CHECK(kq_ple_value_prefill_f32(config,split_state,&weights,&provider,hidden,2U,intents,8U,
          split_output,32U,scratch,kq_ple_value_config_scratch_bytes(config),NULL,NULL,NULL,&diagnostic)==KQ_STATUS_OK);
    CHECK(kq_ple_value_decode_f32(config,split_state,&weights,&provider,hidden+32U,intents+8U,4U,
          split_output+32U,16U,scratch,kq_ple_value_config_scratch_bytes(config),NULL,NULL,&decode_metrics,&diagnostic)==KQ_STATUS_OK);
    CHECK(memcmp(output,split_output,sizeof(output))==0);
    CHECK(kq_ple_value_state_export_f32(config,split_state,split_after,144U,&count,&position,&diagnostic)==KQ_STATUS_OK);
    CHECK(position==3U&&memcmp(after,split_after,sizeof(after))==0);
    CHECK(kq_ple_value_state_reset(config,state,&diagnostic)==KQ_STATUS_OK);
    CHECK(kq_ple_value_state_export_f32(config,state,before,144U,&count,&position,&diagnostic)==KQ_STATUS_OK);
    synthetic.call_count=0U;synthetic.fail_at=2U;
    CHECK(kq_ple_value_prefill_f32(config,state,&weights,&provider,hidden,1U,intents,4U,
          output,16U,scratch,kq_ple_value_config_scratch_bytes(config),NULL,NULL,NULL,&diagnostic)==KQ_STATUS_PLE_LOOKUP_FAILED);
    CHECK(kq_ple_value_state_export_f32(config,state,after,144U,&count,&position,&diagnostic)==KQ_STATUS_OK);
    CHECK(position==0U&&memcmp(before,after,sizeof(before))==0);
    synthetic.fail_at=UINT64_MAX;synthetic.call_count=0U;intents[0].global_head=1U;
    CHECK(kq_ple_value_prefill_f32(config,state,&weights,&provider,hidden,1U,intents,4U,
          output,16U,scratch,kq_ple_value_config_scratch_bytes(config),NULL,NULL,NULL,&diagnostic)==KQ_STATUS_INCOMPATIBLE_PLE_VALUE);
    CHECK(kq_ple_value_state_export_f32(config,state,after,144U,&count,&position,&diagnostic)==KQ_STATUS_OK&&position==0U);
    intents[0].global_head=0U;
    CHECK(kq_ple_value_prefill_f32(config,state,&weights,&provider,hidden,1U,intents,3U,
          output,16U,scratch,kq_ple_value_config_scratch_bytes(config),NULL,NULL,NULL,&diagnostic)==KQ_STATUS_INCOMPATIBLE_PLE_VALUE);
    CHECK(kq_ple_value_prefill_f32(config,state,&weights,&provider,hidden,1U,intents,4U,
          output,15U,scratch,kq_ple_value_config_scratch_bytes(config),NULL,NULL,NULL,&diagnostic)==KQ_STATUS_BUFFER_TOO_SMALL);
    provider.row_width=3U;
    CHECK(kq_ple_value_prefill_f32(config,state,&weights,&provider,hidden,1U,intents,4U,
          output,16U,scratch,kq_ple_value_config_scratch_bytes(config),NULL,NULL,NULL,&diagnostic)==KQ_STATUS_INCOMPATIBLE_PLE_VALUE);
    provider.row_width=2U;before[0]=NAN;
    CHECK(kq_ple_value_state_import_f32(config,state,before,144U,0U,&diagnostic)==KQ_STATUS_NUMERIC_DOMAIN);
    memset(before,0,sizeof(before));
    CHECK(kq_ple_value_state_import_f32(config,state,before,144U,UINT64_MAX,&diagnostic)==KQ_STATUS_OK);
    CHECK(kq_ple_value_decode_f32(config,state,&weights,&provider,hidden,intents,4U,
          output,16U,scratch,kq_ple_value_config_scratch_bytes(config),NULL,NULL,NULL,&diagnostic)==KQ_STATUS_ARITHMETIC_OVERFLOW);
    CHECK(kq_ple_value_state_reset(config,state,&diagnostic)==KQ_STATUS_OK);
    CHECK(kq_ple_value_decode_f32(config,state,&weights,&provider,hidden,intents,4U,
          (float *)scratch,16U,scratch,kq_ple_value_config_scratch_bytes(config),NULL,NULL,NULL,&diagnostic)==KQ_STATUS_ALIASING_VIOLATION);
    fill_target_source(&target_source,target_dense,target_dense_physical,target_tables,&target_table_physical);
    CHECK(kq_ple_value_config_create_from_source(&target_source,KQ_PLE_VALUE_ACTIVATION_BF16,&target,&diagnostic)==KQ_STATUS_OK);
    kq_ple_value_config_close(target);target=NULL;
    target_source.hidden_size=2559U;
    CHECK(kq_ple_value_config_create_from_source(&target_source,KQ_PLE_VALUE_ACTIVATION_BF16,&target,&diagnostic)==KQ_STATUS_INCOMPATIBLE_PLE_VALUE&&target==NULL);
    target_source.hidden_size=2560U;target_source.dense[0]=NULL;
    CHECK(kq_ple_value_config_create_from_source(&target_source,KQ_PLE_VALUE_ACTIVATION_BF16,&target,&diagnostic)==KQ_STATUS_INCOMPATIBLE_PLE_VALUE&&target==NULL);
    target_source.dense[0]=&target_dense[0];target_dense[0].canonical_dimensions[0]=10239U;
    CHECK(kq_ple_value_config_create_from_source(&target_source,KQ_PLE_VALUE_ACTIVATION_BF16,&target,&diagnostic)==KQ_STATUS_INCOMPATIBLE_PLE_VALUE&&target==NULL);
    target_dense[0].canonical_dimensions[0]=10240U;target_source.tables[7]=NULL;
    CHECK(kq_ple_value_config_create_from_source(&target_source,KQ_PLE_VALUE_ACTIVATION_BF16,&target,&diagnostic)==KQ_STATUS_INCOMPATIBLE_PLE_VALUE&&target==NULL);
    target_source.tables[7]=&target_tables[7];target_table_physical.type_id=KQ_GGUF_TYPE_Q8_0;
    CHECK(kq_ple_value_config_create_from_source(&target_source,KQ_PLE_VALUE_ACTIVATION_BF16,&target,&diagnostic)==KQ_STATUS_TENSOR_LAYOUT_MISMATCH&&target==NULL);
    target_table_physical.type_id=KQ_GGUF_TYPE_IQ4_NL;target_tables[7].bindings[0].fused_member_index=8U;
    CHECK(kq_ple_value_config_create_from_source(&target_source,KQ_PLE_VALUE_ACTIVATION_BF16,&target,&diagnostic)==KQ_STATUS_INCOMPATIBLE_PLE_VALUE&&target==NULL);
    target_tables[7].bindings[0].fused_member_index=7U;other_table_physical=target_table_physical;target_tables[7].bindings[0].physical=&other_table_physical;
    CHECK(kq_ple_value_config_create_from_source(&target_source,KQ_PLE_VALUE_ACTIVATION_BF16,&target,&diagnostic)==KQ_STATUS_TENSOR_LAYOUT_MISMATCH&&target==NULL);
    target_tables[7].bindings[0].physical=&target_table_physical;target_source.address_info.logical_member_count=127U;
    CHECK(kq_ple_value_config_create_from_source(&target_source,KQ_PLE_VALUE_ACTIVATION_BF16,&target,&diagnostic)==KQ_STATUS_INCOMPATIBLE_PLE_VALUE&&target==NULL);
    target_source.address_info.logical_member_count=128U;
    CHECK(kq_ple_value_config_create_from_source(&target_source,(kq_ple_value_activation_dtype)99,&target,&diagnostic)==KQ_STATUS_INCOMPATIBLE_PLE_VALUE&&target==NULL);
    for(i=0U;i<6U;++i)free(owned[i]);free(scratch);kq_ple_value_config_close(target);kq_ple_value_state_close(split_state);kq_ple_value_state_close(state);kq_ple_value_config_close(config);
    if(failures!=0){fprintf(stderr,"PLE value failures: %d\n",failures);return 1;}
    printf("PLE value synthetic: PASS prefill_ns=%llu decode_ns=%llu scratch=%llu\n",
           (unsigned long long)metrics.elapsed_nanoseconds,
           (unsigned long long)decode_metrics.elapsed_nanoseconds,
           (unsigned long long)metrics.scratch_bytes);return 0;
}
