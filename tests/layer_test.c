#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_gdn_internal.h"
#include "kq_layer_internal.h"
#include "kq_moe_internal.h"
#include "kq_ple_internal.h"
#include "kq_ple_value_internal.h"
#include "kq_qsa_internal.h"

static int failures;
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); ++failures; } } while (0)
#ifndef KQ_LAYER_FIXTURE_ONLY
static int close_arrays(const float *a,const float *b,uint64_t n){uint64_t i;float max=0.0f;for(i=0;i<n;++i){float d=fabsf(a[i]-b[i]);if(d>max)max=d;}if(max>2.0e-6f){fprintf(stderr,"max layer split diff=%g\n",(double)max);return 0;}return 1;}
#endif

typedef struct fixture {
    kq_gdn_config *gdn; kq_qsa_config *qsa; kq_moe_config *moe;
    kq_ple_config *ple; kq_ple_value_config *ple_value;
    kq_layer_config *layer; kq_layer_state *state;
    kq_layer_weights_f32 weights;
    float *owned[40]; uint32_t owned_count;
} fixture;

static float *make_values(fixture *f,uint64_t n,float scale){uint64_t i;float *p=(float *)calloc((size_t)n,sizeof(float));CHECK(p!=NULL);if(p==NULL)return NULL;
    for(i=0U;i<n;++i)p[i]=(float)((int)(i%17U)-8)*scale;f->owned[f->owned_count++]=p;return p;}
static void gr_weights(fixture *f,kq_layer_gr_weights_f32 *w){w->norm=make_values(f,32U,0.003f);w->norm_count=32U;w->down=make_values(f,128U,0.002f);w->down_count=128U;
    w->up=make_values(f,128U,0.0025f);w->up_count=128U;w->inject=make_values(f,128U,0.0015f);w->inject_count=128U;}

static void descriptor(kq_ple_compatibility_descriptor *d){static const uint64_t m[3]={23703573157769ULL,20109073645365ULL,8052911324071ULL};
    static const uint64_t o[16]={0,20000003,40000026,60000059,80000106,100000165,120000228,140000297,160000374,180000455,200000548,220000655,240000802,260000955,280001114,300001275};
    static const uint64_t v[16]={20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171};
    memset(d,0,sizeof(*d));d->hidden_size=2560;d->vocabulary_size=248320;d->context_length=262144;d->layer_count=48;d->gdn_layer_count=36;d->qsa_layer_count=12;d->ple_layer_id=1;d->ple_layer_type=KQ_MODEL_LAYER_GDN;
    d->ngram_size=3;d->heads_per_order=8;d->eos_token_id=248044;d->logical_member_count=128;d->member_rows=2500012;d->table_width=160;d->ple_table_semantic_count=128;d->ple_dense_semantic_count=6;d->ple_metadata_semantic_count=3;d->table_semantics_valid=1;d->metadata_semantics_valid=1;
    memcpy(d->multipliers,m,sizeof(m));memcpy(d->head_offsets,o,sizeof(o));memcpy(d->head_vocab_sizes,v,sizeof(v));}

static kq_status lookup(void *u,uint32_t member,uint64_t row,float *out,uint64_t cap,kq_diagnostic *d){uint32_t i;(void)u;(void)d;if(cap<2U)return KQ_STATUS_BUFFER_TOO_SMALL;
    for(i=0U;i<2U;++i)out[i]=(float)((member%7U)+(uint32_t)(row%13U)+i)*0.015625f;return KQ_STATUS_OK;}

static int setup(fixture *f,kq_layer_family family){kq_diagnostic d;kq_status s;kq_gdn_dimensions gd={8,2,4,4,4,4,1e-6f,KQ_GDN_ACTIVATION_F32};
    kq_qsa_dimensions qd={8,2,1,4,2,4,4,8,32,4,1e-6f,10000000.0f,KQ_QSA_ACTIVATION_F32};kq_moe_dimensions md={8,4,2,4,4,KQ_MOE_ACTIVATION_F32};
    kq_layer_dimensions ld={family==KQ_LAYER_FAMILY_PLE_GDN?1U:family==KQ_LAYER_FAMILY_QSA?3U:0U,8,4,4,1e-6f,family};
    memset(f,0,sizeof(*f));if(family==KQ_LAYER_FAMILY_QSA)s=kq_qsa_test_config_create(&qd,&f->qsa,&d);else s=kq_gdn_test_config_create(&gd,&f->gdn,&d);if(s!=KQ_STATUS_OK)return 0;
    if(kq_moe_test_config_create(&md,&f->moe,&d)!=KQ_STATUS_OK)return 0;
    if(family==KQ_LAYER_FAMILY_PLE_GDN){kq_ple_compatibility_descriptor pd;kq_ple_value_dimensions vd={8,4,8,16,2,128,2500012,4,3,9,1e-6f,KQ_PLE_VALUE_ACTIVATION_F32};descriptor(&pd);
        if(kq_ple_config_open_from_descriptor_for_test(&pd,&f->ple,&d)!=KQ_STATUS_OK||kq_ple_value_test_config_create(&vd,&f->ple_value,&d)!=KQ_STATUS_OK)return 0;}
    if(kq_layer_test_config_create(&ld,f->gdn,f->qsa,f->moe,f->ple,f->ple_value,&f->layer,&d)!=KQ_STATUS_OK||kq_layer_state_create(f->layer,16U,&f->state,&d)!=KQ_STATUS_OK)return 0;
    gr_weights(f,&f->weights.attention_gr);gr_weights(f,&f->weights.moe_gr);
    {kq_moe_weights_f32 *w=(kq_moe_weights_f32 *)calloc(1,sizeof(*w));f->owned[f->owned_count++]=(float *)w;f->weights.moe=w;
     w->router=make_values(f,32,0.002f);w->router_count=32;w->routed_gate=make_values(f,128,0.002f);w->routed_gate_count=128;w->routed_up=make_values(f,128,0.0022f);w->routed_up_count=128;w->routed_down=make_values(f,128,0.0024f);w->routed_down_count=128;
     w->shared_gate=make_values(f,32,0.002f);w->shared_gate_count=32;w->shared_up=make_values(f,32,0.002f);w->shared_up_count=32;w->shared_down=make_values(f,32,0.002f);w->shared_down_count=32;w->shared_gate_weight=make_values(f,8,0.002f);w->shared_gate_weight_count=8;}
    if(f->gdn){kq_gdn_weights_f32 *w=(kq_gdn_weights_f32 *)calloc(1,sizeof(*w));f->owned[f->owned_count++]=(float *)w;f->weights.gdn=w;
     w->a_log=make_values(f,4,0.02f);w->a_log_count=4;w->conv=make_values(f,128,0.003f);w->conv_count=128;w->dt_bias=make_values(f,4,0.01f);w->dt_bias_count=4;w->alpha=make_values(f,32,0.003f);w->alpha_count=32;w->beta=make_values(f,32,0.003f);w->beta_count=32;w->qkv=make_values(f,256,0.002f);w->qkv_count=256;w->gate=make_values(f,128,0.002f);w->gate_count=128;w->norm=make_values(f,4,0.005f);w->norm_count=4;w->output=make_values(f,128,0.002f);w->output_count=128;}
    if(f->qsa){kq_qsa_weights_f32 *w=(kq_qsa_weights_f32 *)calloc(1,sizeof(*w));f->owned[f->owned_count++]=(float *)w;f->weights.qsa=w;
     w->query=make_values(f,128,0.002f);w->query_count=128;w->key=make_values(f,32,0.002f);w->key_count=32;w->value=make_values(f,32,0.002f);w->value_count=32;w->output=make_values(f,64,0.002f);w->output_count=64;w->query_norm=make_values(f,4,0.002f);w->query_norm_count=4;w->key_norm=make_values(f,4,0.002f);w->key_norm_count=4;w->index_query=make_values(f,64,0.002f);w->index_query_count=64;w->index_key=make_values(f,32,0.002f);w->index_key_count=32;w->index_query_norm=make_values(f,4,0.002f);w->index_query_norm_count=4;w->index_key_norm=make_values(f,4,0.002f);w->index_key_norm_count=4;}
    if(f->ple_value){kq_ple_value_weights_f32 *w=(kq_ple_value_weights_f32 *)calloc(1,sizeof(*w));kq_ple_value_lookup_provider *p=(kq_ple_value_lookup_provider *)calloc(1,sizeof(*p));f->owned[f->owned_count++]=(float *)w;f->owned[f->owned_count++]=(float *)p;f->weights.ple_value=w;f->weights.ple_provider=p;
     w->key_projection=make_values(f,1024,0.001f);w->key_projection_count=1024;w->value_projection=make_values(f,256,0.001f);w->value_projection_count=256;w->norm_key=make_values(f,32,0.001f);w->norm_key_count=32;w->norm_query=make_values(f,32,0.001f);w->norm_query_count=32;w->norm_conv=make_values(f,32,0.001f);w->norm_conv_count=32;w->convolution=make_values(f,128,0.001f);w->convolution_count=128;p->lookup_row=lookup;p->logical_member_count=128;p->member_rows=2500012;p->row_width=2;}
    return 1;}
static void teardown(fixture *f){uint32_t i;kq_layer_state_close(f->state);kq_layer_config_close(f->layer);kq_gdn_config_close(f->gdn);kq_qsa_config_close(f->qsa);kq_moe_config_close(f->moe);kq_ple_value_config_close(f->ple_value);kq_ple_config_close(f->ple);for(i=0;i<f->owned_count;++i)free(f->owned[i]);}

#ifndef KQ_LAYER_FIXTURE_ONLY
static void family_test(kq_layer_family family){fixture a,b,c;float input[160],whole[128],split[160],control[32];uint32_t ids[5]={11,22,33,44,55};uint64_t sa=0,sb=0,sc=0;void *wa=NULL,*wb=NULL,*wc=NULL;kq_diagnostic d;kq_layer_metrics m,dm;uint32_t i;kq_moe_weights_f32 *poison;const float *saved_router;float saved_gate;
    CHECK(setup(&a,family));CHECK(setup(&b,family));CHECK(setup(&c,family));if(a.layer==NULL||b.layer==NULL||c.layer==NULL)return;for(i=0;i<160;++i)input[i]=(float)((int)(i%19)-9)*0.03125f;
    CHECK(kq_layer_required_scratch_bytes(a.layer,a.state,4,&sa,&d)==KQ_STATUS_OK);CHECK(kq_layer_required_scratch_bytes(b.layer,b.state,2,&sb,&d)==KQ_STATUS_OK);CHECK(kq_layer_required_scratch_bytes(c.layer,c.state,4,&sc,&d)==KQ_STATUS_OK);wa=calloc(1,(size_t)sa);wb=calloc(1,(size_t)(sa>sb?sa:sb));wc=calloc(1,(size_t)sc);
    CHECK(kq_layer_required_scratch_bytes(a.layer,a.state,UINT64_MAX,&sc,&d)==KQ_STATUS_ARITHMETIC_OVERFLOW);
    CHECK(kq_layer_prefill_f32(a.layer,&a.weights,input,family==KQ_LAYER_FAMILY_PLE_GDN?ids:NULL,4,NULL,whole,127,a.state,wa,sa,NULL,NULL,NULL,&d)==KQ_STATUS_BUFFER_TOO_SMALL);
    CHECK(kq_layer_prefill_f32(a.layer,&a.weights,input,family==KQ_LAYER_FAMILY_PLE_GDN?ids:NULL,4,NULL,whole,128,a.state,wa,sa-1U,NULL,NULL,NULL,&d)==KQ_STATUS_BUFFER_TOO_SMALL);
    CHECK(kq_layer_prefill_f32(a.layer,&a.weights,input,family==KQ_LAYER_FAMILY_PLE_GDN?ids:NULL,1,NULL,input,32,a.state,wa,sa,NULL,NULL,NULL,&d)==KQ_STATUS_INVALID_ARGUMENT);
    {const float *saved=a.weights.attention_gr.norm;float finite=input[0];a.weights.attention_gr.norm=NULL;CHECK(kq_layer_prefill_f32(a.layer,&a.weights,input,family==KQ_LAYER_FAMILY_PLE_GDN?ids:NULL,1,NULL,whole,32,a.state,wa,sa,NULL,NULL,NULL,&d)==KQ_STATUS_INCOMPATIBLE_LAYER);a.weights.attention_gr.norm=saved;input[0]=NAN;CHECK(kq_layer_prefill_f32(a.layer,&a.weights,input,family==KQ_LAYER_FAMILY_PLE_GDN?ids:NULL,1,NULL,whole,32,a.state,wa,sa,NULL,NULL,NULL,&d)==KQ_STATUS_NUMERIC_DOMAIN);input[0]=finite;}
    {uint32_t magic=a.state->magic;a.state->magic=0U;CHECK(kq_layer_state_reset(a.state,&d)==KQ_STATUS_INVALID_LAYER_STATE);a.state->magic=magic;}
    if(family==KQ_LAYER_FAMILY_PLE_GDN){kq_ple_value_lookup_provider *provider=(kq_ple_value_lookup_provider *)a.weights.ple_provider;kq_ple_value_lookup_row_f32 saved_lookup=provider->lookup_row;CHECK(kq_layer_prefill_f32(a.layer,&a.weights,input,NULL,1,NULL,whole,32,a.state,wa,sa,NULL,NULL,NULL,&d)==KQ_STATUS_INCOMPATIBLE_LAYER);provider->lookup_row=NULL;CHECK(kq_layer_prefill_f32(a.layer,&a.weights,input,ids,1,NULL,whole,32,a.state,wa,sa,NULL,NULL,NULL,&d)!=KQ_STATUS_OK);provider->lookup_row=saved_lookup;}
    CHECK(kq_layer_state_position(a.state)==0);
    {kq_status first=kq_layer_prefill_f32(a.layer,&a.weights,input,family==KQ_LAYER_FAMILY_PLE_GDN?ids:NULL,4,NULL,whole,128,a.state,wa,sa,NULL,NULL,&m,&d);if(first!=KQ_STATUS_OK)fprintf(stderr,"family %s first prefill: %s (%s)\n",kq_layer_family_name(family),kq_status_string(first),d.message);CHECK(first==KQ_STATUS_OK);}CHECK(kq_layer_state_position(a.state)==4);
    CHECK(kq_layer_prefill_f32(b.layer,&b.weights,input,family==KQ_LAYER_FAMILY_PLE_GDN?ids:NULL,2,NULL,split,64,b.state,wb,sa,NULL,NULL,NULL,&d)==KQ_STATUS_OK);
    CHECK(kq_layer_decode_f32(b.layer,&b.weights,input+64,ids[2],split+64,32,b.state,wb,sa,NULL,NULL,NULL,&d)==KQ_STATUS_OK);CHECK(kq_layer_decode_f32(b.layer,&b.weights,input+96,ids[3],split+96,32,b.state,wb,sa,NULL,NULL,&dm,&d)==KQ_STATUS_OK);CHECK(close_arrays(whole,split,128));CHECK(kq_layer_state_position(b.state)==4);
    CHECK(kq_layer_prefill_f32(c.layer,&c.weights,input,family==KQ_LAYER_FAMILY_PLE_GDN?ids:NULL,4,NULL,split,128,c.state,wc,sc,NULL,NULL,NULL,&d)==KQ_STATUS_OK);
    poison=(kq_moe_weights_f32 *)b.weights.moe;saved_router=poison->router;saved_gate=((float *)poison->shared_gate)[0];poison->router=poison->shared_gate;((float *)poison->shared_gate)[0]=NAN;
    CHECK(kq_layer_decode_f32(b.layer,&b.weights,input+128,ids[4],split+128,32,b.state,wb,sa,NULL,NULL,NULL,&d)==KQ_STATUS_NUMERIC_DOMAIN);CHECK(kq_layer_state_position(b.state)==4);
    ((float *)poison->shared_gate)[0]=saved_gate;poison->router=saved_router;
    CHECK(kq_layer_decode_f32(b.layer,&b.weights,input+128,ids[4],split+128,32,b.state,wb,sa,NULL,NULL,NULL,&d)==KQ_STATUS_OK);CHECK(kq_layer_decode_f32(c.layer,&c.weights,input+128,ids[4],control,32,c.state,wc,sc,NULL,NULL,NULL,&d)==KQ_STATUS_OK);CHECK(close_arrays(split+128,control,32));CHECK(kq_layer_state_position(b.state)==5);
    CHECK(kq_layer_state_reset(b.state,&d)==KQ_STATUS_OK);CHECK(kq_layer_prefill_f32(b.layer,&b.weights,input,family==KQ_LAYER_FAMILY_PLE_GDN?ids:NULL,4,NULL,split,128,b.state,wb,sa,NULL,NULL,NULL,&d)==KQ_STATUS_OK);CHECK(memcmp(whole,split,sizeof(whole))==0);
    printf("layer metrics: family=%s prefill4_ns=%llu decode1_ns=%llu state_bytes=%llu scratch4=%llu gr_workspace=%llu transaction_staging=%llu\n",kq_layer_family_name(family),(unsigned long long)m.elapsed_nanoseconds,(unsigned long long)dm.elapsed_nanoseconds,(unsigned long long)kq_layer_state_owned_bytes(a.state),(unsigned long long)sa,(unsigned long long)m.gr_workspace_bytes,(unsigned long long)m.transaction_staging_bytes);
    free(wa);free(wb);free(wc);teardown(&a);teardown(&b);teardown(&c);}

int main(void){fixture f;kq_layer_dimensions bad={0,7,4,4,1e-6f,KQ_LAYER_FAMILY_GDN};kq_diagnostic d;kq_layer_config *c=NULL;kq_layer_state *limited=NULL;uint64_t scratch=0;
    family_test(KQ_LAYER_FAMILY_GDN);family_test(KQ_LAYER_FAMILY_QSA);family_test(KQ_LAYER_FAMILY_PLE_GDN);
    CHECK(setup(&f,KQ_LAYER_FAMILY_GDN));CHECK(kq_layer_test_config_create(&bad,f.gdn,NULL,f.moe,NULL,NULL,&c,&d)==KQ_STATUS_INCOMPATIBLE_LAYER);CHECK(c==NULL);bad.hidden_size=8;bad.family=KQ_LAYER_FAMILY_QSA;CHECK(kq_layer_test_config_create(&bad,f.gdn,NULL,f.moe,NULL,NULL,&c,&d)==KQ_STATUS_INCOMPATIBLE_LAYER);CHECK(c==NULL);teardown(&f);
    CHECK(setup(&f,KQ_LAYER_FAMILY_PLE_GDN));bad.layer_id=1;bad.hidden_size=8;bad.family=KQ_LAYER_FAMILY_GDN;CHECK(kq_layer_test_config_create(&bad,f.gdn,NULL,f.moe,f.ple,f.ple_value,&c,&d)==KQ_STATUS_INCOMPATIBLE_LAYER);CHECK(c==NULL);bad.family=KQ_LAYER_FAMILY_PLE_GDN;CHECK(kq_layer_test_config_create(&bad,f.gdn,NULL,f.moe,NULL,NULL,&c,&d)==KQ_STATUS_INCOMPATIBLE_LAYER);CHECK(c==NULL);teardown(&f);
    CHECK(setup(&f,KQ_LAYER_FAMILY_QSA));kq_layer_state_close(f.state);f.state=NULL;CHECK(kq_layer_state_create(f.layer,2U,&limited,&d)==KQ_STATUS_OK);CHECK(kq_layer_required_scratch_bytes(f.layer,limited,3U,&scratch,&d)!=KQ_STATUS_OK);kq_layer_state_close(limited);teardown(&f);
    if(failures!=0){fprintf(stderr,"layer tests failed: %d\n",failures);return 1;}printf("layer synthetic: PASS; families=3 transactional_rollback=PASS prefill_decode=PASS\n");return 0;}
#endif
