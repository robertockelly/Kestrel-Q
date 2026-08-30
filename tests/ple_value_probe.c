#define _CRT_SECURE_NO_WARNINGS
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_ple_value.h"
#include "kq_ple_value_internal.h"

typedef struct rows { const kq_ple_address_intent *intents; const float *values; uint64_t count,index; uint32_t width; } rows;

static int read_array(const char *expected,float **out,uint64_t *count){char label[64];unsigned long long n,i;unsigned int bits;
    if(scanf("%63s %llu",label,&n)!=2||strcmp(label,expected)!=0)return 0;*out=(float *)calloc((size_t)n,sizeof(float));if(*out==NULL)return 0;*count=(uint64_t)n;
    for(i=0;i<n;++i){if(scanf("%x",&bits)!=1)return 0;memcpy(*out+i,&bits,sizeof(bits));}return 1;}
static kq_status lookup(void *u,uint32_t member,uint64_t row,float *out,uint64_t cap,kq_diagnostic *d){rows *r=(rows *)u;const kq_ple_address_intent *in;
    if(r==NULL||r->index>=r->count||cap<r->width)return KQ_STATUS_PLE_LOOKUP_FAILED;in=&r->intents[r->index];
    if(in->logical_member!=member||in->member_row!=row){(void)d;return KQ_STATUS_PLE_LOOKUP_FAILED;}
    memcpy(out,r->values+r->index*r->width,r->width*sizeof(float));r->index+=1U;return KQ_STATUS_OK;}
static void observer(const kq_ple_value_checkpoint *c,void *u){uint64_t i;(void)u;printf("TRACE %s %llu %u",kq_ple_value_checkpoint_kind_name(c->kind),(unsigned long long)c->token_index,c->rank);
    for(i=0U;i<c->rank;++i)printf(" %llu",(unsigned long long)c->dimensions[i]);printf(" %llu",(unsigned long long)c->value_count);
    for(i=0U;i<c->value_count;++i){uint32_t bits;memcpy(&bits,c->values+i,sizeof(bits));printf(" %08x",bits);}printf("\n");}
int main(void){char header[32],label[32];unsigned hidden,branches,heads_per,row_width,members;unsigned long long member_rows,tokens,intent_n,i;
    kq_ple_value_dimensions dims; kq_ple_value_config *config=NULL;kq_ple_value_state *state=NULL;kq_ple_value_weights_f32 w={0};
    float *arrays[9]={0};uint64_t counts[9]={0};kq_ple_address_intent *intents=NULL;rows provider_rows={0};kq_ple_value_lookup_provider provider;
    float *output=NULL,*exported=NULL;void *scratch=NULL;uint64_t state_count=0U,state_position=0U; kq_ple_value_run_metrics metrics;kq_diagnostic diagnostic;kq_status status;int rc=1;
    if(scanf("%31s",header)!=1||strcmp(header,"KQPLEVALUE1")!=0)goto cleanup;
    if(scanf("%31s %u %u %u %u %u %llu %llu",label,&hidden,&branches,&heads_per,&row_width,&members,&member_rows,&tokens)!=8||strcmp(label,"CONFIG")!=0)goto cleanup;
    memset(&dims,0,sizeof(dims));dims.hidden_size=hidden;dims.residual_branches=branches;dims.heads_per_order=heads_per;dims.head_count=heads_per*2U;dims.row_width=row_width;
    dims.logical_member_count=members;dims.member_rows=(uint64_t)member_rows;dims.convolution_kernel=4U;dims.convolution_dilation=3U;dims.history_length=9U;dims.rms_epsilon=1.0e-6f;dims.activation_dtype=KQ_PLE_VALUE_ACTIVATION_F32;
    if(kq_ple_value_test_config_create(&dims,&config,&diagnostic)!=KQ_STATUS_OK||kq_ple_value_state_create(config,&state,&diagnostic)!=KQ_STATUS_OK)goto cleanup;
    if(!read_array("KEY",&arrays[0],&counts[0])||!read_array("VALUE",&arrays[1],&counts[1])||!read_array("NORM_KEY",&arrays[2],&counts[2])||
       !read_array("NORM_QUERY",&arrays[3],&counts[3])||!read_array("NORM_CONV",&arrays[4],&counts[4])||!read_array("CONV",&arrays[5],&counts[5])||
       !read_array("INPUT",&arrays[6],&counts[6])||!read_array("INITIAL_STATE",&arrays[7],&counts[7]))goto cleanup;
    w.key_projection=arrays[0];w.key_projection_count=counts[0];w.value_projection=arrays[1];w.value_projection_count=counts[1];
    w.norm_key=arrays[2];w.norm_key_count=counts[2];w.norm_query=arrays[3];w.norm_query_count=counts[3];w.norm_conv=arrays[4];w.norm_conv_count=counts[4];w.convolution=arrays[5];w.convolution_count=counts[5];
    if(kq_ple_value_state_import_f32(config,state,arrays[7],counts[7],0U,&diagnostic)!=KQ_STATUS_OK)goto cleanup;
    if(scanf("%31s %llu",label,&intent_n)!=2||strcmp(label,"INTENTS")!=0)goto cleanup;intents=(kq_ple_address_intent *)calloc((size_t)intent_n,sizeof(*intents));if(intents==NULL)goto cleanup;
    for(i=0U;i<(uint64_t)intent_n;++i){unsigned order,local,global,member;unsigned long long position,row;
        if(scanf("%llu %u %u %u %u %llu",&position,&order,&local,&global,&member,&row)!=6)goto cleanup;
        intents[i].position=(uint64_t)position;intents[i].ngram_order=order;intents[i].local_head=local;intents[i].global_head=global;intents[i].logical_member=member;intents[i].member_row=(uint64_t)row;}
    if(!read_array("ROWS",&arrays[8],&counts[8])||counts[8]!=(uint64_t)intent_n*row_width)goto cleanup;
    provider_rows.intents=intents;provider_rows.values=arrays[8];provider_rows.count=(uint64_t)intent_n;provider_rows.width=row_width;
    provider.user_data=&provider_rows;provider.lookup_row=lookup;provider.logical_member_count=members;provider.member_rows=(uint64_t)member_rows;provider.row_width=row_width;
    output=(float *)calloc((size_t)(tokens*hidden*branches),sizeof(float));scratch=calloc(1U,(size_t)kq_ple_value_config_scratch_bytes(config));
    exported=(float *)calloc((size_t)(hidden*branches*9U),sizeof(float));if(output==NULL||scratch==NULL||exported==NULL)goto cleanup;
    status=kq_ple_value_prefill_f32(config,state,&w,&provider,arrays[6],(uint64_t)tokens,intents,(uint64_t)intent_n,output,(uint64_t)(tokens*hidden*branches),
        scratch,kq_ple_value_config_scratch_bytes(config),observer,NULL,&metrics,&diagnostic);if(status!=KQ_STATUS_OK){fprintf(stderr,"%s: %s\n",kq_status_string(status),diagnostic.message);goto cleanup;}
    printf("OUTPUT %llu",tokens*hidden*branches);for(i=0U;i<tokens*hidden*branches;++i){uint32_t bits;memcpy(&bits,output+i,sizeof(bits));printf(" %08x",bits);}printf("\n");
    if(kq_ple_value_state_export_f32(config,state,exported,hidden*branches*9U,&state_count,&state_position,&diagnostic)!=KQ_STATUS_OK)goto cleanup;
    printf("STATE %llu %llu",(unsigned long long)state_position,(unsigned long long)state_count);for(i=0U;i<state_count;++i){uint32_t bits;memcpy(&bits,exported+i,sizeof(bits));printf(" %08x",bits);}printf("\n");
    printf("METRICS scratch=%llu lookups=%llu\n",(unsigned long long)metrics.scratch_bytes,(unsigned long long)metrics.lookups_performed);rc=0;
cleanup:for(i=0U;i<9U;++i)free(arrays[i]);free(intents);free(output);free(exported);free(scratch);kq_ple_value_state_close(state);kq_ple_value_config_close(config);return rc;}
