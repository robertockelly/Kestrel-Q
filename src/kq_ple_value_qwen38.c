#include "kq_ple_value_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "kq_internal.h"
#include "kq_numeric.h"
#include "kq_weight_provider.h"

static kq_status fail(kq_diagnostic *d,kq_status s,const char *m){kq_diagnostic_set(d,s,"%s",m);return s;}
static int finite_array(const float *v,uint64_t n){uint64_t i;if(v==NULL)return 0;for(i=0;i<n;++i)if(!isfinite(v[i]))return 0;return 1;}
static int overlap(const void *a,uint64_t an,const void *b,uint64_t bn){
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(an==0U||bn==0U)return 0;
    if(an>(uint64_t)(UINTPTR_MAX-x)||bn>(uint64_t)(UINTPTR_MAX-y))return 1;
    return x+(uintptr_t)an>y&&y+(uintptr_t)bn>x;
}

static kq_status dot_rows(const kq_ple_value_config *config,
                          const float *weight,kq_weight_provider *provider,
                          uint32_t role_index,uint64_t rows,uint64_t cols,
                          const float *input,float *output,
                          void *weight_scratch,uint64_t weight_scratch_bytes,
                          kq_diagnostic *d){
    uint64_t r;if(provider!=NULL)return kq_weight_provider_linear_f32(
        provider,config->dense[role_index],KQ_BINDING_PART_WHOLE,
        KQ_WEIGHT_PROVIDER_NO_EXPERT,rows,cols,input,output,rows,
        weight_scratch,weight_scratch_bytes,d);
    for(r=0;r<rows;++r){kq_status s=kq_f32_dot(weight+r*cols,input,cols,&output[r],d);if(s!=KQ_STATUS_OK)return s;}return KQ_STATUS_OK;}

static void emit(kq_ple_value_checkpoint_observer observer,void *user,
                 kq_ple_value_checkpoint_kind kind,uint64_t token,
                 const float *values,uint64_t count,uint32_t rank,
                 uint64_t d0,uint64_t d1,uint64_t d2){
    kq_ple_value_checkpoint c;if(observer==NULL)return;memset(&c,0,sizeof(c));c.kind=kind;c.token_index=token;
    c.rank=rank;c.dimensions[0]=d0;c.dimensions[1]=d1;c.dimensions[2]=d2;c.values=values;c.value_count=count;observer(&c,user);
}

static kq_status validate_weights(const kq_ple_value_config *c,const kq_ple_value_weights_f32 *w,kq_diagnostic *d){
    uint64_t key_count,value_count,conv_count,b=c->branch_width,e=c->embedding_width,h=c->dimensions.hidden_size;
    if(w==NULL||!kq_ple_value_u64_mul(b,e,&key_count)||!kq_ple_value_u64_mul(h,e,&value_count)||
       !kq_ple_value_u64_mul(b,4U,&conv_count))return fail(d,KQ_STATUS_ARITHMETIC_OVERFLOW,"PLE weight geometry overflows");
    if(w->key_projection_count!=key_count||w->value_projection_count!=value_count||w->norm_key_count!=b||
       w->norm_query_count!=b||w->norm_conv_count!=b||w->convolution_count!=conv_count||
       !finite_array(w->key_projection,key_count)||!finite_array(w->value_projection,value_count)||
       !finite_array(w->norm_key,b)||!finite_array(w->norm_query,b)||!finite_array(w->norm_conv,b)||
       !finite_array(w->convolution,conv_count))return fail(d,KQ_STATUS_DIMENSION_MISMATCH,"PLE weight count or finite-domain mismatch");
    return KQ_STATUS_OK;
}

static kq_status run(const kq_ple_value_config *c,kq_ple_value_state *state,
 const kq_ple_value_weights_f32 *w,kq_weight_provider *weight_provider,
 const kq_ple_value_lookup_provider *p,
 const float *hidden,uint64_t tokens,const kq_ple_address_intent *intents,uint64_t intent_count,
 float *output,uint64_t output_capacity,void *scratch,uint64_t scratch_bytes,
 void *weight_scratch,uint64_t weight_scratch_bytes,
 kq_ple_value_checkpoint_observer observer,void *user,kq_ple_value_run_metrics *metrics,kq_diagnostic *d){
    const kq_ple_value_dimensions *g;float *at,*staged,*embedding,*row,*value,*key,*key_norm,*query_norm,*gated,*conv_norm,*conv_pre,*conv_out;
    uint64_t needed_intents,needed_output,output_bytes,next_position,t,i,j,branch,position;
    uint64_t b,e,h,state_count,key_count,value_count,conv_count,key_bytes,value_bytes,branch_bytes,conv_bytes; kq_status status;
#ifdef _WIN32
    LARGE_INTEGER f={0},start={0},end={0};QueryPerformanceFrequency(&f);QueryPerformanceCounter(&start);
#endif
    kq_diagnostic_clear(d);if(metrics!=NULL)memset(metrics,0,sizeof(*metrics));
    if(!kq_ple_value_config_valid(c)||state==NULL||state->magic!=KQ_PLE_VALUE_STATE_MAGIC||state->config!=c||
       c->dimensions.activation_dtype!=KQ_PLE_VALUE_ACTIVATION_F32||p==NULL||p->lookup_row==NULL||
       hidden==NULL||tokens==0U||intents==NULL||output==NULL||scratch==NULL)
        return fail(d,KQ_STATUS_INVALID_ARGUMENT,"invalid PLE value execution arguments");
    g=&c->dimensions;b=c->branch_width;e=c->embedding_width;h=g->hidden_size;state_count=c->state_elements;
    if(!kq_ple_value_u64_mul(tokens,g->head_count,&needed_intents)||!kq_ple_value_u64_mul(tokens,b,&needed_output)||
       !kq_ple_value_u64_mul(needed_output,sizeof(float),&output_bytes)||
       !kq_ple_value_u64_mul(b,e,&key_count)||!kq_ple_value_u64_mul(h,e,&value_count)||
       !kq_ple_value_u64_mul(b,4U,&conv_count)||!kq_ple_value_u64_mul(key_count,sizeof(float),&key_bytes)||
       !kq_ple_value_u64_mul(value_count,sizeof(float),&value_bytes)||!kq_ple_value_u64_mul(b,sizeof(float),&branch_bytes)||
       !kq_ple_value_u64_mul(conv_count,sizeof(float),&conv_bytes)||!kq_ple_value_u64_add(state->position,tokens,&next_position))
        return fail(d,KQ_STATUS_ARITHMETIC_OVERFLOW,"PLE value run count overflows");
    if(intent_count!=needed_intents)return fail(d,KQ_STATUS_INCOMPATIBLE_PLE_VALUE,"PLE intent count mismatch");
    if(output_capacity<needed_output||scratch_bytes<c->scratch_bytes)return fail(d,KQ_STATUS_BUFFER_TOO_SMALL,"PLE output or scratch capacity is too small");
    if(p->logical_member_count!=g->logical_member_count||p->member_rows!=g->member_rows||p->row_width!=g->row_width)
        return fail(d,KQ_STATUS_INCOMPATIBLE_PLE_VALUE,"lookup provider geometry mismatch");
    if(weight_provider==NULL){status=validate_weights(c,w,d);if(status!=KQ_STATUS_OK)return status;}
    if(!finite_array(hidden,needed_output))return fail(d,KQ_STATUS_NUMERIC_DOMAIN,"non-finite PLE hidden input");
    if(overlap(output,output_bytes,hidden,output_bytes)||overlap(output,output_bytes,state->history,c->state_bytes)||
       overlap(output,output_bytes,scratch,c->scratch_bytes)||overlap(output,output_bytes,w->key_projection,key_bytes)||
       overlap(output,output_bytes,w->value_projection,value_bytes)||overlap(output,output_bytes,w->norm_key,branch_bytes)||
       overlap(output,output_bytes,w->norm_query,branch_bytes)||overlap(output,output_bytes,w->norm_conv,branch_bytes)||
       overlap(output,output_bytes,w->convolution,conv_bytes)||overlap(scratch,c->scratch_bytes,hidden,output_bytes)||
       overlap(scratch,c->scratch_bytes,state->history,c->state_bytes)||overlap(scratch,c->scratch_bytes,w->key_projection,key_bytes)||
       overlap(scratch,c->scratch_bytes,w->value_projection,value_bytes)||overlap(scratch,c->scratch_bytes,w->norm_key,branch_bytes)||
       overlap(scratch,c->scratch_bytes,w->norm_query,branch_bytes)||overlap(scratch,c->scratch_bytes,w->norm_conv,branch_bytes)||
       overlap(scratch,c->scratch_bytes,w->convolution,conv_bytes))
        return fail(d,KQ_STATUS_ALIASING_VIOLATION,"PLE writable output/scratch aliases input, state, or weights");
    at=(float *)scratch;staged=at;at+=state_count;embedding=at;at+=e;row=at;at+=g->row_width;value=at;at+=h;
    key=at;at+=b;key_norm=at;at+=b;query_norm=at;at+=b;gated=at;at+=b;conv_norm=at;at+=b;conv_pre=at;at+=b;conv_out=at;
    memcpy(staged,state->history,(size_t)c->state_bytes);position=state->position;
    for(t=0U;t<tokens;++t){
        memset(embedding,0,(size_t)(e*sizeof(float)));
        for(i=0U;i<g->head_count;++i){const kq_ple_address_intent *in=&intents[t*g->head_count+i];
            uint32_t order=2U+(uint32_t)(i/g->heads_per_order),local=(uint32_t)(i%g->heads_per_order);
            if(in->position!=position+t||in->global_head!=i||in->ngram_order!=order||in->local_head!=local||
               in->logical_member>=g->logical_member_count||in->member_row>=g->member_rows)
                return fail(d,KQ_STATUS_INCOMPATIBLE_PLE_VALUE,"PLE intent order/member/row mismatch");
            status=p->lookup_row(p->user_data,in->logical_member,in->member_row,row,g->row_width,d);
            if(status!=KQ_STATUS_OK)return status==KQ_STATUS_PLE_LOOKUP_FAILED?status:fail(d,KQ_STATUS_PLE_LOOKUP_FAILED,"PLE lookup provider failed");
            if(!finite_array(row,g->row_width))return fail(d,KQ_STATUS_NUMERIC_DOMAIN,"PLE lookup returned non-finite data");
            memcpy(embedding+i*g->row_width,row,(size_t)(g->row_width*sizeof(float)));
        }
        emit(observer,user,KQ_PLE_VALUE_CHECKPOINT_RAW_LOOKUPS,t,embedding,e,2U,g->head_count,g->row_width,0U);
        emit(observer,user,KQ_PLE_VALUE_CHECKPOINT_EMBEDDING,t,embedding,e,1U,e,0U,0U);
        status=dot_rows(c,w->key_projection,weight_provider,0U,b,e,embedding,key,weight_scratch,weight_scratch_bytes,d);if(status!=KQ_STATUS_OK)return status;
        status=dot_rows(c,w->value_projection,weight_provider,1U,h,e,embedding,value,weight_scratch,weight_scratch_bytes,d);if(status!=KQ_STATUS_OK)return status;
        emit(observer,user,KQ_PLE_VALUE_CHECKPOINT_KEY_PROJECTION,t,key,b,2U,g->residual_branches,h,0U);
        emit(observer,user,KQ_PLE_VALUE_CHECKPOINT_VALUE_PROJECTION,t,value,h,1U,h,0U,0U);
        for(branch=0U;branch<g->residual_branches;++branch){float raw,transformed,sig;
            status=kq_f32_rms_norm(key+branch*h,w->norm_key+branch*h,h,g->rms_epsilon,key_norm+branch*h,d);if(status!=KQ_STATUS_OK)return status;
            status=kq_f32_rms_norm(hidden+t*b+branch*h,w->norm_query+branch*h,h,g->rms_epsilon,query_norm+branch*h,d);if(status!=KQ_STATUS_OK)return status;
            status=kq_f32_dot(key_norm+branch*h,query_norm+branch*h,h,&raw,d);if(status!=KQ_STATUS_OK)return status;
            raw=(float)(raw/sqrtf((float)h));conv_pre[branch]=raw;
            transformed=copysignf(sqrtf(fmaxf(fabsf(raw),1.0e-6f)),raw);conv_out[branch]=transformed;
            status=kq_f32_sigmoid(&transformed,1U,&sig,d);if(status!=KQ_STATUS_OK)return status;
            for(j=0U;j<h;++j)gated[branch*h+j]=(float)(sig*value[j]);
        }
        emit(observer,user,KQ_PLE_VALUE_CHECKPOINT_KEY_NORM,t,key_norm,b,2U,g->residual_branches,h,0U);
        emit(observer,user,KQ_PLE_VALUE_CHECKPOINT_QUERY_NORM,t,query_norm,b,2U,g->residual_branches,h,0U);
        emit(observer,user,KQ_PLE_VALUE_CHECKPOINT_GATE_RAW,t,conv_pre,g->residual_branches,1U,g->residual_branches,0U,0U);
        emit(observer,user,KQ_PLE_VALUE_CHECKPOINT_GATE_TRANSFORMED,t,conv_out,g->residual_branches,1U,g->residual_branches,0U,0U);
        emit(observer,user,KQ_PLE_VALUE_CHECKPOINT_GATED_VALUE,t,gated,b,2U,g->residual_branches,h,0U);
        for(branch=0U;branch<g->residual_branches;++branch){status=kq_f32_rms_norm(gated+branch*h,w->norm_conv+branch*h,h,g->rms_epsilon,conv_norm+branch*h,d);if(status!=KQ_STATUS_OK)return status;}
        emit(observer,user,KQ_PLE_VALUE_CHECKPOINT_CONV_NORM,t,conv_norm,b,2U,g->residual_branches,h,0U);
        for(i=0U;i<b;++i){float sum=0.0f;uint32_t k;for(k=0U;k<4U;++k){
                float source=(k==3U)?conv_norm[i]:staged[i*g->history_length+(uint64_t)k*3U];
                sum=(float)(sum+(float)(source*w->convolution[i*4U+k]));}conv_pre[i]=sum;}
        status=kq_f32_silu(conv_pre,b,conv_out,d);if(status!=KQ_STATUS_OK)return status;
        for(i=0U;i<b;++i)output[t*b+i]=(float)(gated[i]+conv_out[i]);
        emit(observer,user,KQ_PLE_VALUE_CHECKPOINT_CONV_PRE_ACTIVATION,t,conv_pre,b,2U,g->residual_branches,h,0U);
        emit(observer,user,KQ_PLE_VALUE_CHECKPOINT_CONV_OUTPUT,t,conv_out,b,2U,g->residual_branches,h,0U);
        emit(observer,user,KQ_PLE_VALUE_CHECKPOINT_OPERATOR_OUTPUT,t,output+t*b,b,2U,g->residual_branches,h,0U);
        for(i=0U;i<b;++i){memmove(staged+i*g->history_length,staged+i*g->history_length+1U,(g->history_length-1U)*sizeof(float));staged[i*g->history_length+g->history_length-1U]=conv_norm[i];}
    }
    memcpy(state->history,staged,(size_t)c->state_bytes);state->position=next_position;
    if(metrics!=NULL){metrics->tokens_processed=tokens;metrics->lookups_performed=needed_intents;metrics->scratch_bytes=c->scratch_bytes;
#ifdef _WIN32
        QueryPerformanceCounter(&end);if(f.QuadPart>0)metrics->elapsed_nanoseconds=(uint64_t)(((end.QuadPart-start.QuadPart)*1000000000ULL)/(uint64_t)f.QuadPart);
#endif
    }return KQ_STATUS_OK;
}

kq_status kq_ple_value_prefill_f32(const kq_ple_value_config *c,kq_ple_value_state *s,const kq_ple_value_weights_f32 *w,const kq_ple_value_lookup_provider *p,const float *h,uint64_t n,const kq_ple_address_intent *i,uint64_t ic,float *o,uint64_t oc,void *sc,uint64_t sb,kq_ple_value_checkpoint_observer ob,void *u,kq_ple_value_run_metrics *m,kq_diagnostic *d){return run(c,s,w,NULL,p,h,n,i,ic,o,oc,sc,sb,NULL,0U,ob,u,m,d);}
kq_status kq_ple_value_decode_f32(const kq_ple_value_config *c,kq_ple_value_state *s,const kq_ple_value_weights_f32 *w,const kq_ple_value_lookup_provider *p,const float *h,const kq_ple_address_intent *i,uint64_t ic,float *o,uint64_t oc,void *sc,uint64_t sb,kq_ple_value_checkpoint_observer ob,void *u,kq_ple_value_run_metrics *m,kq_diagnostic *d){return run(c,s,w,NULL,p,h,1U,i,ic,o,oc,sc,sb,NULL,0U,ob,u,m,d);}

kq_status kq_ple_value_execute_quantized(
    const kq_ple_value_config *c,kq_ple_value_state *state,
    kq_weight_provider *provider,const kq_ple_value_lookup_provider *lookup,
    const float *hidden,uint64_t tokens,const kq_ple_address_intent *intents,
    uint64_t intent_count,float *output,uint64_t output_capacity,
    void *scratch,uint64_t scratch_bytes,void *weight_scratch,
    uint64_t weight_scratch_bytes,kq_ple_value_checkpoint_observer observer,
    void *user,kq_ple_value_run_metrics *metrics,kq_diagnostic *d){
    kq_ple_value_weights_f32 w;float *cursor=(float *)weight_scratch,*temporary;
    uint64_t vector_count=UINT64_C(10240)*3U+UINT64_C(40960);
    uint64_t vector_bytes=vector_count*sizeof(float);
    uint64_t temporary_bytes=UINT64_C(40960)*sizeof(float);kq_status status;
    if(provider==NULL||weight_scratch==NULL||weight_scratch_bytes<vector_bytes+temporary_bytes+65536U)
        return fail(d,KQ_STATUS_BUFFER_TOO_SMALL,"PLE provider weight scratch is too small");
    memset(&w,0,sizeof(w));
    w.norm_key=cursor;w.norm_key_count=10240U;cursor+=10240U;
    w.norm_query=cursor;w.norm_query_count=10240U;cursor+=10240U;
    w.norm_conv=cursor;w.norm_conv_count=10240U;cursor+=10240U;
    w.convolution=cursor;w.convolution_count=40960U;cursor+=40960U;
    temporary=cursor;
#define LOAD_PLE_VECTOR(field,role_index) do { \
    status=kq_weight_provider_vector_f32(provider,c->dense[role_index],w.field##_count,(float *)w.field,w.field##_count,temporary,temporary_bytes,d); \
    if(status!=KQ_STATUS_OK)return status; \
} while(0)
    LOAD_PLE_VECTOR(norm_key,2U);LOAD_PLE_VECTOR(norm_query,3U);
    LOAD_PLE_VECTOR(norm_conv,4U);LOAD_PLE_VECTOR(convolution,5U);
#undef LOAD_PLE_VECTOR
    /* The provider is the storage-to-canonical boundary for the three
       converter-folded PLE gammas; they are already zero-centred here. */
    cursor=(float *)((unsigned char *)temporary+temporary_bytes);
    return run(c,state,&w,provider,lookup,hidden,tokens,intents,intent_count,
        output,output_capacity,scratch,scratch_bytes,cursor,
        weight_scratch_bytes-vector_bytes-temporary_bytes,observer,user,metrics,d);
}
