#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_ple.h"
#include "kq_ple_value.h"

static wchar_t *load_path(void){DWORD n=GetEnvironmentVariableW(L"KQ_GGUF_PATH",NULL,0U);wchar_t *p;DWORD copied;if(n==0U)return NULL;
    p=(wchar_t *)malloc((size_t)n*sizeof(*p));if(p==NULL)return NULL;copied=GetEnvironmentVariableW(L"KQ_GGUF_PATH",p,n);if(copied==0U||copied>=n){free(p);return NULL;}return p;}

int wmain(void){wchar_t *path=load_path();kq_file *file=NULL;kq_gguf *gguf=NULL;kq_model *model=NULL;kq_ple_config *address=NULL;
    kq_ple_value_config *value=NULL;kq_ple_value_gguf_provider *storage=NULL;kq_ple_value_lookup_provider provider;kq_ple_stream_state address_state;
    kq_ple_address_intent intents[16];float row[160];uint64_t required=0U;uint32_t token=1U,i;kq_diagnostic diagnostic;kq_status status;const kq_ple_value_gguf_metrics *metrics;
    if(path==NULL){printf("KQ_GGUF_PATH unavailable; skipping PLE value real integration\n");return 77;}
    if(kq_file_open_readonly(path,&file,&diagnostic)!=KQ_STATUS_OK||kq_gguf_open(file,&gguf,&diagnostic)!=KQ_STATUS_OK||
       kq_model_open_from_gguf(gguf,&model,&diagnostic)!=KQ_STATUS_OK||kq_ple_config_open_from_model(model,&address,&diagnostic)!=KQ_STATUS_OK||
       kq_ple_value_config_create_reference_f32(model,address,&value,&diagnostic)!=KQ_STATUS_OK||
       kq_ple_value_gguf_provider_open(gguf,model,value,UINT64_C(8)*1024U*1024U,&storage,&diagnostic)!=KQ_STATUS_OK){
        fprintf(stderr,"PLE value real setup failed: %s\n",diagnostic.message);free(path);path=NULL;goto fail;}
    free(path);path=NULL;
    if(kq_ple_state_reset(address,&address_state,&diagnostic)!=KQ_STATUS_OK||
       kq_ple_generate_prefill(address,&address_state,&token,1U,intents,16U,&required,NULL,&diagnostic)!=KQ_STATUS_OK||required!=16U)goto fail;
    provider=kq_ple_value_gguf_provider_interface(storage);
    for(i=0U;i<16U;++i){status=provider.lookup_row(provider.user_data,intents[i].logical_member,intents[i].member_row,row,160U,&diagnostic);if(status!=KQ_STATUS_OK)goto fail;}
    metrics=kq_ple_value_gguf_provider_metrics(storage);
    if(metrics==NULL||metrics->logical_payload_bytes_touched!=1440U||metrics->blocks_touched!=80U||metrics->rows_read!=16U)goto fail;
    printf("PLE value real: members=128 rows=2500012 row_width=160 member_span=225001080 table_bytes=28800138240 total_ple_bytes=28835240960 bytes_touched=%llu blocks=%llu config_bytes=%llu semantic_state_bytes=%llu reference_state_bytes=%llu scratch_bytes=%llu\n",
           (unsigned long long)metrics->logical_payload_bytes_touched,(unsigned long long)metrics->blocks_touched,
           (unsigned long long)kq_ple_value_config_owned_bytes(value),
           (unsigned long long)kq_ple_value_config_semantic_state_bytes(value),
           (unsigned long long)kq_ple_value_config_state_bytes(value),
           (unsigned long long)kq_ple_value_config_scratch_bytes(value));
    kq_ple_value_gguf_provider_close(storage);kq_ple_value_config_close(value);kq_ple_config_close(address);kq_model_close(model);kq_gguf_close(gguf);kq_file_close(file);return 0;
fail:free(path);fprintf(stderr,"PLE value real integration failed: %s\n",diagnostic.message);kq_ple_value_gguf_provider_close(storage);kq_ple_value_config_close(value);kq_ple_config_close(address);kq_model_close(model);kq_gguf_close(gguf);kq_file_close(file);return 1;}
