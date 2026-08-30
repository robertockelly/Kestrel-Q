#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_layer.h"
#include "kq_model.h"

static wchar_t *path_from_env(void){DWORD n=GetEnvironmentVariableW(L"KQ_GGUF_PATH",NULL,0);wchar_t *p;DWORD got;if(!n)return NULL;p=(wchar_t *)malloc((size_t)n*sizeof(*p));if(!p)return NULL;got=GetEnvironmentVariableW(L"KQ_GGUF_PATH",p,n);if(!got||got>=n){free(p);return NULL;}return p;}
int wmain(void){wchar_t *path=path_from_env();kq_file *file=NULL;kq_gguf *gguf=NULL;kq_model *model=NULL;kq_layer_config *config=NULL;kq_diagnostic d;kq_status s=KQ_STATUS_OK;uint32_t i,g=0,q=0,p=0;uint64_t bytes=0,gbytes=0,qbytes=0,pbytes=0,current=0,before=0;if(!path){puts("KQ_GGUF_PATH unavailable; layer integration skipped");return 77;}
    s=kq_file_open_readonly(path,&file,&d);free(path);if(s==KQ_STATUS_OK)s=kq_gguf_open(file,&gguf,&d);if(s==KQ_STATUS_OK){before=kq_gguf_payload_bytes_accessed(gguf);s=kq_model_open_from_gguf(gguf,&model,&d);}if(s!=KQ_STATUS_OK)goto fail;
    for(i=0;i<48;++i){s=kq_layer_config_create(model,i,&config,&d);if(s!=KQ_STATUS_OK)goto fail;current=kq_layer_config_owned_bytes(config);bytes+=current;if(kq_layer_config_family(config)==KQ_LAYER_FAMILY_QSA){++q;qbytes+=current;}else if(kq_layer_config_family(config)==KQ_LAYER_FAMILY_PLE_GDN){++p;pbytes+=current;}else if(kq_layer_config_family(config)==KQ_LAYER_FAMILY_GDN){++g;gbytes+=current;}else goto fail;
        if(kq_layer_config_hidden_size(config)!=2560||kq_layer_config_branch_count(config)!=4||kq_layer_config_gr_rank(config)!=320)goto fail;
        if((kq_layer_config_family(config)==KQ_LAYER_FAMILY_GDN&&kq_layer_config_persistent_semantic_bytes(config,16)!=UINT64_C(3227648))||
           (kq_layer_config_family(config)==KQ_LAYER_FAMILY_QSA&&kq_layer_config_persistent_semantic_bytes(config,16)!=UINT64_C(36864))||
           (kq_layer_config_family(config)==KQ_LAYER_FAMILY_PLE_GDN&&kq_layer_config_persistent_semantic_bytes(config,16)!=UINT64_C(3412000)))goto fail;
        kq_layer_config_close(config);config=NULL;}
    if(g!=35||q!=12||p!=1||before!=0||kq_gguf_payload_bytes_accessed(gguf)!=0)goto fail;
    printf("layer real integration: PASS; layers=48 ordinary_gdn=35 qsa=12 ple_gdn=1 gr_branches=4 gr_rank=320 config_bytes=%llu ordinary_gdn_config_bytes=%llu qsa_config_bytes=%llu ple_gdn_config_bytes=%llu real_model_payload_logical_bytes_touched=0\n",(unsigned long long)bytes,(unsigned long long)(gbytes/g),(unsigned long long)(qbytes/q),(unsigned long long)pbytes);
    kq_model_close(model);kq_gguf_close(gguf);kq_file_close(file);return 0;
fail:fprintf(stderr,"layer integration failed: %s\n",d.message);kq_layer_config_close(config);kq_model_close(model);kq_gguf_close(gguf);kq_file_close(file);return 1;}
