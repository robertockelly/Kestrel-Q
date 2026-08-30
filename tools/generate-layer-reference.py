#!/usr/bin/env python3
"""Generate Task 2.10 complete-layer evidence from pinned Transformers code."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, struct, sys
from pathlib import Path
from typing import Any

MODEL_REVISION="de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_REVISION="805a9e939fa8c1bff8d8ffdf041c051b71a914aa"
SOURCE_SHA="91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f"
CONFIG_SHA="889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
FILES=("layer-contract.json","layer-calibration.json","layer-holdout.json","layer-state-vectors.json","layer-family-vectors.json")

def write(path:Path,obj:Any)->None:path.write_bytes((json.dumps(obj,indent=2,sort_keys=True)+"\n").encode())
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def fhex(v:float)->str:return struct.pack("<f",float(v)).hex()
def tensor(torch:Any,v:Any)->dict[str,Any]:
    x=v.detach().cpu().float().contiguous();return {"dtype":"float32","shape":list(x.shape),"f32_le_hex":[fhex(a) for a in x.flatten().tolist()]}
def values(torch:Any,n:int,scale:float)->Any:return torch.tensor([((i%17)-8)*scale for i in range(n)],dtype=torch.float32)

def config(Config:Any,family:str)->Any:
    c=Config(hidden_size=8,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=4,
        layer_types=["qwen_sparse_attention" if family=="QSA" else "linear_attention"],
        linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=4,linear_value_head_dim=4,
        linear_conv_kernel_dim=4,output_gate_type="sigmoid",hidden_act="silu",rms_norm_eps=1e-6,
        num_experts=4,num_experts_per_tok=2,moe_intermediate_size=4,shared_expert_intermediate_size=4,
        norm_topk_prob=True,hc_count=4,hc_lowrank=4,vocab_size=248320,eos_token_id=248044,
        max_position_embeddings=32,indexer_n_heads=2,indexer_kv_heads=1,indexer_head_dim=4,
        indexer_budget=8,indexer_compress_ratio=4,attention_bias=False,attention_dropout=0.0,
        rope_parameters={"rope_type":"default","rope_theta":10000000.0,"partial_rotary_factor":1.0,
                         "mrope_section":[2,0,0],"mrope_interleaved":True},
        ple_layer_ids=[1] if family=="PLE_GDN" else [],ple_embed_dim=32,ngram_size=3,
        heads_per_ngram=8,ngram_vocab_size_base=11,make_ngram_vocab_size_divisible_by=4,
        ple_conv_kernel_size=4)
    c._attn_implementation="eager";c._experts_implementation="eager";return c

def fill(module:Any,torch:Any)->dict[str,Any]:
    records={}
    def assign(p:Any,scale:float)->None:p.copy_(values(torch,p.numel(),scale).reshape(p.shape))
    with torch.no_grad():
      for name,p in module.named_parameters():
        scale=0.002
        if "hc_norm" in name:scale=0.003
        elif "input_mix_weight_down" in name:scale=0.002
        elif "input_mix_weight_up" in name:scale=0.0025
        elif "block_inject" in name:scale=0.0015
        elif "conv1d.weight" in name and "ple" not in name:scale=0.003
        elif name.endswith("A_log"):scale=0.02
        elif name.endswith("dt_bias"):scale=0.01
        elif ".in_proj_a." in name or ".in_proj_b." in name:scale=0.003
        elif ".norm.weight" in name and "hyper" not in name:scale=0.005
        elif "ple." in name:scale=0.001
        elif ".experts.gate_up_proj" in name:
            flat=torch.empty_like(p).flatten();half=p.shape[-2]//2
            shaped=flat.reshape(p.shape)
            for expert in range(p.shape[0]):
                shaped[expert,:half].copy_(values(torch,half*p.shape[-1],0.002).reshape(half,p.shape[-1]))
                shaped[expert,half:].copy_(values(torch,half*p.shape[-1],0.0022).reshape(half,p.shape[-1]))
            p.copy_(shaped);records[name]=tensor(torch,p);continue
        elif ".experts.down_proj" in name:scale=0.0024
        assign(p,scale);records[name]=tensor(torch,p)
    return records

MULT=(23703573157769,20109073645365,8052911324071)
OFF=(0,20000003,40000026,60000059,80000106,100000165,120000228,140000297,160000374,180000455,200000548,220000655,240000802,260000955,280001114,300001275)
MOD=(20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171)
def ple_rows(torch:Any,ids:list[int])->Any:
    history=[248044,248044];rows=[]
    for token in ids:
      old1=history[1] if history[1]!=248044 else 248044;old2=history[0] if history[1]!=248044 and history[0]!=248044 else 248044
      for head in range(16):
        order=2 if head<8 else 3;m=(token*MULT[0])^(old1*MULT[1]);m=m if order==2 else m^(old2*MULT[2]);address=OFF[head]+m%MOD[head];member=address//2500012;row=address%2500012
        rows.extend([((member%7)+(row%13)+i)*0.015625 for i in range(2)])
      history=[history[1],token]
    return torch.tensor(rows,dtype=torch.float32).reshape(1,len(ids),32)

def fixed_embedding(torch:Any,ids:list[int])->Any:
    class Fixed(torch.nn.Module):
        def __init__(self)->None:super().__init__();self.ids=ids
        def forward(self,input_ids:Any,_cache:Any)->Any:
            got=input_ids.detach().cpu().tolist()[0]
            if got!=self.ids[-len(got):]:raise RuntimeError("unexpected PLE token suffix")
            return ple_rows(torch,got)
    return Fixed()

def capture(torch:Any,module:Any,records:dict[str,Any])->list[Any]:
    handles=[]
    names={"ple":"ple_output","attn_hyper_connection":"attn_gr","linear_attn":"mixer_output","self_attn":"mixer_output","mlp_hyper_connection":"moe_gr","mlp":"moe_output"}
    for name,label in names.items():
      child=dict(module.named_modules()).get(name)
      if child is None:continue
      def hook(_m:Any,_a:Any,out:Any,key:str=label)->None:
        value=out[0] if isinstance(out,(tuple,list)) else out
        records[key]=tensor(torch,value)
      handles.append(child.register_forward_hook(hook))
    return handles

def run_case(torch:Any,modeling:Any,DynamicCache:Any,Config:Any,family:str,case_id:str,seed:int)->dict[str,Any]:
    c=config(Config,family);module=modeling.Qwen4ExpTextDecoderLayer(c,0).float().eval();weights=fill(module,torch)
    ids=[11,22,33,44];hidden=torch.tensor([((i%19)-9)*0.03125 for i in range(128)],dtype=torch.float32).reshape(1,4,32)
    if seed&1:hidden=hidden+values(torch,128,0.0001).reshape(1,4,32)
    if family=="PLE_GDN":module.ple.ple_embedding=fixed_embedding(torch,ids)
    rope=modeling.Qwen4ExpTextRotaryEmbedding(c);positions=torch.arange(4,dtype=torch.long).reshape(1,-1);pos=rope(hidden,positions)
    mask=torch.full((1,1,4,4),torch.finfo(torch.float32).min)
    for q in range(4):mask[0,0,q,:q+1]=0.0
    cache=DynamicCache(config=c);records={};handles=capture(torch,module,records)
    with torch.no_grad():out=module(hidden,pos,attention_mask=mask,conv_mask=None,past_key_values=cache,ple_input_ids=torch.tensor([ids]))
    for h in handles:h.remove()
    return {"id":case_id,"family":family,"input":tensor(torch,hidden),"token_ids":ids,"weights":weights,"checkpoints":records,"output":tensor(torch,out),"state_position":4}

def generate(out:Path,checkout:Path)->None:
    os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1";sys.path.insert(0,str(checkout/"src"))
    import torch,tokenizers,transformers
    from transformers import DynamicCache
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
    if transformers.__version__!="5.16.0.dev0":raise SystemExit("unexpected Transformers revision")
    if tokenizers.__version__!="0.23.1":raise SystemExit("unexpected tokenizers version")
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.set_grad_enabled(False);out.mkdir(parents=True,exist_ok=True)
    contract={"schema":"kq-layer-contract-v1","authority":{"model_revision":MODEL_REVISION,"transformers_revision":TRANSFORMERS_REVISION,"source_sha256":SOURCE_SHA,"config_sha256":CONFIG_SHA,"license":"Apache-2.0"},
      "canonical":{"class":"Qwen4ExpTextDecoderLayer","gr_class":"Qwen4ExpTextGatedResidual","branches":4,"rank":320,"target_hidden":2560,"ple_zero_based_layer":1,"gdn_layers":36,"qsa_layers":12},
      "reduced":{"hidden":8,"branches":4,"rank":4,"experts":4,"top_k":2,"intermediate":4},"oracle_environment":{"python":platform.python_version(),"torch":torch.__version__,"tokenizers":tokenizers.__version__,"transformers":transformers.__version__}}
    write(out/"layer-contract.json",contract)
    calibration=[run_case(torch,modeling,DynamicCache,Qwen4ExpTextConfig,f,f"KQ-LAYER-CAL-{f}",0x2100+i*2) for i,f in enumerate(("GDN","QSA","PLE_GDN"))]
    holdout=[run_case(torch,modeling,DynamicCache,Qwen4ExpTextConfig,f,f"KQ-LAYER-HOLD-{f}",0x2201+i*2) for i,f in enumerate(("GDN","QSA","PLE_GDN"))]
    write(out/"layer-calibration.json",{"schema":"kq-layer-calibration-v1","cases":calibration})
    write(out/"layer-holdout.json",{"schema":"kq-layer-holdout-v1","cases":holdout})
    write(out/"layer-state-vectors.json",{"schema":"kq-layer-state-v1","cases":[{"family":c["family"],"initial_state":"ZERO","prefill_length":2,"decode_steps":2,"final_position":4,"reset_replay":"REQUIRED","post_failure_continuation":"REQUIRED","persistent_state":"suboperator-only; exact state tensors remain governed by the pinned GDN/QSA/PLE operator evidence"} for c in calibration]})
    write(out/"layer-family-vectors.json",{"schema":"kq-layer-family-v1","families":[{"family":c["family"],"case_id":c["id"],"output_sha256":hashlib.sha256(bytes.fromhex("".join(c["output"]["f32_le_hex"]))).hexdigest()} for c in calibration]})
    hashes={name:sha(out/name) for name in FILES};write(out/"layer-manifest.json",{"schema":"kq-layer-manifest-v1","status":"ORACLE_GENERATED_NATIVE_PENDING","files":hashes,"generation":{"network":False,"self_oracle":False}})

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--checkout",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();generate(a.output_dir.resolve(),a.checkout.resolve())
if __name__=="__main__":main()
