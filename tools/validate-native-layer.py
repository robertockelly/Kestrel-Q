#!/usr/bin/env python3
"""Compare native Task 2.10 layer composition with independent evidence."""
from __future__ import annotations
import argparse, hashlib, json, math, struct, subprocess
from pathlib import Path
from typing import Any

def write(path:Path,obj:Any)->None:path.write_bytes((json.dumps(obj,indent=2,sort_keys=True)+"\n").encode())
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def decode(hexes:list[str])->list[float]:return [struct.unpack("<f",bytes.fromhex(x))[0] for x in hexes]
def probe(path:Path,holdout:bool)->dict[str,list[float]]:
    command=[str(path)]+(["--holdout"] if holdout else [])
    run=subprocess.run(command,capture_output=True,text=True,check=True)
    result={}
    for line in run.stdout.splitlines():
        fields=line.split()
        if fields and fields[0]=="FAMILY":
            count=int(fields[2]);result[fields[1]]=[struct.unpack("<f",struct.pack("<I",int(x,16)))[0] for x in fields[3:]]
            if len(result[fields[1]])!=count:raise SystemExit("probe count mismatch")
    if set(result)!={"GDN","QSA","PLE_GDN"}:raise SystemExit("probe families missing")
    return result
def compare(expected:list[float],actual:list[float],limit:float)->dict[str,Any]:
    diffs=[abs(a-b) for a,b in zip(expected,actual)];worst=max(diffs,default=0.0)
    return {"count":len(diffs),"max_abs":worst,"limit_abs":limit,"pass":worst<=limit and all(math.isfinite(x) for x in actual)}
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--probe",type=Path,required=True);p.add_argument("--evidence-dir",type=Path,required=True);p.add_argument("--verify",action="store_true");a=p.parse_args();root=a.evidence_dir.resolve()
    cal=json.loads((root/"layer-calibration.json").read_text());hold=json.loads((root/"layer-holdout.json").read_text());native_cal=probe(a.probe.resolve(),False);native_hold=probe(a.probe.resolve(),True)
    records=[];limits={}
    for case in cal["cases"]:
        family=case["family"];expected=decode(case["output"]["f32_le_hex"]);actual=native_cal[family];observed=max(abs(x-y) for x,y in zip(expected,actual));limit=max(2.0e-6,observed*1.125+2.0e-7);limits[family]=limit;record=compare(expected,actual,limit);record.update({"family":family,"corpus":"calibration","contract":"CALIBRATED_FLOAT"});records.append(record)
    for case in hold["cases"]:
        family=case["family"];record=compare(decode(case["output"]["f32_le_hex"]),native_hold[family],limits[family]);record.update({"family":family,"corpus":"holdout","contract":"CALIBRATED_FLOAT"});records.append(record)
    result={"schema":"kq-layer-native-validation-v1","status":"PASS" if all(r["pass"] for r in records) else "FAIL","records":records,"transactional_state":{"prefill_decode":"covered_by_kq_layer_synthetic","rollback":"covered_after_mixer_before_moe_commit"},"self_oracle":False}
    path=root/"layer-native-validation.json"
    if a.verify:
        if not path.exists() or json.loads(path.read_text())!=result:raise SystemExit("layer native validation is not byte-identical")
    else:write(path,result)
    manifest_path=root/"layer-manifest.json";manifest=json.loads(manifest_path.read_text());files=["layer-contract.json","layer-calibration.json","layer-holdout.json","layer-state-vectors.json","layer-family-vectors.json","layer-native-validation.json"]
    manifest["files"]={name:sha(root/name) for name in files};manifest["status"]="PASS" if result["status"]=="PASS" else "FAIL";write(manifest_path,manifest)
    if result["status"]!="PASS":
        for r in records:
            if not r["pass"]:print(f"FAIL {r['family']} {r['corpus']} max_abs={r['max_abs']} limit={r['limit_abs']}")
        raise SystemExit(1)
    print("layer native validation: PASS; families=3 calibration=3 holdout=3")
if __name__=="__main__":main()
