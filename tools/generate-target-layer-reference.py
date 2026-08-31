#!/usr/bin/env python3
"""Generate Task 2.11 real-quantized target layer oracle evidence.

Packed weights are decoded only by the pinned llama.cpp helper.  Layer
semantics execute through the pinned Transformers components plus a small
selected-expert adapter, so native Kestrel-Q output is never an input.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_REVISION = "805a9e939fa8c1bff8d8ffdf041c051b71a914aa"
LLAMA_REVISION = "90c26fcd4b2114b4aa39d09d69318cb8f438d27a"
GGUF_SHA256 = "8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2"
GGUF_SIZE = 111_334_654_400
FILES = (
    "target-layer-contract.json",
    "target-layer-calibration.json",
    "target-layer-holdout.json",
    "target-layer-state-vectors.json",
)


def write(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32_hex(torch: Any, value: Any) -> list[str]:
    raw = value.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
    return [bytes(x).hex() for x in raw.reshape(-1).view("V4")]


def tensor(torch: Any, value: Any) -> dict[str, Any]:
    item = value.detach().cpu().float().contiguous()
    return {"dtype": "float32", "shape": list(item.shape), "f32_le_hex": f32_hex(torch, item)}


class Weights:
    def __init__(self, torch: Any, numpy: Any, gguf: Path, inventory: Path,
                 helper: Path, cache: Path) -> None:
        self.torch = torch
        self.np = numpy
        self.gguf = gguf
        self.helper = helper
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        with inventory.open(newline="", encoding="utf-8") as stream:
            self.rows = {row["gguf_tensor_name"]: row for row in csv.DictReader(stream)}
        self.loaded: dict[tuple[str, int | None], Any] = {}

    @staticmethod
    def dims(row: dict[str, str]) -> list[int]:
        return [int(x) for x in row["dimensions"].split("x")]

    def _decode(self, label: str, row: dict[str, str], offset: int,
                elements: int) -> Any:
        key = hashlib.sha256(f"{label}|{offset}|{elements}".encode()).hexdigest()[:24]
        output = self.cache / f"{key}.f32"
        if not output.exists() or output.stat().st_size != elements * 4:
            subprocess.run([
                str(self.helper), "dequant-file", row["type_name"],
                str(self.gguf), str(offset), str(elements), str(output)
            ], check=True, capture_output=True, text=True)
        if output.stat().st_size != elements * 4:
            raise RuntimeError(f"oracle decode length mismatch for {label}")
        values = self.np.fromfile(output, dtype="<f4")
        return self.torch.from_numpy(values.copy())

    def get(self, name: str, expert: int | None = None) -> Any:
        key = (name, expert)
        if key in self.loaded:
            return self.loaded[key]
        row = self.rows[name]
        dims = self.dims(row)
        offset = int(row["absolute_offset"])
        elements = int(row["parameter_count"])
        if expert is not None:
            if len(dims) != 3 or dims[2] != 512 or not 0 <= expert < 512:
                raise RuntimeError(f"invalid expert request for {name}")
            elements //= 512
            packed = int(row["packed_bytes"])
            if packed % 512:
                raise RuntimeError(f"non-contiguous expert packing for {name}")
            offset += expert * (packed // 512)
            shape = (dims[1], dims[0])
        elif len(dims) == 2:
            shape = (dims[1], dims[0])
        elif len(dims) == 1:
            shape = (dims[0],)
        else:
            raise RuntimeError(f"whole stacked tensor decode forbidden for {name}")
        result = self._decode(f"{name}:{expert}", row, offset, elements).reshape(shape)
        self.loaded[key] = result
        return result

    def ple_row(self, member: int, member_row: int) -> Any:
        name = "per_layer_token_embd.weight"
        row = self.rows[name]
        row_bytes = 5 * int(row["bytes_per_block"])
        offset = int(row["absolute_offset"]) + (member * 2_500_012 + member_row) * row_bytes
        return self._decode(f"ple:{member}:{member_row}", row, offset, 160)

    def clear_loaded(self) -> None:
        self.loaded.clear()
        gc.collect()


def grouped_to_tiled(torch: Any, count: int, width: int) -> Any:
    canonical = torch.arange(count, dtype=torch.long)
    group = canonical // (3 * width)
    within = canonical % (3 * width)
    repeat = within // width
    feature = within % width
    return (repeat * 16 + group) * width + feature


def assign_gr(torch: Any, module: Any, weights: Weights, prefix: str) -> None:
    with torch.no_grad():
        # The pinned qwen4exp converter inherits Qwen's zero-centred norm rule:
        # every *norm.weight except linear_attn.norm is stored as 1 + delta.
        module.hc_norm.weight.copy_(weights.get(f"{prefix}_norm.weight") - 1.0)
        module.input_mix_weight_down.weight.copy_(weights.get(f"{prefix}_down.weight"))
        module.input_mix_weight_up.weight.copy_(weights.get(f"{prefix}_up.weight"))
        module.block_inject_weight.weight.copy_(weights.get(f"{prefix}_inject.weight"))


def assign_gdn(torch: Any, module: Any, weights: Weights, layer: int) -> None:
    prefix = f"blk.{layer}"
    value_index = grouped_to_tiled(torch, 6144, 128)
    scalar_index = grouped_to_tiled(torch, 48, 1)
    qkv_index = torch.cat((torch.arange(4096), 4096 + value_index))
    conv_index = torch.cat((torch.arange(4096), 4096 + value_index))
    with torch.no_grad():
        module.in_proj_qkv.weight.copy_(weights.get(f"{prefix}.attn_qkv.weight")[qkv_index])
        module.in_proj_z.weight.copy_(weights.get(f"{prefix}.attn_gate.weight")[value_index])
        module.in_proj_a.weight.copy_(weights.get(f"{prefix}.ssm_alpha.weight")[scalar_index])
        module.in_proj_b.weight.copy_(weights.get(f"{prefix}.ssm_beta.weight")[scalar_index])
        module.conv1d.weight.copy_(weights.get(f"{prefix}.ssm_conv1d.weight")[conv_index].unsqueeze(1))
        stored_a = weights.get(f"{prefix}.ssm_a")[scalar_index]
        if not bool(torch.all(stored_a < 0)):
            raise RuntimeError("stored GDN decay base is not negative")
        module.A_log.copy_(torch.log(-stored_a))
        module.dt_bias.copy_(weights.get(f"{prefix}.ssm_dt.bias")[scalar_index])
        module.norm.weight.copy_(weights.get(f"{prefix}.ssm_norm.weight"))
        raw_out = weights.get(f"{prefix}.ssm_out.weight")
        module.out_proj.weight.copy_(raw_out[:, value_index])


def assign_qsa(torch: Any, module: Any, weights: Weights, layer: int) -> None:
    prefix = f"blk.{layer}"
    with torch.no_grad():
        module.q_proj.weight.copy_(weights.get(f"{prefix}.attn_q.weight"))
        module.k_proj.weight.copy_(weights.get(f"{prefix}.attn_k.weight"))
        module.v_proj.weight.copy_(weights.get(f"{prefix}.attn_v.weight"))
        module.o_proj.weight.copy_(weights.get(f"{prefix}.attn_output.weight"))
        module.q_norm.weight.copy_(weights.get(f"{prefix}.attn_q_norm.weight") - 1.0)
        module.k_norm.weight.copy_(weights.get(f"{prefix}.attn_k_norm.weight") - 1.0)
        index_q = weights.get(f"{prefix}.indexer.q_proj.weight")
        index_k = weights.get(f"{prefix}.indexer.k_proj.weight")
        module.indexer.index_qk_proj.weight.copy_(torch.cat((index_q, index_k), dim=0))
        module.indexer.q_layernorm.weight.copy_(
            weights.get(f"{prefix}.indexer.q_norm.weight") - 1.0)
        module.indexer.k_layernorm.weight.copy_(
            weights.get(f"{prefix}.indexer.k_norm.weight") - 1.0)


class SelectedMoe:
    def __init__(self, torch: Any, functional: Any, weights: Weights,
                 layer: int) -> None:
        self.torch = torch
        self.F = functional
        self.weights = weights
        self.layer = layer
        prefix = f"blk.{layer}"
        self.router = weights.get(f"{prefix}.ffn_gate_inp.weight")
        self.shared_gate = weights.get(f"{prefix}.ffn_gate_shexp.weight")
        self.shared_up = weights.get(f"{prefix}.ffn_up_shexp.weight")
        self.shared_down = weights.get(f"{prefix}.ffn_down_shexp.weight")
        self.shared_control = weights.get(f"{prefix}.ffn_gate_inp_shexp.weight")
        self.routes: list[list[int]] = []
        self.route_weights: list[list[float]] = []

    def __call__(self, hidden: Any) -> Any:
        flat = hidden.reshape(-1, 2560)
        logits = self.F.linear(flat, self.router)
        probabilities = self.torch.softmax(logits, dim=-1, dtype=self.torch.float32)
        top, indices = self.torch.topk(probabilities, 10, dim=-1)
        top = top / top.sum(dim=-1, keepdim=True)
        self.routes.extend(indices.detach().cpu().tolist())
        self.route_weights.extend(top.detach().cpu().tolist())
        routed = self.torch.zeros_like(flat)
        prefix = f"blk.{self.layer}"
        for token in range(flat.shape[0]):
            for slot in range(10):
                expert = int(indices[token, slot])
                gate = self.F.linear(flat[token], self.weights.get(
                    f"{prefix}.ffn_gate_exps.weight", expert))
                up = self.F.linear(flat[token], self.weights.get(
                    f"{prefix}.ffn_up_exps.weight", expert))
                value = self.F.silu(gate) * up
                value = self.F.linear(value, self.weights.get(
                    f"{prefix}.ffn_down_exps.weight", expert))
                routed[token] += value * top[token, slot]
        shared = self.F.silu(self.F.linear(flat, self.shared_gate))
        shared = shared * self.F.linear(flat, self.shared_up)
        shared = self.F.linear(shared, self.shared_down)
        shared = self.torch.sigmoid(self.F.linear(flat, self.shared_control)) * shared
        return (routed + shared).reshape(hidden.shape)


MULTIPLIERS = (23703573157769, 20109073645365, 8052911324071)
OFFSETS = (0, 20000003, 40000026, 60000059, 80000106, 100000165,
           120000228, 140000297, 160000374, 180000455, 200000548,
           220000655, 240000802, 260000955, 280001114, 300001275)
MODULI = (20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
          20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
          20000153, 20000159, 20000161, 20000171)


class PleEmbedding:
    def __init__(self, torch: Any, weights: Weights) -> None:
        self.torch = torch
        self.weights = weights
        self.history = [248044, 248044]
        self.requests: list[tuple[int, int]] = []

    def __call__(self, input_ids: Any, _cache: Any) -> Any:
        rows = []
        for token_value in input_ids.detach().cpu().reshape(-1).tolist():
            token = int(token_value)
            old1 = self.history[1] if self.history[1] != 248044 else 248044
            old2 = self.history[0] if self.history[1] != 248044 and self.history[0] != 248044 else 248044
            pieces = []
            for head in range(16):
                mixed = token * MULTIPLIERS[0] ^ old1 * MULTIPLIERS[1]
                if head >= 8:
                    mixed ^= old2 * MULTIPLIERS[2]
                address = OFFSETS[head] + mixed % MODULI[head]
                member, member_row = divmod(address, 2_500_012)
                self.requests.append((member, member_row))
                pieces.append(self.weights.ple_row(member, member_row))
            rows.append(self.torch.cat(pieces))
            self.history = [self.history[1], token]
        return self.torch.stack(rows).reshape(1, len(rows), 2560)


def assign_ple(torch: Any, module: Any, weights: Weights) -> PleEmbedding:
    embedding = PleEmbedding(torch, weights)
    class Adapter(torch.nn.Module):
        def forward(self, input_ids: Any, cache: Any) -> Any:
            return embedding(input_ids, cache)
    prefix = "blk.1"
    module.ple_embedding = Adapter()
    with torch.no_grad():
        module.key_proj.weight.copy_(weights.get(f"{prefix}.ple_key.weight"))
        module.value_proj.weight.copy_(weights.get(f"{prefix}.ple_value.weight"))
        module.norm_key.weight.copy_(weights.get(f"{prefix}.ple_norm_key.weight") - 1.0)
        module.norm_query.weight.copy_(weights.get(f"{prefix}.ple_norm_query.weight") - 1.0)
        module.norm_conv.weight.copy_(weights.get(f"{prefix}.ple_norm_conv.weight") - 1.0)
        module.conv1d.weight.copy_(weights.get(f"{prefix}.ple_conv1d.weight").unsqueeze(1))
    return embedding


def state_summary(torch: Any, cache: Any, layer_id: int) -> dict[str, Any]:
    layer = cache.layers[layer_id]
    summary: dict[str, Any] = {}
    for name, value in vars(layer).items():
        tensors = []
        if torch.is_tensor(value):
            tensors = [value]
        elif isinstance(value, (list, tuple)):
            tensors = [item for item in value if torch.is_tensor(item)]
        if not tensors:
            continue
        raw = b"".join(item.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False).tobytes()
                       for item in tensors)
        summary[name] = {
            "tensor_count": len(tensors),
            "elements": sum(item.numel() for item in tensors),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return summary


def configure(torch: Any, Config: Any, source: dict[str, Any]) -> Any:
    config = Config.from_dict(source["text_config"])
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    config.use_cache = True
    return config


def make_input(torch: Any, phase: str, profile: str) -> Any:
    if profile == "holdout":
        modulus = 41 if phase == "prefill" else 47
        center = 20 if phase == "prefill" else 23
        scale = 2.0 ** (-12 if phase == "prefill" else -13)
        multiplier = 1
    elif profile == "calibration-d":
        modulus = 67 if phase == "prefill" else 71
        center = 33 if phase == "prefill" else 35
        scale = 2.0 ** (-12 if phase == "prefill" else -13)
        multiplier = 17 if phase == "prefill" else 29
    elif profile == "calibration-c":
        modulus = 43 if phase == "prefill" else 61
        center = 21 if phase == "prefill" else 30
        scale = 2.0 ** (-12 if phase == "prefill" else -13)
        multiplier = 1
    elif profile == "calibration-b":
        modulus = 53 if phase == "prefill" else 59
        center = 26 if phase == "prefill" else 29
        scale = 2.0 ** (-12 if phase == "prefill" else -13)
        multiplier = 1
    else:
        modulus = 31 if phase == "prefill" else 37
        center = 15 if phase == "prefill" else 18
        scale = 2.0 ** (-10 if phase == "prefill" else -11)
        multiplier = 1
    values = [(((index * multiplier) % modulus) - center) * scale
              for index in range(10240)]
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 10240)


def run_family(torch: Any, modeling: Any, Config: Any, DynamicCache: Any,
               weights: Weights, config_json: dict[str, Any], family: str,
               layer_id: int, profile: str) -> dict[str, Any]:
    config = configure(torch, Config, config_json)
    attention_gr = modeling.Qwen4ExpTextGatedResidual(config).float().eval()
    moe_gr = modeling.Qwen4ExpTextGatedResidual(config).float().eval()
    assign_gr(torch, attention_gr, weights, f"blk.{layer_id}.hc_attn")
    assign_gr(torch, moe_gr, weights, f"blk.{layer_id}.hc_ffn")
    if family == "QSA":
        mixer = modeling.Qwen4ExpTextAttention(config, layer_id).float().eval()
        assign_qsa(torch, mixer, weights, layer_id)
    else:
        mixer = modeling.Qwen4ExpTextGatedDeltaNet(config, layer_id).float().eval()
        assign_gdn(torch, mixer, weights, layer_id)
    ple = None
    ple_embedding = None
    if family == "PLE_GDN":
        reduced = configure(torch, Config, config_json)
        reduced.ngram_vocab_size_base = 11
        reduced.make_ngram_vocab_size_divisible_by = 4
        ple = modeling.Qwen4ExpTextPLELayer(reduced, 1, 0).float().eval()
        ple_embedding = assign_ple(torch, ple, weights)
    moe = SelectedMoe(torch, torch.nn.functional, weights, layer_id)
    cache = DynamicCache(config=config)
    rotary = modeling.Qwen4ExpTextRotaryEmbedding(config)
    results = []
    with torch.no_grad():
        for step, phase in enumerate(("prefill", "decode")):
            hidden = make_input(torch, phase, profile)
            token_id = 100 + layer_id + step
            if ple is not None:
                hidden = hidden + ple(hidden, torch.tensor([[token_id]], dtype=torch.long), cache)
            mixed, hyper, inject = attention_gr(hidden)
            if family == "QSA":
                positions = torch.arange(step + 1, dtype=torch.long).reshape(1, -1)
                position_embeddings = rotary(mixed, positions)
                mask = torch.zeros((1, 1, 1, step + 1), dtype=torch.float32)
                block, _ = mixer(mixed, position_embeddings, mask, cache)
            else:
                block = mixer(mixed, cache_params=cache, attention_mask=None)
            hidden = hyper + (block.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)
            mixed, hyper, inject = moe_gr(hidden)
            block = moe(mixed)
            hidden = hyper + (block.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)
            results.append({
                "phase": phase,
                "input": tensor(torch, make_input(torch, phase, profile)),
                "output": tensor(torch, hidden),
                "selected_expert_ids": moe.routes[-1],
                "selected_weights": moe.route_weights[-1],
                "token_id": token_id,
            })
    result = {
        "id": f"KQ-TARGET-LAYER-{profile.upper()}-{family}",
        "profile": profile,
        "family": family,
        "layer_id": layer_id,
        "steps": results,
        "state": state_summary(torch, cache, layer_id),
    }
    if ple_embedding is not None:
        result["ple_requests"] = [
            {"member": member, "row": row}
            for member, row in ple_embedding.requests
        ]
    del attention_gr, moe_gr, mixer, ple, moe, cache
    weights.clear_loaded()
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--llama-helper", type=Path, required=True)
    parser.add_argument("--transformers-checkout", type=Path, required=True)
    parser.add_argument("--oracle-site-packages", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    gguf = args.gguf.resolve()
    if gguf.stat().st_size != GGUF_SIZE:
        raise SystemExit("registered GGUF size mismatch")
    checkout = args.transformers_checkout.resolve()
    if subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                      check=True, capture_output=True, text=True).stdout.strip() != TRANSFORMERS_REVISION:
        raise SystemExit("Transformers checkout revision mismatch")
    sys.path.insert(0, str(checkout / "src"))
    sys.path.insert(1, str(args.oracle_site_packages.resolve()))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy
    import torch
    import transformers
    from transformers import DynamicCache
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_config = json.loads(args.config.read_text(encoding="utf-8"))
    weights = Weights(torch, numpy, gguf, args.inventory.resolve(),
                      args.llama_helper.resolve(), args.cache_dir.resolve())
    contract = {
        "schema": "kq-target-layer-contract-v1",
        "status": "INDEPENDENT_ORACLE_GENERATED",
        "authority": {
            "model_revision": MODEL_REVISION,
            "transformers_revision": TRANSFORMERS_REVISION,
            "llama_cpp_revision": LLAMA_REVISION,
            "gguf_sha256": GGUF_SHA256,
            "gguf_size": GGUF_SIZE,
            "licenses": {"transformers": "Apache-2.0", "llama.cpp": "MIT"},
        },
        "method": "pinned llama.cpp dequant-file plus pinned Transformers components; no native output input",
        "converter_zero_centered_gamma_inverse": {
            "operation": "canonical_delta = stored_gamma - 1",
            "source_rule": "qwen inherited *norm.weight except linear_attn.norm plus qwen4exp indexer/PLE exceptions",
            "applied_roles": [
                "attention_hc_norm", "mlp_hc_norm", "qsa_q_norm",
                "qsa_k_norm", "qsa_index_q_norm", "qsa_index_k_norm",
                "ple_norm_key", "ple_norm_query", "ple_norm_conv"
            ],
            "excluded_role": "gdn_linear_attn_norm"
        },
        "representative_layers": {"GDN": 0, "QSA": 3, "PLE_GDN": 1},
        "input_generation": {
            "calibration": "prefill ((i%31)-15)*2^-10; decode ((i%37)-18)*2^-11",
            "calibration_secondary": "prefill ((i%53)-26)*2^-12; decode ((i%59)-29)*2^-13",
            "calibration_tertiary": "prefill ((i%43)-21)*2^-12; decode ((i%61)-30)*2^-13",
            "calibration_permuted": "prefill (((17*i)%67)-33)*2^-12; decode (((29*i)%71)-35)*2^-13",
            "holdout": "prefill ((i%41)-20)*2^-12; decode ((i%47)-23)*2^-13",
        },
        "environment": {
            "python": platform.python_version(), "numpy": numpy.__version__,
            "torch": torch.__version__, "transformers": transformers.__version__,
        },
    }
    write(output / "target-layer-contract.json", contract)
    families = (("GDN", 0), ("QSA", 3), ("PLE_GDN", 1))
    calibration = [run_family(torch, modeling, Qwen4ExpTextConfig, DynamicCache,
                              weights, source_config, family, layer, profile)
                   for profile in ("calibration-a", "calibration-b",
                                   "calibration-c", "calibration-d")
                   for family, layer in families]
    holdout = [run_family(torch, modeling, Qwen4ExpTextConfig, DynamicCache,
                         weights, source_config, family, layer, "holdout")
               for family, layer in families]
    write(output / "target-layer-calibration.json",
          {"schema": "kq-target-layer-calibration-v1", "cases": calibration})
    write(output / "target-layer-holdout.json",
          {"schema": "kq-target-layer-holdout-v1", "cases": holdout})
    write(output / "target-layer-state-vectors.json", {
        "schema": "kq-target-layer-state-vectors-v1",
        "calibration": [{"family": c["family"], "state": c["state"]} for c in calibration],
        "holdout": [{"family": c["family"], "state": c["state"]} for c in holdout],
    })
    hashes = {name: sha256(output / name) for name in FILES}
    write(output / "target-layer-manifest.json", {
        "schema": "kq-target-layer-manifest-v1",
        "status": "ORACLE_GENERATED_NATIVE_PENDING",
        "files": hashes,
        "self_oracle": False,
    })
    print("target layer independent oracle: generated")


if __name__ == "__main__":
    main()
