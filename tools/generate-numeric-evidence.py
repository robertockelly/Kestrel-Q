#!/usr/bin/env python3
"""Generate Task 2.5 synthetic and scalar numeric evidence.

Expected dequantized values come from the pinned llama.cpp helper. Expected
primitive values come from the explicitly ordered NumPy oracle below. Native
Kestrel-Q output is recorded only as an observation and never defines an
expected value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import subprocess
from pathlib import Path

import numpy as np


MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
LLAMA_REVISION = "90c26fcd4b2114b4aa39d09d69318cb8f438d27a"
TRANSFORMERS_REVISION = "805a9e939fa8c1bff8d8ffdf041c051b71a914aa"
TYPE_GEOMETRY = {
    "F32": (1, 4),
    "BF16": (1, 2),
    "Q5_1": (32, 24),
    "Q8_0": (32, 34),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "IQ4_NL": (32, 18),
}


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def parse_bits_line(text: str) -> list[str]:
    if not text.startswith("bits="):
        raise RuntimeError(f"unexpected probe output: {text!r}")
    return text[5:].split(",")


def f32(value: float | np.float32) -> np.float32:
    return np.float32(value)


def f32_bits(value: float | np.float32) -> str:
    return f"{struct.unpack('<I', struct.pack('<f', float(f32(value))))[0]:08x}"


def bits_to_f32(bits: str) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", int(bits, 16)))[0])


def bits_list(values: list[np.float32]) -> list[str]:
    return [f32_bits(value) for value in values]


def csv_bits(values: list[np.float32]) -> str:
    return ",".join(bits_list(values))


def ordered_ulp_key(bits: int) -> int:
    if bits & 0x80000000:
        return 0x80000000 - (bits & 0x7FFFFFFF)
    return 0x80000000 + bits


def ulp_distance(left: str, right: str) -> int:
    return abs(ordered_ulp_key(int(left, 16)) - ordered_ulp_key(int(right, 16)))


def float_metrics(expected: list[str], observed: list[str]) -> dict[str, float | int]:
    if len(expected) != len(observed):
        raise RuntimeError("native output length differs from oracle")
    maximum_abs = 0.0
    maximum_rel = 0.0
    maximum_ulp = 0
    for expected_bits, observed_bits in zip(expected, observed, strict=True):
        expected_value = float(bits_to_f32(expected_bits))
        observed_value = float(bits_to_f32(observed_bits))
        absolute = abs(observed_value - expected_value)
        relative = absolute / abs(expected_value) if expected_value != 0.0 else (0.0 if absolute == 0.0 else math.inf)
        maximum_abs = max(maximum_abs, absolute)
        maximum_rel = max(maximum_rel, relative)
        maximum_ulp = max(maximum_ulp, ulp_distance(expected_bits, observed_bits))
    return {"max_abs": maximum_abs, "max_rel": maximum_rel, "max_ulp": maximum_ulp}


def synthetic_blocks() -> list[dict[str, str]]:
    rng = random.Random(0x4B51543235)
    cases: list[dict[str, str]] = []

    for index, bits in enumerate((0x00000000, 0x80000000, 0x3F800000,
                                  0x00000001, 0x7F800000, 0x7FC12345)):
        cases.append({"id": f"f32-edge-{index}", "type": "F32",
                      "pattern": "ieee-edge", "packed_hex": struct.pack("<I", bits).hex()})
    for index, bits in enumerate((0x0000, 0x8000, 0x3F80, 0x0001, 0x7F80, 0x7FC1)):
        cases.append({"id": f"bf16-edge-{index}", "type": "BF16",
                      "pattern": "ieee-edge", "packed_hex": struct.pack("<H", bits).hex()})

    for type_name, (_, block_bytes) in TYPE_GEOMETRY.items():
        if type_name in ("F32", "BF16"):
            continue
        zero = bytearray(block_bytes)
        cases.append({"id": f"{type_name.lower()}-zero", "type": type_name,
                      "pattern": "zero", "packed_hex": zero.hex()})

        alternating = bytearray((0xAA if i % 2 == 0 else 0x55) for i in range(block_bytes))
        alternating[0:2] = struct.pack("<e", 1.0)
        if type_name in ("Q5_1", "Q4_K", "Q5_K"):
            alternating[2:4] = struct.pack("<e", 0.25)
        cases.append({"id": f"{type_name.lower()}-alternating", "type": type_name,
                      "pattern": "alternating-bits", "packed_hex": alternating.hex()})

        if type_name != "IQ4_NL":
            extrema = bytearray(block_bytes)
            extrema[0:2] = struct.pack("<e", 1.0)
            if type_name == "Q5_1":
                extrema[4:8] = b"\xff" * 4
                extrema[8:] = b"\xf0" * 16
            elif type_name == "Q8_0":
                extrema[2:] = bytes(0x80 if i % 2 == 0 else 0x7F for i in range(32))
            elif type_name == "Q4_K":
                extrema[4:16] = b"\xff" * 12
                extrema[16:] = b"\xf0" * 128
            elif type_name == "Q5_K":
                extrema[4:16] = b"\xff" * 12
                extrema[16:48] = b"\xff" * 32
                extrema[48:] = b"\xf0" * 128
            cases.append({"id": f"{type_name.lower()}-code-extrema", "type": type_name,
                          "pattern": "quant-code-extrema", "packed_hex": extrema.hex()})

        negative = bytearray(rng.getrandbits(8) for _ in range(block_bytes))
        negative[0:2] = struct.pack("<e", -0.5)
        if type_name in ("Q5_1", "Q4_K", "Q5_K"):
            negative[2:4] = struct.pack("<e", -0.125)
        cases.append({"id": f"{type_name.lower()}-negative-scale", "type": type_name,
                      "pattern": "negative-scale", "packed_hex": negative.hex()})

        random_block = bytearray(rng.getrandbits(8) for _ in range(block_bytes))
        random_block[0:2] = struct.pack("<e", 0.03125)
        if type_name in ("Q5_1", "Q4_K", "Q5_K"):
            random_block[2:4] = struct.pack("<e", 0.015625)
        cases.append({"id": f"{type_name.lower()}-random", "type": type_name,
                      "pattern": "seeded-random", "packed_hex": random_block.hex()})

        if type_name in ("Q4_K", "Q5_K"):
            metadata_edges = bytearray(block_bytes)
            metadata_edges[0:2] = struct.pack("<e", 0.125)
            metadata_edges[2:4] = struct.pack("<e", 0.0625)
            metadata_edges[4:16] = b"\xff" * 12
            metadata_edges[16:] = bytes((i * 37 + 11) & 0xFF for i in range(block_bytes - 16))
            cases.append({"id": f"{type_name.lower()}-metadata-extrema", "type": type_name,
                          "pattern": "six-bit-scale-min-extrema", "packed_hex": metadata_edges.hex()})
        if type_name == "IQ4_NL":
            coverage = bytearray(block_bytes)
            coverage[0:2] = struct.pack("<e", 1.0)
            coverage[2:] = bytes((15 - i) << 4 | i for i in range(16))
            cases.append({"id": "iq4_nl-codebook-coverage", "type": type_name,
                          "pattern": "all-codebook-indices", "packed_hex": coverage.hex()})
    return cases


def generate_dequant(llama_oracle: Path, native_probe: Path) -> dict[str, object]:
    cases = synthetic_blocks()
    for case in cases:
        expected = parse_bits_line(run([str(llama_oracle), case["type"], case["packed_hex"]]))
        observed = parse_bits_line(run([str(native_probe), "dequant", case["type"], case["packed_hex"]]))
        if expected != observed:
            raise RuntimeError(f"dequant mismatch for {case['id']}")
        case["comparison"] = "EXACT_BITS"
        case["expected_f32_bits"] = expected
        case["observed_native_f32_bits"] = observed

    row_dot_cases: list[dict[str, object]] = []
    for type_name in TYPE_GEOMETRY:
        candidates = [case for case in cases if case["type"] == type_name and
                      case["pattern"] not in ("ieee-edge", "zero")]
        if not candidates:
            # F32/BF16 finite one-element source.
            source = next(case for case in cases if case["type"] == type_name and
                          case["id"].endswith("-2"))
        else:
            source = candidates[0]
        expected_values = [bits_to_f32(bits) for bits in source["expected_f32_bits"]]
        activation = [f32(((index % 11) - 5) / 8.0) for index in range(len(expected_values))]
        accumulator = f32(0.0)
        for weight, value in zip(expected_values, activation, strict=True):
            product = f32(weight * value)
            accumulator = f32(accumulator + product)
        observed = parse_bits_line(run([str(native_probe), "rowdot", type_name,
                                        source["packed_hex"], csv_bits(activation)]))
        expected = [f32_bits(accumulator)]
        if expected != observed:
            raise RuntimeError(f"row-dot mismatch for {type_name}")
        row_dot_cases.append({
            "id": f"rowdot-{type_name.lower()}",
            "type": type_name,
            "source_case_id": source["id"],
            "activation_f32_bits": bits_list(activation),
            "expected_f32_bits": expected,
            "observed_native_f32_bits": observed,
            "comparison": "EXACT_BITS",
            "scratch_bytes": 1024,
        })

    return {
        "asset_id": "KQ-NUMERIC-DEQUANT-001",
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "oracle": {"implementation": "ggml-org/llama.cpp",
                   "revision": LLAMA_REVISION, "license": "MIT",
                   "method": "ggml_get_type_traits(type)->to_float; F32 bit-copy"},
        "type_geometry": {name: {"elements_per_block": values[0], "bytes_per_block": values[1]}
                          for name, values in TYPE_GEOMETRY.items()},
        "cases": cases,
        "row_dot_cases": row_dot_cases,
        "summary": {"types": 7, "dequant_cases": len(cases),
                    "row_dot_cases": len(row_dot_cases), "mismatches": 0},
    }


def primitive_inputs(corpus: str) -> list[dict[str, object]]:
    seed = 0xC411B if corpus == "CALIBRATION" else 0xA01D0
    rng = random.Random(seed)
    cases: list[dict[str, object]] = []
    lengths = (9, 17, 31) if corpus == "CALIBRATION" else (13, 29)
    for ordinal, length in enumerate(lengths):
        left = [f32(rng.uniform(-6.0, 6.0)) for _ in range(length)]
        right = [f32(rng.uniform(-3.0, 3.0)) for _ in range(length)]
        if ordinal == 0:
            left[:5] = [f32(-12.0), f32(-1.0), f32(-0.0), f32(1.0), f32(12.0)]
            right[:5] = [f32(0.5), f32(-2.0), f32(3.0), f32(-0.25), f32(0.125)]
        suffix = f"{corpus.lower()}-{ordinal}"
        for operation in ("add", "multiply", "dot", "sigmoid", "silu", "swiglu",
                          "rmsnorm", "softmax"):
            cases.append({"id": f"{operation}-{suffix}", "operation": operation,
                          "left": left, "right": right,
                          "epsilon": f32(1.0e-6)})
        positive = [f32(abs(float(value)) + 0.01) for value in right[: min(length, 10)]]
        cases.append({"id": f"renormalize-{suffix}", "operation": "renormalize",
                      "left": positive, "right": [], "epsilon": f32(0.0)})
    scale_values = [f32(-0.75), f32(1.125), f32(0.03125)] if corpus == "CALIBRATION" else [f32(-1.25), f32(0.625)]
    for ordinal, scale in enumerate(scale_values):
        values = [f32(rng.uniform(-8.0, 8.0)) for _ in range(11 + ordinal)]
        cases.append({"id": f"scale-{corpus.lower()}-{ordinal}", "operation": "scale",
                      "left": values, "right": [scale], "epsilon": f32(0.0)})
    ties = [f32(3.0), f32(-1.0), f32(3.0), f32(2.0), f32(2.0), f32(0.0)]
    cases.append({"id": f"topk-{corpus.lower()}-ties", "operation": "topk",
                  "left": ties, "right": [], "epsilon": f32(0.0), "k": 4})
    return cases


def oracle_primitive(case: dict[str, object]) -> tuple[list[str], list[int] | None]:
    operation = str(case["operation"])
    left = list(case["left"])
    right = list(case["right"])
    result: list[np.float32] = []
    if operation == "add":
        result = [f32(a + b) for a, b in zip(left, right, strict=True)]
    elif operation == "multiply":
        result = [f32(a * b) for a, b in zip(left, right, strict=True)]
    elif operation == "scale":
        result = [f32(a * right[0]) for a in left]
    elif operation == "dot":
        accumulator = f32(0.0)
        for a, b in zip(left, right, strict=True):
            accumulator = f32(accumulator + f32(a * b))
        result = [accumulator]
    elif operation in ("sigmoid", "silu", "swiglu"):
        for index, value in enumerate(left):
            if value >= f32(0.0):
                exponential = f32(np.exp(f32(-value)))
                sigmoid = f32(f32(1.0) / f32(f32(1.0) + exponential))
            else:
                exponential = f32(np.exp(value))
                sigmoid = f32(exponential / f32(f32(1.0) + exponential))
            if operation == "sigmoid": result.append(sigmoid)
            elif operation == "silu": result.append(f32(value * sigmoid))
            else: result.append(f32(f32(value * sigmoid) * right[index]))
    elif operation == "rmsnorm":
        accumulator = f32(0.0)
        for value in left:
            accumulator = f32(accumulator + f32(value * value))
        mean = f32(accumulator / f32(len(left)))
        inverse = f32(f32(1.0) / f32(np.sqrt(f32(mean + case["epsilon"]))))
        for value, weight in zip(left, right, strict=True):
            normalized = f32(value * inverse)
            result.append(f32(normalized * f32(f32(1.0) + weight)))
    elif operation == "softmax":
        maximum = max(left)
        total = f32(0.0)
        for value in left:
            exponential = f32(np.exp(f32(value - maximum)))
            result.append(exponential)
            total = f32(total + exponential)
        result = [f32(value / total) for value in result]
    elif operation == "renormalize":
        total = f32(0.0)
        for value in left: total = f32(total + value)
        result = [f32(value / total) for value in left]
    elif operation == "topk":
        indices = sorted(range(len(left)), key=lambda index: (-float(left[index]), index))[: int(case["k"])]
        return bits_list([left[index] for index in indices]), indices
    else:
        raise RuntimeError(f"unsupported oracle operation {operation}")
    return bits_list(result), None


def native_primitive(native_probe: Path, case: dict[str, object]) -> tuple[list[str], list[int] | None]:
    operation = str(case["operation"])
    left = list(case["left"])
    right = list(case["right"])
    command = [str(native_probe), operation, csv_bits(left)]
    if operation in ("add", "multiply", "dot", "swiglu"):
        command.append(csv_bits(right))
    elif operation == "scale":
        command.append(csv_bits(right))
    elif operation == "rmsnorm":
        command.extend((csv_bits(right), csv_bits([case["epsilon"]])))
    elif operation == "topk":
        command.append(str(case["k"]))
    text = run(command)
    if operation == "topk":
        prefix, bit_text = text.split(";", 1)
        indices = [int(value) for value in prefix.removeprefix("indices=").split(",")]
        return parse_bits_line(bit_text), indices
    return parse_bits_line(text), None


def generate_primitive_corpus(native_probe: Path, corpus: str,
                              contracts: dict[str, object] | None = None) -> dict[str, object]:
    output_cases: list[dict[str, object]] = []
    aggregate: dict[str, dict[str, float | int]] = {}
    for raw_case in primitive_inputs(corpus):
        expected, expected_indices = oracle_primitive(raw_case)
        observed, observed_indices = native_primitive(native_probe, raw_case)
        if expected_indices != observed_indices:
            raise RuntimeError(f"discrete top-k mismatch for {raw_case['id']}")
        metrics = float_metrics(expected, observed)
        operation = str(raw_case["operation"])
        totals = aggregate.setdefault(operation, {"max_abs": 0.0, "max_rel": 0.0, "max_ulp": 0})
        totals["max_abs"] = max(float(totals["max_abs"]), float(metrics["max_abs"]))
        totals["max_rel"] = max(float(totals["max_rel"]), float(metrics["max_rel"]))
        totals["max_ulp"] = max(int(totals["max_ulp"]), int(metrics["max_ulp"]))
        item: dict[str, object] = {
            "id": raw_case["id"], "operation": operation,
            "input_left_f32_bits": bits_list(list(raw_case["left"])),
            "expected_f32_bits": expected,
            "observed_native_f32_bits": observed,
            "observed_difference": metrics,
        }
        if raw_case["right"]:
            item["input_right_f32_bits"] = bits_list(list(raw_case["right"]))
        if operation == "rmsnorm": item["epsilon_f32_bits"] = f32_bits(raw_case["epsilon"])
        if expected_indices is not None:
            item["k"] = raw_case["k"]
            item["expected_indices"] = expected_indices
            item["observed_native_indices"] = observed_indices
        output_cases.append(item)

    if contracts is None:
        generated_contracts: dict[str, object] = {}
        for operation, metrics in aggregate.items():
            max_ulp = int(metrics["max_ulp"])
            classification = "EXACT_DISCRETE" if operation == "topk" else (
                "EXACT_BITS" if max_ulp == 0 else "CALIBRATED_FLOAT")
            generated_contracts[operation] = {
                "classification": classification,
                "calibration_observed": metrics,
                "acceptance": {
                    "ulp_limit": 0 if max_ulp == 0 else max_ulp + 1,
                    "margin_basis": "exact calibration maximum plus one adjacent F32 representable step" if max_ulp else "bit equality",
                    "nan_policy": "FORBID",
                    "inf_policy": "FORBID",
                },
            }
        contracts = generated_contracts
    else:
        for operation, metrics in aggregate.items():
            limit = int(contracts[operation]["acceptance"]["ulp_limit"])
            if int(metrics["max_ulp"]) > limit:
                raise RuntimeError(f"holdout {operation} exceeded calibrated ULP contract")

    return {
        "asset_id": f"KQ-NUMERIC-PRIMITIVE-{corpus}-001",
        "schema_version": 1,
        "corpus": corpus,
        "model_revision": MODEL_REVISION,
        "oracle": {"implementation": "Python/NumPy explicit F32 scalar order",
                   "python": "3.13.12", "numpy": np.__version__,
                   "license": "BSD-3-Clause", "model_semantics_revision": TRANSFORMERS_REVISION},
        "contracts": contracts,
        "cases": output_cases,
        "summary": {"cases": len(output_cases), "operations": len(aggregate),
                    "all_discrete_exact": True,
                    "within_contract": True},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-oracle", type=Path, required=True)
    parser.add_argument("--native-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.llama_oracle.is_file() or not args.native_probe.is_file():
        raise SystemExit("oracle and native probe executables are required")

    dequant = generate_dequant(args.llama_oracle.resolve(), args.native_probe.resolve())
    calibration = generate_primitive_corpus(args.native_probe.resolve(), "CALIBRATION")
    holdout = generate_primitive_corpus(args.native_probe.resolve(), "HOLDOUT", calibration["contracts"])
    write_json(args.output_dir / "dequant-vectors.json", dequant)
    write_json(args.output_dir / "primitive-calibration.json", calibration)
    write_json(args.output_dir / "primitive-holdout.json", holdout)
    real_path = args.output_dir / "real-gguf-samples.json"
    if real_path.is_file():
        tool_root = Path(__file__).resolve().parent
        asset_names = ("dequant-vectors.json", "primitive-calibration.json",
                       "primitive-holdout.json", "real-gguf-samples.json")
        manifest = {
            "asset_id": "KQ-NUMERIC-MANIFEST-001",
            "schema_version": 1,
            "model_revision": MODEL_REVISION,
            "oracles": [
                {"class": "Q", "implementation": "ggml-org/llama.cpp",
                 "revision": LLAMA_REVISION, "license": "MIT"},
                {"class": "primitive", "implementation": "Python/NumPy explicit F32 scalar order",
                 "python": "3.13.12", "numpy": np.__version__, "license": "BSD-3-Clause",
                 "model_semantics_revision": TRANSFORMERS_REVISION},
            ],
            "assets": [{"path": name, "sha256": sha256(args.output_dir / name)}
                       for name in asset_names],
            "tools": [{"path": name, "sha256": sha256(tool_root / name)} for name in (
                "generate-numeric-evidence.py", "capture-real-numeric-samples.py",
                "validate-native-numerics.py", "llama-dequant-oracle.cpp")],
            "generation": {
                "synthetic": "python tools/generate-numeric-evidence.py --llama-oracle <PINNED_HELPER> --native-probe <KQ_PROBE> --output-dir <OUTPUT>",
                "real": "python tools/capture-real-numeric-samples.py --real-probe <KQ_REAL_PROBE> --llama-oracle <PINNED_HELPER> --output <OUTPUT>/real-gguf-samples.json",
                "regeneration_order": ["synthetic", "real", "synthetic (manifest refresh)"],
            },
            "safety": {"raw_real_model_bytes_committed": 0,
                       "real_payload_sample_budget_bytes": 1048576},
        }
        write_json(args.output_dir / "manifest.json", manifest)
    print(f"dequant_cases={dequant['summary']['dequant_cases']}")
    print(f"row_dot_cases={dequant['summary']['row_dot_cases']}")
    print(f"calibration_cases={calibration['summary']['cases']}")
    print(f"holdout_cases={holdout['summary']['cases']}")
    for name in ("dequant-vectors.json", "primitive-calibration.json", "primitive-holdout.json"):
        print(f"sha256.{name}={sha256(args.output_dir / name)}")
    if (args.output_dir / "manifest.json").is_file():
        print(f"sha256.manifest.json={sha256(args.output_dir / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
