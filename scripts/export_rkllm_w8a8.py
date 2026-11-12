#!/usr/bin/env python3
"""Convert a HuggingFace LLM to .rkllm for Rockchip NPUs (run on x86_64 Linux).

RK3588S reality check: the NPU/ RKLLM stack supports W8A8 (and rk-llama.cpp
exposes W4A4). Native W4A8 is NOT available on RK3588S; it only exists as
w4a8_g32 on RK3562. This script enforces that matrix instead of producing a
model that cannot load.
"""
from __future__ import annotations

import argparse
import os
import sys

RK3588 = {"rk3588", "rk3588s"}
SUPPORTED = {
    "rk3588": {"w8a8", "w8a8_g128", "w8a8_g256", "w8a8_g512"},
    "rk3588s": {"w8a8", "w8a8_g128", "w8a8_g256", "w8a8_g512"},
    "rk3576": {"w8a8", "w8a8_g128", "w8a8_g256", "w8a8_g512", "w4a16"},
    "rk3562": {"w8a8", "w8a8_g128", "w8a8_g256", "w8a8_g512", "w4a8_g32"},
}


def validate_quant(target: str, dtype: str):
    target = target.lower()
    dtype = dtype.lower()
    allowed = SUPPORTED.get(target)
    if allowed is None:
        return False, f"unknown target platform: {target}"
    if dtype not in allowed:
        if target in RK3588 and dtype.startswith("w4"):
            return (
                False,
                f"{dtype} is not supported on {target}: RK3588S has no W4A8 kernel; "
                "use w8a8/w8a8_g128 (RKLLM) or rk-llama.cpp W4A4, or reduce model size.",
            )
        return False, f"{dtype} is not supported on {target}; allowed: {sorted(allowed)}"
    return True, "ok"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert HF -> .rkllm (RK3588S defaults to W8A8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dataset", default="data/calibration.json")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--quantized-dtype",
        default="w8a8",
        choices=["w8a8", "w8a8_g128", "w8a8_g256", "w8a8_g512", "w4a16", "w4a8_g32"],
    )
    parser.add_argument("--quantized-algorithm", default=None)
    parser.add_argument(
        "--target-platform",
        default="rk3588",
        choices=["rk3588", "rk3588s", "rk3576", "rk3562"],
    )
    parser.add_argument("--num-npu-core", type=int, default=3)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--optimization-level", type=int, default=1, choices=[0, 1])
    parser.add_argument("--hybrid-rate", type=float, default=0.0)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ok, message = validate_quant(args.target_platform, args.quantized_dtype)
    print(f"quant check: {message}")
    if not ok:
        return 2
    if args.check_only:
        return 0

    if not os.path.isdir(args.model_dir):
        print(f"ERROR: model directory not found: {args.model_dir}")
        return 2

    try:
        from rkllm.api import RKLLM
    except ImportError:
        print("ERROR: rkllm-toolkit is not installed in this environment.")
        print("Install the wheel from rknn-llm/rkllm-toolkit/packages/ on x86_64 Linux.")
        return 2

    if not os.path.exists(args.dataset):
        print(f"ERROR: calibration dataset not found: {args.dataset}")
        print("Run: python scripts/generate_calibration.py --output data/calibration.json")
        return 2

    model_name = os.path.basename(os.path.normpath(args.model_dir))
    output = args.output or (
        f"{model_name}_{args.quantized_dtype.upper()}_{args.target_platform.upper()}.rkllm"
    )
    algorithm = args.quantized_algorithm
    if algorithm is None:
        algorithm = "grq" if args.quantized_dtype == "w4a16" else "normal"

    print(f"model:      {args.model_dir}")
    print(f"output:     {output}")
    print(f"quantized:  {args.quantized_dtype} / {algorithm}")
    print(f"platform:   {args.target_platform} (npu cores: {args.num_npu_core})")
    print(f"context:    {args.max_context}")
    if args.quantized_dtype.startswith("w8"):
        print("WARN: a 7B W8A8 .rkllm is ~8.2 GB and will not fit in 8 GB RAM; use <=3B or Huong A.")

    llm = RKLLM()
    ret = llm.load_huggingface(
        model=args.model_dir,
        model_lora=None,
        device=args.device,
        dtype=args.dtype,
        custom_config=None,
        load_weight=True,
    )
    if ret != 0:
        print("ERROR: load_huggingface failed")
        return int(ret)

    ret = llm.build(
        do_quantization=True,
        optimization_level=args.optimization_level,
        quantized_dtype=args.quantized_dtype,
        quantized_algorithm=algorithm,
        target_platform=args.target_platform,
        num_npu_core=args.num_npu_core,
        extra_qparams=None,
        dataset=args.dataset,
        hybrid_rate=args.hybrid_rate,
        max_context=args.max_context,
    )
    if ret != 0:
        print("ERROR: build failed")
        return int(ret)

    ret = llm.export_rkllm(output)
    if ret != 0:
        print("ERROR: export_rkllm failed")
        return int(ret)

    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"OK: exported {output} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
