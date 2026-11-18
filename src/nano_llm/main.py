from __future__ import annotations

import argparse
import errno
import importlib.util
import os
import random
import socket
import sys
from typing import List, Optional

from nano_llm.bench.metrics import read_jsonl
from nano_llm.bench.runner import format_table, run_bench
from nano_llm.config import Config, load_config
from nano_llm.logging_setup import setup_logging
from nano_llm.models.slm import build_slm
from nano_llm.utils.system import board_model, file_size_gb, read_meminfo_gb

DEMO_PROMPTS = [
    "Tóm tắt trong 2 câu: NanoPi-R6C dùng NPU nào và có bao nhiêu TOPS?",
    "Giải thích ngắn gọn sự khác nhau giữa W8A8 và W4A16 khi lượng tử hoá LLM.",
    "Viết một cảnh báo an ninh ngắn bằng tiếng Việt khi phát hiện 3 người xâm nhập.",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NanoPi-R6C-LLM: quantized LLM benchmark/serving toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/config.yaml", help="YAML config path")
    parser.add_argument("--base-dir", default=None, help="resolve relative paths against this directory")
    parser.add_argument("--backend", default=None, help="override model.backend (mock|hf|gguf|rkllm)")
    parser.add_argument("--force-mock", action="store_true", help="always use the mock backend")
    parser.add_argument("--check", action="store_true", help="print resolved config and exit")
    parser.add_argument("--demo", action="store_true", help="run a few prompts and exit")
    parser.add_argument("--prompt", default=None, help="run a single prompt and print the answer")
    parser.add_argument("--prompts-file", default=None, help="JSONL prompts for --bench")
    parser.add_argument("--bench", action="store_true", help="run the benchmark and write JSON")
    parser.add_argument("--runs", type=int, default=None, help="measured runs per case")
    parser.add_argument("--warmup", type=int, default=None, help="warmup runs per case")
    parser.add_argument("--prompt-tokens", default=None, help="comma-separated context sizes")
    parser.add_argument("--gen-tokens", type=int, default=None, help="tokens to generate per run")
    parser.add_argument("--out", default=None, help="output JSON path for --bench")
    parser.add_argument("--serve", dest="serve", action="store_true", default=None)
    parser.add_argument("--no-serve", dest="serve", action="store_false")
    parser.add_argument("--host", default=None, help="API host override")
    parser.add_argument("--port", type=int, default=None, help="API port override")
    return parser.parse_args(argv)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_api_bind(host: str, port: int) -> Optional[str]:
    """Return an actionable error if the API address is already occupied."""
    try:
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as sock:
            if ":" in host:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            sock.bind((host, port))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            return (
                f"API address {host}:{port} is already in use. Stop the process using it, "
                f"or start this server on another port with --port <port>."
            )
        return f"Cannot bind API address {host}:{port}: {exc}"
    return None


def print_check(cfg: Config) -> int:
    summary = cfg.summary()
    mem = read_meminfo_gb()
    print("== NanoPi-R6C-LLM check ==")
    print(f"base_dir:        {summary['base_dir']}")
    print(f"board:           {board_model()}")
    if mem:
        print(f"memory:          {mem.get('MemTotal', 0):.2f} GB total, {mem.get('MemAvailable', 0):.2f} GB available")
    print(f"backend:         {summary['backend']}")
    print(f"model:           {summary['model']}")
    if summary["model_path"]:
        exists = os.path.exists(summary["model_path"])
        size = file_size_gb(summary["model_path"])
        print(f"model path:      {summary['model_path']} ({'found' if exists else 'MISSING'}, {size:.2f} GB)")
    print(f"max_context:     {summary['max_context']}")
    print(f"max_new_tokens:  {summary['max_new_tokens']}")
    print(f"results dir:     {summary['results_dir']}")
    print(f"rkllm library:   {summary['rkllm_library']} ({'found' if os.path.exists(summary['rkllm_library']) else 'not found'})")
    print("-- python modules --")
    for name in ("yaml", "fastapi", "uvicorn", "transformers", "torch", "llama_cpp", "datasets", "lm_eval"):
        print(f"{name:<14} {'OK' if _has_module(name) else 'missing'}")
    print("-- runtime notes --")
    print("RK3588S NPU supports W8A8 (and W4A4 via rk-llama.cpp); W4A8 is not a native RKLLM target.")
    print("See KE_HOACH_QUANTIZE_QWEN7B_4BIT_NANOPI_R6C.md for the decision gates.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.base_dir)
    if args.backend:
        cfg.model.backend = args.backend
    if args.host:
        cfg.api.host = args.host
    if args.port:
        cfg.api.port = args.port
    random.seed(cfg.app.seed)
    setup_logging(cfg.app.log_level, cfg.resolve(cfg.app.log_file))

    if args.check:
        return print_check(cfg)

    serving = not (args.prompt or args.demo or args.bench or args.serve is False)
    if serving:
        bind_error = _check_api_bind(cfg.api.host, cfg.api.port)
        if bind_error:
            print(bind_error, file=sys.stderr)
            return 1

    llm = build_slm(cfg, force_mock=args.force_mock)
    try:
        if args.prompt:
            result = llm.generate(args.prompt)
            print(result.text)
            print(
                f"\n[ttft={result.ttft_s:.3f}s prefill={result.prefill_tok_s:.2f} tok/s "
                f"decode={result.decode_tok_s:.2f} tok/s tokens={result.generated_tokens}]",
                file=sys.stderr,
            )
            return 0

        if args.demo:
            for prompt in DEMO_PROMPTS:
                print(f"\n### {prompt}")
                result = llm.generate(prompt, max_new_tokens=128)
                print(result.text)
            return 0

        if args.bench:
            prompt_tokens = None
            if args.prompt_tokens:
                prompt_tokens = [int(value) for value in args.prompt_tokens.split(",") if value.strip()]
            prompts = None
            if args.prompts_file:
                prompts = [item.get("prompt", "") for item in read_jsonl(cfg.resolve(args.prompts_file))]
            report = run_bench(
                llm,
                cfg,
                out_path=args.out,
                prompt_tokens=prompt_tokens,
                gen_tokens=args.gen_tokens,
                runs=args.runs,
                warmup=args.warmup,
                prompts=prompts,
            )
            print(format_table(report))
            print(f"\nwritten: {report['output']}")
            return 0

        if args.serve is False:
            print(llm.info())
            return 0

        import uvicorn

        from nano_llm.api.app import create_app

        app = create_app(llm, cfg)
        uvicorn.run(app, host=cfg.api.host, port=cfg.api.port, log_level=cfg.app.log_level.lower())
        return 0
    finally:
        llm.close()


if __name__ == "__main__":
    sys.exit(main())
