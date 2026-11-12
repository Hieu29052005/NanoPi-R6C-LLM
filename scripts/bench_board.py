#!/usr/bin/env python3
"""Benchmark TTFT / prefill / decode / peak RAM / load time on the board."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nano_llm.bench.metrics import read_jsonl  # noqa: E402
from nano_llm.bench.runner import format_table, run_bench  # noqa: E402
from nano_llm.config import load_config  # noqa: E402
from nano_llm.logging_setup import setup_logging  # noqa: E402
from nano_llm.models.slm import build_slm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the configured LLM backend")
    parser.add_argument("--config", default=os.path.join(ROOT, "config", "config.yaml"))
    parser.add_argument("--backend", default=None, help="mock|hf|gguf|rkllm")
    parser.add_argument("--force-mock", action="store_true")
    parser.add_argument("--prompt-tokens", default=None, help="comma-separated context sizes")
    parser.add_argument("--gen-tokens", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--prompts-file", default=None, help="JSONL with a 'prompt' field")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    args = parser.parse_args()

    cfg = load_config(args.config, ROOT)
    if args.backend:
        cfg.model.backend = args.backend
    if args.temperature is not None:
        cfg.model.temperature = args.temperature
    setup_logging(cfg.app.log_level, cfg.resolve(cfg.app.log_file))

    prompt_tokens = None
    if args.prompt_tokens:
        prompt_tokens = [int(value) for value in args.prompt_tokens.split(",") if value.strip()]
    prompts = None
    if args.prompts_file:
        rows = read_jsonl(cfg.resolve(args.prompts_file))
        prompts = [row.get("prompt", "") for row in rows if row.get("prompt")]

    llm = build_slm(cfg, force_mock=args.force_mock)
    try:
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
    finally:
        llm.close()

    print(format_table(report))
    print(f"\nbackend: {report['backend']}  model: {report['model']}")
    print(f"model size: {report['model_size_gb']:.2f} GB  load: {report['load_time_s']:.2f} s")
    print(f"written: {report['output']}")
    if args.json:
        import json

        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
