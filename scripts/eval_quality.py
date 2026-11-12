#!/usr/bin/env python3
"""Quality evaluation orchestrator: lm-eval-harness (+ VMLU/PPL separately).

Quality gate from the plan:
  - WikiText-2 PPL delta <= +5% (W4A8) or +10% (W4A4)
  - MMLU/ARC-C/HellaSwag/VMLU delta >= -2 points
  - GSM8K delta >= -5, IFEval delta >= -3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nano_llm.bench.metrics import find_latest_results, now_iso, write_json  # noqa: E402
from nano_llm.config import load_config  # noqa: E402


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in text).strip("-")[:80]


def build_lm_eval_cmd(
    model_backend: str,
    model: str,
    tasks,
    limit: int,
    output_path: str,
    device: str = "cuda",
    batch_size: str = "auto",
    extra_model_args: str = "",
) -> list:
    if model_backend == "gguf":
        model_args = f"pretrained={model}"
    else:
        model_args = f"pretrained={model},dtype=float16,trust_remote_code=True"
    if extra_model_args:
        model_args = f"{model_args},{extra_model_args}"
    return [
        "lm_eval",
        "--model",
        model_backend,
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--num_fewshot",
        "0",
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--limit",
        str(limit),
        "--output_path",
        output_path,
        "--log_samples",
    ]


def extract_task_scores(payload: dict) -> dict:
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, dict):
        results = payload if isinstance(payload, dict) else {}
    scores = {}
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        task_scores = {}
        for key, value in metrics.items():
            if "stderr" in key.lower() or not isinstance(value, (int, float)):
                continue
            task_scores[key.split(",")[0]] = float(value)
        if task_scores:
            scores[task] = task_scores
    return scores


def run_logged(cmd: list, log_path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lm-eval quality suite")
    parser.add_argument("--config", default=os.path.join(ROOT, "config", "config.yaml"))
    parser.add_argument("--model", default=None, help="model id or local path")
    parser.add_argument("--model-backend", default="auto", choices=["auto", "hf", "gguf", "vllm"])
    parser.add_argument("--tasks", default=None, help="comma-separated lm-eval tasks")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--extra-model-args", default="")
    parser.add_argument("--output", default=None, help="summary JSON path")
    parser.add_argument("--dry-run", action="store_true", help="print command only")
    args = parser.parse_args()

    cfg = load_config(args.config, ROOT)
    model_backend = args.model_backend
    if model_backend == "auto":
        configured = cfg.model.backend.lower()
        model_backend = "gguf" if configured in ("gguf", "llamacpp", "llama.cpp") else "hf"

    model = args.model or (cfg.resolve(cfg.model.path) if cfg.model.path else cfg.model.name)
    tasks = [item.strip() for item in (args.tasks.split(",") if args.tasks else cfg.eval.tasks) if item.strip()]
    limit = args.limit if args.limit is not None else cfg.eval.limit
    outdir = cfg.resolve(cfg.eval.output_dir)
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    lm_out = os.path.join(outdir, f"lm_eval_{slug(os.path.basename(model))}_{stamp}.json")
    cmd = build_lm_eval_cmd(
        model_backend=model_backend,
        model=model,
        tasks=tasks,
        limit=limit,
        output_path=lm_out,
        device=args.device,
        batch_size=args.batch_size,
        extra_model_args=args.extra_model_args,
    )

    print("$ " + " ".join(cmd))
    print("\nQuality gate: PPL +5%/+10%; MMLU/ARC-C/HellaSwag/VMLU >= -2 dp; GSM8K >= -5; IFEval >= -3")
    if args.dry_run:
        return 0

    if importlib.util.find_spec("lm_eval") is None:
        print("ERROR: lm-eval is not installed. Run: pip install -r requirements-host.txt")
        return 2

    log_path = os.path.join(outdir, f"lm_eval_{slug(os.path.basename(model))}_{stamp}.log")
    code = run_logged(cmd, log_path)
    if code != 0:
        print(f"ERROR: lm-eval exited with {code}; see {log_path}")
        return code

    latest = find_latest_results(outdir, "lm_eval_*.json")
    scores = {}
    if latest:
        try:
            with open(latest, "r", encoding="utf-8") as fh:
                scores = extract_task_scores(json.load(fh))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN: could not parse {latest}: {exc}")

    summary = {
        "timestamp": now_iso(),
        "model": model,
        "model_backend": model_backend,
        "tasks": tasks,
        "limit": limit,
        "command": cmd,
        "log": log_path,
        "lm_eval_output": latest,
        "scores": scores,
        "quality_gate": {
            "ppl_delta_max_pct_w4a8": 5,
            "ppl_delta_max_pct_w4a4": 10,
            "benchmark_delta_min_points": -2,
            "gsm8k_delta_min_points": -5,
            "ifeval_delta_min_points": -3,
        },
    }
    output = args.output or os.path.join(outdir, f"quality_{slug(os.path.basename(model))}_{stamp}.json")
    print(f"OK: {write_json(output, summary)}")
    if scores:
        print(json.dumps(scores, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
