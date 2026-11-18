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
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nano_llm.bench.metrics import find_latest_results, now_iso, write_json  # noqa: E402
from nano_llm.config import load_config  # noqa: E402


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in text).strip("-")[:80]


def cuda_device_count() -> int:
    """Number of visible CUDA GPUs; 0 if torch/CUDA is unavailable."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def find_llama_server() -> str | None:
    candidates = [
        os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
        shutil.which("llama-server"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(base_url: str, proc: subprocess.Popen, log_path: str, timeout: int = 300) -> None:
    # urlopen() has no `proxies` kwarg in Python < 3.11; use a no-proxy opener instead.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {proc.returncode}); see {log_path}")
        try:
            with opener.open(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"llama-server did not become healthy in {timeout}s; see {log_path}")


def build_lm_eval_cmd(
    model_backend: str,
    model: str,
    tasks,
    limit: int,
    output_path: str,
    device: str = "cuda",
    batch_size: str = "auto",
    extra_model_args: str = "",
    parallelize: bool = False,
    max_memory_per_gpu: str = "10GiB",
    base_url: str = "",
    max_length: int = 2048,
) -> list:
    if model_backend == "gguf":
        # lm-eval's GGUF backend is an HTTP client of llama-server, not a local loader.
        model_args = f"pretrained={model}"
        if base_url:
            model_args += f",base_url={base_url},max_length={max_length}"
    else:
        model_args = f"pretrained={model},dtype=float16,trust_remote_code=True"
        if parallelize:
            # 7B fp16 ~= 15.2 GB: one 16 GB card has no room for logits/activations.
            # Cap per GPU so `device_map="auto"` actually splits the layers across cards
            # (accelerate's default is the *whole* free VRAM, which keeps everything on GPU 0).
            model_args += f",parallelize=True,max_memory_per_gpu={max_memory_per_gpu}"
    if extra_model_args:
        model_args = f"{model_args},{extra_model_args}"
    # Use the same interpreter as this script: a bare `lm_eval` on PATH can come from
    # a different install (user site / system) with an incompatible numpy/scipy stack.
    # GGUF needs a compat wrapper: lm-eval 0.4.13 expects legacy `echo` logprobs that
    # modern llama-server no longer returns.
    launcher = (
        [sys.executable, os.path.join(ROOT, "scripts", "lm_eval_gguf_compat.py")]
        if model_backend == "gguf"
        else [sys.executable, "-m", "lm_eval"]
    )
    return launcher + [
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
    parser.add_argument(
        "--parallelize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Split HF model across GPUs (auto: on when >=2 CUDA GPUs). "
        "Needed for 7B fp16 on 16 GB cards.",
    )
    parser.add_argument(
        "--max-memory-per-gpu",
        default="10GiB",
        help="VRAM cap per GPU when --parallelize is on (default: 10GiB)",
    )
    parser.add_argument("--output", default=None, help="summary JSON path")
    parser.add_argument(
        "--llama-server", default=None,
        help="path to llama-server binary (default: ~/llama.cpp/build/bin/llama-server or PATH)",
    )
    parser.add_argument("--server-port", type=int, default=0, help="port for auto-started llama-server (0 = free port)")
    parser.add_argument("--server-ctx", type=int, default=2048, help="context size for llama-server / GGUF backend")
    parser.add_argument("--server-ngl", type=int, default=99, help="GPU layers offloaded by llama-server")
    parser.add_argument(
        "--no-auto-server", action="store_true",
        help="do not start llama-server; supply base_url via --extra-model-args instead",
    )
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
    parallelize = args.parallelize
    if parallelize is None:
        parallelize = model_backend == "hf" and str(args.device).startswith("cuda") and cuda_device_count() >= 2
    if parallelize:
        print(
            f"[info] HF model will be split across GPUs "
            f"(max_memory_per_gpu={args.max_memory_per_gpu}); "
            "use --no-parallelize to force a single device."
        )
    outdir = cfg.resolve(cfg.eval.output_dir)
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    lm_out = os.path.join(outdir, f"lm_eval_{slug(os.path.basename(model))}_{stamp}.json")

    server_proc = None
    server_fh = None
    server_log = None
    base_url = ""
    try:
        if model_backend == "gguf" and "base_url=" not in args.extra_model_args and not args.no_auto_server:
            if args.dry_run:
                base_url = f"http://127.0.0.1:{args.server_port or 8080}"
                print("[info] dry-run: llama-server sẽ được khởi động khi chạy thật")
            else:
                server_bin = args.llama_server or find_llama_server()
                if not server_bin:
                    print("ERROR: llama-server not found. Build it or pass --llama-server /path/to/llama-server")
                    return 2
                port = args.server_port or free_port()
                base_url = f"http://127.0.0.1:{port}"
                server_log = os.path.join(
                    outdir, f"llama_server_{slug(os.path.basename(model))}_{stamp}.log"
                )
                server_fh = open(server_log, "w", encoding="utf-8")
                server_cmd = [
                    server_bin,
                    "-m", model,
                    "--host", "127.0.0.1",
                    "--port", str(port),
                    "-c", str(args.server_ctx),
                    "-ngl", str(args.server_ngl),
                ]
                print("$ " + " ".join(server_cmd))
                server_proc = subprocess.Popen(
                    server_cmd, stdout=server_fh, stderr=subprocess.STDOUT
                )
                wait_for_server(base_url, server_proc, server_log)
                print(f"[info] llama-server ready at {base_url} (log: {server_log})")

        cmd = build_lm_eval_cmd(
            model_backend=model_backend,
            model=model,
            tasks=tasks,
            limit=limit,
            output_path=lm_out,
            device=args.device,
            batch_size=args.batch_size,
            extra_model_args=args.extra_model_args,
            parallelize=parallelize,
            max_memory_per_gpu=args.max_memory_per_gpu,
            base_url=base_url,
            max_length=args.server_ctx,
        )

        print("$ " + " ".join(cmd))
        print("\nQuality gate: PPL +5%/+10%; MMLU/ARC-C/HellaSwag/VMLU >= -2 dp; GSM8K >= -5; IFEval >= -3")
        if args.dry_run:
            return 0

        if importlib.util.find_spec("lm_eval") is None:
            print("ERROR: lm-eval is not installed. Run: uv pip install -r requirements.txt")
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
            "parallelize": parallelize,
            "max_memory_per_gpu": args.max_memory_per_gpu if parallelize else None,
            "base_url": base_url or None,
            "server_log": server_log,
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
    except (RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 2
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()
        if server_fh is not None:
            server_fh.close()


if __name__ == "__main__":
    sys.exit(main())
