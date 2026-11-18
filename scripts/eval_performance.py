#!/usr/bin/env python3
"""So sanh model goc va model da quantize (SpinQuant -> GGUF) theo cac tieu chi:

  Model size (GB) | Peak RAM (GB) | Model loading time (s) | TTFT (s)
  Prefill speed (tok/s) | Decode speed (tok/s) | Power (W)

Moi model chay trong MOT PROCESS RIENG, nen Peak RAM (peak RSS) va load time
khong bi anh huong boi model khac.

Backend:
  hf    model HuggingFace (transformers)  - dung cho model goc
  gguf  file .gguf (llama-cpp-python)     - dung cho model da quantize that su

Cu phap model:  --model <ten>=<backend>:<duong_dan>

Vi du (tren NanoPi R6C, CPU, cung runtime llama.cpp cho ca hai model):
  python bench_llm.py \
    --model f16=gguf:/data/qwen2.5-7b-f16.gguf \
    --model spinquant_q4=gguf:/data/qwen2.5-7b-spinquant-q4_0.gguf \
    --threads 4 --prompt-tokens 512 --gen-tokens 128 --runs 3 \
    --power cmd --power-cmd "python read_meter.py"

Vi du (GPU, model goc HF, do cong suat bang NVML):
  python bench_llm.py --model orig=hf:/path/Qwen2.5-3B-Instruct \
    --device cuda:0 --power nvml --power-interval 0.05
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import threading
import time

GIB = 1024 ** 3  # "GB" trong bao cao = GiB (1024^3 byte), giong llama.cpp

BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. Edge devices run large language "
    "models under tight memory and power budgets, so quantization is used to reduce "
    "model size, memory traffic and energy per token. "
)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def parse_spec(spec):
    name, rest = spec.split("=", 1)
    backend, path = rest.split(":", 1)
    if backend not in ("hf", "gguf"):
        raise ValueError(f"backend phai la hf hoac gguf: {spec}")
    return name, backend, path


def model_size_gb(path):
    """Tong dung luong file trong so (khong tinh tokenizer/config)."""
    if os.path.isfile(path):
        return os.path.getsize(path) / GIB
    names = os.listdir(path)
    for exts in ((".safetensors",), (".bin", ".pt", ".pth"), (".gguf",)):
        files = [f for f in names if f.endswith(exts) and "training_args" not in f]
        if files:
            return sum(os.path.getsize(os.path.join(path, f)) for f in files) / GIB
    return 0.0


# ----------------------------------------------------------------- cong suat
class PowerSampler:
    """Lay mau cong suat (W) o thread nen trong luc model chay.

    mode:
      nvml  cong suat GPU (nvidia-ml-py)
      file  doc 1 so tu file (vd hwmon power1_input), nhan voi --power-scale
      cmd   chay lenh shell, lenh in ra so W (vd doc dong ho do USB / o cam thong minh)
    """

    def __init__(self, ns):
        self.mode = ns.power
        self.interval = ns.power_interval
        self.path = ns.power_file
        self.scale = ns.power_scale
        self.cmd = ns.power_cmd
        self._samples = []
        self._mem = []
        self._stop = threading.Event()
        self._thread = None
        if self.mode == "nvml":
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(ns.nvml_index)
        elif self.mode == "file" and not self.path:
            raise ValueError("--power file can --power-file")
        elif self.mode == "cmd" and not self.cmd:
            raise ValueError("--power cmd can --power-cmd")

    def _read(self):
        try:
            if self.mode == "nvml":
                return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            if self.mode == "file":
                with open(self.path) as f:
                    return float(f.read().split()[0]) * self.scale
            if self.mode == "cmd":
                out = subprocess.run(
                    self.cmd, shell=True, capture_output=True, text=True, timeout=10
                ).stdout
                return float(out.split()[0])
        except Exception:
            return None
        return None

    def _read_mem(self):
        """VRAM dang dung (GiB) cua ca GPU, chi co o che do nvml."""
        if self.mode != "nvml":
            return None
        try:
            return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / GIB
        except Exception:
            return None

    def used_gb(self):
        return self._read_mem()

    def _loop(self):
        while not self._stop.is_set():
            w = self._read()
            if w is not None:
                self._samples.append(w)
            m = self._read_mem()
            if m is not None:
                self._mem.append(m)
            self._stop.wait(self.interval)

    def start(self):
        self._samples = []
        self._mem = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        s = self._samples
        out = {"avg": sum(s) / len(s), "peak": max(s), "n": len(s)} if s else {}
        if self._mem:
            out["mem_peak_gb"] = max(self._mem)
        return out

    def measure_idle(self, seconds):
        self.start()
        time.sleep(seconds)
        return self.stop().get("avg")


# -------------------------------------------------------------------- do luong
def summarize(t0, stamps, prompt_tokens, p):
    """stamps = thoi diem moi token sinh ra. Token dau = het prefill."""
    n = len(stamps)
    ttft = stamps[0] - t0
    dec = stamps[-1] - stamps[0]
    avg = p.get("avg")
    return {
        "n_gen": n,
        "ttft_s": ttft,
        "prefill_tps": prompt_tokens / ttft,
        "decode_tps": (n - 1) / dec if n > 1 and dec > 0 else None,
        "power_avg_w": avg,
        "power_peak_w": p.get("peak"),
        "vram_used_peak_gb": p.get("mem_peak_gb"),
        "energy_j_per_token": avg * (stamps[-1] - t0) / n if avg else None,
    }


def run_hf(ns, path, power):
    import torch
    import transformers
    from packaging import version
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[ns.dtype]
    new_api = version.parse(transformers.__version__) >= version.parse("4.56")
    kw = {"dtype": dtype} if new_api else {"torch_dtype": dtype}
    cuda = ns.device.startswith("cuda")

    def sync():
        if cuda:
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True, **kw)
    model = model.to(ns.device).eval()
    sync()
    load_s = time.perf_counter() - t0
    log(f"loaded in {load_s:.1f}s")

    base = tok(BASE_TEXT, add_special_tokens=False).input_ids
    ids = (base * (ns.prompt_tokens // len(base) + 1))[: ns.prompt_tokens]
    input_ids = torch.tensor([ids], device=ns.device)
    attn = torch.ones_like(input_ids)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    class Timer(StoppingCriteria):
        def __init__(self):
            self.t = []

        def __call__(self, input_ids, scores, **kwargs):
            sync()
            self.t.append(time.perf_counter())
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    def once(n_new, measure_power=True):
        timer = Timer()
        sync()
        t_start = time.perf_counter()
        use_power = power is not None and measure_power
        if use_power:
            power.start()
        with torch.inference_mode():
            model.generate(
                input_ids,
                attention_mask=attn,
                max_new_tokens=n_new,
                min_new_tokens=n_new,  # ep sinh du n_new token, khong dung som o EOS
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                stopping_criteria=StoppingCriteriaList([timer]),
                pad_token_id=pad_id,
            )
        sync()
        p = power.stop() if use_power else {}
        return summarize(t_start, timer.t, ns.prompt_tokens, p)

    for _ in range(ns.warmup):
        once(8, measure_power=False)
    runs = [once(ns.gen_tokens) for _ in range(ns.runs)]
    out = {"load_time_s": load_s, "runs": runs}
    if cuda:
        out["peak_vram_gb"] = torch.cuda.max_memory_allocated() / GIB
    return out


def run_gguf(ns, path, power):
    from llama_cpp import Llama

    n_ctx = ns.prompt_tokens + ns.gen_tokens + 64
    t0 = time.perf_counter()
    llm = Llama(
        model_path=path,
        n_ctx=n_ctx,
        n_threads=ns.threads,
        n_gpu_layers=ns.n_gpu_layers,
        n_batch=ns.n_batch,
        use_mmap=not ns.no_mmap,
        verbose=False,
    )
    load_s = time.perf_counter() - t0
    log(f"loaded in {load_s:.1f}s (mmap={'off' if ns.no_mmap else 'on'})")

    base = llm.tokenize(BASE_TEXT.encode("utf-8"), add_bos=False)
    ids = (base * (ns.prompt_tokens // len(base) + 1))[: ns.prompt_tokens]

    def once(n_new, measure_power=True):
        # reset() bat buoc: neu khong llama-cpp-python se tai su dung KV cache
        # cua prompt lan truoc va bo qua prefill, lam prefill speed ao.
        llm.reset()
        stamps = []
        t_start = time.perf_counter()
        use_power = power is not None and measure_power
        if use_power:
            power.start()
        for _ in llm.generate(ids, temp=0.0, reset=True):
            stamps.append(time.perf_counter())
            if len(stamps) >= n_new:  # khong dung o EOS, sinh du n_new token
                break
        p = power.stop() if use_power else {}
        return summarize(t_start, stamps, ns.prompt_tokens, p)

    for _ in range(ns.warmup):
        once(8, measure_power=False)
    runs = [once(ns.gen_tokens) for _ in range(ns.runs)]
    return {"load_time_s": load_s, "runs": runs}


# ---------------------------------------------------------------------- worker
def worker(cfg):
    ns = argparse.Namespace(**cfg)
    name, backend, path = parse_spec(ns.model)
    power = PowerSampler(ns) if ns.power != "none" else None

    idle = None
    if power is not None and ns.idle_seconds > 0:
        log(f"measuring idle power for {ns.idle_seconds}s ...")
        idle = power.measure_idle(ns.idle_seconds)

    vram_base = power.used_gb() if power is not None else None  # truoc khi load model
    res = (run_hf if backend == "hf" else run_gguf)(ns, path, power)
    runs = res["runs"]
    out = {
        "name": name,
        "backend": backend,
        "path": path,
        "model_size_gb": model_size_gb(path),
        # ru_maxrss tinh bang KB tren Linux; gom ca luc load model
        "peak_ram_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
        "load_time_s": res["load_time_s"],
        "power_idle_w": idle,
        "prompt_tokens": ns.prompt_tokens,
        "gen_tokens": ns.gen_tokens,
        "runs": runs,
    }
    if "peak_vram_gb" in res:
        out["peak_vram_gb"] = res["peak_vram_gb"]
    else:
        peaks = [r.get("vram_used_peak_gb") for r in runs if r.get("vram_used_peak_gb") is not None]
        if peaks and vram_base is not None:
            out["peak_vram_gb"] = max(max(peaks) - vram_base, 0.0)
    for k in ("ttft_s", "prefill_tps", "decode_tps", "power_avg_w", "power_peak_w", "energy_j_per_token"):
        out[k] = mean([r.get(k) for r in runs])
    print("RESULT_JSON:" + json.dumps(out), flush=True)


# ---------------------------------------------------------------------- bang
ROWS = [
    ("Model size (GB)", "model_size_gb", 2),
    ("Peak RAM (GB)", "peak_ram_gb", 2),
    ("Peak VRAM (GB)", "peak_vram_gb", 2),
    ("Peak VRAM (GB)", "peak_vram_gb", 2),
    ("Model loading time (s)", "load_time_s", 2),
    ("TTFT (s)", "ttft_s", 3),
    ("Prefill speed (tok/s)", "prefill_tps", 1),
    ("Decode speed (tok/s)", "decode_tps", 2),
    ("Power avg (W)", "power_avg_w", 2),
    ("Power peak (W)", "power_peak_w", 2),
    ("Power idle (W)", "power_idle_w", 2),
    ("Energy (J/token)", "energy_j_per_token", 2),
]


def fmt(v, nd):
    return "n/a" if v is None else f"{v:.{nd}f}"


def print_table(results):
    ok = [r for r in results if "error" not in r]
    if not ok:
        return
    base = ok[0]
    header = ["Metric"] + [r["name"] for r in ok]
    table = [header]
    for label, key, nd in ROWS:
        if all(r.get(key) is None for r in ok):
            continue
        row = [label]
        for i, r in enumerate(ok):
            v = r.get(key)
            cell = fmt(v, nd)
            b = base.get(key)
            if i > 0 and v is not None and b:
                cell += f" (x{v / b:.2f})"
            row.append(cell)
        table.append(row)
    widths = [max(len(row[c]) for row in table) for c in range(len(header))]
    line = "-+-".join("-" * w for w in widths)
    print()
    for i, row in enumerate(table):
        print(" | ".join(cell.ljust(widths[c]) for c, cell in enumerate(row)))
        if i == 0:
            print(line)
    print(f"\n(x..) = ti le so voi model dau tien '{base['name']}'. "
          f"prompt={base['prompt_tokens']} tokens, gen={base['gen_tokens']} tokens.")


def save(results, out_json):
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    out_csv = os.path.splitext(out_json)[0] + ".csv"
    keys = [k for _, k, _ in ROWS]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "backend"] + keys)
        for r in results:
            if "error" not in r:
                w.writerow([r["name"], r["backend"]] + [r.get(k) for k in keys])
    print(f"saved: {out_json}, {out_csv}")


# ---------------------------------------------------------------------- parent
def parent(ns):
    results = []
    for spec in ns.model:
        cfg = vars(ns).copy()
        cfg["model"] = spec
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--cfg", json.dumps(cfg)]
        log(f"\n=== {spec} ===")
        p = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)  # stderr hien thang
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
        if p.returncode != 0 or line is None:
            log(f"FAILED: {spec} (exit {p.returncode})")
            results.append({"name": spec.split("=", 1)[0], "error": f"exit {p.returncode}"})
            continue
        results.append(json.loads(line[len("RESULT_JSON:"):]))
    print_table(results)
    save(results, ns.out)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=BACKEND:PATH")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="bench_results.json")
    # gguf
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--n-batch", type=int, default=512)
    ap.add_argument("--no-mmap", action="store_true", help="doc het model vao RAM (load time day du hon)")
    # hf
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    # power
    ap.add_argument("--power", default="none", choices=["none", "nvml", "file", "cmd"])
    ap.add_argument("--power-interval", type=float, default=0.2)
    ap.add_argument("--nvml-index", type=int, default=0)
    ap.add_argument("--power-file", default=None)
    ap.add_argument("--power-scale", type=float, default=1.0, help="nhan gia tri doc tu file de ra W (vd 1e-6 cho uW)")
    ap.add_argument("--power-cmd", default=None)
    ap.add_argument("--idle-seconds", type=float, default=0.0, help="do cong suat idle truoc khi load model")
    return ap


def main():
    if "--worker" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--worker", action="store_true")
        ap.add_argument("--cfg", required=True)
        a = ap.parse_args()
        try:
            worker(json.loads(a.cfg))
        except Exception:
            import traceback

            traceback.print_exc()
            sys.exit(1)
        return
    parent(build_parser().parse_args())


if __name__ == "__main__":
    main()
