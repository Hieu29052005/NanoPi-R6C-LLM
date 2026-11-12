from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from typing import List, Optional

from nano_llm.config import Config
from nano_llm.models.prompting import user_prompt
from nano_llm.models.slm import BaseSLM, GenerationResult, estimate_tokens
from nano_llm.utils.system import file_size_gb, read_vm_hwm_gb

log = logging.getLogger(__name__)


def find_gguf(path: str) -> str:
    if not path:
        return path
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidates: List[str] = []
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if name.endswith(".gguf") and os.path.isfile(full):
                candidates.append(full)
        if candidates:
            return max(candidates, key=os.path.getsize)
    return path


def parse_llama_perf(output: str) -> dict:
    """Parse timings from llama.cpp's stderr (llama_perf_context_print)."""
    perf: dict = {}
    patterns = {
        "prompt_eval_ms": r"prompt eval time\s*=\s*([0-9.]+)\s*ms",
        "prompt_eval_tokens": r"prompt eval time\s*=.*?/\s*(\d+)\s*tokens",
        "eval_ms": r"\beval time\s*=\s*([0-9.]+)\s*ms",
        "eval_tokens": r"\beval time\s*=.*?/\s*(\d+)\s*(?:runs|tokens)",
        "prompt_tok_s": r"prompt eval time\s*=.*?([0-9.]+)\s*tokens per second",
        "eval_tok_s": r"\beval time\s*=.*?([0-9.]+)\s*tokens per second",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if not match:
            continue
        value = float(match.group(1))
        perf[key] = int(value) if key.endswith("tokens") else value
    return perf


class LlamaCppBackend(BaseSLM):
    """GGUF backend: llama-cpp-python when available, llama-cli subprocess otherwise."""

    backend = "gguf"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        model_cfg = cfg.model
        self.model_path = find_gguf(cfg.resolve(model_cfg.path) if model_cfg.path else "")
        if not self.model_path or not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"GGUF model not found: {model_cfg.path!r}")
        if cfg.llamacpp.rknpu_hybrid:
            os.environ["RKNPU_HYBRID"] = cfg.llamacpp.rknpu_hybrid
        self.mode = ""
        self.llm = None
        start = time.perf_counter()
        try:
            self._init_python()
            self.mode = "python"
        except ImportError:
            self._init_cli()
            self.mode = "cli"
        self.load_time_s = time.perf_counter() - start
        self.model_name = os.path.basename(self.model_path)
        self.model_size_gb = file_size_gb(self.model_path)
        log.info("GGUF backend ready (%s): %s", self.mode, self.model_name)

    def _init_python(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError("llama-cpp-python is not installed") from exc
        cfg = self.cfg
        kwargs = dict(
            model_path=self.model_path,
            n_ctx=cfg.model.max_context,
            n_threads=cfg.llamacpp.n_threads,
            n_batch=cfg.llamacpp.n_batch,
            n_gpu_layers=cfg.llamacpp.n_gpu_layers,
            verbose=False,
        )
        if cfg.llamacpp.flash_attn:
            kwargs["flash_attn"] = True
        self.llm = Llama(**kwargs)

    def _init_cli(self) -> None:
        binary = self.cfg.llamacpp.binary
        if not shutil.which(binary) and not os.path.isfile(binary):
            raise FileNotFoundError(
                "llama-cpp-python not installed and llama-cli not found on PATH"
            )

    def count_tokens(self, text: str) -> int:
        if self.mode == "python" and self.llm is not None:
            try:
                return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))
            except Exception:
                pass
        return estimate_tokens(text)

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> GenerationResult:
        if self.mode == "python":
            return self._generate_python(prompt, max_new_tokens)
        return self._generate_cli(prompt, max_new_tokens)

    def _generate_python(self, prompt: str, max_new_tokens: Optional[int]) -> GenerationResult:
        cfg = self.cfg
        max_new = int(max_new_tokens or cfg.model.max_new_tokens)
        text = user_prompt(prompt, cfg.model.system_prompt)
        start = time.perf_counter()
        stream = self.llm.create_completion(
            prompt=text,
            max_tokens=max_new,
            temperature=cfg.model.temperature,
            top_p=cfg.model.top_p,
            top_k=cfg.model.top_k,
            repeat_penalty=cfg.model.repeat_penalty,
            stream=True,
            echo=False,
        )
        pieces: List[str] = []
        first_token_at = None
        for chunk in stream:
            piece = chunk["choices"][0].get("text", "")
            if piece and first_token_at is None:
                first_token_at = time.perf_counter()
            pieces.append(piece)
        end = time.perf_counter()
        output = "".join(pieces).strip()
        prompt_tokens = self.count_tokens(text)
        generated_tokens = self.count_tokens(output)
        ttft = (first_token_at - start) if first_token_at else (end - start)
        decode_time = max(end - start - ttft, 1e-6)
        return GenerationResult(
            text=output,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            load_time_s=self.load_time_s,
            ttft_s=ttft,
            prefill_time_s=ttft,
            prefill_tok_s=prompt_tokens / ttft if ttft else 0.0,
            decode_time_s=decode_time,
            decode_tok_s=max(generated_tokens - 1, 0) / decode_time,
            total_time_s=end - start,
            peak_ram_gb=read_vm_hwm_gb(),
            backend=self.backend,
            model=self.model_name,
            extra={"mode": self.mode, "model_size_gb": round(self.model_size_gb, 3)},
        )

    def _generate_cli(self, prompt: str, max_new_tokens: Optional[int]) -> GenerationResult:
        cfg = self.cfg
        max_new = int(max_new_tokens or cfg.model.max_new_tokens)
        text = user_prompt(prompt, cfg.model.system_prompt)
        cmd = [
            cfg.llamacpp.binary,
            "-m",
            self.model_path,
            "-p",
            text,
            "-n",
            str(max_new),
            "-c",
            str(cfg.model.max_context),
            "-t",
            str(cfg.llamacpp.n_threads),
            "--temp",
            str(cfg.model.temperature),
            "--top-p",
            str(cfg.model.top_p),
            "--top-k",
            str(cfg.model.top_k),
            "--repeat-penalty",
            str(cfg.model.repeat_penalty),
            "--no-display-prompt",
        ]
        cmd.extend(cfg.llamacpp.extra_args)
        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(cfg.rkllm.timeout_seconds, 60.0),
        )
        end = time.perf_counter()
        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-cli failed ({proc.returncode}): {proc.stderr[-800:]}"
            )
        output = proc.stdout.strip()
        perf = parse_llama_perf(proc.stderr)
        prompt_tokens = int(perf.get("prompt_eval_tokens") or self.count_tokens(text))
        generated_tokens = int(perf.get("eval_tokens") or self.count_tokens(output))
        prefill_time = float(perf.get("prompt_eval_ms", 0.0)) / 1000.0
        decode_time = float(perf.get("eval_ms", 0.0)) / 1000.0
        total = end - start
        if not prefill_time:
            prefill_time = total * 0.3
        if not decode_time:
            decode_time = max(total - prefill_time, 1e-6)
        return GenerationResult(
            text=output,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            load_time_s=self.load_time_s,
            ttft_s=prefill_time,
            prefill_time_s=prefill_time,
            prefill_tok_s=prompt_tokens / prefill_time if prefill_time else 0.0,
            decode_time_s=decode_time,
            decode_tok_s=max(generated_tokens - 1, 0) / decode_time if decode_time else 0.0,
            total_time_s=total,
            peak_ram_gb=read_vm_hwm_gb(),
            backend=self.backend,
            model=self.model_name,
            extra={"mode": self.mode, "model_size_gb": round(self.model_size_gb, 3)},
        )
