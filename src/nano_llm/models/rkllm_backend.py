from __future__ import annotations

import logging
import time
from typing import Optional

from nano_llm.config import Config
from nano_llm.models.slm import BaseSLM, GenerationResult
from nano_llm.utils.system import file_size_gb, read_vm_hwm_gb

log = logging.getLogger(__name__)


class RKLLMBackend(BaseSLM):
    """Rockchip NPU backend using librkllmrt.so (W8A8 on RK3588S)."""

    backend = "rkllm"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        from nano_llm.models.rkllm_bindings import RKLLMRuntime

        model_cfg = cfg.model
        if not model_cfg.path:
            raise ValueError("model.path must point to a .rkllm file for the rkllm backend")
        model_path = cfg.resolve(model_cfg.path)
        library_path = cfg.resolve(cfg.rkllm.library_path)
        start = time.perf_counter()
        self.runtime = RKLLMRuntime(
            library_path=library_path,
            model_path=model_path,
            target_platform=cfg.rkllm.target_platform,
            max_context=model_cfg.max_context,
            max_new_tokens=model_cfg.max_new_tokens,
            temperature=model_cfg.temperature,
            top_k=model_cfg.top_k,
            top_p=model_cfg.top_p,
            repeat_penalty=model_cfg.repeat_penalty,
            enable_thinking=cfg.rkllm.enable_thinking,
            num_npu_core=cfg.rkllm.num_npu_core,
            system_prompt=model_cfg.system_prompt,
        )
        if model_cfg.system_prompt:
            try:
                self.runtime.set_chat_template(model_cfg.system_prompt)
            except Exception as exc:
                log.warning("rkllm_set_chat_template failed: %s", exc)
        self.load_time_s = time.perf_counter() - start
        self.model_name = model_cfg.name or self.runtime.model_name
        self.model_size_gb = file_size_gb(model_path)
        log.info("RKLLM backend ready: %s (%.1fs)", self.model_name, self.load_time_s)

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> GenerationResult:
        start = time.perf_counter()
        text, perf = self.runtime.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            timeout=self.cfg.rkllm.timeout_seconds,
        )
        elapsed = time.perf_counter() - start
        prefill_time = float(perf.get("prefill_time_ms", 0.0)) / 1000.0
        decode_time = float(perf.get("generate_time_ms", 0.0)) / 1000.0
        prompt_tokens = int(perf.get("prefill_tokens", 0))
        generated_tokens = int(perf.get("generate_tokens", 0))
        if not prefill_time:
            prefill_time = elapsed * 0.3
        if not decode_time:
            decode_time = max(elapsed - prefill_time, 1e-6)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            load_time_s=self.load_time_s,
            ttft_s=prefill_time,
            prefill_time_s=prefill_time,
            prefill_tok_s=prompt_tokens / prefill_time if prefill_time else 0.0,
            decode_time_s=decode_time,
            decode_tok_s=generated_tokens / decode_time if decode_time else 0.0,
            total_time_s=elapsed,
            peak_ram_gb=read_vm_hwm_gb(),
            memory_usage_mb=float(perf.get("memory_usage_mb", 0.0)),
            backend=self.backend,
            model=self.model_name,
            extra={"model_size_gb": round(self.model_size_gb, 3)},
        )

    def close(self) -> None:
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            runtime.release()
