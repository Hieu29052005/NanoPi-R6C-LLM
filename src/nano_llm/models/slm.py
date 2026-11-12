from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from nano_llm.config import Config, ModelConfig
from nano_llm.utils.system import read_vm_hwm_gb

log = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate used when no tokenizer is available."""
    if not text:
        return 0
    return max(1, int(len(text) / 4))


@dataclass
class GenerationResult:
    text: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    load_time_s: float = 0.0
    ttft_s: float = 0.0
    prefill_time_s: float = 0.0
    prefill_tok_s: float = 0.0
    decode_time_s: float = 0.0
    decode_tok_s: float = 0.0
    total_time_s: float = 0.0
    peak_ram_gb: float = 0.0
    memory_usage_mb: float = 0.0
    backend: str = ""
    model: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseSLM(ABC):
    backend = "base"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_cfg: ModelConfig = cfg.model
        self.model_name = cfg.model.name
        self.load_time_s = 0.0

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> GenerationResult:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "max_context": self.model_cfg.max_context,
            "max_new_tokens": self.model_cfg.max_new_tokens,
            "load_time_s": round(self.load_time_s, 3),
        }

    def close(self) -> None:
        return None

    def __enter__(self) -> "BaseSLM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class MockSLM(BaseSLM):
    """Deterministic, dependency-free backend for tests, CI and laptop demos."""

    backend = "mock"

    def __init__(
        self,
        cfg: Optional[Config] = None,
        prefill_tok_s: float = 40.0,
        decode_tok_s: float = 6.0,
        max_mock_tokens: int = 64,
    ):
        super().__init__(cfg or Config())
        self.prefill_tok_s = prefill_tok_s
        self.decode_tok_s = decode_tok_s
        self.max_mock_tokens = max_mock_tokens
        self.load_time_s = 0.01

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> GenerationResult:
        start = time.perf_counter()
        prompt_tokens = self.count_tokens(prompt)
        requested = max_new_tokens or self.model_cfg.max_new_tokens
        generated_tokens = max(1, min(int(requested), self.max_mock_tokens))
        sentence = "Đây là phản hồi mô phỏng từ NanoPi-R6C-LLM. "
        repeats = max(1, math.ceil(generated_tokens / 8))
        text = (sentence * repeats).strip()

        ttft = prompt_tokens / max(self.prefill_tok_s, 1e-6)
        decode_time = max(generated_tokens - 1, 1) / max(self.decode_tok_s, 1e-6)
        total = ttft + decode_time
        # Keep the API honest: no wall-clock sleeping, only modelled timings.
        elapsed = time.perf_counter() - start
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            load_time_s=self.load_time_s,
            ttft_s=ttft,
            prefill_time_s=ttft,
            prefill_tok_s=prompt_tokens / ttft if ttft else 0.0,
            decode_time_s=decode_time,
            decode_tok_s=generated_tokens / decode_time if decode_time else 0.0,
            total_time_s=max(total, elapsed),
            peak_ram_gb=read_vm_hwm_gb(),
            memory_usage_mb=0.0,
            backend=self.backend,
            model=self.model_name,
            extra={"simulated": True},
        )


def build_slm(cfg: Config, force_mock: bool = False) -> BaseSLM:
    """Build the configured backend, falling back to MockSLM on failure."""
    backend = (cfg.model.backend or "mock").lower()
    if force_mock or backend in ("mock", "none"):
        return MockSLM(cfg)

    try:
        if backend in ("hf", "transformers"):
            from nano_llm.models.hf_backend import HFBackend

            return HFBackend(cfg)
        if backend in ("gguf", "llamacpp", "llama.cpp", "llama_cpp"):
            from nano_llm.models.llamacpp_backend import LlamaCppBackend

            return LlamaCppBackend(cfg)
        if backend in ("rkllm", "npu"):
            from nano_llm.models.rkllm_backend import RKLLMBackend

            return RKLLMBackend(cfg)
        raise ValueError(f"unknown backend: {cfg.model.backend}")
    except Exception as exc:
        log.warning("backend %r unavailable (%s); falling back to mock", backend, exc)
        mock = MockSLM(cfg)
        mock.model_name = f"mock:{cfg.model.name}"
        return mock
