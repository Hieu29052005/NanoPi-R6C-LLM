from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from nano_llm.config import Config
from nano_llm.models.prompting import render_chat
from nano_llm.models.slm import BaseSLM, GenerationResult, estimate_tokens
from nano_llm.utils.system import file_size_gb, read_vm_hwm_gb

log = logging.getLogger(__name__)

_DTYPES = {
    "float16": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
}


class HFBackend(BaseSLM):
    """Transformers baseline (FP16/BF16) used for quality and performance reference."""

    backend = "hf"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        model_cfg = cfg.model
        source = model_cfg.path or model_cfg.name
        if model_cfg.path:
            source = cfg.resolve(model_cfg.path)
        tokenizer_source = model_cfg.tokenizer_path or source
        if model_cfg.tokenizer_path:
            tokenizer_source = cfg.resolve(model_cfg.tokenizer_path)

        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, trust_remote_code=model_cfg.trust_remote_code
        )
        dtype_name = _DTYPES.get(model_cfg.dtype.lower(), model_cfg.dtype)
        dtype = getattr(torch, dtype_name)
        kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": model_cfg.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if model_cfg.device.startswith("cuda") or model_cfg.device == "auto":
            kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
            self.device = next(self.model.parameters()).device
        else:
            self.model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
            self.device = torch.device(model_cfg.device)
            self.model.to(self.device)
        self.model.eval()
        self.load_time_s = time.perf_counter() - start
        self.model_name = model_cfg.name or self.model.__class__.__name__
        self.model_size_gb = file_size_gb(source) if model_cfg.path else 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        log.info("HF backend ready: %s (%.1fs)", self.model_name, self.load_time_s)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return estimate_tokens(text)

    def _render(self, prompt: str) -> str:
        messages = []
        if self.model_cfg.system_prompt:
            messages.append({"role": "system", "content": self.model_cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})
        template = getattr(self.tokenizer, "apply_chat_template", None)
        if template is not None:
            try:
                return template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as exc:
                log.debug("chat template failed (%s); using ChatML fallback", exc)
        return render_chat(messages, add_generation_prompt=True)

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> GenerationResult:
        from transformers import TextIteratorStreamer

        max_new = int(max_new_tokens or self.model_cfg.max_new_tokens)
        max_new = max(1, min(max_new, self.model_cfg.max_context - 8))
        text = self._render(prompt)
        max_prompt = max(1, self.model_cfg.max_context - max_new)
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_prompt
        )
        prompt_tokens = int(inputs["input_ids"].shape[1])
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new,
            streamer=streamer,
            do_sample=self.model_cfg.temperature > 0,
            temperature=max(self.model_cfg.temperature, 1e-5),
            top_p=self.model_cfg.top_p,
            top_k=self.model_cfg.top_k or 50,
            repetition_penalty=self.model_cfg.repeat_penalty,
        )
        start = time.perf_counter()
        error = []

        def _run() -> None:
            try:
                with self.torch.inference_mode():
                    self.model.generate(**generate_kwargs)
            except Exception as exc:  # propagate to caller
                error.append(exc)

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        pieces = []
        first_token_at = None
        for piece in streamer:
            if piece and first_token_at is None:
                first_token_at = time.perf_counter()
            pieces.append(piece)
        worker.join()
        end = time.perf_counter()
        if error:
            raise RuntimeError(f"HF generation failed: {error[0]}") from error[0]

        output = "".join(pieces).strip()
        generated_tokens = len(
            self.tokenizer.encode(output, add_special_tokens=False)
        )
        ttft = (first_token_at - start) if first_token_at else (end - start)
        decode_time = max(end - start - ttft, 1e-6)
        peak_ram = read_vm_hwm_gb()
        if self.torch.cuda.is_available():
            peak_ram = self.torch.cuda.max_memory_allocated() / (1024**3)
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
            peak_ram_gb=peak_ram,
            backend=self.backend,
            model=self.model_name,
            extra={"model_size_gb": round(self.model_size_gb, 3)},
        )
