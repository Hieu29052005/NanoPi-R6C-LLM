from __future__ import annotations

import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, get_args, get_origin, get_type_hints

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when PyYAML is absent
    yaml = None


DEFAULT_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chạy cục bộ trên thiết bị biên NanoPi-R6C. "
    "Trả lời ngắn gọn, chính xác bằng tiếng Việt; nếu không chắc chắn hãy nói rõ."
)


@dataclass
class AppConfig:
    name: str = "nano-llm-r6c"
    log_level: str = "INFO"
    log_file: str = "logs/nano_llm.log"
    seed: int = 42


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    backend: str = "mock"
    path: str = ""
    tokenizer_path: str = ""
    device: str = "cpu"
    dtype: str = "float16"
    max_context: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    trust_remote_code: bool = False


@dataclass
class LlamaCppConfig:
    binary: str = "llama-cli"
    n_threads: int = 4
    n_gpu_layers: int = 0
    n_batch: int = 512
    flash_attn: bool = False
    rknpu_hybrid: str = ""
    extra_args: List[str] = field(default_factory=list)


@dataclass
class RKLLMConfig:
    library_path: str = "models/librkllmrt.so"
    target_platform: str = "rk3588"
    num_npu_core: int = 3
    timeout_seconds: float = 180.0
    enable_thinking: bool = False


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: str = ""


@dataclass
class BenchConfig:
    prompt_tokens: List[int] = field(default_factory=lambda: [128, 512, 1024, 2048, 4096])
    gen_tokens: int = 256
    runs: int = 3
    warmup: int = 1
    results_dir: str = "data/results"


@dataclass
class EvalConfig:
    tasks: List[str] = field(
        default_factory=lambda: ["mmlu", "arc_challenge", "hellaswag", "gsm8k", "ifeval"]
    )
    limit: int = 200
    output_dir: str = "data/results"
    ppl_corpus: str = "data/corpus/wikitext-2-raw-val.txt"
    ppl_chunks: int = 64
    vmlu_dataset: str = ""
    vmlu_split: str = "validation"
    vmlu_limit: int = 500
    vmlu_subjects: List[str] = field(default_factory=list)
    prompts_vi: str = "eval/prompts_vi.jsonl"
    prompts_en: str = "eval/prompts_en.jsonl"


@dataclass
class PowerConfig:
    meter: str = "synthetic"
    serial_port: str = "/dev/ttyUSB0"
    baud: int = 115200
    input_file: str = ""
    adc_raw_path: str = "/sys/devices/platform/fec10000.saradc/iio:device0/in_voltage2_raw"
    adc_scale: float = 3.0
    adc_vref: float = 1.8
    adc_bits: int = 12
    interval_s: float = 1.0
    out: str = "data/results/power.csv"


@dataclass
class Config:
    base_dir: Path = field(default_factory=Path.cwd)
    app: AppConfig = field(default_factory=AppConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    llamacpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    rkllm: RKLLMConfig = field(default_factory=RKLLMConfig)
    api: APIConfig = field(default_factory=APIConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    power: PowerConfig = field(default_factory=PowerConfig)

    def resolve(self, path: str) -> str:
        if not path:
            return path
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str((self.base_dir / p).resolve())

    def ensure_dirs(self) -> None:
        targets = [
            self.app.log_file,
            self.bench.results_dir,
            self.eval.output_dir,
            self.power.out,
        ]
        for path in targets:
            if not path:
                continue
            Path(self.resolve(path)).parent.mkdir(parents=True, exist_ok=True)

    def summary(self) -> Dict[str, Any]:
        return {
            "base_dir": str(self.base_dir),
            "app": self.app.name,
            "backend": self.model.backend,
            "model": self.model.name,
            "model_path": self.resolve(self.model.path) if self.model.path else "",
            "max_context": self.model.max_context,
            "max_new_tokens": self.model.max_new_tokens,
            "rkllm_library": self.resolve(self.rkllm.library_path),
            "results_dir": self.resolve(self.bench.results_dir),
        }


def _list_item_type(tp: Any) -> Optional[type]:
    origin = get_origin(tp)
    if origin in (list, List):
        args = get_args(tp)
        if args and hasattr(args[0], "__dataclass_fields__"):
            return args[0]
    return None


def _apply(obj: Any, data: Dict[str, Any]) -> None:
    hints = get_type_hints(type(obj))
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        field_type = hints.get(key)
        item_type = _list_item_type(field_type) if field_type is not None else None
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _apply(current, value)
        elif isinstance(value, list) and item_type is not None:
            items = []
            for item in value:
                if isinstance(item, dict):
                    inst = item_type()
                    _apply(inst, item)
                    items.append(inst)
                else:
                    items.append(item)
            setattr(obj, key, items)
        else:
            setattr(obj, key, value)


def load_config(path: Optional[str] = None, base_dir: Optional[str] = None) -> Config:
    cfg = Config()
    cfg.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
    if path and os.path.exists(path):
        if yaml is None:
            raise RuntimeError("PyYAML is required to read config files: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _apply(cfg, data)
    cfg.ensure_dirs()
    return cfg
