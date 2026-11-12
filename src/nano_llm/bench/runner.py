from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from nano_llm.bench.metrics import now_iso, summarize, write_json
from nano_llm.config import Config
from nano_llm.models.slm import BaseSLM, GenerationResult

log = logging.getLogger(__name__)

FILLER = (
    "Thiết bị biên NanoPi-R6C dùng SoC RK3588S với NPU 6 TOPS và 8 GB RAM. "
    "Mục tiêu của dự án là lượng tử hoá mô hình ngôn ngữ 7B xuống 4-bit, "
    "đo chất lượng và hiệu năng suy luận trên phần cứng thực tế. "
    "Câu hỏi đánh giá được viết bằng tiếng Việt để phản ánh use case chatbot. "
)


def slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return value[:80] or "model"


def make_prompt(llm: BaseSLM, target_tokens: int) -> str:
    target = max(1, int(target_tokens))
    text = FILLER
    tokens = llm.count_tokens(text)
    guard = 0
    while tokens < target and guard < 512:
        text += FILLER
        tokens = llm.count_tokens(text)
        guard += 1
    if tokens > target:
        ratio = target / max(tokens, 1)
        text = text[: max(1, int(len(text) * ratio))]
    return text


def _summarize_runs(runs: List[GenerationResult]) -> Dict[str, Any]:
    keys = [
        "prompt_tokens",
        "generated_tokens",
        "ttft_s",
        "prefill_tok_s",
        "decode_tok_s",
        "total_time_s",
        "peak_ram_gb",
        "memory_usage_mb",
    ]
    summary = {key: summarize(getattr(run, key) for run in runs) for key in keys}
    summary["text_sample"] = runs[-1].text[:300] if runs else ""
    return summary


def run_bench(
    llm: BaseSLM,
    cfg: Config,
    out_path: Optional[str] = None,
    prompt_tokens: Optional[List[int]] = None,
    gen_tokens: Optional[int] = None,
    runs: Optional[int] = None,
    warmup: Optional[int] = None,
    prompts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    tokens_list = prompt_tokens or cfg.bench.prompt_tokens
    max_new = int(gen_tokens or cfg.bench.gen_tokens)
    n_runs = max(1, int(runs if runs is not None else cfg.bench.runs))
    n_warmup = max(0, int(warmup if warmup is not None else cfg.bench.warmup))

    if prompts:
        cases = [
            {"label": f"prompt_{index}", "target_tokens": llm.count_tokens(prompt), "prompt": prompt}
            for index, prompt in enumerate(prompts)
        ]
    else:
        cases = [
            {
                "label": f"ctx_{target}",
                "target_tokens": target,
                "prompt": make_prompt(llm, target),
            }
            for target in tokens_list
        ]

    results = []
    for case in cases:
        for _ in range(n_warmup):
            llm.generate(case["prompt"], max_new_tokens=max_new)
        measured: List[GenerationResult] = []
        for _ in range(n_runs):
            result = llm.generate(case["prompt"], max_new_tokens=max_new)
            measured.append(result)
            log.info(
                "%s: prompt=%d gen=%d ttft=%.3fs decode=%.2f tok/s",
                case["label"],
                result.prompt_tokens,
                result.generated_tokens,
                result.ttft_s,
                result.decode_tok_s,
            )
        results.append(
            {
                "label": case["label"],
                "target_tokens": case["target_tokens"],
                "prompt_chars": len(case["prompt"]),
                "max_new_tokens": max_new,
                "summary": _summarize_runs(measured),
                "runs": [run.to_dict() for run in measured],
            }
        )

    info = llm.info()
    report = {
        "timestamp": now_iso(),
        "backend": info.get("backend"),
        "model": info.get("model"),
        "model_size_gb": round(float(getattr(llm, "model_size_gb", 0.0)), 3),
        "load_time_s": round(float(getattr(llm, "load_time_s", 0.0)), 3),
        "max_context": cfg.model.max_context,
        "runs_per_case": n_runs,
        "warmup_per_case": n_warmup,
        "cases": results,
    }
    target = out_path or (
        cfg.resolve(cfg.bench.results_dir)
        + f"/bench_{slug(str(info.get('backend')))}_"
        + f"{slug(str(info.get('model')))}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + ".json"
    )
    report["output"] = write_json(target, report)
    return report


def format_table(report: Dict[str, Any]) -> str:
    header = f"{'case':<12}{'prompt':>8}{'gen':>6}{'ttft_s':>10}{'prefill':>10}{'decode':>10}{'total_s':>10}"
    lines = [header, "-" * len(header)]
    for case in report.get("cases", []):
        summary = case.get("summary", {})
        lines.append(
            f"{case.get('label', ''):<12}"
            f"{summary.get('prompt_tokens', {}).get('median', 0):>8.0f}"
            f"{summary.get('generated_tokens', {}).get('median', 0):>6.0f}"
            f"{summary.get('ttft_s', {}).get('median', 0):>10.3f}"
            f"{summary.get('prefill_tok_s', {}).get('median', 0):>10.2f}"
            f"{summary.get('decode_tok_s', {}).get('median', 0):>10.2f}"
            f"{summary.get('total_time_s', {}).get('median', 0):>10.2f}"
        )
    return "\n".join(lines)
