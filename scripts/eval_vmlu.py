#!/usr/bin/env python3
"""VMLU (Vietnamese multiple-choice) evaluation with any configured backend.

The loader accepts a local JSON/JSONL file or a HuggingFace dataset id, and is
tolerant about column names (question/prompt, choices/options, answer/label).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from nano_llm.bench.metrics import now_iso, write_json  # noqa: E402
from nano_llm.config import load_config  # noqa: E402
from nano_llm.models.slm import build_slm  # noqa: E402

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def letter(index: int) -> str:
    return LETTERS[index] if 0 <= index < len(LETTERS) else "?"


def extract_choice(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else None


def _pick(row: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def normalize_choices(value: Any) -> List[str]:
    if isinstance(value, dict):
        if "text" in value:
            value = value["text"]
        elif "choices" in value:
            value = value["choices"]
        else:
            return [str(value[key]) for key in sorted(value.keys())]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            return [str(item) for item in json.loads(value)]
        except json.JSONDecodeError:
            return [item.strip() for item in value.split("|") if item.strip()]
    return []


def normalize_answer(value: Any, choices: Optional[List[str]] = None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return normalize_answer(value[0], choices)
    if isinstance(value, int):
        return letter(value)
    text = str(value).strip()
    if len(text) == 1 and text.upper() in LETTERS[: len(choices or LETTERS)]:
        return text.upper()
    if text.isdigit():
        return letter(int(text))
    match = re.match(r"^([A-Za-z])[\.\)\:\s]", text)
    if match:
        return match.group(1).upper()
    if choices:
        for index, choice in enumerate(choices):
            if text == choice.strip() or text.lower() == choice.strip().lower():
                return letter(index)
    return text.upper()


def format_mcq(sample: Dict[str, Any]) -> str:
    lines = [f"Câu hỏi: {sample['question']}"]
    for index, choice in enumerate(sample["choices"]):
        lines.append(f"{letter(index)}. {choice}")
    lines.append("Chỉ trả lời bằng một chữ cái A, B, C hoặc D.")
    lines.append("Đáp án:")
    return "\n".join(lines)


def normalize_sample(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = _pick(row, ["question", "prompt", "query", "instruction", "text"])
    choices = normalize_choices(_pick(row, ["choices", "options", "answers", "candidates"]))
    answer = _pick(row, ["answer", "label", "target", "correct_answer", "correct"])
    if not question or not choices or answer is None:
        return None
    subject = str(_pick(row, ["subject", "category", "topic", "field"], "unknown"))
    return {
        "question": str(question),
        "choices": choices,
        "answer": normalize_answer(answer, choices),
        "answer_raw": answer,
        "subject": subject,
    }


def load_samples(
    source: str,
    split: str = "validation",
    limit: int = 0,
    subjects: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    looks_like_path = source.endswith((".json", ".jsonl")) or source.startswith((".", "/", "~"))
    if looks_like_path and not os.path.exists(os.path.expanduser(source)):
        raise RuntimeError(f"dataset file not found: {source}")
    if source and os.path.exists(os.path.expanduser(source)):
        source = os.path.expanduser(source)
        if source.endswith(".jsonl"):
            with open(source, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        else:
            with open(source, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                payload = payload.get("data", payload.get("samples", []))
            rows = list(payload)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "datasets is not installed; pip install -r requirements-host.txt or pass --dataset <file>"
            ) from exc
        last_error = None
        for candidate in [split, "validation", "test", "train"]:
            try:
                dataset = load_dataset(source, split=candidate)
                rows = [dict(row) for row in dataset]
                break
            except Exception as exc:  # dataset/split unavailable
                last_error = exc
        if not rows:
            raise RuntimeError(f"could not load dataset {source!r}: {last_error}")

    samples = []
    wanted = {item.strip().lower() for item in (subjects or []) if item.strip()}
    for row in rows:
        sample = normalize_sample(row)
        if sample is None:
            continue
        if wanted and sample["subject"].lower() not in wanted:
            continue
        samples.append(sample)
        if limit and len(samples) >= limit:
            break
    return samples


def is_correct(prediction: Optional[str], answer: str) -> bool:
    if not prediction:
        return False
    return prediction.strip().upper() == answer.strip().upper()


def evaluate(llm, samples: List[Dict[str, Any]], max_new_tokens: int = 8) -> Dict[str, Any]:
    per_subject: Dict[str, List[int]] = defaultdict(list)
    details = []
    for index, sample in enumerate(samples, start=1):
        prompt = format_mcq(sample)
        result = llm.generate(prompt, max_new_tokens=max_new_tokens)
        prediction = extract_choice(result.text)
        correct = is_correct(prediction, sample["answer"])
        per_subject[sample["subject"]].append(1 if correct else 0)
        details.append(
            {
                "index": index,
                "subject": sample["subject"],
                "question": sample["question"][:240],
                "answer": sample["answer"],
                "prediction": prediction,
                "raw_output": result.text[:160],
                "correct": correct,
            }
        )
        if index % 20 == 0:
            print(f"  ... {index}/{len(samples)}", file=sys.stderr)

    total = len(details)
    correct_total = sum(1 for item in details if item["correct"])
    subject_scores = {
        subject: {
            "correct": sum(values),
            "total": len(values),
            "accuracy": sum(values) / len(values) if values else 0.0,
        }
        for subject, values in sorted(per_subject.items())
    }
    return {
        "samples": total,
        "correct": correct_total,
        "accuracy": correct_total / total if total else 0.0,
        "per_subject": subject_scores,
        "details": details[:200],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate VMLU with the configured backend")
    parser.add_argument("--config", default=os.path.join(ROOT, "config", "config.yaml"))
    parser.add_argument("--backend", default=None, help="override model.backend (hf/gguf/mock)")
    parser.add_argument("--model", default=None, help="override model.path (local dir or .gguf)")
    parser.add_argument("--device", default=None, help="override model.device (cpu/cuda/auto)")
    parser.add_argument(
        "--ngl",
        type=int,
        default=None,
        help="override llamacpp.n_gpu_layers for the gguf backend (default: 99 on CUDA)",
    )
    parser.add_argument("--force-mock", action="store_true")
    parser.add_argument("--dataset", default=None, help="local JSON/JSONL or HF dataset id")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--subjects", default=None, help="comma-separated subject filter")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true", help="load and print dataset stats only")
    args = parser.parse_args()

    cfg = load_config(args.config, ROOT)
    if args.backend:
        cfg.model.backend = args.backend
    if args.model:
        cfg.model.path = args.model
    if args.device:
        cfg.model.device = args.device
    if args.ngl is not None:
        cfg.llamacpp.n_gpu_layers = args.ngl
    elif cfg.model.backend in ("gguf", "llamacpp", "llama.cpp", "llama_cpp"):
        # Host có CUDA -> offload mọi layer. Build CPU-only sẽ tự bỏ qua giá trị này.
        cfg.llamacpp.n_gpu_layers = 99
    source = args.dataset or cfg.eval.vmlu_dataset
    if not source:
        print("ERROR: no dataset. Download VMLU and pass --dataset <file>, or set eval.vmlu_dataset.")
        print("Ref: https://github.com/ZaloAI-Jaist/VMLU")
        return 2
    subjects = (args.subjects.split(",") if args.subjects else cfg.eval.vmlu_subjects) or []
    try:
        samples = load_samples(
            source,
            split=args.split or cfg.eval.vmlu_split,
            limit=args.limit if args.limit is not None else cfg.eval.vmlu_limit,
            subjects=subjects,
        )
    except (RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2

    if not samples:
        print("ERROR: no usable samples (check question/choices/answer columns)")
        return 2
    print(f"loaded {len(samples)} samples from {source}")
    if args.dry_run:
        print(json.dumps(samples[0], ensure_ascii=False, indent=2)[:1200])
        return 0

    llm = build_slm(cfg, force_mock=args.force_mock)
    try:
        report = evaluate(llm, samples, max_new_tokens=args.max_new_tokens)
    finally:
        llm.close()

    payload = {
        "timestamp": now_iso(),
        "dataset": source,
        "backend": llm.info().get("backend"),
        "model": llm.info().get("model"),
        "limit": len(samples),
        **report,
    }
    outdir = cfg.resolve(cfg.eval.output_dir)
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = args.out or os.path.join(outdir, f"vmlu_{stamp}.json")
    write_json(output, payload)
    print(f"VMLU accuracy: {payload['accuracy'] * 100:.2f}% ({payload['correct']}/{payload['samples']})")
    for subject, score in payload["per_subject"].items():
        print(f"  {subject:<24} {score['accuracy'] * 100:5.1f}%  ({score['correct']}/{score['total']})")
    print(f"OK: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
