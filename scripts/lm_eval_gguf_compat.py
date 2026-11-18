#!/usr/bin/env python3
"""Chạy lm-eval CLI với backend ``gguf`` tương thích llama-server hiện đại.

lm-eval 0.4.13 ``GGUFLM`` mong đợi logprobs dạng legacy của OpenAI Completions
(``token_logprobs`` / ``text_offset`` / ``top_logprobs``) trả về khi ``echo=true``.
llama.cpp build >= b4xxx (đã thử build 12476) **từ chối ``echo``** và chỉ trả
``logprobs.content`` cho token được sinh ra. Hệ quả: mọi request loglikelihood
bị lm-eval bỏ qua, rồi crash ``ValueError: zip() argument 2 is longer than
argument 1`` tại ``evaluator.py``.

Wrapper này monkeypatch ``GGUFLM.loglikelihood``: chấm điểm continuation từng
token qua endpoint native ``/completion`` của llama-server, dùng
``completion_probabilities[0].top_logprobs`` để lấy logprob xác suất của đúng
token cần tính. Các task ``generate_until`` (GSM8K, IFEval) giữ nguyên luồng cũ.

Biến môi trường:
  GGUF_COMPAT_NPROBS   số token trong top_logprobs cho mỗi lần hỏi (mặc định 100)
  GGUF_COMPAT_TIMEOUT  timeout HTTP, giây (mặc định 180)
"""

from __future__ import annotations

import json
import os
import urllib.request

from lm_eval.__main__ import cli_evaluate
from lm_eval.models import gguf as gguf_mod

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_NPROBS = int(os.environ.get("GGUF_COMPAT_NPROBS", "100"))
_TIMEOUT = int(os.environ.get("GGUF_COMPAT_TIMEOUT", "180"))
_FALLBACK_LOGPROB = -1e6  # token nằm ngoài top-N: coi như gần như không thể


def _post(base_url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _OPENER.open(req, timeout=_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _tokenize(base_url: str, text: str) -> list[int]:
    return _post(
        base_url,
        "/tokenize",
        {"content": text, "add_special": True, "parse_special": False},
    )["tokens"]


def _split_tokens(base_url: str, context: str, continuation: str):
    """Token ID của context và của continuation, khớp tokenizer của server."""
    full = _tokenize(base_url, context + continuation)
    ctx = _tokenize(base_url, context)
    if full[: len(ctx)] == ctx:
        return ctx, full[len(ctx) :]
    # BPE có thể gộp khác nhau ở ranh giới: lấy tiền tố chung dài nhất.
    n = 0
    while n < min(len(ctx), len(full)) and ctx[n] == full[n]:
        n += 1
    return full[:n], full[n:]


def _score_continuation(base_url: str, ctx_ids: list[int], cont_ids: list[int]):
    """Tổng log P(cont | ctx) và cờ greedy (mọi token đều là argmax)."""
    prefix = list(ctx_ids)
    total = 0.0
    greedy = True
    for token_id in cont_ids:
        payload = {
            "prompt": prefix,
            "n_predict": 1,
            "n_probs": _NPROBS,
            "temperature": 0.0,
            "cache_prompt": True,
        }
        data = _post(base_url, "/completion", payload)
        probs = data.get("completion_probabilities") or []
        top = (probs[0].get("top_logprobs") or []) if probs else []
        found = next((p for p in top if p.get("id") == token_id), None)

        if found is None and _NPROBS < 1000:
            # Lần đầu không thấy token mục tiêu trong top-N: thử N lớn hơn.
            payload["n_probs"] = 1000
            data = _post(base_url, "/completion", payload)
            probs = data.get("completion_probabilities") or []
            top = (probs[0].get("top_logprobs") or []) if probs else []
            found = next((p for p in top if p.get("id") == token_id), None)

        if found is None:
            return _FALLBACK_LOGPROB, False

        total += float(found["logprob"])
        greedy = greedy and bool(top and top[0].get("id") == token_id)
        prefix.append(token_id)
    return total, greedy


def _patched_loglikelihood(self, requests, disable_tqdm: bool = False):
    from tqdm import tqdm

    if not requests:
        return []
    results = []
    for request in tqdm(requests, disable=disable_tqdm):
        context, continuation = request.args
        ctx_ids, cont_ids = _split_tokens(self.base_url, context, continuation)
        results.append(_score_continuation(self.base_url, ctx_ids, cont_ids))
    return results


# Áp dụng patch trước khi lm-eval khởi tạo model.
gguf_mod.GGUFLM.loglikelihood = _patched_loglikelihood

if __name__ == "__main__":
    cli_evaluate()
