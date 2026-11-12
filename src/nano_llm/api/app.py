from __future__ import annotations

import json
import time
import uuid
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from nano_llm.config import Config
from nano_llm.models.prompting import normalize_messages, render_chat, user_prompt
from nano_llm.models.slm import BaseSLM, GenerationResult


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]] = ""
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False


def _usage(result: GenerationResult) -> dict:
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.generated_tokens,
        "total_tokens": result.prompt_tokens + result.generated_tokens,
    }


def _check_auth(request: Request, cfg: Config) -> None:
    if not cfg.api.api_key:
        return
    header = request.headers.get("authorization", "")
    expected = f"Bearer {cfg.api.api_key}"
    if header != expected:
        raise HTTPException(status_code=401, detail="invalid API key")


def _chat_prompt(cfg: Config, messages: List[ChatMessage]) -> str:
    normalized = normalize_messages(messages)
    has_system = any(message["role"] == "system" for message in normalized)
    system = "" if has_system else cfg.model.system_prompt
    return render_chat(normalized, system_prompt=system)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(slm: BaseSLM, cfg: Config) -> FastAPI:
    app = FastAPI(title="NanoPi-R6C-LLM", version="0.1.0")
    app.state.slm = slm
    app.state.cfg = cfg

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "backend": slm.info().get("backend"),
            "model": slm.info().get("model"),
            "max_context": cfg.model.max_context,
            "uptime_s": round(time.monotonic(), 1),
        }

    @app.get("/v1/models")
    def models(request: Request) -> dict:
        _check_auth(request, cfg)
        info = slm.info()
        return {
            "object": "list",
            "data": [
                {
                    "id": info.get("model") or cfg.model.name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "backend": info.get("backend"),
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(payload: ChatCompletionRequest, request: Request):
        _check_auth(request, cfg)
        if not payload.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        prompt = _chat_prompt(cfg, payload.messages)
        result = slm.generate(prompt, max_new_tokens=payload.max_tokens)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        model_id = payload.model or cfg.model.name
        if not payload.stream:
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(result),
                "timings": {
                    "ttft_s": round(result.ttft_s, 4),
                    "prefill_tok_s": round(result.prefill_tok_s, 3),
                    "decode_tok_s": round(result.decode_tok_s, 3),
                    "total_time_s": round(result.total_time_s, 4),
                },
            }

        def stream():
            base = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
            }
            yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            yield _sse({**base, "choices": [{"index": 0, "delta": {"content": result.text}, "finish_reason": None}]})
            yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/completions")
    def completions(payload: CompletionRequest, request: Request):
        _check_auth(request, cfg)
        prompt = payload.prompt[0] if isinstance(payload.prompt, list) else payload.prompt
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt must not be empty")
        result = slm.generate(prompt, max_new_tokens=payload.max_tokens)
        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        model_id = payload.model or cfg.model.name
        if not payload.stream:
            return {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {"index": 0, "text": result.text, "finish_reason": "stop", "logprobs": None}
                ],
                "usage": _usage(result),
            }

        def stream():
            base = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
            }
            yield _sse({**base, "choices": [{"index": 0, "text": result.text, "finish_reason": None}]})
            yield _sse({**base, "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        info = slm.info()
        return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>NanoPi-R6C-LLM</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#16202a}}
code,pre{{background:#f2f5f8;border-radius:6px}}pre{{padding:1rem;overflow:auto}}
textarea{{width:100%;min-height:100px}}button{{margin-top:.5rem;padding:.5rem 1rem}}
</style></head><body>
<h1>NanoPi-R6C-LLM</h1>
<p>Backend: <b>{info.get("backend")}</b> · Model: <b>{info.get("model")}</b> ·
Context: <b>{info.get("max_context")}</b></p>
<textarea id="q" placeholder="Nhập câu hỏi tiếng Việt..."></textarea><br>
<button onclick="ask()">Gửi</button>
<pre id="out">...</pre>
<script>
async function ask(){{
  const q=document.getElementById('q').value;
  const out=document.getElementById('out');
  out.textContent='...';
  const r=await fetch('/v1/chat/completions',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{messages:[{{role:'user',content:q}}],max_tokens:256}})}});
  const j=await r.json();
  out.textContent=j.choices?.[0]?.message?.content || JSON.stringify(j,null,2);
}}
</script></body></html>"""

    return app
