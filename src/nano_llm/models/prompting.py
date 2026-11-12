from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _get(message: Any, key: str) -> str:
    if isinstance(message, dict):
        return str(message.get(key, ""))
    return str(getattr(message, key, ""))


def normalize_messages(messages: Iterable[Any]) -> List[Dict[str, str]]:
    normalized = []
    for message in messages:
        role = _get(message, "role") or "user"
        content = _get(message, "content")
        normalized.append({"role": role, "content": content})
    return normalized


def render_chat(
    messages: Iterable[Any],
    system_prompt: str = "",
    add_generation_prompt: bool = True,
) -> str:
    """Render ChatML (Qwen2.5) without requiring a tokenizer.

    Llama-3 and other families use different special tokens; when running those
    models through HuggingFace or llama-server, the model's own chat template is
    used instead. This fallback keeps the mock/rkllm paths usable offline.
    """
    parts: List[str] = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
    for message in normalize_messages(messages):
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def user_prompt(text: str, system_prompt: str = "") -> str:
    return render_chat([{"role": "user", "content": text}], system_prompt=system_prompt)
