from nano_llm.config import Config
from nano_llm.models.prompting import render_chat, user_prompt
from nano_llm.models.slm import MockSLM, build_slm, estimate_tokens


def make_cfg() -> Config:
    cfg = Config()
    cfg.model.backend = "mock"
    cfg.model.max_new_tokens = 32
    return cfg


def test_mock_generate_returns_metrics():
    cfg = make_cfg()
    llm = MockSLM(cfg)
    result = llm.generate("Xin chào, hãy tóm tắt tài liệu.", max_new_tokens=16)

    assert result.text
    assert result.backend == "mock"
    assert result.prompt_tokens > 0
    assert result.generated_tokens == 16
    assert result.ttft_s > 0
    assert result.prefill_tok_s > 0
    assert result.decode_tok_s > 0
    assert result.to_dict()["backend"] == "mock"


def test_factory_falls_back_to_mock_for_unknown_backend():
    cfg = make_cfg()
    cfg.model.backend = "does-not-exist"
    llm = build_slm(cfg)

    assert isinstance(llm, MockSLM)


def test_force_mock_overrides_backend():
    cfg = make_cfg()
    cfg.model.backend = "rkllm"
    llm = build_slm(cfg, force_mock=True)

    assert isinstance(llm, MockSLM)


def test_estimate_tokens_is_positive():
    assert estimate_tokens("") == 0
    assert estimate_tokens("một hai ba bốn") >= 1


def test_chat_render_contains_system_and_user():
    text = render_chat([{"role": "user", "content": "Chào"}], system_prompt="Bạn là trợ lý")

    assert "<|im_start|>system" in text
    assert "Bạn là trợ lý" in text
    assert "Chào" in text
    assert text.endswith("<|im_start|>assistant\n")
    assert user_prompt("Hỏi").startswith("<|im_start|>user")
