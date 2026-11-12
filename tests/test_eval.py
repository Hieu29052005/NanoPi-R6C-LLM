from conftest import load_script

export_rkllm = load_script("export_rkllm_w8a8")
quality = load_script("eval_quality")
vmlu = load_script("eval_vmlu")


def test_validate_quant_rk3588_rejects_w4a8():
    ok, message = export_rkllm.validate_quant("rk3588", "w4a8_g32")
    assert not ok
    assert "W4A8" in message or "w4a8" in message


def test_validate_quant_matrix():
    assert export_rkllm.validate_quant("rk3588", "w8a8")[0]
    assert export_rkllm.validate_quant("rk3588s", "w8a8_g128")[0]
    assert export_rkllm.validate_quant("rk3576", "w4a16")[0]
    assert export_rkllm.validate_quant("rk3562", "w4a8_g32")[0]
    assert not export_rkllm.validate_quant("rk3576", "w4a8_g32")[0]


def test_extract_choice_and_answer_normalization():
    assert vmlu.extract_choice("Đáp án: B") == "B"
    assert vmlu.extract_choice("a. vì trời mưa") == "A"
    assert vmlu.extract_choice("không có") is None

    choices = ["Hà Nội", "Đà Nẵng", "Huế", "Cần Thơ"]
    assert vmlu.normalize_answer(1, choices) == "B"
    assert vmlu.normalize_answer("C", choices) == "C"
    assert vmlu.normalize_answer("Huế", choices) == "C"
    assert vmlu.is_correct("b", "B")


def test_normalize_sample_and_format():
    sample = vmlu.normalize_sample(
        {
            "question": "Thủ đô của Việt Nam là gì?",
            "choices": ["Hà Nội", "Đà Nẵng", "Huế", "Cần Thơ"],
            "answer": 0,
            "subject": "địa lý",
        }
    )
    assert sample["answer"] == "A"
    prompt = vmlu.format_mcq(sample)
    assert "Câu hỏi:" in prompt
    assert "A. Hà Nội" in prompt
    assert "Đáp án:" in prompt


def test_normalize_choices_hf_arc_shape():
    assert vmlu.normalize_choices({"text": ["a", "b"], "label": ["A", "B"]}) == ["a", "b"]
    assert vmlu.normalize_choices(["x", "y"]) == ["x", "y"]


def test_extract_task_scores_ignores_stderr():
    payload = {
        "results": {
            "mmlu": {"acc,none": 0.62, "acc_stderr,none": 0.01},
            "arc_challenge": {"acc_norm,none": 0.44},
        }
    }
    scores = quality.extract_task_scores(payload)
    assert scores["mmlu"]["acc"] == 0.62
    assert "acc_stderr" not in scores["mmlu"]
    assert scores["arc_challenge"]["acc_norm"] == 0.44


def test_build_lm_eval_cmd_contains_tasks_and_limit():
    cmd = quality.build_lm_eval_cmd(
        "hf", "Qwen/Qwen2.5-7B-Instruct", ["mmlu", "vmlu"], 50, "/tmp/out.json"
    )
    assert cmd[0] == "lm_eval"
    assert "mmlu,vmlu" in cmd
    assert "50" in cmd
    assert "trust_remote_code=True" in " ".join(cmd)
