import json

from conftest import load_script

export_rkllm = load_script("export_rkllm_w8a8")
quality = load_script("eval_quality")
vmlu = load_script("eval_vmlu")
prepare_vmlu = load_script("prepare_vmlu")


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


def test_vmlu_ngl_override_and_gguf_default(monkeypatch, capsys):
    from nano_llm.config import Config

    captured = {}

    def fake_load_config(path, root):
        captured["cfg"] = Config()
        return captured["cfg"]

    monkeypatch.setattr(vmlu, "load_config", fake_load_config)
    monkeypatch.setattr(
        vmlu.sys,
        "argv",
        [
            "eval_vmlu.py",
            "--dataset", "data/vmlu/vmlu_valid.jsonl",
            "--backend", "gguf",
            "--ngl", "42",
            "--limit", "1", "--dry-run",
        ],
    )
    assert vmlu.main() == 0
    assert captured["cfg"].llamacpp.n_gpu_layers == 42

    monkeypatch.setattr(
        vmlu.sys,
        "argv",
        [
            "eval_vmlu.py",
            "--dataset", "data/vmlu/vmlu_valid.jsonl",
            "--backend", "gguf",
            "--limit", "1", "--dry-run",
        ],
    )
    assert vmlu.main() == 0
    assert captured["cfg"].llamacpp.n_gpu_layers == 99


def test_prepare_vmlu_strips_choice_prefix_and_maps_subject(tmp_path):
    assert prepare_vmlu.strip_prefix("A. Hà Nội") == "Hà Nội"
    assert prepare_vmlu.strip_prefix("D) Cần Thơ") == "Cần Thơ"
    assert prepare_vmlu.SUBJECTS["28"] == ("Macroeconomics", "Social Science")
    assert prepare_vmlu.SUBJECTS["51"] == ("Clinical Pharmacology", "Other")

    src = tmp_path / "valid.jsonl"
    src.write_text(
        "\n".join(
            [
                '{"id":"28-0007","question":"Q1","choices":["A. a","B. b","C. c","D. d"],"answer":"C"}',
                '{"id":"51-0001","question":"Q2","choices":["A. a","B. b","C. c"],"answer":"A"}',
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "prepared.jsonl"
    assert prepare_vmlu.convert(str(src), str(out)) == 0

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["choices"] == ["a", "b", "c", "d"]
    assert rows[0]["subject"] == "28_Macroeconomics"
    assert rows[0]["answer"] == "C"
    assert rows[1]["subject"] == "51_Clinical_Pharmacology"
    # format_mcq() thêm prefix A./B./... -> prepared data không được có prefix cũ
    assert "A. a" in vmlu.format_mcq(vmlu.normalize_sample(rows[0]))
    assert "A. A." not in vmlu.format_mcq(vmlu.normalize_sample(rows[0]))


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
    assert cmd[:3] == [quality.sys.executable, "-m", "lm_eval"]
    assert "mmlu,vmlu" in cmd
    assert "50" in cmd
    assert "trust_remote_code=True" in " ".join(cmd)
    # default: single-GPU path (parallelize is opt-in at build_lm_eval_cmd level)
    assert "parallelize=True" not in " ".join(cmd)


def test_build_lm_eval_cmd_parallelize_splits_across_gpus():
    cmd = quality.build_lm_eval_cmd(
        "hf", "Qwen/Qwen2.5-7B-Instruct", ["mmlu"], 1, "/tmp/out.json",
        parallelize=True, max_memory_per_gpu="10GiB",
    )
    joined = " ".join(cmd)
    assert "parallelize=True" in joined
    assert "max_memory_per_gpu=10GiB" in joined


def test_build_lm_eval_cmd_gguf_passes_base_url():
    cmd = quality.build_lm_eval_cmd(
        "gguf", "data/gguf/model.gguf", ["mmlu"], 1, "/tmp/out.json",
        base_url="http://127.0.0.1:18080", max_length=4096,
    )
    joined = " ".join(cmd)
    assert "lm_eval_gguf_compat.py" in joined
    assert "base_url=http://127.0.0.1:18080" in joined
    assert "max_length=4096" in joined
    assert "parallelize" not in joined
