from nano_llm.config import Config, load_config


def test_nested_config_loaded(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model:
  backend: gguf
  name: Qwen/Qwen2.5-7B-Instruct
  path: models/qwen.gguf
  max_context: 2048
bench:
  prompt_tokens: [128, 512]
  runs: 5
eval:
  tasks: [mmlu, vmlu]
power:
  meter: serial
  out: data/results/p.csv
""",
        encoding="utf-8",
    )

    cfg = load_config(str(path), str(tmp_path))

    assert cfg.model.backend == "gguf"
    assert cfg.model.max_context == 2048
    assert cfg.bench.prompt_tokens == [128, 512]
    assert cfg.bench.runs == 5
    assert cfg.eval.tasks == ["mmlu", "vmlu"]
    assert cfg.power.meter == "serial"


def test_relative_paths_resolve_against_base_dir(tmp_path):
    cfg = load_config(None, str(tmp_path))

    assert cfg.resolve("models/model.gguf") == str((tmp_path / "models/model.gguf").resolve())
    assert cfg.resolve("/tmp/absolute.gguf") == "/tmp/absolute.gguf"
    assert cfg.resolve("") == ""


def test_defaults_and_dirs(tmp_path):
    cfg = load_config(None, str(tmp_path))
    cfg.ensure_dirs()

    assert isinstance(cfg, Config)
    assert cfg.model.max_context == 4096
    assert cfg.model.backend == "mock"
    assert (tmp_path / "data/results").is_dir()


def test_summary_reports_backend(tmp_path):
    cfg = load_config(None, str(tmp_path))
    summary = cfg.summary()

    assert summary["backend"] == "mock"
    assert summary["model"] == cfg.model.name
    assert summary["max_context"] == 4096
