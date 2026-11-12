from nano_llm.bench.metrics import percentile, summarize
from nano_llm.bench.runner import format_table, make_prompt, run_bench, slug
from nano_llm.config import Config
from nano_llm.models.slm import MockSLM


def test_percentile_and_summary():
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 0) == 1
    assert percentile(values, 50) == 3
    assert percentile(values, 100) == 5
    summary = summarize(values)
    assert summary["count"] == 5
    assert summary["mean"] == 3
    assert summary["min"] == 1
    assert summary["max"] == 5


def test_slug_is_safe():
    assert slug("Qwen/Qwen2.5-7B Instruct") == "Qwen-Qwen2.5-7B-Instruct"


def test_make_prompt_hits_target_roughly():
    llm = MockSLM(Config())
    prompt = make_prompt(llm, 100)
    tokens = llm.count_tokens(prompt)

    assert 50 <= tokens <= 130


def test_run_bench_writes_report(tmp_path):
    cfg = Config()
    cfg.base_dir = tmp_path
    cfg.bench.runs = 2
    cfg.bench.warmup = 0
    cfg.bench.results_dir = "data/results"
    cfg.model.max_new_tokens = 8
    llm = MockSLM(cfg)

    report = run_bench(llm, cfg, prompt_tokens=[32, 64], gen_tokens=4)

    assert report["backend"] == "mock"
    assert len(report["cases"]) == 2
    assert report["cases"][0]["summary"]["ttft_s"]["count"] == 2
    assert (tmp_path / "data/results").is_dir()
    assert "ctx_32" in format_table(report)
