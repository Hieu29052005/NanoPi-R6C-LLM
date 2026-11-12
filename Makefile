PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: help venv install install-dev install-host test check demo run bench api \
        download calibrate spinquant export-gguf export-rkllm setup-rkllama \
        eval eval-vmlu ppl power deploy clean

help:
	@echo "make venv           - create virtualenv"
	@echo "make install        - install runtime requirements"
	@echo "make install-dev    - install runtime + test requirements"
	@echo "make install-host   - install host tooling (torch/transformers/lm-eval)"
	@echo "make test           - run unit tests"
	@echo "make check          - print resolved configuration"
	@echo "make demo           - run mock prompts"
	@echo "make api            - serve OpenAI-compatible API on :8080"
	@echo "make bench          - benchmark configured backend"
	@echo "make download       - download the candidate models"
	@echo "make calibrate      - generate calibration JSON for RKLLM"
	@echo "make spinquant      - W4A8 rotation+PTQ on a CUDA host (see script help)"
	@echo "make export-gguf    - convert HF -> GGUF + quantize"
	@echo "make export-rkllm   - convert HF -> .rkllm (W8A8 on RK3588)"
	@echo "make setup-rkllama  - build rk-llama.cpp for the NPU (Huong B)"
	@echo "make eval           - lm-eval quality suite"
	@echo "make eval-vmlu      - VMLU (Vietnamese) evaluation"
	@echo "make ppl            - WikiText-2 perplexity via llama-perplexity"
	@echo "make power          - synthetic power monitor smoke test"
	@echo "make deploy         - rsync repo to BOARD"
	@echo "make clean          - remove caches"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

install-host:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-host.txt

test:
	$(PIP) install -q -r requirements-dev.txt
	$(PY) -m pytest -q

check:
	$(PY) run.py --check --config config/config.yaml

demo:
	$(PY) run.py --demo --force-mock

run:
	$(PY) run.py --config config/config.yaml

api:
	$(PY) run.py --config config/config.yaml

bench:
	$(PY) run.py --bench --config config/config.yaml

download:
	$(PY) scripts/download_model.py --model qwen2.5-7b

calibrate:
	$(PY) scripts/generate_calibration.py --output data/calibration_qwen.json --count 128

spinquant:
	bash scripts/spinquant_w4a8.sh --help

export-gguf:
	bash scripts/export_gguf.sh --help

export-rkllm:
	$(PY) scripts/export_rkllm_w8a8.py --help

setup-rkllama:
	bash scripts/rkllama_setup.sh --help

eval:
	$(PY) scripts/eval_quality.py --config config/config.yaml

eval-vmlu:
	$(PY) scripts/eval_vmlu.py --config config/config.yaml

ppl:
	bash scripts/eval_ppl.sh --help

power:
	$(PY) scripts/power_monitor.py --meter synthetic --duration 5 --interval 1 --out data/results/power_smoke.csv

deploy:
	bash scripts/deploy_board.sh

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
