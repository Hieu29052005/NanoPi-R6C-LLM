# NanoPi-R6C-LLM — baseline vs 4-bit trên RK3588S

Toolkit tải model, lượng tử hoá, đánh giá chất lượng/hiệu năng và phục vụ suy luận
cho **Qwen2.5-7B-Instruct** (khuyến nghị) trên **NanoPi-R6C (RK3588S, 8 GB RAM, NPU 6 TOPS)**.
Kế hoạch chi tiết nằm ở `KE_HOACH_QUANTIZE_QWEN7B_4BIT_NANOPI_R6C.md`.
Hướng dẫn **end-to-end** từ quantize trên 2×RTX 5080 (CUDA 13 + SpinQuant W4A8)
→ export GGUF → đẩy code/model lên board → benchmark/đo công suất trên R6C:
**`HUONG_DAN_CHAY_NANOPI_R6C.md`** (có phần Phase 0: các hotfix cần áp trước khi chạy).

Mặc định mọi backend là **mock**, nên repo chạy được ngay trên laptop, không cần NPU/model.

## Hiện trạng phần cứng (đọc trước khi chạy)

| Hướng | Định dạng | Nơi chạy | Ghi chú |
|---|---|---|---|
| A | GGUF Q4_K_M/Q4_0 (sau PTQ ± SpinQuant) | CPU + RAM trên R6C | **Khả thi nhất cho 7B**, ~4.3–4.7 GB |
| B | GGUF W8A8/W4A4 qua rk-llama.cpp | NPU | Giới hạn 4 GB/IOMMU domain, cần `RKNPU_DOMAINS` |
| C | `.rkllm` W8A8 | NPU | 7B ≈ 8.2 GB → **OOM** trên 8 GB; chỉ dùng cho model ≤3B |
| D | W4A8 native | — | **Không có trên RK3588S**; W4A8 chỉ có `w4a8_g32` trên RK3562 |

`RKLLM` trên RK3588S chỉ nhận `w8a8`, `w8a8_g128/256/512`; `w4a16` thuộc RK3576.
Xem `scripts/export_rkllm_w8a8.py` — script sẽ từ chối cấu hình sai thay vì tạo model không load được.

## Kiến trúc

```
                    ┌──────────────── x86_64 host / Kaggle ────────────────┐
HF checkpoint ─────►│ download → (SpinQuant) → GGUF export → lm-eval/VMLU │
                    └───────────────────────────┬─────────────────────────┘
                                                │ .gguf / .rkllm
                                                ▼
                    ┌──────────────── NanoPi-R6C ─────────────────────────┐
                    │ run.py → backend (mock|gguf|rkllm|hf) → FastAPI      │
                    │ bench_board.py + power_monitor.py → data/results/    │
                    └──────────────────────────────────────────────────────┘
```

## Cấu trúc

```
NanoPi-R6C-LLM/
├── config/config.yaml            # model/backend/bench/eval/power
├── run.py                        # entrypoint
├── src/nano_llm/
│   ├── config.py, logging_setup.py, main.py
│   ├── models/
│   │   ├── hf_backend.py         # baseline FP16/BF16 (transformers)
│   │   ├── llamacpp_backend.py   # GGUF: llama-cpp-python hoặc llama-cli
│   │   ├── rkllm_backend.py      # NPU W8A8
│   │   ├── rkllm_bindings.py     # ctypes ABI rknn-llm v1.3.1
│   │   └── slm.py                # factory + MockSLM
│   ├── api/app.py                # FastAPI OpenAI-compatible
│   ├── bench/                    # metrics + benchmark runner
│   └── utils/system.py           # RAM, board, NPU, ADC
├── scripts/                      # download/quantize/eval/deploy/bench/power
├── eval/                         # prompt sets + VMLU artifacts
├── data/{corpus,results}/
├── systemd/nano-llm.service
└── tests/
```

## Quickstart (laptop, mock)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

.venv/bin/python run.py --check                 # cấu hình + module + board info
.venv/bin/python run.py --demo --force-mock     # vài prompt
.venv/bin/python run.py --force-mock            # API :8080 + dashboard
# http://localhost:8080/dashboard
```

Hoặc `make install-dev && make test && make demo && make api`.

## Backends

Chọn trong `config/config.yaml`:

```yaml
model:
  backend: mock   # mock | hf | gguf | rkllm
  path: ""        # thư mục HF, file .gguf hoặc .rkllm
```

- `mock`: deterministic, dùng cho test/CI/demo.
- `hf`: baseline FP16/BF16, cần `transformers` + `torch` (`requirements-host.txt`).
- `gguf`: cần `llama-cpp-python` hoặc `llama-cli`. Với rk-llama.cpp đặt `llamacpp.rknpu_hybrid`.
- `rkllm`: cần `librkllmrt.so` + file `.rkllm`; hỗ trợ **W8A8** trên RK3588S.

Factory tự fallback về `mock` nếu backend không khả dụng (có log cảnh báo).

## Quy trình trên host (x86_64 / Kaggle)

```bash
pip install -r requirements-host.txt

# 1) Tải model
python scripts/download_model.py --list
python scripts/download_model.py --model qwen2.5-7b --out models

# 2) (Tùy chọn) SpinQuant rotation + GPTQ/RTN W4A8
#    LICENSE: CC-BY-NC 4.0 (phi thương mại) — cần legal duyệt trước.
git clone https://github.com/shijiew/QwenSpinQuant ~/QwenSpinQuant
bash scripts/spinquant_w4a8.sh --model models/Qwen2.5-7B-Instruct --out models/qwen-spinquant --yes

# 3) GGUF export + quantize
LLAMA_CPP_DIR=~/llama.cpp bash scripts/export_gguf.sh \
  --model-dir models/Qwen2.5-7B-Instruct --quant Q4_K_M --outdir data/gguf

# 4) Chất lượng
python scripts/eval_quality.py --model models/Qwen2.5-7B-Instruct --device cuda
python scripts/prepare_vmlu.py --split valid   # lần đầu: tải + chuẩn hóa VMLU
python scripts/eval_vmlu.py --dataset data/vmlu/vmlu_valid.jsonl --backend hf --model models/Qwen2.5-7B-Instruct --device cuda --limit 500
python scripts/eval_vmlu.py --dataset data/vmlu/vmlu_valid.jsonl --backend gguf --model data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf --ngl 99 --limit 500
bash scripts/eval_ppl.sh --model data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf
# Đánh giá Δ so với ngưỡng: xem HUONG_DAN_CHAY_NANOPI_R6C.md mục 4.5
# và báo cáo chi tiết data/results/BAO_CAO_DANH_GIA_CHAT_LUONG_PHASE4.md

# 5) Calibration + .rkllm W8A8 (chỉ cho model ≤3B)
python scripts/generate_calibration.py --output data/calibration.json --count 128
python scripts/export_rkllm_w8a8.py --model-dir models/Qwen2.5-1.5B-Instruct \
  --quantized-dtype w8a8 --target-platform rk3588
```

Nếu cần Hướng B (NPU + rk-llama.cpp):

```bash
bash scripts/rkllama_setup.sh --dir ~/rk-llama.cpp
# trên board:
export RKNPU_DOMAINS=...
export RKNPU_HYBRID=...
```

## Trên board

```bash
BOARD=pi@192.168.1.50 bash scripts/deploy_board.sh
ssh pi@192.168.1.50
cd ~/NanoPi-R6C-LLM
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# copy librkllmrt.so -> models/, model .gguf/.rkllm -> models/
bash scripts/check_board.sh

# serve API
bash scripts/run_board.sh --config config/config.yaml
# hoặc systemd
sudo cp systemd/nano-llm.service /etc/systemd/system/
sudo systemctl enable --now nano-llm
```

`run_board.sh` đặt `LD_LIBRARY_PATH`, `RKLLM_LOG_LEVEL=1`, `ulimit -n 65536` và CPU governor.
Với mô hình >4 GB/IOMMU domain, export `RKNPU_DOMAINS` trước khi chạy.

## Benchmark

```bash
# TTFT, prefill, decode, peak RAM, load time (JSON + bảng)
python scripts/bench_board.py --config config/config.yaml \
  --prompt-tokens 128,512,1024,2048,4096 --gen-tokens 256 --runs 3

# Prompt thật (eval/prompts_vi.jsonl có trường "prompt")
python scripts/bench_board.py --prompts-file eval/prompts_vi.jsonl

# Công suất: synthetic (smoke) | serial | file | adc | manual
python scripts/power_monitor.py --meter synthetic --duration 30 --out data/results/qwen4bit_power.csv
python scripts/power_monitor.py --meter serial --serial-port /dev/ttyUSB0 --duration 300 \
  --out data/results/qwen4bit_power.csv
```

Kết quả ghi vào `data/results/` (JSON) và CSV power. ADC kênh 2 trên R6C chỉ đo **điện áp**;
muốn ra watt cần USB-C power meter hoặc shunt.

## Quality gate (theo biên bản họp)

| Metric | Ngưỡng |
|---|---|
| WikiText-2 PPL | Δ ≤ +5% (W4A8) / +10% (W4A4) |
| MMLU, ARC-C, HellaSwag, VMLU | Δ ≥ −2 điểm % |
| GSM8K | Δ ≥ −5 điểm % |
| IFEval | Δ ≥ −3 điểm % |
| Model size | ≤ 4.7 GB |
| Peak RAM | ≤ 6.5 GB |
| Load time | ≤ 30 s |
| TTFT @2048 | ≤ 120 s (CPU) / ≤ 90 s (NPU) |
| Prefill | ≥ 8 tok/s (CPU) / ≥ 12 tok/s (NPU) |
| Decode | ≥ 4 tok/s (CPU) / ≥ 6 tok/s (NPU) |

## API

OpenAI-compatible:

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"Xin chào"}],
  "max_tokens":128
}'
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (hỗ trợ `stream`),
`POST /v1/completions`, `GET /dashboard`. Đặt `api.api_key` để bật Bearer auth.

## License / pháp lý

- Qwen2.5-7B-Instruct: **Apache-2.0**.
- Llama-3.1-8B-Instruct: **Llama 3.1 Community License** (gated).
- DeepSeek-R1-Distill-Qwen-7B: kiểm tra model card trước khi dùng.
- **QwenSpinQuant / SpinQuant: CC-BY-NC 4.0 — phi thương mại.** `spinquant_w4a8.sh`
  từ chối chạy nếu không có `--yes` sau khi legal duyệt.
- VMLU và dataset bên thứ ba: kiểm tra điều khoản riêng.

## Tham chiếu

- SpinQuant — https://arxiv.org/abs/2405.16406
- LiquidGEMM (W4A8 GPU, tham chiếu) — https://arxiv.org/abs/2509.01229
- QwenSpinQuant — https://github.com/shijiew/QwenSpinQuant
- rk-llama.cpp (rknpu2) — https://github.com/invisiofficial/rk-llama.cpp
- RKNN-LLM — https://github.com/airockchip/rknn-llm
- VMLU — https://github.com/ZaloAI-Jaist/VMLU
