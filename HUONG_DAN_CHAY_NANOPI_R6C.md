# Hướng dẫn chạy NanoPi-R6C-LLM

Tài liệu này đi từ đầu đến cuối:

1. Chạy thử trên laptop cá nhân (mock, không cần model/NPU).
2. Chuẩn bị model trên laptop/host (tải, lượng tử hoá, export).
3. Đẩy code, dữ liệu và model lên NanoPi-R6C 8 GB.
4. Cài đặt và chạy trực tiếp trên board.
5. Benchmark, đo công suất, chạy API, systemd và xử lý sự cố.

Kế hoạch kỹ thuật/quality gate: `KE_HOACH_QUANTIZE_QWEN7B_4BIT_NANOPI_R6C.md`.
Tóm tắt kiến trúc và lệnh nhanh: `README.md`.

> **Đọc trước:** RK3588S **không có W4A8 native**. RKLLM chỉ nhận `w8a8`
> (`w8a8_g128/256/512`); `w4a16` thuộc RK3576. Vì vậy với Qwen2.5-7B, hướng
> thực tế nhất là **GGUF 4-bit chạy CPU/RAM** trên board. NPU chỉ dùng cho
> `.rkllm` W8A8 với model ≤3B, hoặc W4A4 qua rk-llama.cpp (Hướng B, có giới hạn
> 4 GB/IOMMU domain). Đừng kỳ vọng `export_rkllm_w8a8.py` tạo được W4A8 cho 7B.

---

## 0. Sơ đồ tổng quát

```
┌──────────── Laptop / host x86_64 ────────────┐
│ 1. venv + chạy mock + test                    │
│ 2. tải model HF                               │
│ 3. (tuỳ chọn) SpinQuant W4A8 trên GPU         │
│ 4. export GGUF Q4_K_M / Q4_0                  │
│ 5. kiểm tra chất lượng (lm-eval, PPL, VMLU)   │
└──────────────────┬───────────────────────────┘
                   │ ssh / rsync / USB
                   ▼
┌──────────── NanoPi-R6C (RK3588S, 8 GB) ──────┐
│ 6. cài Python venv + requirements.txt         │
│ 7. copy model + librkllmrt.so (nếu dùng NPU)  │
│ 8. sửa config/config.yaml                     │
│ 9. check_board.sh                             │
│ 10. run_board.sh: demo | bench | API | systemd│
│ 11. power_monitor.py + copy kết quả về laptop │
└───────────────────────────────────────────────┘
```

---

## 1. Yêu cầu

### Laptop cá nhân

- Linux, macOS hoặc Windows + WSL2.
- Python **3.10–3.12** khuyến nghị (repo chạy được với 3.8+, nhưng tool host
  `rkllm-toolkit`, `torch`, `lm-eval` nên dùng 3.10/3.11).
- `git`, `ssh`, `rsync` (hoặc `scp`).
- Với mock/bench/test: không cần GPU, không cần model.
- Với export GGUF: ~16 GB RAM trống và ~20 GB đĩa cho bản f16 + bản quantized.
- Với SpinQuant/GPTQ: GPU CUDA (Kaggle T4/L4/A100 hoặc máy có GPU).

### NanoPi-R6C

- RK3588S, **8 GB RAM**, NPU 6 TOPS.
- OS Linux aarch64 (FriendlyWrt/Ubuntu/Debian tùy image), có SSH.
- Python 3.9+ và `python3-venv`.
- Trống ít nhất **8 GB** cho model 4-bit + runtime + kết quả (khuyến nghị 16 GB).
- Nếu dùng NPU: driver RKNPU tương thích và `librkllmrt.so` đúng phiên bản.
- Nguồn USB-C PD **65 W** khuyến nghị để tránh sụt áp khi NPU/CPU tải cao.

Kiểm tra nhanh trên board:

```bash
cat /proc/device-tree/model
free -h
df -h
uname -a
python3 --version
```

---

## 2. Phần A — Chạy trên laptop cá nhân (mock mode)

Mặc định `config/config.yaml` để `model.backend: mock`, nên chạy được ngay.

```bash
cd /đường/dẫn/FPT/NanoPi-R6C-LLM

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-dev.txt
```

### A.1 Kiểm tra cấu hình

```bash
python run.py --check
```

Lệnh này in ra:

- board/CPU, RAM;
- backend, model, `max_context`;
- đường dẫn model, `librkllmrt.so` (có/không);
- các module Python đã cài (`transformers`, `torch`, `llama_cpp`, `datasets`, `lm_eval`).

### A.2 Demo và chạy API

```bash
# vài prompt mẫu, không cần model
python run.py --demo --force-mock

# API OpenAI-compatible + dashboard tại http://localhost:8080/dashboard
python run.py --force-mock
```

Gọi thử API:

```bash
curl -s http://localhost:8080/health

curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Xin chào"}],"max_tokens":64}'
```

### A.3 Benchmark thử

```bash
python run.py --bench --force-mock \
  --prompt-tokens 128,512,1024 \
  --gen-tokens 32 \
  --runs 1 \
  --warmup 0
```

Kết quả JSON được ghi vào `data/results/`. Ở mock mode, số liệu **được mô phỏng**
(`extra.simulated = true`), chỉ dùng để kiểm tra pipeline, không dùng để báo cáo.

### A.4 Chạy test

```bash
python -m pytest -q
```

Nếu dùng Makefile:

```bash
make install-dev
make test
make demo
make api
```

> `make` dùng `.venv/bin/python`; phải tạo `.venv` trước.

---

## 3. Phần B — Chuẩn bị model trên laptop/host

Phần này chạy trên laptop/host x86_64 hoặc Kaggle. **Không lượng tử hoá trên board.**

### B.1 Tải model

```bash
pip install -r requirements-host.txt
python scripts/download_model.py --list

python scripts/download_model.py \
  --model qwen2.5-7b \
  --out models
```

Model sẽ nằm ở `models/Qwen2.5-7B-Instruct/`.

Các ứng viên:

| Key                        | Repo                                        | License                     |
| -------------------------- | ------------------------------------------- | --------------------------- |
| `qwen2.5-7b`             | `Qwen/Qwen2.5-7B-Instruct`                | Apache-2.0 — khuyến nghị |
| `llama-3.1-8b`           | `meta-llama/Llama-3.1-8B-Instruct`        | Llama 3.1 Community (gated) |
| `deepseek-r1-distill-7b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | kiểm tra model card        |

Nếu model gated:

```bash
export HF_TOKEN=hf_xxx
python scripts/download_model.py --model llama-3.1-8b --out models
```

### B.2 (Tuỳ chọn) SpinQuant + PTQ W4A8

> **License:** QwenSpinQuant/SpinQuant là **CC-BY-NC 4.0 (phi thương mại)**.
> Phải được legal duyệt trước khi dùng cho sản phẩm/dự án thương mại.

Chạy trên host có GPU CUDA:

```bash
# Idempotent: clone lần đầu, các lần sau chỉ pull (tránh lỗi "already exists")
if [ -d ~/QwenSpinQuant ]; then
  git -C ~/QwenSpinQuant pull --ff-only
else
  git clone https://github.com/shijiew/QwenSpinQuant ~/QwenSpinQuant
fi
export QWEN_SPINQUANT_DIR=~/QwenSpinQuant

# Lần đầu: tạo venv riêng cho QwenSpinQuant (pin transformers==4.49.0, torch CUDA)
bash scripts/spinquant_w4a8.sh \
  --setup-env \
  --model models/Qwen2.5-7B-Instruct \
  --w-bits 4 --a-bits 8 --kv-bits 16 \
  --out models/qwen-spinquant \
  --yes

# Các lần sau (venv đã có, bỏ --setup-env):
bash scripts/spinquant_w4a8.sh \
  --model models/Qwen2.5-7B-Instruct \
  --out models/qwen-spinquant \
  --yes
```

> **Quan trọng — môi trường Python riêng:** QwenSpinQuant import
> `is_torch_compile_available` từ `transformers.utils`, API đã bị bỏ ở các bản
> transformers mới. Nếu dùng chung `.venv` của NanoPi (cài
> `transformers>=4.44` không giới hạn) sẽ lỗi:
>
> ```
> ImportError: cannot import name 'is_torch_compile_available' from 'transformers.utils'
> ```
>
> Vì vậy script mặc định dùng venv riêng
> `$QWEN_SPINQUANT_DIR/.venv-spinquant` và **preflight-check** trước khi chạy,
> báo lỗi rõ ràng nếu thiếu/sai phiên bản. Nếu đã có môi trường tương thích,
> trỏ thẳng vào nó:
>
> ```bash
> export SPINQUANT_PYTHON=/path/to/venv/bin/python   # transformers==4.49.0
> ```

Ghi chú:

- Script từ chối chạy nếu thiếu `--yes`.
- Mặc định dùng GPTQ; thêm `--rtn` để dùng RTN.
- Có thể `--skip-rotation` hoặc `--skip-eval` khi cần chạy lại một bước.
- Kết quả là checkpoint đã rotate/quantize; bước tiếp theo vẫn là export GGUF.
- Script tự `cd` vào `$QWEN_SPINQUANT_DIR` nên `--model`/`--out` được đổi sang
  **đường dẫn tuyệt đối** trước khi gọi `optimize_rotation.py` / `ptq.py`
  (tránh lỗi đường dẫn tương đối). Repo ID HF (vd `Qwen/Qwen2.5-7B-Instruct`)
  được giữ nguyên.
- Script gọi trực tiếp `torch.distributed.run optimize_rotation.py` và
  `ptq.py`, **không** dùng các script mẫu `scripts/*.sh` của QwenSpinQuant
  (chúng còn placeholder `your_path`, `your_output_path/`, `your_log_path/`).
- Thư mục mặc định: `$QWEN_SPINQUANT_DIR/output_rotation` (chứa `R.bin`),
  `$QWEN_SPINQUANT_DIR/output` (checkpoint), `$QWEN_SPINQUANT_DIR/logs`.
  Override bằng `SPINQUANT_ROTATION_DIR`, `SPINQUANT_OUTPUT_DIR`,
  `SPINQUANT_LOG_DIR`.
- Số GPU: mặc định lấy từ `nvidia-smi` (tối đa 8); chỉnh bằng `--nproc <n>`
  hoặc `SPINQUANT_NPROC`. Bước eval mặc định 1 GPU (`SPINQUANT_EVAL_NPROC`).
- Số bước rotation: `SPINQUANT_MAX_STEPS` (mặc định 100);
  độ dài ngữ cảnh: `SPINQUANT_MODEL_MAX_LENGTH` (mặc định 2048).

**GPU Blackwell (RTX 50xx / sm_120):** cần PyTorch ≥ 2.7 với CUDA 12.8.
`--setup-env` mặc định cài từ index `cu128`; đổi bằng
`--torch-index-url` / `SPINQUANT_TORCH_INDEX_URL` nếu máy dùng CUDA khác
(vd `https://download.pytorch.org/whl/cu121`). Có thể bỏ bước cài torch bằng
`SPINQUANT_SKIP_TORCH=1` nếu đã cài sẵn torch phù hợp.

**Muốn tái sử dụng torch CUDA đã có (tránh tải lại vài GB):** tạo venv riêng
trước, cài torch vào đó, rồi chạy `--setup-env` — script tự phát hiện torch CUDA
trong venv và bỏ qua bước cài torch:

```bash
python3 -m venv ~/QwenSpinQuant/.venv-spinquant
~/QwenSpinQuant/.venv-spinquant/bin/pip install --upgrade pip
~/QwenSpinQuant/.venv-spinquant/bin/pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130   # khớp CUDA/torch hiện có

bash scripts/spinquant_w4a8.sh --setup-env \
  --model models/Qwen2.5-7B-Instruct \
  --out models/qwen-spinquant \
  --yes
```

> `fast-hadamard-transform` cần biên dịch CUDA extension → cần `nvcc` khớp với
> torch (load module CUDA hoặc thêm `.../cuda/bin` vào `PATH`). Nếu đã cài sẵn
> gói này trong venv, thêm `SPINQUANT_SKIP_HADAMARD=1`.

Nếu chưa được duyệt license, bỏ qua bước này và dùng PTQ GGUF thông thường (`Q4_K_M`).

**Chạy trên Kaggle (tuỳ chọn):**

1. Tạo notebook có GPU T4/L4, bật Internet.
2. `git clone` repo này và `QwenSpinQuant`.
3. Upload/tải model vào `/kaggle/working/models/`.
4. Chạy lần đầu kèm `--setup-env` (tạo venv riêng; Kaggle dùng CUDA 12.x nên
   có thể cần `--torch-index-url https://download.pytorch.org/whl/cu121`):
   `bash scripts/spinquant_w4a8.sh --setup-env --model /kaggle/working/models/Qwen2.5-7B-Instruct --out /kaggle/working/out --yes`.
5. Tải thư mục output về máy, rồi tiếp tục bước export GGUF.

> Venv nằm trong `~/QwenSpinQuant/.venv-spinquant`, không được lưu giữa các
> phiên Kaggle — phải chạy lại `--setup-env` mỗi phiên mới. Đặt venv ra
> `/kaggle/working` bằng `--venv /kaggle/working/.venv-spinquant` nếu muốn tái
> sử dụng trong cùng phiên.

Kaggle có giới hạn thời gian/phiên; checkpoint/model lớn nên lưu output ra
`/kaggle/working` và tải về ngay sau khi xong.

### B.3 Export GGUF và quantize

```bash
# clone/build llama.cpp trên host (nếu chưa có)
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build -j"$(nproc)"

# convert + quantize
LLAMA_CPP_DIR=~/llama.cpp bash scripts/export_gguf.sh \
  --model-dir models/Qwen2.5-7B-Instruct \
  --outdir data/gguf \
  --quant Q4_K_M
```

Hoặc `Q4_0` nếu muốn ưu tiên tốc độ/độ tương thích với rk-llama.cpp:

```bash
LLAMA_CPP_DIR=~/llama.cpp bash scripts/export_gguf.sh \
  --model-dir models/Qwen2.5-7B-Instruct \
  --outdir data/gguf \
  --quant Q4_0
```

Kích thước tham khảo cho 7B:

| Định dạng | Kích thước xấp xỉ |
| ------------ | ---------------------- |
| FP16         | ~14–15 GB             |
| Q8_0         | ~7.5–8 GB             |
| Q4_K_M       | ~4.3–4.7 GB           |
| Q4_0         | ~4.0–4.4 GB           |

Kiểm tra:

```bash
ls -lh data/gguf/
sha256sum data/gguf/*.gguf | tee data/gguf/SHA256SUMS
```

### B.4 Calibration + `.rkllm` W8A8 (chỉ cho model ≤3B)

Chỉ dùng nếu bạn chọn Hướng C (RKLLM W8A8 trên NPU) với model nhỏ:

```bash
python scripts/generate_calibration.py \
  --output data/calibration.json \
  --count 128

python scripts/export_rkllm_w8a8.py \
  --model-dir models/Qwen2.5-1.5B-Instruct \
  --dataset data/calibration.json \
  --quantized-dtype w8a8 \
  --target-platform rk3588 \
  --output models/Qwen2.5-1.5B-Instruct_W8A8_RK3588.rkllm
```

Script sẽ **từ chối** `w4a8_g32`/`w4a16` trên `rk3588`/`rk3588s`. Đây là hành vi
đúng: 7B W8A8 có kích thước ~8.2 GB nên cũng không vừa 8 GB RAM.

### B.5 (Tuỳ chọn) Đánh giá chất lượng trên host

Chạy trên host có GPU để so baseline FP16 với 4-bit:

```bash
# lm-eval: MMLU, ARC-C, HellaSwag, GSM8K, IFEval
python scripts/eval_quality.py \
  --model models/Qwen2.5-7B-Instruct \
  --device cuda \
  --limit 200

# WikiText-2 PPL
bash scripts/eval_ppl.sh \
  --model data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --corpus data/corpus/wikitext-2-raw-val.txt

# VMLU
python scripts/eval_vmlu.py \
  --dataset /path/to/vmlu.json \
  --limit 500
```

Kết quả nằm trong `data/results/`. Đối chiếu ngưỡng trong `README.md`/plan.

---

## 4. Phần C — Đẩy code, dữ liệu, model lên board

### C.1 Chuẩn bị SSH

```bash
# thay IP/user theo board thực tế
ssh-copy-id pi@192.168.1.50
ssh pi@192.168.1.50 "echo OK"
```

Nếu board dùng user khác (`root`, `friendly`, ...), thay `pi` trong mọi lệnh.

### C.2 Đẩy code (không kèm model)

Cách khuyến nghị:

```bash
cd /đường/dẫn/FPT/NanoPi-R6C-LLM

BOARD=pi@192.168.1.50 \
BOARD_DIR=~/NanoPi-R6C-LLM \
bash scripts/deploy_board.sh
```

Mặc định script:

- đẩy code, config, scripts, tests, eval;
- **bỏ qua** `.venv`, `__pycache__`, `.git`, `logs`, `data/gguf`;
- **bỏ qua** `models/` (vì model rất lớn);
- vẫn đẩy `data/results` (đặt `COPY_RESULTS=0` nếu không muốn).

Muốn đẩy luôn model trong lần deploy:

```bash
COPY_MODELS=1 \
BOARD=pi@192.168.1.50 \
BOARD_DIR=~/NanoPi-R6C-LLM \
bash scripts/deploy_board.sh
```

> Khuyến nghị: model 4 GB nên đẩy riêng bằng `rsync` có `--progress/--partial`
> thay vì gộp vào deploy code.

Nếu board không có `rsync`, dùng `scp`:

```bash
scp -r config src scripts eval data systemd run.py Makefile requirements*.txt \
  pi@192.168.1.50:~/NanoPi-R6C-LLM/

scp data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  pi@192.168.1.50:~/NanoPi-R6C-LLM/models/
```

### C.3 Đẩy model và runtime lên board

```bash
# 1) GGUF 4-bit (Hướng A)
rsync -avz --progress --partial \
  data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  pi@192.168.1.50:~/NanoPi-R6C-LLM/models/

# 2) (Nếu dùng RKLLM) thư viện runtime
#    tìm trong SDK rknn-llm đã tải trên host:
find ~/rknn-llm -name 'librkllmrt.so' 2>/dev/null

rsync -avz --progress --partial \
  /path/to/librkllmrt.so \
  pi@192.168.1.50:~/NanoPi-R6C-LLM/models/

# 3) (Nếu dùng rk-llama.cpp) copy thêm thư viện NPU đi kèm
rsync -avz --progress /path/to/rknn/runtime/lib/ \
  pi@192.168.1.50:~/NanoPi-R6C-LLM/lib/
```

Nếu dùng `.rkllm`:

```bash
rsync -avz --progress --partial \
  models/Qwen2.5-1.5B-Instruct_W8A8_RK3588.rkllm \
  pi@192.168.1.50:~/NanoPi-R6C-LLM/models/
```

### C.4 Đẩy dữ liệu (tuỳ nhu cầu)

Chỉ cần nếu chạy PPL/VMLU trực tiếp trên board:

```bash
# corpus PPL
rsync -avz --progress data/corpus/ pi@192.168.1.50:~/NanoPi-R6C-LLM/data/corpus/

# dataset VMLU
rsync -avz --progress eval/vmlu/ pi@192.168.1.50:~/NanoPi-R6C-LLM/eval/vmlu/
```

Khuyến nghị: **đánh giá chất lượng chạy trên host**, board chỉ đo hiệu năng.

### C.5 Kiểm tra checksum sau khi copy

Trên laptop:

```bash
sha256sum data/gguf/*.gguf | tee /tmp/gguf.sha256
scp /tmp/gguf.sha256 pi@192.168.1.50:/tmp/
```

Trên board:

```bash
cd ~/NanoPi-R6C-LLM
sha256sum -c /tmp/gguf.sha256 --ignore-missing
```

### C.6 Cách khác: USB / thẻ nhớ

Nếu Wi-Fi chậm hoặc file quá lớn:

```bash
# laptop
cp data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf /media/USB/
sync

# board
mount /dev/sda1 /mnt
cp /mnt/Qwen2.5-7B-Instruct-Q4_K_M.gguf ~/NanoPi-R6C-LLM/models/
sync
umount /mnt
```

Luôn kiểm tra `sha256sum` sau khi copy.

---

## 5. Phần D — Cài đặt trên board

### D.1 Cài gói hệ thống

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git rsync \
  build-essential cmake
```

Board tối giản có thể chưa có `python3-venv`; nếu `python3 -m venv` lỗi, cài gói trên.

### D.2 Tạo venv và cài dependency runtime

```bash
cd ~/NanoPi-R6C-LLM
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Chỉ cài `requirements.txt` trên board. **Không** cài `requirements-host.txt`
(torch/transformers/lm-eval rất nặng và không cần để chạy GGUF/RKLLM).

### D.3 Cài llama.cpp cho GGUF (Hướng A)

`requirements.txt` không kèm `llama-cpp-python`, nên backend `gguf` sẽ dùng
`llama-cli`. Build llama.cpp trên board:

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON
cmake --build ~/llama.cpp/build -j4

sudo cp ~/llama.cpp/build/bin/llama-cli /usr/local/bin/
sudo cp ~/llama.cpp/build/bin/llama-perplexity /usr/local/bin/  # tuỳ chọn
llama-cli --version
```

Smoke test trực tiếp:

```bash
llama-cli \
  -m ~/NanoPi-R6C-LLM/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -p "Xin chào, bạn là ai?" \
  -n 64 -c 2048 -t 4
```

Nếu dùng rk-llama.cpp (Hướng B), build theo `scripts/rkllama_setup.sh` trên host
hoặc trực tiếp trên board, rồi đặt `llamacpp.binary` trỏ tới binary NPU.

### D.4 Cấu hình `config/config.yaml`

#### Kịch bản A — Qwen2.5-7B 4-bit trên CPU (khuyến nghị)

```yaml
model:
  name: qwen2.5-7b-q4km
  backend: gguf
  path: models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
  max_context: 4096
  max_new_tokens: 512
  temperature: 0.2
  top_p: 0.9
  top_k: 40
  repeat_penalty: 1.1

llamacpp:
  binary: llama-cli
  n_threads: 4
  n_gpu_layers: 0
  n_batch: 256
  flash_attn: false
```

- `n_threads: 4` hợp lý với 4×Cortex-A76; có thể thử 4–8.
- `max_context: 4096` là mức plan hướng tới; giảm 2048 nếu thiếu RAM.
- `max_new_tokens: 512`; benchmark nên dùng 128–256 để tránh quá lâu.

#### Kịch bản B — NPU qua rk-llama.cpp

```yaml
model:
  backend: gguf
  path: models/Qwen2.5-7B-Instruct-Q4_0.gguf
  max_context: 4096

llamacpp:
  binary: /home/pi/rk-llama.cpp/build/bin/llama-cli
  n_threads: 4
  rknpu_hybrid: "Q4_0:W4A4_HADAMARD_LOWRANK"
```

Chạy:

```bash
export RKNPU_DOMAINS=...   # bắt buộc khi model > 4 GB/IOMMU domain
bash scripts/run_board.sh --config config/config.yaml
```

> Hướng B có rủi ro: giới hạn 4 GB/IOMMU domain, có thể kernel panic nếu chạy
> nhiều process song song. Test kỹ trước khi dùng cho demo dài.

#### Kịch bản C — RKLLM W8A8 (chỉ model ≤3B)

```yaml
model:
  name: qwen2.5-1.5b-w8a8
  backend: rkllm
  path: models/Qwen2.5-1.5B-Instruct_W8A8_RK3588.rkllm
  max_context: 4096

rkllm:
  library_path: models/librkllmrt.so
  target_platform: rk3588
  num_npu_core: 3
  timeout_seconds: 180
```

### D.5 Kiểm tra trước khi chạy

```bash
cd ~/NanoPi-R6C-LLM
bash scripts/check_board.sh
```

Script kiểm tra:

- model board/RAM/NPU;
- driver `rknpu`;
- `librkllmrt.so`;
- file `.gguf`/`.rkllm` trong `models/`;
- ADC điện áp USB-C;
- các module Python runtime.

Xem cấu hình đã resolve:

```bash
bash scripts/run_board.sh --check
```

---

## 6. Phần E — Chạy trực tiếp trên NanoPi-R6C

### E.1 Smoke test

```bash
cd ~/NanoPi-R6C-LLM

# mock để chắc chắn code chạy
bash scripts/run_board.sh --demo --force-mock

# một prompt thật với backend trong config
bash scripts/run_board.sh --prompt "Tóm tắt vai trò của NPU trong thiết bị biên."

# demo vài prompt thật
bash scripts/run_board.sh --demo
```

`run_board.sh` tự đặt:

- `LD_LIBRARY_PATH` gồm `models/` và `lib/`;
- `RKLLM_LOG_LEVEL=1`;
- `RKNPU_DOMAINS` nếu đã export;
- `ulimit -n 65536` (nếu được phép);
- CPU governor `performance` (nếu có quyền).

### E.2 Benchmark hiệu năng

```bash
bash scripts/run_board.sh --bench \
  --prompt-tokens 128,512,1024,2048 \
  --gen-tokens 128 \
  --runs 3 \
  --warmup 1
```

Kết quả JSON ghi vào `data/results/`, gồm:

- model size, load time;
- TTFT, prefill tok/s, decode tok/s;
- peak RAM;
- từng run chi tiết.

Giữ nguyên nhiệt độ/phòng và nguồn điện khi so sánh baseline với 4-bit.

### E.3 Chạy API trên LAN

```bash
bash scripts/run_board.sh --config config/config.yaml
```

Mở từ laptop:

```text
http://<IP-board>:8080/dashboard
http://<IP-board>:8080/health
```

Kiểm tra port/firewall trên board:

```bash
ss -ltnp | grep 8080
```

Nếu chỉ muốn nghe nội bộ:

```bash
bash scripts/run_board.sh --host 127.0.0.1 --port 8080
```

Bật API key trong `config/config.yaml`:

```yaml
api:
  host: 0.0.0.0
  port: 8080
  api_key: "doi-thanh-secret"
```

Khi đó client phải gửi `Authorization: Bearer doi-thanh-secret`.

### E.4 Chạy bằng systemd

```bash
sudo cp systemd/nano-llm.service /etc/systemd/system/
sudo nano /etc/systemd/system/nano-llm.service
```

Sửa `User`, `WorkingDirectory`, `Environment=LD_LIBRARY_PATH` và `ExecStart`
cho đúng user/đường dẫn thực tế. Sau đó:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nano-llm
sudo systemctl status nano-llm
journalctl -u nano-llm -f
```

Service đã đặt `LimitNOFILE=65536` và `MemoryMax=7G` để chừa RAM cho OS.

Khởi động lại sau khi đổi config:

```bash
sudo systemctl restart nano-llm
```

### E.5 Đo công suất

Cài `pyserial` trên board nếu dùng power meter serial:

```bash
.venv/bin/pip install pyserial
sudo usermod -aG dialout "$USER"
# logout/login lại để nhận group
```

Chạy song song với benchmark (terminal khác):

```bash
.venv/bin/python scripts/power_monitor.py \
  --meter serial \
  --serial-port /dev/ttyUSB0 \
  --baud 115200 \
  --duration 300 \
  --interval 1 \
  --out data/results/power_qwen4bit.csv
```

Các chế độ khác:

```bash
# smoke test không cần phần cứng
.venv/bin/python scripts/power_monitor.py --meter synthetic --duration 10

# đọc từ file log đã ghi
.venv/bin/python scripts/power_monitor.py --meter file --input /tmp/power.log --duration 0

# ADC nội bộ: chỉ có điện áp, KHÔNG ra watt
.venv/bin/python scripts/power_monitor.py --meter adc --duration 60
```

Kết quả: CSV + `<tên>.summary.json` trong `data/results/`.

> ADC kênh 2 trên R6C chỉ đo điện áp USB-C. Muốn báo cáo “Power consumption (W)”
> đúng biên bản họp, cần USB-C power meter hoặc shunt đo dòng.

### E.6 (Tuỳ chọn) VMLU/PPL trên board

Chỉ nên chạy với `--limit` nhỏ vì 7B trên CPU rất chậm:

```bash
bash scripts/run_board.sh --backend gguf
.venv/bin/python scripts/eval_vmlu.py \
  --dataset eval/vmlu/vmlu.json \
  --limit 50 \
  --max-new-tokens 8

bash scripts/eval_ppl.sh \
  --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --chunks 16
```

### E.7 Copy kết quả về laptop

```bash
mkdir -p data/results/board
rsync -avz --progress \
  pi@192.168.1.50:~/NanoPi-R6C-LLM/data/results/ \
  data/results/board/
```

Nên đặt tên file kết quả có hậu tố backend/định dạng, ví dụ:

```text
data/results/
├── bench_gguf_qwen2.5-7b-q4km_20261112_....json
├── power_qwen4bit.csv
└── power_qwen4bit.summary.json
```

---

## 7. Phần F — Troubleshooting

| Triệu chứng                        | Nguyên nhân thường gặp            | Cách xử lý                                                                                  |
| ------------------------------------ | -------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: nano_llm`    | Chạy sai entrypoint/cwd               | Chạy `python run.py ...` từ repo root hoặc `bash scripts/run_board.sh`                  |
| `llama-cli not found`              | Chưa build/cài llama.cpp             | Build trên board, copy vào `/usr/local/bin`, hoặc đặt `llamacpp.binary` tuyệt đối  |
| `GGUF model not found`             | `model.path` rỗng/sai               | Sửa `config/config.yaml`, đường dẫn tính từ repo root                                 |
| `librkllmrt.so not found`          | Chưa copy runtime                     | Copy vào `models/` hoặc `lib/`, hoặc set `rkllm.library_path`                         |
| `rkllm_init failed`                | Sai phiên bản ABI/toolkit            | Đối chiếu `rkllm_bindings.py` với `rkllm.h`, dùng đúng release                      |
| Tiến trình bị kill/OOM            | 7B W8A8 ~8.2 GB; context/seq quá lớn | Dùng GGUF 4-bit, giảm `max_context`, đóng dịch vụ khác, tăng swap chỉ để cứu hộ |
| Lỗi IOMMU/domain khi chạy NPU      | Model > 4 GB/domain                    | Export `RKNPU_DOMAINS`, không chạy song song nhiều process                                |
| `W4A8 not supported`               | RK3588S không có kernel W4A8         | Dùng W8A8 (RKLLM), W4A4 (rk-llama.cpp), hoặc GGUF CPU                                        |
| `Too many open files`              | Giới hạn fd thấp                    | `ulimit -n 65536`, systemd đã đặt `LimitNOFILE`                                        |
| `Permission denied /dev/ttyUSB0`   | User chưa thuộc `dialout`          | `sudo usermod -aG dialout $USER`, login lại                                                 |
| API không truy cập được từ LAN | Bind localhost/firewall                | `api.host: 0.0.0.0`, mở port 8080, cùng subnet                                             |
| Benchmark rất chậm                 | Governor powersave/thermal             | Dùng `run_board.sh` (governor performance), kiểm tra nhiệt độ                           |
| Nhiệt độ cao                      | Tản nhiệt kém                       | Thêm quạt/tản, hạ `n_threads`, nghỉ giữa các run                                      |
| rsync đứt giữa chừng             | Wi-Fi chập chờn                      | Dùng `--partial --append-verify`, ưu tiên dây LAN                                        |
| File model hỏng                     | Copy lỗi                              | So `sha256sum` hai phía                                                                     |
| Kết quả mock toàn 0               | Đang dùng `--force-mock`           | Bỏ `--force-mock`, kiểm tra `model.backend`                                              |

Kiểm tra nhiệt độ nhanh:

```bash
for z in /sys/class/thermal/thermal_zone*; do
  echo "$z: $(cat "$z/type") $(cat "$z/temp")"
done
```

---

## 8. Checklist trước khi báo cáo/demo

- [ ] `python run.py --check` trên laptop không lỗi.
- [ ] `python -m pytest -q` pass.
- [ ] Model 4-bit có checksum khớp giữa host và board.
- [ ] `bash scripts/check_board.sh` không có WARN nghiêm trọng.
- [ ] Chạy được một prompt thật trên board.
- [ ] Có JSON benchmark trong `data/results/`.
- [ ] Có CSV + summary công suất (nếu báo cáo W).
- [ ] Ghi rõ backend/định dạng/context/threads khi so sánh.
- [ ] Đã đọc giới hạn license của SpinQuant và model gốc.

---

## 9. Cheat sheet

### Laptop

```bash
# setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# chạy
python run.py --check
python run.py --demo --force-mock
python run.py --force-mock
python -m pytest -q

# model
python scripts/download_model.py --model qwen2.5-7b --out models
LLAMA_CPP_DIR=~/llama.cpp bash scripts/export_gguf.sh \
  --model-dir models/Qwen2.5-7B-Instruct --quant Q4_K_M

# đẩy code
BOARD=pi@192.168.1.50 bash scripts/deploy_board.sh

# đẩy model
rsync -avz --progress --partial \
  data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  pi@192.168.1.50:~/NanoPi-R6C-LLM/models/
```

### Board

```bash
cd ~/NanoPi-R6C-LLM
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

bash scripts/check_board.sh
bash scripts/run_board.sh --demo --force-mock
bash scripts/run_board.sh --prompt "Xin chào"
bash scripts/run_board.sh --bench --prompt-tokens 128,512,1024,2048 --gen-tokens 128
bash scripts/run_board.sh --config config/config.yaml
```

### Lấy kết quả về

```bash
rsync -avz --progress \
  pi@192.168.1.50:~/NanoPi-R6C-LLM/data/results/ \
  data/results/board/
```

---

## 10. Ghi chú quan trọng

1. **Lượng tử hoá chạy trên host, board chỉ inference.** Đừng chạy
   `rkllm-toolkit`/GPTQ trên R6C.
2. **7B W8A8 không vừa 8 GB RAM.** Với 7B, dùng GGUF 4-bit.
3. **Không có W4A8 native trên RK3588S.** Các lệnh/config W4A8 chỉ đúng cho
   RK3562 hoặc môi trường GPU khác.
4. **SpinQuant là CC-BY-NC 4.0.** Cần legal duyệt nếu dùng ngoài mục đích
   nghiên cứu/phi thương mại.
5. **Luôn đo công suất bằng power meter.** ADC nội bộ chỉ cho điện áp.
6. **So sánh công bằng:** cùng prompt set, cùng `max_new_tokens`, cùng
   context, cùng điều kiện nhiệt độ/nguồn.
