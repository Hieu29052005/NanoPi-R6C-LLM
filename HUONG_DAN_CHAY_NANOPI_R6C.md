# Hướng dẫn chạy toàn bộ project: RTX 5080 → NanoPi-R6C 8GB

Tài liệu này là **hướng dẫn duy nhất, chạy từ đầu đến cuối**:

```
[Host x86_64 / 2×RTX 5080]
  Phase 0  Hotfix trước khi chạy
  Phase 1  CUDA 13 + uv + torch + fast-hadamard-transform
  Phase 2  Tải Qwen2.5-7B-Instruct
  Phase 3  SpinQuant: tối ưu rotation (2 GPU) + PTQ/PPL (1 GPU)
  Phase 4  Đánh giá chất lượng baseline FP16 vs quantized (host)
  Phase 5  Export GGUF Q4_K_M / Q4_0 + SHA256
        │  rsync / ssh
        ▼
[NanoPi-R6C / RK3588S / 8GB]
  Phase 6  Đẩy code + model lên board
  Phase 7  Cài runtime tối thiểu + build llama.cpp + check_board.sh
  Phase 8  Smoke test → benchmark W1/W2/W3/W4 → đo công suất
  Phase 9  Copy kết quả về host → báo cáo so sánh
```

Tài liệu liên quan:

- `KE_HOACH_QUANTIZE_QWEN7B_4BIT_NANOPI_R6C.md` — kế hoạch, metrics, decision gate, rủi ro.
- `INSTRUCTION_SPINQUANT_5080.md` — chi tiết lỗi CUDA/Blackwell khi cài SpinQuant.
- `README.md` — tóm tắt kiến trúc và license.

---

## 0. Đọc trước: quyết định kiến trúc và giới hạn cứng

### 0.1 Mặc định = Hướng A

| Việc                              | Chạy ở đâu               | Artifact                                              | Mục đích                                                                    |
| ---------------------------------- | ---------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------ |
| SpinQuant + PTQ **W4A8**     | Host 2×RTX 5080             | `rotation/R.bin`, log PPL, `output_ptq/qmodel.pt` | **Nghiên cứu chất lượng** (đúng yêu cầu PTQ + SpinQuant + W4A8) |
| Baseline FP16 + so sánh 7 metrics | Host GPU                     | `data/results/*.json`                               | Quality gate                                                                   |
| Deploy 4-bit                       | **CPU/RAM NanoPi-R6C** | `data/gguf/*-Q4_K_M.gguf` (hoặc `Q4_0`)          | Performance gate                                                               |
| NPU                                | Chỉ thử nghiệm            | `rk-llama.cpp` W4A4 hoặc `.rkllm` W8A8 (≤3B)    | Không phải đường chính                                                   |

**Lý do:** RK3588S **không có pipeline W4A8 native**. RKLLM chỉ nhận `w8a8`, `w8a8_g128/256/512`; `w4a16` thuộc RK3576, `w4a8_g32` chỉ trên RK3562. 7B W8A8 nặng ≈8.2 GB → OOM trên 8 GB. NPU còn có giới hạn **~4 GB/IOMMU domain**. Do đó W4A8 là kết quả đo trên host, không phải artifact deploy lên NPU.

### 0.2 Bảng khả thi (tóm tắt)

| Hướng         | Định dạng                   | Nơi chạy    | Kết luận                                                       |
| --------------- | ------------------------------ | ------------- | ---------------------------------------------------------------- |
| A (mặc định) | GGUF Q4_K_M/Q4_0               | CPU + RAM R6C | ✅ Khả thi, ~4.3–4.7 GB                                        |
| B               | GGUF Q4_0 → rk-llama.cpp W4A4 | NPU           | ⚠️ Chất lượng rủi ro, >4 GB/domain, cần `RKNPU_DOMAINS` |
| C               | `.rkllm` W8A8                | NPU           | ❌ 7B ≈8.2 GB → OOM; chỉ ≤3B                                 |
| D               | W4A8 native                    | NPU           | ❌ Không tồn tại trên RK3588S                                |

### 0.3 License — phải duyệt trước khi chạy SpinQuant

- Qwen2.5-7B-Instruct: **Apache-2.0**.
- **QwenSpinQuant / SpinQuant: CC-BY-NC 4.0 — phi thương mại.** Script `spinquant_w4a8.sh` từ chối chạy nếu thiếu `--yes`. Chỉ dùng cho R&D nội bộ khi chưa có ý kiến legal.
- VMLU / dataset bên thứ ba: kiểm tra điều khoản riêng.

### 0.4 Biến dùng xuyên suốt tài liệu

Mở terminal trên **host** và dán một lần:

```bash
export PROJECT_DIR=/mnt/hdd/dev/edgeAI/tungbm/NanoPi-R6C-LLM
export QWEN_SPINQUANT_DIR="$PROJECT_DIR/QwenSpinQuant"
export MODEL_DIR="$PROJECT_DIR/models/Qwen2.5-7B-Instruct"
export BOARD=pi@192.168.1.50          # thay bằng IP/user board thật
export BOARD_DIR=~/NanoPi-R6C-LLM
cd "$PROJECT_DIR"
```

> `scripts/spinquant_w4a8.sh` hiện **không `export`** `QWEN_SPINQUANT_DIR` (xem Phase 0.1). Vì vậy phải export ở shell như trên, nếu không script con sẽ nhận đường dẫn rỗng.

---

## Phase 0 — Hotfix bắt buộc trước khi chạy

Đây là các lỗi đã được kiểm chứng bằng thực nghiệm trong repo hiện tại. **Áp hết trước Phase 1**, nếu không kết quả benchmark/eval có thể sai mà không báo lỗi.

### 0.1 `spinquant_w4a8.sh` không export `QWEN_SPINQUANT_DIR`

`scripts/spinquant_w4a8.sh:16` gán biến mà không `export`. Script con `10_optimize_rotation.sh` và `2_eval_ptq.sh` dùng `$QWEN_SPINQUANT_DIR` → nhận chuỗi rỗng nếu bạn chưa export ở shell.

Cách nhanh (không sửa code): luôn export như mục 0.4.

Cách sửa triệt để:

```bash
cd "$PROJECT_DIR"
python3 - <<'PY'
from pathlib import Path
p = Path("scripts/spinquant_w4a8.sh")
s = p.read_text()
s = s.replace(
    'QWEN_SPINQUANT_DIR="${QWEN_SPINQUANT_DIR:-$PROJECT_DIR/QwenSpinQuant}"',
    'export QWEN_SPINQUANT_DIR="${QWEN_SPINQUANT_DIR:-$PROJECT_DIR/QwenSpinQuant}"',
)
p.write_text(s)
PY
grep -n 'QWEN_SPINQUANT_DIR=' scripts/spinquant_w4a8.sh
# kỳ vọng: export QWEN_SPINQUANT_DIR=...
```

### 0.2 `QwenSpinQuant/scripts/2_eval_ptq.sh` hardcode W4A4 + đường dẫn tuyệt đối

File hiện tại bỏ qua tham số truyền vào, luôn chạy

```
--input_model /mnt/hdd/.../models/Qwen2.5-7B-Instruct --w_bits 4 --a_bits 4
```

Nên `--a-bits 8` từ wrapper **không có tác dụng**. Thay toàn bộ nội dung bằng bản nhận positional args:

```bash
cat > "$QWEN_SPINQUANT_DIR/scripts/2_eval_ptq.sh" <<'EOF'
#!/usr/bin/env bash
# Usage: bash scripts/2_eval_ptq.sh <model> <w_bits> <a_bits> <kv_bits>
# QWEN_SPINQUANT_DIR được export từ shell hoặc suy từ vị trí script.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${1:-${MODEL_DIR:-/mnt/hdd/dev/edgeAI/tungbm/NanoPi-R6C-LLM/models/Qwen2.5-7B-Instruct}}"
W_BITS="${2:-4}"
A_BITS="${3:-8}"
KV_BITS="${4:-16}"
QDIR="${QWEN_SPINQUANT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG="${PTQ_LOG:-$QDIR/ptq_run.log}"

cd "$QDIR"
torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
  --input_model "$MODEL" \
  --output_dir "$QDIR/output_ptq" \
  --do_train False --do_eval True \
  --per_device_eval_batch_size 1 \
  --model_max_length 2048 \
  --fp16 False --bf16 True \
  --save_safetensors False \
  --w_bits "$W_BITS" --a_bits "$A_BITS" \
  --k_bits "$KV_BITS" --v_bits "$KV_BITS" \
  --w_clip --a_asym --k_asym --v_asym \
  --k_groupsize 128 --v_groupsize 128 \
  --rotate \
  --optimized_rotation_path "$QDIR/rotation/R.bin" \
  2>&1 | tee "$LOG"
EOF
chmod +x "$QWEN_SPINQUANT_DIR/scripts/2_eval_ptq.sh"
```

> Chạy rotation riêng (`--skip-eval`), rồi gọi script PTQ ở trên với cặp bit mong muốn. Xem Phase 3.2.

### 0.3 Ghim patch local của QwenSpinQuant (lưu ý: các thay đổi này chưa commit)

Fork đang ở commit `e6b0840` nhưng có **6 file đã sửa chưa commit**:

```
optimize_rotation.py            # pad_mlp(model, 19968)
ptq.py                          # pad_mlp + lưu qmodel.pt, bỏ model.cuda()
scripts/10_optimize_rotation.sh
scripts/2_eval_ptq.sh
train_utils/fsdp_trainer.py
utils/hadamard_utils.py         # sửa bias .to(hadK.device) cho phép rotate v_proj/o_proj
```

Hai patch quan trọng nhất:

- `pad_mlp(19968)` ở **cả** `optimize_rotation.py` và `ptq.py`: Qwen2.5-7B có `intermediate_size=18944 = 37×512`, hệ số 37 không được `get_hadK` hỗ trợ → phải pad lên `19968 = 156×128`. Hai bước phải pad **cùng kích thước**, nếu không rotation không khớp model.
- `utils/hadamard_utils.py`: bias của `v_proj` đang nằm CPU trong khi `hadK` nằm CUDA → lỗi `Expected all tensors to be on the same device`. Dòng đã sửa: `b = b.to(hadK.device).to(torch.float64) @ hadK`.

Kiểm tra và **lưu lại patch** trước khi chạy (nếu clone lại máy khác mà quên các sửa này sẽ lỗi ngay):

```bash
cd "$QWEN_SPINQUANT_DIR"
git -c safe.directory='*' status --short
git -c safe.directory='*' diff --stat

# khuyến nghị: commit hoặc xuất patch để mang theo
git -c safe.directory='*' add -A
git -c safe.directory='*' commit -m "local: pad_mlp 19968, bias device fix, W4A8 ptq args"

# hoặc chỉ xuất patch:
git -c safe.directory='*' diff > "$PROJECT_DIR/QwenSpinQuant_local.patch"
```

Kiểm tra Hadamard trước khi chạy rotation:

```bash
cd "$QWEN_SPINQUANT_DIR"
. "$PROJECT_DIR/.venv/bin/activate"
python - <<'PY'
from utils.hadamard_utils import get_hadK
for n in (3584, 128, 19968):
    h, K = get_hadK(n)
    print(n, "K =", K)
PY
# Kỳ vọng: 3584 K=28, 128 K=1, 19968 K=156
grep -n "pad_mlp" optimize_rotation.py ptq.py   # phải thấy định nghĩa + lời gọi ở cả 2 file
```

### 0.4 Không cài `requirements.txt` trên board (sai kiến trúc)

`requirements.txt` là file **host/CUDA 13**: `torch==2.14.1`, `cuda-toolkit`, `nvidia-*`, `fast-hadamard-transform`, `triton`, `lm-eval`… Cài bản này trên aarch64 board sẽ lỗi/không cần thiết. Tạo file riêng cho board:

```bash
cat > "$PROJECT_DIR/requirements-board.txt" <<'EOF'
# Chỉ các dependency runtime cần thiết trên NanoPi-R6C (aarch64).
pyyaml==6.0.3
fastapi==0.162.5
uvicorn==0.54.0
pydantic==2.14.3
psutil==7.2.2
# pytest/httpx chỉ cần nếu chạy test trên board
pytest==9.2.0
httpx==0.28.1
# Bắt buộc nếu dùng power meter serial
pyserial>=3.5
EOF
```

> `scripts/deploy_board.sh` dòng 39 có `echo` gợi ý `pip install -r requirements.txt` trên board — **bỏ qua dòng đó**, dùng `requirements-board.txt` ở Phase 7.2.

### 0.5 `parse_llama_perf` đọc nhầm TTFT/decode (regex)

`src/nano_llm/models/llamacpp_backend.py:35` dùng `\beval time...`, khớp luôn dòng `prompt eval time = ...` → decode tok/s bị tính từ dòng prompt. Đã reproduce: output đúng phải là `eval_ms=250, eval_tokens=5, eval_tok_s=20`, nhưng hàm trả `100, 10, 100`.

Sửa bằng anchor đầu dòng chế độ multiline:

```bash
cd "$PROJECT_DIR"
python3 - <<'PY'
from pathlib import Path
p = Path("src/nano_llm/models/llamacpp_backend.py")
s = p.read_text()
s = s.replace(
    '"eval_ms": r"\\beval time\\s*=\\s*([0-9.]+)\\s*ms",',
    '"eval_ms": r"(?m)^eval time\\s*=\\s*([0-9.]+)\\s*ms",',
)
s = s.replace(
    '"eval_tokens": r"\\beval time\\s*=.*?/\\s*(\\d+)\\s*(?:runs|tokens)",',
    '"eval_tokens": r"(?m)^eval time\\s*=.*?/\\s*(\\d+)\\s*(?:runs|tokens)",',
)
s = s.replace(
    '"eval_tok_s": r"\\beval time\\s*=.*?([0-9.]+)\\s*tokens per second",',
    '"eval_tok_s": r"(?m)^eval time\\s*=.*?([0-9.]+)\\s*tokens per second",',
)
p.write_text(s)
PY
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "src")
from nano_llm.models.llamacpp_backend import parse_llama_perf
s = """prompt eval time =     100.00 ms /    10 tokens (   10.00 tokens per second)
eval time =     250.00 ms /     5 runs (   20.00 tokens per second)"""
print(parse_llama_perf(s))
# kỳ vọng có eval_ms=250.0, eval_tokens=5, eval_tok_s=20.0
PY
```

### 0.6 Fallback `mock` âm thầm + JSON thiếu khóa `output`

- `src/nano_llm/models/slm.py:131` `build_slm()` bắt mọi exception rồi trả `MockSLM` với `model_name = "mock:<tên>"`. Nếu backend lỗi, benchmark vẫn chạy và ra số giả (`extra.simulated=true`). **Luôn kiểm tra `backend` trong JSON** trước khi tin số liệu.
- `src/nano_llm/bench/runner.py:135` gọi `write_json()` rồi gán `report["output"]`, nên file JSON được ghi **trước** khi khóa `output` tồn tại. Sửa:

```bash
cd "$PROJECT_DIR"
python3 - <<'PY'
from pathlib import Path
p = Path("src/nano_llm/bench/runner.py")
s = p.read_text()
s = s.replace(
    '    report["output"] = write_json(target, report)\n',
    '    report["output"] = target\n    write_json(target, report)\n',
)
p.write_text(s)
PY
```

Kiểm tra sau hotfix (trên host):

```bash
cd "$PROJECT_DIR"
.venv/bin/python run.py --check
.venv/bin/python -m pytest -q     # kỳ vọng: 30 passed
```

---

## Phase 1 — Môi trường host RTX 5080 (CUDA 13 + uv)

### 1.1 Cấu hình đã kiểm chứng

| Thành phần      | Giá trị                                  |
| ----------------- | ------------------------------------------ |
| GPU               | 2× RTX 5080 16 GB (Blackwell, `sm_120`) |
| RAM host          | 62 GB (+31 GB swap)                        |
| OS                | Ubuntu 22.04                               |
| Driver            | 580.x (CUDA runtime 13.0)                  |
| Python            | 3.10                                       |
| PyTorch           | 2.14.x **cu130**                     |
| Quản lý package | `uv`                                     |

### 1.2 CUDA toolkit 13.0

RTX 5080 (`sm_120`) cần CUDA ≥12.8. PyTorch bản `cu130` bắt buộc `nvcc` **cùng major 13**. Nếu `nvcc` là 12.2, bước build `fast-hadamard-transform` báo `The detected CUDA version (12.2) mismatches ... (13.0)`.

```bash
nvidia-smi                         # Driver >= 580, "CUDA Version: 13.x"
nvcc --version                     # phải "release 13.0,..."
```

Cài toolkit 13.0 song song bản cũ (**không** cài gói `cuda`/`cuda-13-0`/`cuda-drivers` vì sẽ kéo driver):

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
apt-cache policy cuda-toolkit-13-0
sudo apt install cuda-toolkit-13-0
ls /usr/local | grep cuda          # có cuda-13.0
```

Nếu không có sudo:

```bash
sh cuda_13.0.0_*_linux.run --silent --toolkit \
  --toolkitpath="$HOME/cuda-13.0" --no-drm --no-man-page
```

Trỏ môi trường (làm mỗi lần mở terminal mới):

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version
```

### 1.3 venv + dependencies

```bash
cd "$PROJECT_DIR"
git clone https://github.com/shijiew/QwenSpinQuant "$QWEN_SPINQUANT_DIR" 2>/dev/null || true

uv venv .venv --python 3.10
source .venv/bin/activate
export UV_LINK_MODE=copy
export MAX_JOBS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# torch phải cài trước fast-hadamard-transform
uv pip install -r "$PROJECT_DIR/requirements.txt" || \
  uv pip install -r "$PROJECT_DIR/requirements-host.txt"

# build FHT riêng: --no-build-isolation, đúng ref, chỉ build cho sm_120
export TORCH_CUDA_ARCH_LIST="12.0"
uv pip install --no-build-isolation \
  "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@e7706faf8d1c3b9f241e36860640ad1dac644ede"

python - <<'PY'
import torch, fast_hadamard_transform
print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

> wrapper `spinquant_w4a8.sh` tự build FHT nếu chưa có, tự kiểm tra major CUDA của `nvcc` và `torch`, tự bỏ FHT khỏi lần cài `requirements` đầu tiên. Có thể bỏ qua bước build tay ở trên nếu dùng wrapper.

### 1.4 Giảm RAM host khi chạy 2 rank

Quá trình có thể ăn ~40–60 GB RAM host. Đảm bảo swap:

```bash
free -h
# nếu cần (có sudo):
sudo fallocate -l 32G /swapfile && sudo chmod 600 /swapfile \
  && sudo mkswap /swapfile && sudo swapon /swapfile
```

---

## Phase 2 — Tải Qwen2.5-7B-Instruct

```bash
cd "$PROJECT_DIR"
source .venv/bin/activate

python scripts/download_model.py --list
python scripts/download_model.py --model qwen2.5-7b --out models

ls -lh "$MODEL_DIR/config.json" "$MODEL_DIR"/*.safetensors | head
du -sh "$MODEL_DIR"          # ~15 GB
```

Các ứng viên khác (nếu đổi model): `llama-3.1-8b` (gated, cần `HF_TOKEN`), `deepseek-r1-distill-7b`. Với 7B dùng key `qwen2.5-7b` như mặc định.

> Model gated: `export HF_TOKEN=hf_xxx` trước khi gọi script.

---

## Phase 3 — SpinQuant + PTQ trên RTX 5080

### 3.1 Hai profile cần phân biệt

| Profile                                   | Rotation (`10_optimize_rotation.sh`) | PTQ (`2_eval_ptq.sh`) | Ý nghĩa                                                                                                                                                            |
| ----------------------------------------- | -------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **W4A8 đúng chuẩn** (chạy mới) | `16 8 16` (W16A8KV16)                | `4 8 16` (W4A8KV16)   | Theo README fork: khi PTQ dùng GPTQ, tối ưu rotation trên mạng **chỉ quant activation** (`w_bits=16`). Profile này là số liệu W4A8 để báo cáo. |
| **W4A4 tham chiếu** (đã chạy)   | `4 4 16`                             | `4 4 16`              | Kết quả có sẵn: `wiki2 ppl is: 8.532203674316406`. Dùng làm mốc so sánh, không phải W4A8.                                                                |

GPTQ là mặc định; `--rtn` chuyển sang RTN (nhanh hơn, chất lượng thấp hơn).

### 3.2 Chạy rotation (2 GPU, không dùng `--w_clip`)

```bash
cd "$PROJECT_DIR"
source .venv/bin/activate
export QWEN_SPINQUANT_DIR="$PROJECT_DIR/QwenSpinQuant"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NPROC=2                       # = số GPU thật; KHÔNG để 8
export MAX_LEN=512
export MAX_STEPS=100
export USE_W_CLIP=0                  # --w_clip gây OOM ở bước train rotation với 7B
cd "$QWEN_SPINQUANT_DIR"

# Profile W4A8: rotation W16A8KV16
bash scripts/10_optimize_rotation.sh "$MODEL_DIR" 16 8 16 \
  2>&1 | tee "$QWEN_SPINQUANT_DIR/logs/rotation_w16a8.log"

# Hoặc dùng wrapper từ repo chính (tự setup/build FHT, tự log):
cd "$PROJECT_DIR"
bash scripts/spinquant_w4a8.sh \
  --model "$MODEL_DIR" \
  --w-bits 16 --a-bits 8 --kv-bits 16 \
  --skip-eval \
  --yes
```

Kết quả: `$QWEN_SPINQUANT_DIR/rotation/R.bin` (~51 MB). Kiểm tra:

```bash
ls -lh "$QWEN_SPINQUANT_DIR/rotation/R.bin"
```

Lưu ý đã kiểm chứng:

- `--input_model` bắt buộc; nếu thiếu sẽ báo `test-input is not a local folder`.
- FSDP là bắt buộc với 7B (bf16 ~15 GB > 16 GB/card). Script dùng `--fsdp "full_shard auto_wrap offload"` mặc định; nếu lỗi lẫn device, thử bỏ `offload` (`FSDP_OPTS="full_shard auto_wrap"`) hoặc ngược lại.
- Không tăng `--nproc_per_node` quá số GPU: chạy 8 rank trên máy 2 GPU làm OOM RAM host, process bị kill `exitcode: -9`.
- Nếu vẫn OOM: `MAX_LEN=256`.
- Lệnh `torchrun` đầy đủ nếu muốn tự kiểm soát: xem `INSTRUCTION_SPINQUANT_5080.md` mục 4.

### 3.3 Chạy PTQ + đo PPL (1 GPU)

```bash
cd "$QWEN_SPINQUANT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export QWEN_SPINQUANT_DIR="$PROJECT_DIR/QwenSpinQuant"

# Profile W4A8: PTQ 4/8/16 (chỉ chạy 1 rank, KHÔNG FSDP)
# script đã sửa ở 0.2 tự ghi ptq_run.log; tee để lưu thêm bản tên rõ ràng
bash scripts/2_eval_ptq.sh "$MODEL_DIR" 4 8 16 2>&1 | tee ptq_run_w4a8.log
```

Profile W4A4 tham chiếu (nếu cần tái lập mốc 8.53):

```bash
cd "$QWEN_SPINQUANT_DIR"
MODEL_DIR="$MODEL_DIR" bash scripts/2_eval_ptq.sh "$MODEL_DIR" 4 4 16 \
  2>&1 | tee ptq_run_w4a4.log
grep -i "wiki2 ppl" ptq_run_w4a4.log
# kỳ vọng: wiki2 ppl is: 8.532203674316406
```

Giải thích các flag:

- `--input_model` là **model gốc**, không phải `output/checkpoint-100`; rotation được áp qua `--rotate --optimized_rotation_path .../R.bin`.
- `--do_train False --do_eval True`, `--per_device_eval_batch_size 1`, `--model_max_length 2048`.
- `--fp16 False --bf16 True`, `--save_safetensors False`.
- `--w_clip --a_asym --k_asym --v_asym`, group size 128 cho K/V (đúng cấu hình lúc train rotation).
- **Không** truyền `--fsdp`, `--gradient_checkpointing`, `--max_steps` ở bước PTQ.
- Cảnh báo `Token indices sequence length is longer...` và `barrier()` là vô hại.

Artifact sau PTQ:

| File                     | Kích thước | Dùng được không                                                                                |
| ------------------------ | ------------- | --------------------------------------------------------------------------------------------------- |
| `rotation/R.bin`       | ~51 MB        | Ma trận xoay, đầu vào PTQ                                                                       |
| `output_ptq/qmodel.pt` | ~15 GB        | **Fake-quant checkpoint, KHÔNG phải file deploy** (không convert thẳng sang GGUF được) |
| `ptq_run*.log`         | nhỏ          | Chứa PPL, bằng chứng quality                                                                     |

> Để deploy thật bằng GGUF, Phase 5 convert từ checkpoint HF gốc. Việc fuse R1/R2 vào weight rồi lưu lại HF checkout **chưa được implement** trong repo này — ghi rõ điều đó trong báo cáo.

### 3.4 Tái lập hoàn toàn nếu cần lệnh torchrun thủ công

```bash
cd "$QWEN_SPINQUANT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Rotation (2 GPU)
torchrun --nnodes=1 --nproc_per_node=2 optimize_rotation.py \
  --input_model "$MODEL_DIR" \
  --output_rotation_path "$QWEN_SPINQUANT_DIR/rotation" \
  --output_dir "$QWEN_SPINQUANT_DIR/output" \
  --logging_dir "$QWEN_SPINQUANT_DIR/logs" \
  --model_max_length 512 --fp16 False --bf16 True \
  --log_on_each_node False --per_device_train_batch_size 1 \
  --logging_steps 1 --learning_rate 1.5 --weight_decay 0. \
  --lr_scheduler_type cosine --gradient_checkpointing True \
  --save_safetensors False --max_steps 100 \
  --w_bits 16 --a_bits 8 --k_bits 16 --v_bits 16 \
  --a_asym --k_asym --v_asym \
  --k_groupsize 128 --v_groupsize 128 \
  --fsdp "full_shard auto_wrap offload" \
  --fsdp_transformer_layer_cls_to_wrap "Qwen2DecoderLayer"

# PTQ (1 GPU) — chính là nội dung 2_eval_ptq.sh đã sửa ở 0.2
torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
  --input_model "$MODEL_DIR" \
  --output_dir "$QWEN_SPINQUANT_DIR/output_ptq" \
  --do_train False --do_eval True --per_device_eval_batch_size 1 \
  --model_max_length 2048 --fp16 False --bf16 True \
  --save_safetensors False \
  --w_bits 4 --a_bits 8 --k_bits 16 --v_bits 16 \
  --w_clip --a_asym --k_asym --v_asym \
  --k_groupsize 128 --v_groupsize 128 \
  --rotate --optimized_rotation_path "$QWEN_SPINQUANT_DIR/rotation/R.bin" \
  2>&1 | tee ptq_run_w4a8.log
```

---

## Phase 4 — Đánh giá chất lượng trên host (7 metrics)

Mục tiêu: có bảng **baseline FP16** và **quantized** cùng một bộ tool để tính `Δ`.

### 4.1 Chuẩn bị corpus

```bash
cd "$PROJECT_DIR"
# WikiText-2 raw validation → data/corpus/wikitext-2-raw-val.txt
# (từ Salesforce/wikitext, config wikitext-2-raw-v1, split validation)
ls -lh data/corpus/wikitext-2-raw-val.txt

# VMLU: tải mirror công khai "tridm/VMLU" và chuẩn hóa sang định dạng eval_vmlu.py
# (744 câu valid có đáp án; tự bỏ prefix "A. " và gán tên 58 môn)
python scripts/prepare_vmlu.py --split valid
python scripts/eval_vmlu.py --dataset data/vmlu/vmlu_valid.jsonl --limit 5 --dry-run

# Nguồn gốc: https://github.com/ZaloAI-Jaist/VMLU / https://vmlu.ai
# Lưu ý: test.jsonl (9,833 câu) KHÔNG có đáp án -> không eval được.
```

### 4.2 LM-eval: MMLU, ARC-C, HellaSwag, GSM8K, IFEval

```bash
cd "$PROJECT_DIR"
source .venv/bin/activate

# Baseline FP16
python scripts/eval_quality.py \
  --model "$MODEL_DIR" \
  --model-backend hf --device cuda --limit 200 \
  --output data/results/quality_fp16.json

# GGUF Q4_K_M (sau Phase 5) — so cùng limit
python scripts/eval_quality.py \
  --model data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --model-backend gguf --device cuda --limit 200 \
  --output data/results/quality_q4km.json
```

Script in ra lệnh `lm_eval` thật, ghi log và summary JSON, kèm `quality_gate`. Dùng `--dry-run` để xem lệnh trước khi chạy.

> **Bắt buộc:** `config/config.yaml` bật task `ifeval`, mà `lm-eval` chỉ khai báo `langdetect`/`immutabledict`/`nltk` dưới extra `ifeval`. Nếu cài `lm-eval==0.4.13` thường thì model đã load xong nhưng sẽ crash khi load task: `ModuleNotFoundError: No module named 'langdetect'`. Cài lại bằng `uv pip install 'lm-eval[ifeval]==0.4.13'` (hoặc `uv pip install -r requirements.txt` — file này đã cập nhật extra). Lần đầu import `nltk` sẽ tải `punkt_tab` về `~/nltk_data`; cần internet hoặc tải sẵn gói này.

> **Bắt buộc 2 — tránh OOM:** Qwen2.5-7B fp16 nặng ~15.2 GB, gần hết 1 card 16 GB. Nếu chỉ dùng 1 GPU, lm-eval crash ngay ở `F.log_softmax` trong `loglikelihood`: `torch.OutOfMemoryError: Tried to allocate 290.00 MiB ... 15.17 GiB in use`. `scripts/eval_quality.py` giờ **tự động bật** `parallelize=True,max_memory_per_gpu=10GiB` khi có ≥2 CUDA GPU, ép `device_map="auto"` chia layer sang cả 2 card (mặc định của accelerate là dùng *toàn bộ* VRAM trống nên vẫn dồn hết vào GPU 0). Kiểm tra bằng `--dry-run`; tắt bằng `--no-parallelize`, đổi cap bằng `--max-memory-per-gpu 8GiB`. Script cũng gọi `sys.executable -m lm_eval` để luôn dùng đúng interpreter của `.venv` (tránh nhầm bản cài ở `~/.local`).

```bash
# Xem lệnh lm_eval thật trước khi chạy (không tốn GPU)
python scripts/eval_quality.py \
  --model "$MODEL_DIR" --model-backend hf --device cuda --limit 200 \
  --output data/results/quality_fp16.json --dry-run
# kỳ vọng model_args có: ...,parallelize=True,max_memory_per_gpu=10GiB
```

> **Bắt buộc 3 — GGUF cần `llama-server`:** trong `lm-eval==0.4.13`, backend `--model gguf` **không đọc file GGUF tại chỗ** mà là client HTTP gọi `/v1/completions`; nếu thiếu `base_url` sẽ báo `AssertionError: must pass \`base_url\` to use GGUF LM!` ngay khi khởi tạo model. `scripts/eval_quality.py` giờ tự động: tìm `$HOME/llama.cpp/build/bin/llama-server` (hoặc trong `PATH`), chọn port trống, chạy `llama-server -m <gguf> --host 127.0.0.1 --port <port> -c 2048 -ngl 99`, đợi `/health`, truyền `base_url=...` cho lm-eval, rồi tắt server sau khi xong. Log server riêng tại `data/results/llama_server_<model>_<stamp>.log`. Tùy chỉnh: `--llama-server`, `--server-port`, `--server-ctx`, `--server-ngl`; nếu đã tự chạy server thì dùng `--extra-model-args base_url=http://127.0.0.1:9931` (kèm `--no-auto-server`).

> **Hiệu năng GGUF:** nếu log server có `warning: no usable GPU found, --gpu-layers option will be ignored` thì llama.cpp đang build CPU-only; `-ngl 99` bị bỏ qua và evaluation chạy bằng CPU (rất chậm). Phải build lại với `-DGGML_CUDA=ON` dùng CUDA ≥ 12.8 cho RTX 5080 (`sm_120`) — xem Phase 5.1. `eval_quality.py` mặc định `--server-ngl 99`, nên sau khi build CUDA xong server tự dùng GPU.

> **Bắt buộc 4 — logprobs của llama-server hiện đại:** `lm-eval==0.4.13` mong logprobs dạng legacy (`token_logprobs`/`text_offset`/`top_logprobs`) trả về khi gửi `echo=true`. llama.cpp build 12476 **từ chối `echo`** và chỉ trả `logprobs.content` cho token vừa sinh, nên mọi request loglikelihood bị bỏ qua → crash `ValueError: zip() argument 2 is longer than argument 1` tại `evaluator.py:603`. `scripts/eval_quality.py` giờ tự động dùng `scripts/lm_eval_gguf_compat.py` cho backend `gguf`: script này monkeypatch `GGUFLM.loglikelihood`, chấm điểm từng token qua endpoint native `/completion` (`n_probs=100`, fallback 1000 nếu token mục tiêu nằm ngoài top-N). Task sinh (`gsm8k`, `ifeval`) đi luồng cũ không đổi. Có thể chỉnh qua `GGUF_COMPAT_NPROBS` / `GGUF_COMPAT_TIMEOUT`. Lưu ý: scoring từng token chậm hơn đáng kể so với backend HF; chạy `--limit 200` với 5 task có thể mất 20–40 phút trên CPU-only llama.cpp.

```bash
# Xem trước: kỳ vọng có base_url=http://127.0.0.1:<port>,max_length=2048
python scripts/eval_quality.py \
  --model data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --model-backend gguf --device cuda --limit 200 \
  --output data/results/quality_q4km.json --dry-run
```

### 4.3 WikiText-2 PPL (llama.cpp, mọi biến thể GGUF cùng tool)

```bash
# baseline F16 GGUF
bash scripts/eval_ppl.sh \
  --model data/gguf/Qwen2.5-7B-Instruct-f16.gguf \
  --chunks 64 --out data/results/ppl_fp16.json

# quantized
bash scripts/eval_ppl.sh \
  --model data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --chunks 64 --out data/results/ppl_q4km.json
```

> `scripts/eval_ppl.sh` tự tìm binary theo thứ tự: `$LLAMA_PERPLEXITY` → `$HOME/llama.cpp/build/bin/llama-perplexity` → `PATH`. llama.cpp thường không `make install`, nên binary nằm ở `build/bin`; nếu lỗi `llama-perplexity not found`, truyền `--bin /home/hieunvt/llama.cpp/build/bin/llama-perplexity` hoặc export `LLAMA_PERPLEXITY=...`. Dùng `--dry-run` để kiểm tra đường dẫn trước khi chạy.

> PPL của SpinQuant (`wiki2 ppl is: ...` trong `ptq_run.log`) dùng tool khác với `llama-perplexity`, **không so trực tiếp tuyệt đối**. Ghi rõ tool và limit trong báo cáo.

### 4.4 VMLU (tiếng Việt)

Dataset đã được tải và chuẩn hóa sẵn ở `data/vmlu/vmlu_valid.jsonl` (744 câu, 58 môn, đáp án A–D). Nếu chưa có, chạy lại `python scripts/prepare_vmlu.py --split valid` (xem 4.1).

```bash
cd "$PROJECT_DIR"
source .venv/bin/activate

# Baseline FP16 — dùng backend thật, KHÔNG dùng mock
python scripts/eval_vmlu.py \
  --dataset data/vmlu/vmlu_valid.jsonl \
  --backend hf --model "$MODEL_DIR" --device cuda \
  --limit 500 --max-new-tokens 8 \
  --out data/results/vmlu_fp16.json
```

> **Bắt buộc:** `config/config.yaml` mặc định `model.backend: mock`. Nếu bỏ qua 3 flag `--backend hf --model ... --device cuda`, script vẫn chạy nhưng trả về text giả lập vàaccuracy vô nghĩa. Sau khi chạy, mở JSON kết quả và kiểm tra `"backend": "hf"`, `"model": "Qwen/Qwen2.5-7B-Instruct"` (không phải `mock:...`). Nếu backend lỗi, `build_slm()` tự fallback về `MockSLM` — phải đọc `logs/nano_llm.log` và sửa trước khi tin số liệu.

```bash
# Kiểm tra nhanh không tốn GPU
python -c "import json; d=json.load(open('data/results/vmlu_fp16.json')); print(d['backend'], d['model'], d['accuracy'])"

# Bản quantized GGUF Q4_K_M — so cùng limit
python scripts/eval_vmlu.py \
  --dataset data/vmlu/vmlu_valid.jsonl \
  --backend gguf --model data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --ngl 99 \
  --limit 500 --max-new-tokens 8 \
  --out data/results/vmlu_q4km.json
```

> **GGUF dùng GPU:** `config/config.yaml` mặc định `llamacpp.n_gpu_layers: 0` (chạy CPU). Truyền `--ngl 99` (hoặc bỏ qua — script tự dùng 99 khi `--backend gguf`). Máy này có `llama-cpp-python 0.3.37` **được build có CUDA**: log load sẽ có `ggml_cuda_init: found 2 CUDA devices` và tự split 28 layer sang cả 2 RTX 5080. Kiểm tra bằng `nvidia-smi` trong lúc chạy. Nếu thấy `no usable GPU found`, llama.cpp/llama-cpp-python đang build CPU-only (xem Phase 5.1).
>
> So sánh: `python -c "import json; a=json.load(open('data/results/vmlu_fp16.json')); b=json.load(open('data/results/vmlu_q4km.json')); print('FP16', round(a['accuracy']*100,2), '% | Q4_K_M', round(b['accuracy']*100,2), '% | delta', round((b['accuracy']-a['accuracy'])*100,2), 'điểm %')"`. Ngưỡng: Δ ≥ −2 điểm %.

Ngưỡng nghiệm thu (từ plan):

| Metric                       | Ngưỡng                       |
| ---------------------------- | ------------------------------ |
| WikiText-2 PPL               | Δ ≤ +5% (W4A8) / +10% (W4A4) |
| MMLU, ARC-C, HellaSwag, VMLU | Δ ≥ −2 điểm %             |
| GSM8K                        | Δ ≥ −5 điểm %             |
| IFEval                       | Δ ≥ −3 điểm %             |

### 4.5 Đánh giá chất lượng sau khi chạy Phase 4

Sau khi có đủ 6 file (`quality_fp16.json`, `quality_q4km.json`, `ppl_fp16.json`, `ppl_q4km.json`, `vmlu_fp16.json`, `vmlu_q4km.json`), chạy lệnh dưới đây để tính Δ và đối chiếu ngưỡng. `scripts/eval_quality.py` **chỉ lưu ngưỡng trong `quality_gate`, không tự tính pass/fail**.

```bash
cd "$PROJECT_DIR"
python - <<'PY'
import json
R = "data/results/"
a = json.load(open(R + "quality_fp16.json"))["scores"]
b = json.load(open(R + "quality_q4km.json"))["scores"]
for name, task, key in [
    ("MMLU", "mmlu", "acc"), ("ARC-C", "arc_challenge", "acc_norm"),
    ("HellaSwag", "hellaswag", "acc_norm"), ("GSM8K", "gsm8k", "exact_match"),
    ("IFEval", "ifeval", "prompt_level_strict_acc"),
]:
    x, y = a[task][key], b[task][key]
    print(f"{name:12s} {x*100:6.2f} -> {y*100:6.2f}  delta {(y-x)*100:+5.2f} pp")
p1 = json.load(open(R + "ppl_fp16.json"))["ppl"]
p2 = json.load(open(R + "ppl_q4km.json"))["ppl"]
print(f"PPL          {p1:6.4f} -> {p2:6.4f}  delta {(p2-p1)/p1*100:+5.2f}%")
v1 = json.load(open(R + "vmlu_fp16.json")); v2 = json.load(open(R + "vmlu_q4km.json"))
print(f"VMLU         {v1['accuracy']*100:6.2f} -> {v2['accuracy']*100:6.2f}  delta {(v2['accuracy']-v1['accuracy'])*100:+5.2f} pp")
PY
```

**Kết quả thực tế trên máy này** (Q4_K_M so với FP16 — chi tiết xem `data/results/BAO_CAO_DANH_GIA_CHAT_LUONG_PHASE4.md`):

| Hạng mục | FP16 | Q4_K_M | Δ | Ngưỡng | Đạt? |
| --- | ---: | ---: | ---: | --- | --- |
| WikiText-2 PPL | 6.5786 | 6.6800 | +1.54% | ≤ +5% | ✅ |
| MMLU (acc) | 74.77% | 74.39% | −0.38 điểm % | ≥ −2 | ✅ |
| ARC-C (acc_norm) | 50.00% | 55.00% | +5.00 điểm % | ≥ −2 | ✅ |
| HellaSwag (acc_norm) | 67.50% | 65.00% | **−2.50 điểm %** | ≥ −2 | ❌ |
| VMLU | 63.60% | 61.80% | −1.80 điểm % | ≥ −2 | ⚠️ sát ngưỡng |
| GSM8K | 71.00% | 73.50% | +2.50 điểm % | ≥ −5 | ✅ |
| IFEval (prompt strict) | 50.00% | 47.50% | −2.50 điểm % | ≥ −3 | ⚠️ sát ngưỡng |

6/7 hạng mục đạt; **HellaSwag chưa đạt** (thiếu 0.5 điểm %). Trước khi nghiệm thu phải xử lý 4 điểm sau:

1. **Log `quality_q4km.json` có `no usable GPU found`** — lần đó `llama-server` chạy CPU (trước khi build CUDA 13 lúc 09:59 ngày 18/11). Chạy lại lệnh ở Phase 4.2 với `--server-ngl 99`, kiểm tra log có `CUDA : ARCHS = 1200`.
2. **VMLU `--limit 500` mới phủ 39/58 môn** (thiếu `14` và `32–49`). Chạy `--limit 744` để có điểm đầy đủ.
3. **`--limit 200` của lm-eval chỉ để test** — mỗi câu = 0.5 điểm %, nên mọi Δ < 5 điểm % chưa đủ chắc. Chạy lại HellaSwag với `--limit 1000`.
4. **Ghi rõ metric:** dùng `acc_norm` cho HellaSwag/ARC-C, `prompt_level_strict_acc` cho IFEval. Nếu dùng `acc` thô cho HellaSwag thì Δ chỉ −0.50 và đạt — phải ghi rõ trong biên bản.

> **Báo cáo đầy đủ** (bảng metric chi tiết, phân tích theo từng môn VMLU, các cảnh báo trong log, lệnh chạy lại và kết luận nghiệm thu): `data/results/BAO_CAO_DANH_GIA_CHAT_LUONG_PHASE4.md`.

---

## Phase 5 — Export GGUF + checksum

### 5.1 Build llama.cpp trên host

> **Yêu cầu cmake ≥ 3.18.** llama.cpp hiện tại gọi `scripts/ui-assets.cmake`, file này `cmake_minimum_required(VERSION 3.18)`. Trên máy này `/usr/local/bin/cmake` là bản **3.14.4** (2019) và shadow bản **3.22.1** của hệ thống tại `/usr/bin/cmake`. Dùng `CMAKE=/usr/bin/cmake` như dưới đây, hoặc cài bản mới: `python -m pip install -U 'cmake>=3.18'`.

```bash
cd "$PROJECT_DIR"
git clone https://github.com/ggerganov/llama.cpp "$HOME/llama.cpp" 2>/dev/null || true

# Chọn cmake đủ mới (kiểm tra trước khi build)
CMAKE=/usr/bin/cmake              # 3.22.1; thay bằng "$(command -v cmake)" nếu bản đó >= 3.18
"$CMAKE" --version                # phải >= 3.18
# Nếu đã configure bằng cmake cũ, phải configure lại để ghi đè CMAKE_COMMAND trong cache
"$CMAKE" -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build "$HOME/llama.cpp/build" -j"$(nproc)"

python "$HOME/llama.cpp/convert_hf_to_gguf.py" --help | head -3
ls -lh "$HOME/llama.cpp/build/bin/llama-quantize" "$HOME/llama.cpp/build/bin/llama-cli"
```

> **Bắt buộc để dùng GPU trên RTX 5080:** build mặc định ở trên là CPU-only (`GGML_CUDA=OFF`), khi chạy sẽ có `warning: no usable GPU found, --gpu-layers option will be ignored`. RTX 5080 là `sm_120` (Blackwell) → cần CUDA ≥ 12.8. Trên máy này `nvcc` trên PATH là **11.5** (/usr/bin/nvcc, quá cũ) nhưng đã có **CUDA 13.0** tại `/usr/local/cuda-13.0`. Configure lại với CUDA 13 (NCCL tắt vì không cần multi-GPU cho llama.cpp):
>
> ```bash
> export CUDA_HOME=/usr/local/cuda-13.0
> export PATH="/usr/local/cuda-13.0/bin:$PATH"   # quan trọng: nvcc 13 phải đứng trước /usr/bin/nvcc 11.5
> nvcc --version | tail -2                       # phải là release 13.0
> # Chỉ build đúng kiến trúc của RTX 5080 (sm_120a): mặc định llama.cpp biên dịch
> # 7 kiến trúc (75/80/86/89/90/120a/103/121a) -> 135 template .cu * 7 arch, rất lâu.
> /usr/bin/cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" \
>   -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF \
>   -DCMAKE_CUDA_ARCHITECTURES=120a-real \
>   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
> /usr/bin/cmake --build "$HOME/llama.cpp/build" -j"$(nproc)"
> # log configure phải có: "Using CMAKE_CUDA_ARCHITECTURES=...;120a-real;..."
> # kiểm tra: llama-perplexity -m <gguf> ... -ngl 99 -> không còn warning "no usable GPU"
> ```
>
> `scripts/eval_ppl.sh` mặc định `-ngl 99` (override bằng `--gpu-layers` hoặc `LLAMA_GPU_LAYERS`). Build CPU-only vẫn chạy được nhưng rất chậm: đo `--chunks 2` trên F16 mất ~3.5 phút, `--chunks 64` sẽ mất hàng giờ.

### 5.2 Convert + quantize

```bash
cd "$PROJECT_DIR"
source .venv/bin/activate

# Q4_K_M — artifact deploy mặc định (Hướng A)
LLAMA_CPP_DIR="$HOME/llama.cpp" bash scripts/export_gguf.sh \
  --model-dir "$MODEL_DIR" \
  --outdir data/gguf \
  --quant Q4_K_M \
  --outtype f16

# Q4_0 — nếu muốn thử Hướng B (rk-llama.cpp) hoặc ưu tiên tốc độ
LLAMA_CPP_DIR="$HOME/llama.cpp" bash scripts/export_gguf.sh \
  --model-dir "$MODEL_DIR" \
  --outdir data/gguf \
  --quant Q4_0

# F16 để đo PPL baseline bằng cùng tool
LLAMA_CPP_DIR="$HOME/llama.cpp" bash scripts/export_gguf.sh \
  --model-dir "$MODEL_DIR" \
  --outdir data/gguf \
  --quant f16
```

Script tự đặt tên: `data/gguf/<basename>-<outtype>.gguf` và `data/gguf/<basename>-<quant>.gguf`. Dùng `--skip-f16` để tận dụng file f16 đã có.

Kích thước tham khảo (7B):

| Định dạng | Kích thước |
| ------------ | ------------- |
| FP16         | ~14–15 GB    |
| Q8_0         | ~7.5–8 GB    |
| Q4_K_M       | ~4.3–4.7 GB  |
| Q4_0         | ~4.0–4.4 GB  |

### 5.3 SHA256 — bắt buộc trước khi đẩy

```bash
cd "$PROJECT_DIR"
sha256sum data/gguf/*.gguf | tee data/gguf/SHA256SUMS
ls -lh data/gguf/
```

### 5.4 (Chỉ Hướng C) Calibration + `.rkllm` W8A8 cho model ≤3B

```bash
python scripts/generate_calibration.py --output data/calibration.json --count 128

python scripts/export_rkllm_w8a8.py \
  --model-dir models/Qwen2.5-1.5B-Instruct \
  --dataset data/calibration.json \
  --quantized-dtype w8a8 \
  --target-platform rk3588 \
  --output models/Qwen2.5-1.5B-Instruct_W8A8_RK3588.rkllm
```

Script sẽ **từ chối** `w4a8_g32`/`w4a16` trên `rk3588`/`rk3588s` — đó là hành vi đúng. Không chạy bước này cho 7B.

---

## Phase 6 — Đẩy code + model lên NanoPi-R6C

### 6.1 SSH

```bash
ssh-copy-id "$BOARD"
ssh "$BOARD" "echo OK; cat /proc/device-tree/model; free -h; df -h"
```

Thay `pi`/IP bằng user và địa chỉ thật của board.

### 6.2 Đẩy code (không kèm model — khuyến nghị)

```bash
cd "$PROJECT_DIR"
BOARD="$BOARD" BOARD_DIR="$BOARD_DIR" \
  bash scripts/deploy_board.sh
```

Hành vi mặc định của `scripts/deploy_board.sh`:

- đẩy code, `config/`, `scripts/`, `src/`, `tests/`, `eval/`, `run.py`, `requirements*.txt`;
- **bỏ qua** `.venv`, `__pycache__`, `.pytest_cache`, `.git`, `logs`, `data/gguf`;
- **bỏ qua** `models/` (`COPY_MODELS=0`);
- vẫn đẩy `data/results/` (`COPY_RESULTS=1`).

Nếu muốn gộp model trong lần deploy (file lớn, dễ đứt):

```bash
COPY_MODELS=1 BOARD="$BOARD" BOARD_DIR="$BOARD_DIR" \
  bash scripts/deploy_board.sh
```

Không có `rsync` thì dùng `scp`:

```bash
scp -r config src scripts eval data/results systemd run.py Makefile \
  requirements.txt requirements-board.txt requirements-dev.txt \
  "$BOARD:$BOARD_DIR/"
```

### 6.3 Đẩy riêng GGUF (khuyến nghị: `rsync --partial`)

```bash
cd "$PROJECT_DIR"
rsync -avz --progress --partial \
  data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  "$BOARD:$BOARD_DIR/models/"

# nếu dùng Q4_0 / NPU:
rsync -avz --progress --partial \
  data/gguf/Qwen2.5-7B-Instruct-Q4_0.gguf \
  "$BOARD:$BOARD_DIR/models/"

# corpus/VMLU nếu định chạy eval ngay trên board
rsync -avz --progress data/corpus/ "$BOARD:$BOARD_DIR/data/corpus/"
```

Fallback USB/thẻ nhớ nếu mạng chậm:

```bash
cp data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf /media/USB/ && sync
# trên board:
sudo mount /dev/sda1 /mnt
cp /mnt/Qwen2.5-7B-Instruct-Q4_K_M.gguf ~/NanoPi-R6C-LLM/models/ && sync
sudo umount /mnt
```

### 6.4 Kiểm tra checksum hai phía

```bash
# host
scp data/gguf/SHA256SUMS "$BOARD:/tmp/SHA256SUMS"

# board
ssh "$BOARD"
cd ~/NanoPi-R6C-LLM
sha256sum -c /tmp/SHA256SUMS --ignore-missing
```

---

## Phase 7 — Cài đặt và cấu hình trên board

### 7.1 Gói hệ thống

```bash
ssh "$BOARD"
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git rsync build-essential cmake
python3 --version          # >= 3.9
```

### 7.2 venv tối thiểu (KHÔNG dùng `requirements.txt`)

```bash
cd ~/NanoPi-R6C-LLM
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-board.txt
.venv/bin/python -c "import yaml, fastapi, uvicorn, pydantic; print('runtime OK')"
```

### 7.3 Build llama.cpp trên board (backend `gguf`)

Repo không bundle `llama-cpp-python`, nên backend `gguf` gọi binary `llama-cli` qua subprocess.

```bash
# cmake phải >= 3.18 (llama.cpp dùng scripts/ui-assets.cmake)
cmake --version | head -1
# Nếu board có cmake cũ (< 3.18): sudo apt install -y cmake  # Ubuntu 22.04 có 3.22
# hoặc: python3 -m pip install -U 'cmake>=3.18'

git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build ~/llama.cpp/build -j4

llama-cli --version || {
  sudo cp ~/llama.cpp/build/bin/llama-cli /usr/local/bin/
  sudo cp ~/llama.cpp/build/bin/llama-perplexity /usr/local/bin/   # tùy chọn
}

llama-cli \
  -m ~/NanoPi-R6C-LLM/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -p "Xin chào, bạn là ai?" -n 64 -c 2048 -t 4
```

### 7.4 Cấu hình `config/config.yaml` — kịch bản A (mặc định)

Trên board, tạo bản cấu hình riêng:

```bash
cd ~/NanoPi-R6C-LLM
cp config/config.yaml config/config.board.yaml
python3 - <<'PY'
from pathlib import Path
p = Path("config/config.board.yaml")
s = p.read_text()
s = s.replace("backend: mock", "backend: gguf", 1)
s = s.replace('  path: ""', '  path: models/Qwen2.5-7B-Instruct-Q4_K_M.gguf', 1)
s = s.replace("  max_context: 4096", "  max_context: 4096", 1)
s = s.replace("  n_threads: 4", "  n_threads: 4", 1)
s = s.replace("  n_gpu_layers: 0", "  n_gpu_layers: 0", 1)
s = s.replace("  n_batch: 512", "  n_batch: 256", 1)
p.write_text(s)
PY
grep -A4 '^model:' config/config.board.yaml | head -8
bash scripts/run_board.sh --config config/config.board.yaml --check
```

Giải thích: `n_threads: 4` cho 4×Cortex-A76; `n_gpu_layers: 0` (CPU/RAM); `max_context: 4096` theo token budget 2000–4000, hạ xuống 2048 nếu RAM không đủ; `n_batch: 256` giảm peak RAM.

Kịch bản B (NPU qua rk-llama.cpp, thử nghiệm):

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

Chạy với `export RKNPU_DOMAINS="0-3"` (model >4 GB/domain). **Không chạy nhiều process NPU song song** (nguy cơ kernel panic).

Kịch bản C (RKLLM W8A8, chỉ ≤3B):

```yaml
model:
  backend: rkllm
  path: models/Qwen2.5-1.5B-Instruct_W8A8_RK3588.rkllm
rkllm:
  library_path: models/librkllmrt.so
  target_platform: rk3588
  num_npu_core: 3
```

### 7.5 Preflight

```bash
cd ~/NanoPi-R6C-LLM
bash scripts/check_board.sh
```

Script báo: model board, RAM, driver `rknpu`, `librkllmrt.so`, file `.gguf/.rkllm`, ADC điện áp USB-C, module Python. Một vài WARN (ADC, `librkllmrt.so` khi không dùng NPU) là bình thường.

---

## Phase 8 — Chạy thực nghiệm trên chính NanoPi-R6C

### 8.1 Smoke test & xác nhận không phải mock

```bash
cd ~/NanoPi-R6C-LLM

# 1) mock để chắc code chạy
bash scripts/run_board.sh --config config/config.board.yaml --demo --force-mock

# 2) một prompt thật
bash scripts/run_board.sh --config config/config.board.yaml \
  --prompt "Tóm tắt vai trò của NPU trong thiết bị biên."

# 3) vài prompt thật
bash scripts/run_board.sh --config config/config.board.yaml --demo

# 4) xem config đã resolve
bash scripts/run_board.sh --config config/config.board.yaml --check
```

> Nếu output có dạng `mock:Qwen/...` hoặc text "Đây là phản hồi mô phỏng..." → backend đã fallback. Dừng lại, đọc log `logs/nano_llm.log`, sửa lỗi rồi mới benchmark. Mọi JSON kết quả phải có `"backend": "gguf"` (hoặc `"rkllm"`), không phải `"mock"`.

`run_board.sh` tự đặt: `LD_LIBRARY_PATH=$ROOT/models:$ROOT/lib`, `RKLLM_LOG_LEVEL=1`, `RKNPU_DOMAINS` nếu export, `ulimit -n 65536`, CPU governor `performance`.

### 8.2 API + dashboard trên LAN

```bash
bash scripts/run_board.sh --config config/config.board.yaml
```

Từ laptop:

```bash
curl -s http://192.168.1.50:8080/health
curl -s http://192.168.1.50:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Xin chào"}],"max_tokens":128}'
```

Dashboard: `http://<IP-board>:8080/dashboard`. Nếu không truy cập được: `api.host: 0.0.0.0` trong config, `ss -ltnp | grep 8080`, kiểm tra firewall. Đặt `api.api_key` để bật Bearer auth.

systemd (tùy chọn):

```bash
sudo cp systemd/nano-llm.service /etc/systemd/system/
sudo nano /etc/systemd/system/nano-llm.service   # sửa User/WorkingDirectory/ExecStart
sudo systemctl daemon-reload
sudo systemctl enable --now nano-llm
journalctl -u nano-llm -f
```

Service đã đặt `LimitNOFILE=65536`, `MemoryMax=7G`.

### 8.3 Benchmark hiệu năng W1/W2/W3

Điều kiện công bằng: cùng board, nguồn 65 W, governor `performance`, board idle trước mỗi run, cùng `max_context`, cùng prompt set.

```bash
cd ~/NanoPi-R6C-LLM

# W1 ngắn (128) + W2 trung (2048) + W3 dài (4096)
.venv/bin/python scripts/bench_board.py \
  --config config/config.board.yaml \
  --prompt-tokens 128,2048,4096 \
  --gen-tokens 128 \
  --runs 3 --warmup 1 \
  --out data/results/bench_q4km_W123.json \
  --json

# W4 — prompt thật từ eval/prompts_vi.jsonl (trường "prompt")
.venv/bin/python scripts/bench_board.py \
  --config config/config.board.yaml \
  --prompts-file eval/prompts_vi.jsonl \
  --gen-tokens 256 --runs 3 --warmup 0 \
  --out data/results/bench_q4km_W4.json

# Hoặc qua run_board.sh (bản rút gọn)
bash scripts/run_board.sh --config config/config.board.yaml --bench \
  --prompt-tokens 128,512,1024,2048 --gen-tokens 128 --runs 3 --warmup 1
```

JSON ghi ở `data/results/` chứa: backend, model, model size, load time, từng case (TTFT, prefill tok/s, decode tok/s, peak RAM), median/p95. Kiểm tra ngay:

```bash
.venv/bin/python - <<'PY'
import json, glob
p = sorted(glob.glob("data/results/bench_q4km_W123*.json"))[-1]
d = json.load(open(p))
print("backend:", d["backend"], "| model:", d["model"])
print("output:", d.get("output", "<KHÓA output BỊ THIẾU → chưa áp hotfix 0.6>"))
for c in d["cases"]:
    print(c["label"], {k: round(v, 2) for k, v in c["summary"]["decode_tok_s"].items()})
PY
```

Ngưỡng tham khảo (7B 4-bit, CPU): model ≤4.7 GB, peak RAM ≤6.5 GB, load ≤30 s, TTFT@2048 ≤120 s, prefill ≥8 tok/s, decode ≥4 tok/s.

> Chế độ `llama-cli` spawn một process cho mỗi request → `load_time_s` và `VmHWM` **underreport** chi phí thật. Ghi chú khi báo cáo. Decode tok/s đã đúng sau hotfix 0.5.

### 8.4 Đo công suất (7 metrics cuối cùng)

Chuẩn bị: nguồn USB-C PD 65 W + power meter inline (FNIRSI FNB58, AVHzY CT-3, WITRN…). ADC kênh 2 trên R6C **chỉ đo điện áp**, không ra watt.

```bash
# pyserial + quyền serial (lần đầu)
.venv/bin/pip install pyserial
sudo usermod -aG dialout "$USER"     # logout/login lại
```

Chạy power monitor ở **terminal thứ hai**, song song với benchmark ở 8.3:

```bash
cd ~/NanoPi-R6C-LLM

# Power meter serial — dùng cho báo cáo W
.venv/bin/python scripts/power_monitor.py \
  --meter serial --serial-port /dev/ttyUSB0 --baud 115200 \
  --duration 300 --interval 1 \
  --out data/results/power_q4km.csv

# Smoke không cần phần cứng
.venv/bin/python scripts/power_monitor.py --meter synthetic --duration 10

# Đọc log có sẵn
.venv/bin/python scripts/power_monitor.py --meter file \
  --input /tmp/power.log --duration 0 --out data/results/power_from_file.csv

# ADC nội bộ — CHỈ điện áp, power_w để trống
.venv/bin/python scripts/power_monitor.py --meter adc --duration 60
```

Quy trình 3 lần: idle 5 phút → prefill W2 → decode 5 phút. Ghi nhiệt phòng, lấy trung bình idle/prefill/decode + peak. Kết quả: CSV + `<tên>.summary.json` trong `data/results/`.

### 8.5 (Tùy chọn) Chất lượng ngay trên board — chỉ limit nhỏ

```bash
bash scripts/eval_ppl.sh \
  --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --corpus data/corpus/wikitext-2-raw-val.txt \
  --chunks 16 --threads 4 \
  --out data/results/ppl_board_q4km.json

.venv/bin/python scripts/eval_vmlu.py \
  --dataset eval/vmlu/vmlu.json --limit 50 --max-new-tokens 8 \
  --out data/results/vmlu_board_q4km.json
```

Khuyến nghị vẫn chạy quality evaluation trên host; board chỉ đo performance.

### 8.6 Nhiệt độ / tài nguyên trong lúc chạy

```bash
for z in /sys/class/thermal/thermal_zone*; do
  echo "$z: $(cat "$z/type") $(cat "$z/temp")"
done
free -m
cat /sys/kernel/debug/rknpu/load 2>/dev/null      # nếu có
```

---

## Phase 9 — Copy kết quả về host và báo cáo

```bash
# host
mkdir -p data/results/board
rsync -avz --progress \
  "$BOARD:$BOARD_DIR/data/results/" \
  data/results/board/
```

Đặt tên file rõ ràng:

```
data/results/
├── quality_fp16.json
├── quality_q4km.json
├── ppl_fp16.json / ppl_q4km.json
├── vmlu_fp16.json
├── ptq_run_w4a8.log / ptq_run_w4a4.log
└── board/
    ├── bench_q4km_W123.json
    ├── bench_q4km_W4.json
    ├── power_q4km.csv / power_q4km.summary.json
    └── ppl_board_q4km.json
```

Bảng báo cáo tối thiểu (điền số thật, ghi rõ tool/limit):

| Nhóm       | Metric                                           | FP16 host             | SpinQuant W4A8 host      | GGUF Q4_K_M board                               |
| ----------- | ------------------------------------------------ | --------------------- | ------------------------ | ----------------------------------------------- |
| Quality     | WikiText-2 PPL                                   |                       | từ `ptq_run_w4a8.log` | `ppl_q4km.json`                               |
| Quality     | MMLU / ARC-C / HellaSwag / GSM8K / IFEval / VMLU | `quality_fp16.json` | —                       | `quality_q4km.json`, `vmlu_board_q4km.json` |
| Performance | Model size / Peak RAM / Load time                |                       |                          | `bench_q4km_W123.json`                        |
| Performance | TTFT @128/2048/4096                              |                       |                          | `bench_q4km_W123.json`                        |
| Performance | Prefill / Decode tok/s                           |                       |                          | `bench_q4km_W123.json`                        |
| Performance | Power idle/prefill/decode                        |                       |                          | `power_q4km.summary.json`                     |

Luôn ghi: backend, định dạng, context, threads, governor, tool đo PPL, license artifacts.

---

## Troubleshooting

### Host / SpinQuant

| Triệu chứng                                                                | Nguyên nhân                               | Cách xử lý                                                         |
| ---------------------------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------- |
| `CMake 3.18 or higher is required. You are running version 3.14.4` khi build llama.cpp | `/usr/local/bin/cmake` 3.14 shadow `/usr/bin/cmake` 3.22 | Dùng `CMAKE=/usr/bin/cmake`; configure lại rồi build (xem Phase 5.1) |
| `Precompiled wheel not found` + `CUDA version mismatches (12.2 vs 13.0)` | `nvcc` 12.2, torch cu130                  | Cài `cuda-toolkit-13-0`, export `CUDA_HOME=/usr/local/cuda-13.0` |
| `Unable to locate package cuda-toolkit-13-0`                               | Chưa thêm repo NVIDIA                     | `dpkg -i cuda-keyring_1.1-1_all.deb && apt update`                  |
| `cannot import name 'LossKwargs'`                                          | transformers quá mới                      | Ghim `transformers==4.51.3` (đã có trong requirements)           |
| `exitcode: -9`                                                             | OOM RAM host do quá nhiều rank            | `NPROC=2` đúng số GPU                                            |
| `test-input is not a local folder`                                         | Thiếu `--input_model`                    | Truyền `$MODEL_DIR`                                                |
| `AssertionError` tại `get_hadK`                                         | `intermediate_size=18944`                 | Pad MLP lên 19968 ở cả 2 file (hotfix 0.3)                         |
| `FSDPTrainer ... no attribute ...`                                         | Bản sao `Trainer` cũ                    | Patch class default trong `train_utils/fsdp_trainer.py` (đã có)  |
| `Linear.forward() takes 2 positional arguments but 3 were given`           | Thay `QuantizeLinear` bằng `nn.Linear` | Pad trực tiếp trọng số (`_pad_linear`)                          |
| `CUDA OOM` trong `find_params`/`sym_quant`                             | `--w_clip` lúc train rotation            | `USE_W_CLIP=0`, `MAX_LEN=512` (hoặc 256)                         |
| `CUDA OOM` ở `ptq.py`                                                   | `model.cuda()` đưa cả 7B lên 1 card   | Patch đã comment dòng này (hotfix 0.3)                            |
| `Expected all tensors to be on the same device`                            | Bias `v_proj` CPU, `hadK` CUDA          | `.to(hadK.device)` trong `hadamard_utils.py` (hotfix 0.3)         |
| Không thấy `output_ptq`                                                  | `ptq.py` gốc không lưu model           | Patch đã thêm `torch.save(qmodel.pt)`                            |
| `QWEN_SPINQUANT_DIR` rỗng trong script con                                | Biến không được export                 | Hotfix 0.1 + export ở shell                                          |
| PTQ vẫn chạy W4A4 dù truyền `--a-bits 8`                               | `2_eval_ptq.sh` hardcode                  | Hotfix 0.2                                                            |
| `ModuleNotFoundError: No module named 'langdetect'` khi chạy `eval_quality.py` | `lm-eval` cài thiếu extra `ifeval` | `uv pip install 'lm-eval[ifeval]==0.4.13'` (đã ghi trong `requirements.txt`) |
| `torch.OutOfMemoryError` tại `F.log_softmax` khi eval 7B fp16 | 15.2 GB weights chiếm gần hết 1 card 16 GB | Tự động dùng 2 GPU (`parallelize=True,max_memory_per_gpu=10GiB`); kiểm tra `--dry-run`, chỉnh bằng `--max-memory-per-gpu` |
| `ImportError: numpy.core.multiarray failed to import` khi chạy `eval_quality.py` | `lm_eval` trên PATH là bản ~/.local, không phải `.venv` | Script giờ gọi `sys.executable -m lm_eval`; hoặc `source .venv/bin/activate` |
| `AssertionError: must pass \`base_url\` to use GGUF LM!` | backend `gguf` của lm-eval là HTTP client, không load file locally | Script tự chạy `llama-server` và truyền `base_url` (xem Phase 4.2); hoặc tự chạy server + `--extra-model-args base_url=...` |
| `llama-server did not become healthy in 300s` dù log có `listening on ...` | `urlopen(proxies={})` sai kwargs trên Python 3.10 | Đã fix: dùng `build_opener(ProxyHandler({}))` trong `wait_for_server` |
| `ValueError: zip() argument 2 is longer than argument 1` tại `evaluator.py` | llama-server hiện đại bỏ `echo`, lm-eval 0.4.13 không đọc được `logprobs.content` | Đã fix: backend `gguf` dùng `scripts/lm_eval_gguf_compat.py` (xem Phase 4.2) |
| `ERROR: llama-perplexity not found: llama-perplexity` | llama.cpp không `make install`, binary chỉ có ở `build/bin` | `scripts/eval_ppl.sh` tự tìm `$HOME/llama.cpp/build/bin`; hoặc `--bin` / `LLAMA_PERPLEXITY` (xem Phase 4.3) |
| `no usable GPU found, --gpu-layers option will be ignored` | llama.cpp build CPU-only (`GGML_CUDA=OFF`) hoặc **build CUDA đang chưa xong** (`libggml-cuda.so` chưa có) hoặc `nvcc` 11.5 quá cũ cho `sm_120` | Configure lại CUDA 13.0 + `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real`, build xong hẳn rồi chạy lại (xem Phase 5.1). Kiểm tra: `ls ~/llama.cpp/build/bin/libggml-cuda.so*` |
| `ERROR: dataset file not found: /path/to/vmlu.json` | Để nguyên path placeholder trong guide | Chạy `python scripts/prepare_vmlu.py --split valid`, dùng `--dataset data/vmlu/vmlu_valid.jsonl` (xem Phase 4.4) |
| `ERROR: no usable samples (check question/choices/answer columns)` | Dùng `test.jsonl` (không có cột `answer`) hoặc HF split `validation` (không tồn tại) | Dùng `valid.jsonl` đã chuẩn hóa, hoặc `--dataset tridm/VMLU --split valid` |
| VMLU accuracy ~25% dù model tốt | Prompt bị `A. A. ...` do `choices` giữ prefix gốc, hoặc backend `mock` | `prepare_vmlu.py` đã bỏ prefix; truyền `--backend hf --model "$MODEL_DIR" --device cuda` và kiểm tra `backend` trong JSON |

### NanoPi-R6C

| Triệu chứng                      | Nguyên nhân                       | Cách xử lý                                                                         |
| ---------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: nano_llm`  | Sai cwd/entrypoint                  | Chạy từ repo root hoặc `bash scripts/run_board.sh ...`                           |
| `llama-cli not found`            | Chưa build llama.cpp               | Build/copy vào `/usr/local/bin` hoặc set `llamacpp.binary`                      |
| `GGUF model not found`           | `model.path` sai                  | Đường dẫn tính từ repo root; dùng `--check`                                  |
| `librkllmrt.so not found`        | Chưa copy runtime                  | Copy vào `models/` hoặc `lib/` (chỉ cần cho backend `rkllm`)                |
| `rkllm_init failed`              | Sai ABI/toolkit/driver              | Đối chiếu release `rknn-llm` với driver 0.9.8                                   |
| Bị kill / OOM                     | 7B W8A8 ≈8.2 GB, context quá lớn | Dùng GGUF 4-bit, hạ `max_context=2048`, đóng service khác                      |
| IOMMU/domain error khi chạy NPU   | Model >4 GB/domain                  | `export RKNPU_DOMAINS="0-3"`, không chạy song song                                |
| `W4A8 not supported`             | RK3588S không có kernel W4A8      | Dùng CPU GGUF, W8A8 hoặc W4A4                                                       |
| `Too many open files`            | `ulimit -n` thấp                 | `ulimit -n 65536`; systemd `LimitNOFILE=65536`                                    |
| `Permission denied /dev/ttyUSB0` | User không thuộc `dialout`      | `sudo usermod -aG dialout $USER`, login lại                                        |
| API không truy cập từ LAN       | Bind localhost/firewall             | `api.host: 0.0.0.0`, mở port 8080                                                  |
| Benchmark chậm / nhiệt cao       | Governor powersave/tản kém        | `run_board.sh` đặt performance; thêm quạt, giảm `n_threads`, nghỉ giữa run |
| `rsync` đứt giữa chừng       | Wi-Fi                               | `--partial --append-verify`, ưu tiên dây LAN                                     |
| File model hỏng                   | Copy lỗi                           | `sha256sum -c` hai phía                                                            |
| Kết quả mock / decode tok/s sai  | Chưa áp hotfix 0.5/0.6            | Kiểm tra `"backend"` trong JSON, áp patch                                         |

---

## Checklist cuối cùng

Trước khi báo cáo/demo, đảm bảo:

- [ ] Đã đọc và chấp nhận license CC-BY-NC của SpinQuant ( hoặc có ý kiến legal).
- [ ] Hotfix Phase 0.1–0.6 đã áp; `.venv/bin/python -m pytest -q` → `30 passed`.
- [ ] `nvcc` và `torch.version.cuda` cùng major 13; `fast_hadamard_transform` import OK.
- [ ] `R.bin` tồn tại; log PTQ có `wiki2 ppl` (W4A8 profile hoặc W4A4 tham chiếu 8.53).
- [ ] Có JSON quality baseline + quantized cùng tool/limit.
- [ ] GGUF Q4_K_M (và Q4_0 nếu cần) đã export; SHA256 khớp host ↔ board.
- [ ] Trên board dùng `requirements-board.txt`, **không** dùng `requirements.txt`.
- [ ] `bash scripts/check_board.sh` chạy; một prompt thật trả về backend `gguf` (không mock).
- [ ] Có JSON benchmark W1/W2/W3(+W4) với `"output"` key và decode tok/s đúng.
- [ ] Có CSV + summary công suất từ power meter (không dùng ADC làm watt).
- [ ] Báo cáo ghi rõ: W4A8 là kết quả host, board chạy Q4_K_M/W4A16 hiệu dụng trên CPU.

---

## Cheat sheet

### Host

```bash
export PROJECT_DIR=/mnt/hdd/dev/edgeAI/tungbm/NanoPi-R6C-LLM
export QWEN_SPINQUANT_DIR=$PROJECT_DIR/QwenSpinQuant
export MODEL_DIR=$PROJECT_DIR/models/Qwen2.5-7B-Instruct
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
cd $PROJECT_DIR && source .venv/bin/activate

python scripts/download_model.py --model qwen2.5-7b --out models

# rotation W16A8 (2 GPU) → R.bin
bash scripts/spinquant_w4a8.sh --model "$MODEL_DIR" \
  --w-bits 16 --a-bits 8 --kv-bits 16 --skip-eval --yes

# PTQ W4A8 (1 GPU)
cd $QWEN_SPINQUANT_DIR && bash scripts/2_eval_ptq.sh "$MODEL_DIR" 4 8 16

cd $PROJECT_DIR
LLAMA_CPP_DIR=~/llama.cpp bash scripts/export_gguf.sh \
  --model-dir "$MODEL_DIR" --quant Q4_K_M
sha256sum data/gguf/*.gguf | tee data/gguf/SHA256SUMS

BOARD=pi@192.168.1.50 bash scripts/deploy_board.sh
rsync -avz --progress --partial data/gguf/*-Q4_K_M.gguf \
  $BOARD:~/NanoPi-R6C-LLM/models/
```

### Board

```bash
cd ~/NanoPi-R6C-LLM
python3 -m venv .venv
.venv/bin/pip install -r requirements-board.txt
bash scripts/check_board.sh
bash scripts/run_board.sh --config config/config.board.yaml --demo --force-mock
bash scripts/run_board.sh --config config/config.board.yaml --prompt "Xin chào"

.venv/bin/python scripts/bench_board.py --config config/config.board.yaml \
  --prompt-tokens 128,2048,4096 --gen-tokens 128 --runs 3 --warmup 1 \
  --out data/results/bench_q4km_W123.json
.venv/bin/python scripts/power_monitor.py --meter serial \
  --serial-port /dev/ttyUSB0 --duration 300 --out data/results/power_q4km.csv
bash scripts/run_board.sh --config config/config.board.yaml
```

### Lấy kết quả về

```bash
rsync -avz --progress pi@192.168.1.50:~/NanoPi-R6C-LLM/data/results/ \
  data/results/board/
```

*Cập nhật: 2026-11-17 — viết lại thành quy trình end-to-end, đối chiếu cú pháp script hiện tại và bổ sung các hotfix đã kiểm chứng.*
