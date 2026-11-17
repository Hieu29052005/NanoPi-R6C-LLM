# Hướng dẫn chạy QwenSpinQuant trên RTX 5080

Từ cài môi trường, CUDA, tối ưu rotation đến PTQ (đo perplexity) cho **Qwen2.5-7B-Instruct**.

> **License:** SpinQuant / QwenSpinQuant là **CC-BY-NC 4.0** (phi thương mại). Cần xem lại license trước khi dùng cho sản phẩm thương mại.

## 0. Cấu hình máy đã dùng

| Thành phần | Giá trị |
|---|---|
| GPU | 2 x NVIDIA GeForce RTX 5080 (16 GB, kiến trúc Blackwell, sm_120) |
| RAM | 62 GB (+ swap 31 GB) |
| OS | Ubuntu 22.04 (jammy) |
| Driver | 580.x (CUDA runtime 13.0) |
| Python | 3.10 |
| PyTorch | 2.14.x **cu130** |
| Quản lý package | `uv` |

Đường dẫn dùng trong tài liệu:

```bash
export PROJECT_DIR=/mnt/hdd/dev/edgeAI/tungbm/NanoPi-R6C-LLM
export QWEN_SPINQUANT_DIR=$PROJECT_DIR/QwenSpinQuant
export MODEL_DIR=$PROJECT_DIR/models/Qwen2.5-7B-Instruct
```

Để biến `QWEN_SPINQUANT_DIR` giữ lại sau khi đóng terminal:

```bash
echo "export QWEN_SPINQUANT_DIR=$QWEN_SPINQUANT_DIR" >> ~/.bashrc
```

## 1. Yêu cầu CUDA

RTX 5080 (sm_120) cần **CUDA 12.8 trở lên**. Với PyTorch bản `cu130`, toolkit (`nvcc`) phải cùng **major** với torch, tức là **CUDA 13.x**. Nếu `nvcc` là 12.2 trong khi torch là cu130, bước build `fast-hadamard-transform` sẽ báo lỗi `The detected CUDA version (12.2) mismatches the version that was used to compile PyTorch (13.0)`.

Kiểm tra:

```bash
nvidia-smi          # Driver >= 580, "CUDA Version: 13.x"
nvcc --version      # phải là release 13.x
```

### Cài CUDA Toolkit 13.0 song song với bản cũ (Ubuntu 22.04)

Chỉ cài **toolkit**, không cài `cuda`, `cuda-13-0` hay `cuda-drivers` vì các gói này kéo theo driver và có thể thay driver đang chạy.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
apt-cache policy cuda-toolkit-13-0      # phải có Candidate
sudo apt install cuda-toolkit-13-0
ls /usr/local | grep cuda               # phải có cuda-13.0
```

Nếu không có quyền sudo, dùng runfile: `sh cuda_13.0.0_*_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-13.0 --no-drm --no-man-page`.

Trỏ môi trường sang toolkit mới (làm mỗi lần mở terminal mới trước khi build):

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
nvcc --version      # phải ra release 13.0
```

## 2. Clone repo và tạo môi trường

```bash
cd $PROJECT_DIR
git clone https://github.com/shijiew/QwenSpinQuant QwenSpinQuant

# venv (dùng uv)
uv venv .venv --python 3.10
source .venv/bin/activate
export UV_LINK_MODE=copy

# cài requirements (torch phải được cài trước khi build fast-hadamard-transform)
uv pip install -r requirements.txt
```

`fast-hadamard-transform` phải build **sau** torch và với `--no-build-isolation`, nên nếu nó nằm trong `requirements.txt` thì cài riêng:

```bash
export TORCH_CUDA_ARCH_LIST="12.0"     # chỉ build cho RTX 5080, nhanh hơn nhiều
export MAX_JOBS=8                      # giảm nếu thiếu RAM
uv pip install --no-build-isolation \
  "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@e7706faf8d1c3b9f241e36860640ad1dac644ede"

python -c "import torch, fast_hadamard_transform; print(torch.__version__, torch.version.cuda)"
```

### Tải model

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $MODEL_DIR
ls $MODEL_DIR/config.json
```

## 3. Vì sao 7B cần xử lý thêm

SpinQuant chỉ tạo được ma trận Hadamard cho kích thước là lũy thừa của 2, hoặc lũy thừa của 2 nhân với một trong các hệ số cố định (12, 20, 28, 36, 40, 44, 52, 60, 108, 140, 156, 172).

| Model | hidden | intermediate | Dùng được ngay? |
|---|---|---|---|
| Qwen2.5-0.5B | 896 | 4864 | Không |
| Qwen2.5-1.5B | 1536 | 8960 | Có |
| Qwen2.5-3B | 2048 | 11008 | Có |
| **Qwen2.5-7B** | 3584 (28·128) | **18944 (37·512)** | **Không** (37 không được hỗ trợ) |
| Qwen3-0.6B / 1.7B | 1024 / 2048 | 3072 / 6144 | Có |

Với 7B: **zero-pad MLP từ 18944 lên 19968 (= 156·128)**. Phép pad không đổi kết quả vì nơ-ron bị pad tính ra `silu(0) * 0 = 0`. Code của bạn đã có `pad_mlp(model, 19968)` trong cả `optimize_rotation.py` và `ptq.py` (hai bước phải dùng cùng kích thước pad).

> Nếu dùng Qwen2.5-3B hoặc 1.5B thì bỏ `pad_mlp` khỏi hai file, và không cần FSDP vì model vừa một card.

Kiểm tra trước khi chạy:

```bash
cd $QWEN_SPINQUANT_DIR
python - <<'EOF'
from utils.hadamard_utils import get_hadK
for n in (3584, 128, 19968):
    h, K = get_hadK(n); print(n, "K =", K)
EOF
# Kỳ vọng: K = 28, K = 1, K = 156

grep -n "pad_mlp" optimize_rotation.py ptq.py   # phải thấy cả hàm lẫn lời gọi ở mỗi file
```

## 4. Chạy tối ưu rotation (`optimize_rotation.py`)

Dùng 2 GPU. Không đặt `--nproc_per_node` lớn hơn số GPU thật: lần đầu chạy 8 process trên máy 2 GPU làm hết RAM host và bị kill (`exitcode: -9`).

```bash
cd $QWEN_SPINQUANT_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nnodes=1 --nproc_per_node=2 optimize_rotation.py \
  --input_model "$MODEL_DIR" \
  --output_rotation_path "$QWEN_SPINQUANT_DIR/rotation" \
  --output_dir "$QWEN_SPINQUANT_DIR/output" \
  --logging_dir "$QWEN_SPINQUANT_DIR/logs" \
  --model_max_length 512 \
  --fp16 False --bf16 True \
  --log_on_each_node False \
  --per_device_train_batch_size 1 \
  --logging_steps 1 \
  --learning_rate 1.5 \
  --weight_decay 0. \
  --lr_scheduler_type cosine \
  --gradient_checkpointing True \
  --save_safetensors False \
  --max_steps 100 \
  --w_bits 4 --a_bits 4 --k_bits 16 --v_bits 16 \
  --a_asym --k_asym --v_asym \
  --k_groupsize 128 --v_groupsize 128 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "Qwen2DecoderLayer"
```

Lưu ý quan trọng:

- **`--input_model` bắt buộc.** Nếu thiếu, giá trị mặc định là `test-input` và bị báo `test-input is not a local folder`.
- **Không thêm `--w_clip` khi train rotation với 7B.** Nó tìm scale tối ưu ở mỗi forward và tạo nhiều bản sao tạm, gây `CUDA out of memory` trong `find_params`. Có thể bật lại ở bước PTQ.
- **FSDP là bắt buộc với 7B:** bf16 khoảng 15 GB không vừa một card 16 GB. FSDP shard trọng số qua 2 card.
- Nếu vẫn OOM: giảm `--model_max_length` (256), hoặc thử thêm `offload` vào `--fsdp` (`"full_shard auto_wrap offload"`), nhưng cách này chậm hơn và có thể gây lỗi lẫn device vì các ma trận R và optimizer SGDG nằm trên GPU.
- Có thể dùng script bọc `scripts/10_optimize_rotation.sh` (nhận `<model> <w_bits> <a_bits> <kv_bits>`) thay cho lệnh dài này.

Kết quả: `$QWEN_SPINQUANT_DIR/rotation/R.bin`. Kiểm tra:

```bash
ls -lh $QWEN_SPINQUANT_DIR/rotation/R.bin
```

## 5. Chạy PTQ (`ptq.py`)

Dùng **1 process**, không dùng FSDP. Giữ cấu hình quantize **giống lúc train rotation** (w4/a4/k16/v16, asym, groupsize 128). `--model_max_length` có thể là 2048 vì bước này không cần backward.

```bash
cd $QWEN_SPINQUANT_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
  --input_model "$MODEL_DIR" \
  --output_dir "$QWEN_SPINQUANT_DIR/output_ptq" \
  --do_train False \
  --do_eval True \
  --per_device_eval_batch_size 1 \
  --model_max_length 2048 \
  --fp16 False --bf16 True \
  --save_safetensors False \
  --w_bits 4 --a_bits 4 --k_bits 16 --v_bits 16 \
  --w_clip --a_asym --k_asym --v_asym \
  --k_groupsize 128 --v_groupsize 128 \
  --rotate \
  --optimized_rotation_path "$QWEN_SPINQUANT_DIR/rotation/R.bin" \
  2>&1 | tee ptq_run.log
```

- `--input_model` là **model gốc** (không phải `output/checkpoint-100`). Rotation được áp dụng từ `R.bin` nhờ `--rotate`.
- Bỏ các flag chỉ dùng khi train: `--fsdp`, `--gradient_checkpointing`, `--max_steps`, `--learning_rate`, ...
- Nếu repo có `--w_rtn`, có thể chạy thử với RTN trước để kiểm tra pipeline nhanh, rồi chạy GPTQ (mặc định) để lấy chất lượng tốt nhất.

### Kết quả tham chiếu

Lần chạy W4A4 (k16/v16, có `--w_clip`) trên Qwen2.5-7B-Instruct cho:

```
wiki2 ppl is: 8.53
```

Cảnh báo `Token indices sequence length is longer than the specified maximum` và cảnh báo `barrier()` đều vô hại. Bước eval mất khoảng 2 phút (28 layer).

## 6. Bảng lỗi thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `Precompiled wheel not found` rồi `CUDA version mismatches` (12.2 vs 13.0) | nvcc 12.2, torch cu130 | Cài toolkit 13.0 (mục 1), đặt `CUDA_HOME` |
| `Unable to locate package cuda-toolkit-13-0` | Chưa thêm repo NVIDIA | Cài `cuda-keyring`, `apt update` (mục 1) |
| `cannot import name 'LossKwargs'` | transformers mới | Ghim `transformers==4.51.3`, hoặc định nghĩa lại `LossKwargs` trong `modeling_qwen2_quant.py` |
| `exitcode: -9` | OOM RAM host (quá nhiều rank) | `--nproc_per_node` = số GPU thật |
| `test-input is not a local folder` | Thiếu `--input_model` | Truyền đường dẫn model |
| `AssertionError` tại `get_hadK` (`is_pow2`) | `intermediate_size` 18944 không hỗ trợ | Pad MLP lên 19968 trong cả `optimize_rotation.py` và `ptq.py` |
| `FSDPTrainer ... no attribute ...` | Bản sao `Trainer` cũ | Thêm giá trị mặc định cho thuộc tính thiếu ở mức class `FSDPTrainer`, hoặc ghim transformers |
| `Linear.forward() takes 2 positional arguments but 3 were given` | Thay `QuantizeLinear` bằng `nn.Linear` | Pad trực tiếp trọng số (`_pad_linear`) |
| `CUDA out of memory` trong `find_params` / `sym_quant` | `--w_clip` khi train, 7B quá sát VRAM | Bỏ `--w_clip`, `--model_max_length 512` |
| `CUDA out of memory` ở `ptq.py` | `model.cuda()` đưa cả 7B lên 1 card | Comment dòng `model.cuda()` trong `ptq.py` |
| `Expected all tensors to be on the same device` ở `hadamard_utils.py` | Bias nằm trên CPU, `hadK` trên GPU | Thêm `.to(hadK.device)` cho bias trong `apply_exact_had_to_linear` |
| Không thấy folder `output_ptq` | `ptq.py` không lưu model | `ptq.py` chỉ đo ppl; muốn lưu phải tự thêm `torch.save` |

## 7. Bước tiếp theo để đưa lên NanoPi R6C

Phần trên chỉ cho kết quả **fake-quant** (đo perplexity). Để chạy trên board cần thêm:

1. **Gộp R1, R2** vào trọng số rồi xuất model ở định dạng của runtime (rkllm, llama.cpp/GGUF, ...). SpinQuant không làm sẵn bước này.
2. **R3, R4 là online Hadamard** nên runtime phải hỗ trợ. Phần lớn runtime trên board không có, khi đó phải bỏ R3/R4 và chất lượng sẽ khác con số 8.53.
3. **Activation 4-bit hiếm khi có kernel trên board.** Nhiều khả năng bạn sẽ dùng weight-only (W4A16), nghĩa là nên chạy lại với `--a_bits 16` ở cả bước rotation và PTQ.
4. **7B ở W4 khoảng 4 GB** và sẽ chạy chậm trên RK3588. Qwen2.5-3B hoặc 1.5B (không cần pad, không cần FSDP) thường là lựa chọn thực tế hơn cho board.
