# Kế hoạch Quantize LLM 7B xuống 4-bit (PTQ + SpinQuant + W4A8) chạy trên NanoPi-R6C (8GB)

> **Ngày tạo:** 2026-11-12
> **Cập nhật lần 1:** 2026-11-12 — bổ sung yêu cầu từ buổi họp với sếp (chọn model, token budget 2000–4000, bộ metrics 2 nhóm, NPU inference, PTQ + SpinQuant + W4A8)
> **Cập nhật lần 2:** 2026-11-12 — rà soát kỹ thuật: license SpinQuant (CC-BY-NC), sửa lệnh QwenSpinQuant, ràng buộc 4GB/IOMMU domain của NPU, phiên bản RKLLM v1.3.1, ngưỡng performance, tool đo chính thức của RKLLM
> **Thiết bị đích:** FriendlyElec NanoPi-R6C — Rockchip RK3588S (4× Cortex-A76 + 4× Cortex-A55), 8GB LPDDR4X 2133MHz, NPU 6 TOPS (3 core, hỗ trợ INT4/INT8/INT16/FP16), Ubuntu + kernel FriendlyElec 6.1.y, NPU driver **0.9.8**, nguồn USB-C PD 5–20V
> **Host convert:**
> - **Kaggle CPU notebook** (x86_64, 30GB RAM) — convert `.rkllm`, build llama.cpp, quantize GGUF
> - **Kaggle GPU notebook** (T4 16GB) hoặc Colab Pro/VPS GPU — chạy SpinQuant + W4A8 (rotation optimization + GPTQ) cho 7B
> - Laptop Latitude 3380 (~7GB RAM) **không đủ** cho cả hai bước trên
> **Trạng thái:** Kế hoạch đã cập nhật theo biên bản họp — chưa triển khai. **2 điểm cần chốt: (1) W4A8 không có trên NPU RK3588S (mục 1.1); (2) license SpinQuant/QwenSpinQuant là CC-BY-NC 4.0, phi thương mại (mục 1.5).**

---

## 0. Yêu cầu chốt từ buổi họp (ghi chú gốc)

> **Ghi chú buổi họp:**
> - Tiến hành chọn **một model 7B**, quantize xuống **4-bit** trên chip **NanoPi R6C**.
> - **Tokens sử dụng: 2000 – 4000 tokens**.
> - Tiến hành **đánh giá giữa 2 model: Model 7B (baseline)** và **Model 7B (quantize 4-bit)**.
> - Đánh giá trên **2 tiêu chí**: (1) Model Quality Evaluation, (2) Model Performance Evaluation (danh sách bên dưới).
> - Model có thể sử dụng: **Qwen2.5-7B-Instruct (Khuyến nghị — Recommend)**, **Llama-3.1-8B-Instruct**, **DeepSeek-R1-Distill-Qwen-7B**.
> - Hướng triển khai: **NPU Inference**, phương pháp **PTQ**, thuật toán **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)), quantize dạng **W4A8** ([arXiv:2509.01229](https://arxiv.org/pdf/2509.01229)).

### 0.1 Đối chiếu ghi chú → mục triển khai

| Yêu cầu từ ghi chú | Mục trong plan |
|---|---|
| Chọn 1 model 7B | §2 (3 ứng viên + tiêu chí chọn) |
| Quantize 4-bit trên NanoPi R6C | §3 (phương pháp) + §5–6 (export/deploy) |
| Tokens 2000–4000 | §4 (`max_context=4096`, workload W2/W3) |
| Đánh giá baseline vs quantized | §7 (2 nhóm metrics) + Phase 3/5/8 |
| Model Quality Evaluation (7 mục) | §7.1 |
| Model Performance Evaluation (7 mục) | §7.2 |
| NPU Inference | §1 (khả thi) + Phase 2.5/6 |
| PTQ | §3.1 |
| SpinQuant | §3.2 |
| W4A8 | §3.3 (kèm lưu ý NPU RK3588 không có W4A8 — §1.1) |

### 0.2 Model ứng viên (theo ghi chú)

| Model | Vai trò |
|---|---|
| **Qwen2.5-7B-Instruct** | **Khuyến nghị — Recommend** (model chính) |
| Llama-3.1-8B-Instruct | Ứng viên / fallback |
| DeepSeek-R1-Distill-Qwen-7B | Ứng viên |

### 0.3 Bộ metrics bắt buộc (đúng ghi chú)

**1. Model Quality Evaluation**

| Tiêu chí (ghi chú) | Benchmark |
|---|---|
| Language modeling | WikiText-2 (PPL) |
| Knowledge | MMLU |
| Reasoning | ARC-Challenge |
| Commonsense | HellaSwag |
| Mathematics | GSM8K |
| Instruction Following | IFEval |
| Vietnamese | VMLU |

**2. Model Performance Evaluation**

| Tiêu chí (ghi chú) | Đơn vị |
|---|---|
| Kích thước model — Model size | GB |
| Bộ nhớ RAM đỉnh — Peak RAM | GB |
| Thời gian tải model — Model loading time | Seconds |
| Thời gian sinh token đầu tiên — TTFT (Time to first token) | Seconds |
| Tốc độ xử lý prompt — Prefill speed | Token/s |
| Tốc độ sinh token — Decode speed | Token/s |
| Công suất tiêu thụ — Power consumption | W |

### 0.4 Ghi chú triển khai theo từng yêu cầu

| # | Yêu cầu | Ghi chú triển khai |
|---|---|---|
| 1 | Chọn **một model 7B**, quantize xuống **4-bit** | 3 ứng viên: Qwen2.5-7B-Instruct (khuyến nghị), Llama-3.1-8B-Instruct, DeepSeek-R1-Distill-Qwen-7B |
| 2 | Chạy trên **NanoPi-R6C (RK3588S, 8GB)** | W8A8 7B ≈ 8.2GB → không vừa; 4-bit ≈ 4.3–4.7GB → vừa |
| 3 | **Token sử dụng: 2000–4000 tokens** | Đặt `max_context=4096`; benchmark TTFT/prefill ở prompt ~2000–4096 token |
| 4 | **NPU Inference** | RKLLM (chính thức) hoặc rk-llama.cpp RKNPU2 backend (cộng đồng) — xem mục 1 |
| 5 | Phương pháp **PTQ** | Post-Training Quantization, không fine-tune toàn phần |
| 6 | Thuật toán **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)) | Learned rotations (R1/R2, Cayley optimization) để giảm outlier trước khi quantize |
| 7 | Quantize dạng **W4A8** ([arXiv:2509.01229](https://arxiv.org/pdf/2509.01229)) | 4-bit weights, 8-bit activations. **Lưu ý: paper này là kernel CUDA (LiquidGEMM), không dùng được trên NPU Rockchip** — xem mục 1.1 |
| 8 | **Đánh giá 2 model**: baseline 7B (FP16) vs quantized 4-bit | Trên **2 nhóm tiêu chí** ở mục 7 |

---

## 1. PHÂN TÍCH KHẢ THI — đọc trước khi làm

### 1.1 Phát hiện quan trọng: W4A8 KHÔNG chạy được trên NPU RK3588S

Đây là điểm mâu thuẫn giữa yêu cầu họp và năng lực phần cứng/toolchain, cần báo sếp trước khi triển khai:

| Bằng chứng | Nguồn |
|---|---|
| **RKLLM trên RK3588 chỉ hỗ trợ `w8a8`, `w8a8_g128`, `w8a8_g256`, `w8a8_g512`**. 4-bit: `w4a16`/`w4a16_g*` chỉ trên RK3576 (và một phần RK3562); `w4a8` chỉ có dạng `w4a8_g32` trên RK3562 | Tài liệu RKLLM SDK 1.2.3 (ArmSoM dẫn nguồn Rockchip), mục Table 3-3 |
| Lỗi thực tế khi thử: `target_platform: rk3588 not support quantized_dtype: w4a16!` (kể cả toolkit v1.3.0, 09/2026) | [rknn-llm issue #535](https://github.com/airockchip/rknn-llm/issues/535), [#186](https://github.com/airockchip/rknn-llm/issues/186) |
| NPU RK3588 phần cứng có INT4, nhưng **matmul yêu cầu A/B cùng độ chính xác → không có pipeline W4A8** (chỉ W4A4 hoặc W8A8) | [rk-llama.cpp ggml-rknpu2 README](https://github.com/invisiofficial/rk-llama.cpp/blob/rknpu2/ggml/src/ggml-rknpu2/README.md) |
| RKNN-Toolkit2 v2.5: W4A16 chỉ cho RK3576; w4a8/w4a16 MatMul chỉ cho RK3562 | [rknn-toolkit2 CHANGELOG](https://github.com/airockchip/rknn-toolkit2/blob/master/CHANGELOG.md) |
| 7B W8A8 `.rkllm` ≈ **8.17GB** → lớn hơn RAM 8GB | Model pre-converted công khai (HF `jsntwdj/rkllm/Qwen2.5-7B.rkllm`) |
| Paper W4A8 được dẫn (`2509.01229`) là **LiquidGEMM — kernel CUDA cho NVIDIA Tensor Cores**, không phải phương pháp quantize tổng quát và không áp dụng được cho NPU Rockchip | arXiv abstract |

**Kết luận cứng:** Tổ hợp *"7B + W4A8 + NPU RK3588S"* **không khả thi** với cả toolchain chính thức (RKLLM) lẫn cộng đồng (rk-llama.cpp). Không phải vấn đề code — đây là giới hạn đường tính toán của NPU + danh sách dtype được toolkit hỗ trợ.

### 1.2 Ma trận yêu cầu ↔ khả năng thực tế

| Thành phần yêu cầu | Khả năng trên R6C | Đạt? |
|---|---|---|
| Model 7B | Có (Qwen2.5-7B / Llama-3.1-8B / DeepSeek-R1-Distill-Qwen-7B) | ✅ |
| PTQ | Có (trên host GPU/x86) | ✅ |
| SpinQuant (learned rotation) | Có — `QwenSpinQuant` (Qwen2/Qwen2.5, tối ưu cho **W4A8**) hoặc `facebookresearch/SpinQuant` (LLaMA-2/3) | ✅ trên host |
| W4A8 | Host: có (QwenSpinQuant / AIMET / GPTQ+rotation) — **dùng để đánh giá chất lượng** | ⚠️ chỉ host |
| W4A8 runtime trên NPU | ❌ Không có (RK3588 không có pipeline W4A8) | ❌ |
| 4-bit weights runtime trên NPU | Chỉ **W4A4** qua rk-llama.cpp (Q4_0 → mặc định `W4A4_HADAMARD_LOWRANK`, 4 bpw); chất lượng rủi ro cao | ⚠️ |
| NPU inference 7B | W8A8 → ~8.2GB (OOM). W4A4 → ~4.3GB nhưng **vượt giới hạn 4GB/IOMMU domain** → phải chia domain (`RKNPU_DOMAINS`) và test thực tế | ⚠️ |
| 4-bit chất lượng tốt trên board | **CPU** llama.cpp GGUF Q4_K_M (~4.7GB, ~5 tok/s) | ✅ (không NPU) |
| Context 2000–4000 token | Có (RKLLM max_context ≤16K; llama.cpp `-c 4096`; KV cache ~0.2GB @4K) | ✅ |
| Đo công suất (W) | Có — USB-C PD inline power meter + ADC điện áp trên board | ✅ |

### 1.3 Các hướng triển khai có thể chọn

**Hướng A (KHUYẾN NGHỊ) — "Nghiên cứu quantize đúng yêu cầu + deploy thực tế"**
1. **Accuracy study trên host GPU:** PTQ + **SpinQuant W4A8** cho Qwen2.5-7B / Llama-3.1-8B bằng `QwenSpinQuant`/`SpinQuant` + GPTQ/RTN → đo **toàn bộ nhóm Quality metrics** so với FP16 baseline. → Đáp ứng đúng "PTQ + SpinQuant + W4A8".
2. **Deploy 4-bit lên board:** GGUF **Q4_0 / Q4_K_M** chạy llama.cpp → đo **toàn bộ nhóm Performance metrics** (kể cả công suất) so với baseline.
3. **NPU (tùy chọn, có điều kiện):** thử `rk-llama.cpp` RKNPU2 backend với Q4_0 → NPU chạy pipeline `W4A4_HADAMARD`; đo chất lượng + hiệu năng, so CPU. Nếu chất lượng đạt ngưỡng mới dùng NPU; nếu không, báo cáo trung thực và dùng CPU cho LLM (NPU vẫn dùng cho vision ở project khác).
   - *Lợi ích:* Đáp ứng tất cả yêu cầu trừ "W4A8 runtime trên NPU"; có số liệu chất lượng W4A8 thật, deploy được trên 8GB.
   - *Hạn chế:* W4A8 chỉ là kết quả nghiên cứu trên host, không phải artifact chạy trên NPU.

**Hướng B — "Ưu tiên NPU, chấp nhận W4A4"**
- Deploy `rk-llama.cpp` + GGUF Q4_0 → mặc định NPU pipeline **`W4A4_HADAMARD_LOWRANK`** (4 bpw; muốn `W4A4_HADAMARD` phải override `RKNPU_HYBRID`).
- *Lợi ích:* đúng "4-bit + NPU inference".
- *Hạn chế:* không phải W4A8; benchmark tham khảo của chính backend cho thấy W4A4_HADAMARD tăng PPL rất mạnh (gemma-3-1b: 26.17 → 40.93, +56%; LOWRANK 36.54, +40%) → nguy cơ trượt Quality metrics. **Phải đo và có quality gate.**
- *Hạn chế phần cứng:* NPU có **giới hạn 4GB/IOMMU domain**; 7B Q4_0 (~4.3GB) vượt mức này → phải cấu hình `RKNPU_DOMAINS` để trải model qua nhiều domain và test kỹ (chạy đa process song song trên NPU có thể gây kernel panic).
- *Ghi chú:* Q4_0 → chạy `RKNPU_HYBRID=W8A8_STANDARD` sẽ tốt hơn nhưng weights thành INT8 (~2× bộ nhớ ≈ không vừa 8GB cho 7B).
- *Cơ chế nạp:* backend **không chạy GGUF native** — weights được dequant FP32 trên CPU, calibrate rồi requant sang định dạng NPU; vì vậy cặp GGUF ↔ pipeline phải khớp (Q4_0 ↔ INT4, Q8_0 ↔ INT8).

**Hướng C — "Giữ NPU + W8A8, giảm model xuống ≤3B"**
- RKLLM W8A8 (`Qwen2.5-3B`) vừa 8GB, chạy NPU chính thức, ổn định.
- *Hạn chế:* không còn 7B → **không đúng yêu cầu họp** (chỉ dùng làm phương án dự phòng nếu sếp ưu tiên NPU hơn kích thước model).

**Hướng D — "Đổi phần cứng"**: Board RK3576 hỗ trợ W4A16 trong RKLLM (vẫn không W4A8). Ngoài phạm vi (R6C cố định) — chỉ nêu để đủ bức tranh.

### 1.4 Decision Gate (chốt trước Phase 5)

> **Cần sếp chọn:**
> - **A** (khuyến nghị): W4A8/SpinQuant = kết quả nghiên cứu trên host; deploy 4-bit trên board bằng CPU GGUF; NPU chỉ dùng nếu spike rk-llama.cpp đạt chất lượng.
> - **B**: ép NPU 4-bit = W4A4 (khác W4A8), chấp nhận rủi ro chất lượng, cần quality gate.
> - **C**: giữ NPU + W8A8, giảm model xuống 3B (bỏ yêu cầu 7B).
>
> Nếu không chốt, mặc định chạy **Hướng A** và ghi rõ giới hạn trong báo cáo.

### 1.5 Ràng buộc pháp lý (license) — cần legal duyệt

| Thành phần | License | Ảnh hưởng |
|---|---|---|
| `facebookresearch/SpinQuant` / `shijiew/QwenSpinQuant` | **CC-BY-NC 4.0** (README ghi rõ) | **Phi thương mại** → không dùng trực tiếp cho sản phẩm/dịch vụ thương mại khi chưa có ý kiến legal. Chỉ nên dùng cho R&D nội bộ, đo đạc học thuật. |
| Qwen2.5-7B-Instruct | Apache-2.0 | Dùng thương mại được (giữ notice). |
| Llama-3.1-8B-Instruct | Llama 3.1 Community License | Dùng thương mại được nhưng có điều kiện (đặt tên, >700M MAU phải xin phép Meta). |
| DeepSeek-R1-Distill-Qwen-7B | Cần kiểm tra model card | Xác nhận trước khi dùng. |
| VMLU (`ZaloAI-Jaist/VMLU`) | Cần kiểm tra điều khoản | Khả năng chỉ cho mục đích nghiên cứu; không dùng làm marketing/thương mại khi chưa rõ. |

**Phương án nếu license SpinQuant không được duyệt:**
- Dùng **QuaRot** (rotation Hadamard cố định) hoặc **LLMC** (có config SpinQuant) — kiểm tra license từng repo.
- Dùng **AIMET** (R1/R2 Hadamard) — kiểm tra license.
- Hoặc bỏ bước SpinQuant, chỉ làm PTQ W4A8 bằng GPTQ/RTN và ghi rõ khác biệt.
- Trong mọi trường hợp: **xin xác nhận bằng văn bản từ legal** trước khi đưa artifact vào sản phẩm.

---

## 2. Chọn model (mục 1 trong biên bản họp)

### 2.1 Ứng viên

| Model | Params | Kiến trúc | Context | SpinQuant support | RKLLM support | Tiếng Việt | License | File 4-bit (xấp xỉ) |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-7B-Instruct** (Khuyến nghị — Recommend) | 7.61B | Qwen2 (28 layer, GQA 28Q/4KV) | 32K | ✅ `QwenSpinQuant` (W4A8) | ✅ (Qwen2.5) | Tốt | Apache-2.0 | Q4_0 ~4.3GB |
| Llama-3.1-8B-Instruct | 8.03B | Llama (32 layer, GQA 32Q/8KV) | 128K | ✅ `SpinQuant` gốc (LLaMA-3 8B) | ✅ (LLAMA) | Trung bình | Llama 3.1 Community | Q4_0 ~4.5GB |
| DeepSeek-R1-Distill-Qwen-7B | 7.62B | Qwen2 (28 layer) | 32K | ⚠️ qua `QwenSpinQuant` (cùng kiến trúc Qwen2) | ✅ (DeepSeek-R1-Distill) | Khá | Kiểm tra model card | Q4_0 ~4.3GB |

### 2.2 Tiêu chí chọn (có trọng số)

| Tiêu chí | Trọng số | Qwen2.5-7B | Llama-3.1-8B | DeepSeek-R1-7B |
|---|---|---|---|---|
| Chất lượng tiếng Việt (VMLU/chat) | 30% | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Vừa 8GB (kích thước + KV @4K) | 25% | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| SpinQuant W4A8 khả thi | 20% | ⭐⭐⭐ (repo Qwen chuyên biệt) | ⭐⭐⭐ (repo gốc) | ⭐⭐ (chưa kiểm chứng) |
| RKLLM/NPU compatibility | 15% | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Phù hợp token budget 2000–4000 | 10% | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ (CoT dài ăn hết budget) |

**Quyết định:** dùng **Qwen2.5-7B-Instruct** làm model chính.
**Fallback:** nếu pipeline `QwenSpinQuant` cho kiến trúc Qwen2.5-7B gặp lỗi Hadamard non-power-of-2 (xem rủi ro 8.1) → chuyển sang **Llama-3.1-8B-Instruct** (được `SpinQuant` gốc hỗ trợ đầy đủ W4A8), chấp nhận tiếng Việt kém hơn.

### 2.3 Dữ liệu so sánh công khai (điền sau khi tự đo)

| Model | MMLU | ARC-C | HellaSwag | GSM8K | IFEval | VMLU | WikiText-2 PPL |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct (baseline) | — | — | — | — | — | — | — |
| Llama-3.1-8B-Instruct (baseline) | — | — | — | — | — | — | — |
| DeepSeek-R1-Distill-Qwen-7B (baseline) | — | — | — | — | — | — | — |

---

## 3. Phương pháp quantize: PTQ + SpinQuant + W4A8

### 3.1 PTQ trong project

- PTQ = quantize sau huấn luyện, không train lại model.
- **Phạm vi chạy:** bước quantize/convert chạy trên **host** (Kaggle GPU cho SpinQuant/W4A8, Kaggle CPU cho GGUF/.rkllm); **NanoPi-R6C chỉ chạy inference 4-bit**, không quantize trên board (8GB không đủ).
- Hai loại artifact cần phân biệt rõ:
  1. **Artifact nghiên cứu độ chính xác**: `SpinQuant W4A8` (fake-quant/GPTQ checkpoint) — chạy trên host GPU để đo chất lượng. **Không deploy được lên NPU R6C.**
  2. **Artifact triển khai**: `.gguf` (Q4_K_M/Q4_0) chạy llama.cpp trên CPU (và Q4_0 trên NPU qua rk-llama.cpp nếu chọn Hướng B).

### 3.2 SpinQuant pipeline (từ paper 2405.16406)

SpinQuant áp ma trận trực giao (rotation) vào các vị trí chiến lược để làm phẳng phân bố, triệt outlier trước khi quantize. Rotation được **học** bằng Cayley optimization trên Stiefel manifold (khác QuaRot dùng Hadamard cố định):

1. **R1** — rotation toàn bộ residual stream (fuse vào embedding + các ma trận ghi residual như `o_proj`, `down_proj`).
2. **R2** — rotation theo head dimension của attention (fuse vào `v_proj`/`o_proj`).
3. **R3/R4** — rotation trong MLP (down/up projection).
4. **Bước quantize**: sau rotation, dùng GPTQ (mặc định) hoặc RTN (`--w_rtn`) cho weights; activation 8-bit (W4A8).

Công cụ:
- `facebookresearch/SpinQuant` — LLaMA-2/3 (7B/13B/70B, LLaMA-3 8B), hỗ trợ W4A8 & export ExecuTorch.
- `shijiew/QwenSpinQuant` — fork hỗ trợ **Qwen2/Qwen2.5**, được tác giả nêu rõ **tối ưu cho W4A8** (W4A4 không chạy ổn do outlier kênh).
- `qualcomm/aimet` — SpinQuant R1/R2 (Hadamard cố định) cho ONNX/PyTorch; tham chiếu cách fuse rotation sạch vào đồ thị model.

Lệnh tham khảo (trên Kaggle GPU, theo README của fork `shijiew/QwenSpinQuant`):

```bash
# Yêu cầu: Python 3.9, torch >= 2.0 (CUDA) + fast-hadamard-transform (build cần CUDA)
git clone https://github.com/shijiew/QwenSpinQuant.git
cd QwenSpinQuant
git checkout <commit-hash>            # ghim commit để tái lập
pip install -r requirement.txt
pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git

# Bước 1: tối ưu rotation — tham số: $model_name $w_bit $a_bit $kv_bit
bash scripts/10_optimize_rotation.sh Qwen/Qwen2.5-7B-Instruct 4 8 16

# Bước 2: PTQ + đánh giá với rotation đã tối ưu
# (đặt optimized_rotation_path/output_dir/logging_dir trong script trước khi chạy)
bash scripts/2_eval_ptq.sh Qwen/Qwen2.5-7B-Instruct 4 8 16
```

> README gốc: Step 1 dùng `scripts/10_optimize_rotation.sh $model_name $w_bit $a_bit $kv_bit`, Step 2 dùng `scripts/2_eval_ptq.sh ...`; GPTQ là mặc định, thêm `--w_rtn` để dùng RTN. Luôn đọc README đúng fork tại thời điểm chạy.

### 3.3 W4A8 — ý nghĩa và giới hạn thực tế

- **W4A8** = 4-bit weights + 8-bit activations. Ưu điểm lý thuyết: giảm bộ nhớ weight 4× so FP16, activation 8-bit giảm chi phí compute/bandwidth so FP16.
- Trên **GPU NVIDIA**: có kernel W4A8 hiệu quả (ví dụ LiquidGEMM/LiquidQuant trong paper 2509.01229 — nhưng đây là kernel CUDA, không liên quan NPU Rockchip).
- Trên **NPU RK3588S**: không có pipeline W4A8 → chỉ có W4A4 (INT4×INT4) hoặc W8A8. Vì vậy W4A8 dùng để **đánh giá chất lượng thuật toán**, không phải artifact chạy NPU.
- **rk-llama.cpp không chạy GGUF native**: weights được dequant ra FP32 trên CPU → calibrate → requant sang pipeline NPU. Mapping mặc định: `F16→W16A16_STANDARD`, `Q8_0→W8A8_LOWRANK`, `Q6_K→[W8A8_LOWRANK, W4A4_HADAMARD_LOWRANK]`, `Q4_0→W4A4_HADAMARD_LOWRANK`. Vì vậy phải ghép GGUF cùng mức precision với pipeline (Q4_0 ↔ INT4, Q8_0 ↔ INT8).
- **Giới hạn bộ nhớ NPU:** 4GB/IOMMU domain; model 7B 4-bit (~4.3GB) vượt ngưỡng → phải thử `RKNPU_DOMAINS` và kiểm tra OOM/panic.
- Trên **CPU llama.cpp**: GGUF Q4_K_M là weight 4-bit + activation FP16 (tương đương W4A16), không phải W4A8 → cần ghi chú khi so sánh.

### 3.4 Kế hoạch artifact

| Artifact | Mục đích | Môi trường | Ghi chú |
|---|---|---|---|
| FP16 baseline (`Qwen2.5-7B-Instruct`) | Mốc chất lượng | Host GPU | Dùng lm-eval/transformers |
| `SpinQuant_W4A8` (checkpoint + log) | Nghiên cứu chất lượng W4A8 | Host GPU | Không deploy R6C |
| `qwen2.5-7b-Q4_0.gguf` | Deploy 4-bit (NPU thử nghiệm) | R6C CPU/NPU | NPU pipeline W4A4 nếu dùng rk-llama.cpp |
| `qwen2.5-7b-Q4_K_M.gguf` | Deploy 4-bit (CPU, chất lượng cao) | R6C CPU | W4A16 hiệu dụng |
| `qwen2.5-7b-Q8_0.gguf` | Baseline gần gốc để tính PPL cùng tool | Host | Tham chiếu |

---

## 4. Token budget 2000–4000 (mục 3 biên bản)

- Cấu hình: **`max_context = 4096`** (chia hết cho 32), đáp ứng prompt + generation trong khoảng 2000–4000 token.
- Workload benchmark chuẩn (dùng cho TTFT/Prefill/Decode):
  - **W1 — ngắn:** prompt 128 token, sinh 256 token (đo decode ổn định).
  - **W2 — trung:** prompt ~2048 token, sinh 256 token (đo TTFT/prefill vùng 2000).
  - **W3 — dài:** prompt ~4096 token, sinh 128 token (đo TTFT/prefill vùng 4000).
  - **W4 — end-to-end:** hội thoại 2000–4000 token thực tế, sinh đến hết (dùng cho soak/integration, không dùng để tính median vì quá lâu).
- Ước tính KV cache (Qwen2.5-7B, GQA 4 KV head, 28 layer, head_dim 128, FP16): ~56KB/token → 4096 token ≈ **~224MB**; Llama-3.1-8B (8 KV head, 32 layer): ~128KB/token → 4096 token ≈ **~512MB**. Cộng vào ngân sách RAM.
- Lưu ý: với decode ~3–5 tok/s, sinh 4000 token mất 13–22 phút → các bài soak dài phải chạy nền và log, không chặn pipeline.

---

## 5. Cấu trúc repo `NanoPi-R6C-LLM/`

Tái dùng pattern từ `../edge_ai_r6c` (src-layout, dataclass + YAML config, factory với mock fallback, script CLI standalone, Makefile, pytest):

```
NanoPi-R6C-LLM/
├── README.md, Makefile, run.py
├── config/config.yaml                  # model, backend (hf|gguf|rkllm), bench, eval
├── src/nano_llm/
│   ├── config.py, logging_setup.py, main.py
│   ├── models/
│   │   ├── hf_backend.py               # baseline FP16 (transformers)
│   │   ├── llamacpp_backend.py         # llama-server / llama-cli wrapper
│   │   ├── rkllm_bindings.py           # copy từ edge_ai_r6c (kiểm tra ABI)
│   │   └── slm.py                      # build_slm factory + MockSLM
│   └── api/app.py                      # FastAPI OpenAI-compatible
├── scripts/
│   ├── download_model.py               # 3 ứng viên
│   ├── generate_calibration.py         # calibration VN (cho RKLLM nếu dùng)
│   ├── spinquant_w4a8.sh               # QwenSpinQuant: rotate + GPTQ/RTN
│   ├── export_gguf.sh                  # convert_hf_to_gguf + Q4_0/Q4_K_M/Q8_0
│   ├── export_rkllm_w8a8.py            # nếu cần NPU W8A8 (model ≤3B)
│   ├── eval_quality.py                 # điều phối lm-eval + VMLU + PPL
│   ├── eval_vmlu.py                    # runner VMLU (ZaloAI)
│   ├── eval_ppl.sh                     # llama-perplexity WikiText-2
│   ├── bench_board.py                  # TTFT/prefill/decode/load-time/RAM/NPU
│   ├── power_monitor.py                # đọc power meter + ADC, ghi CSV
│   ├── rkllama_setup.sh                # build rk-llama.cpp (Hướng B)
│   └── deploy_board.sh / run_board.sh  # rsync/scp → pi@NanoPi-R6C
├── eval/
│   ├── vmlu/                           # dataset/code VMLU
│   ├── prompts_vi.jsonl, prompts_en.jsonl
│   └── gold/
├── data/
│   ├── corpus/wikitext-2-raw-val.txt
│   └── results/                        # JSON kết quả (commit)
├── systemd/nano-llm.service
├── tests/
└── KE_HOACH_QUANTIZE_QWEN7B_4BIT_NANOPI_R6C.md
```

---

## 6. Các Phase triển khai

### Phase 0 — Kiểm tra nền tảng (0.5 ngày)

```bash
# Board
cat /proc/device-tree/model                              # RK3588S
sudo cat /sys/kernel/debug/rknpu/version                 # RKNPU driver: v0.9.8
free -h && df -h                                         # ≥12GB trống, ưu tiên NVMe
cat /sys/class/thermal/thermal_zone*/temp
nproc
# Điện áp USB-C (ADC) — dùng cho đo công suất
cat /sys/devices/platform/fec10000.saradc/iio:device0/in_voltage2_raw
```

- Chuẩn bị **USB-C PD inline power meter** (VD: FNIRSI FNB58, AVHzY CT-3, WITRN) đặt giữa charger 65W và board — nguồn board là USB-C PD 5–20V, *full load khuyến nghị 65W*.
- Kaggle: tạo 1 CPU notebook (convert GGUF/RKLLM) + 1 GPU T4 notebook (SpinQuant W4A8).

**Checklist:** driver 0.9.8; ≥12GB trống; power meter hoạt động; ghi nhiệt/điện idle làm mốc.

### Phase 1 — Scaffold repo (0.5 ngày)

- Copy pattern `config.py`, `run.py`, Makefile, tests từ `../edge_ai_r6c`; backend mock chạy được.
- **Checklist:** `make check` + `make test` pass.

### Phase 2 — Môi trường convert (0.5 ngày)

- Kaggle CPU: clone `rknn-llm` — **chốt phiên bản sau khi đối chiếu driver**: bản mới nhất là `release-v1.3.1` (trùng bindings hiện có trong `edge_ai_r6c`), nhưng board đang dùng driver 0.9.8 (đi kèm `release-v1.2.3`). Kiểm tra `rknpu-driver` của từng release trước khi chọn; nếu dùng v1.3.1 phải xác nhận driver tương thích hoặc nâng driver. Cài `rkllm_toolkit` + `setuptools<81`; build llama.cpp (ghim release tag).
- Kaggle GPU: clone `QwenSpinQuant` (+ commit hash), Python 3.9, `torch>=2.0` (CUDA), `transformers`, `auto-gptq`/`gptqmodel`, `fast-hadamard-transform` (build cần CUDA), `lm-eval`, `rouge-score`.
- **Checklist:** `from rkllm.api import RKLLM` OK; `llama-quantize` OK; `import fast_hadamard_transform` OK; `QwenSpinQuant` rotation script chạy được với model nhỏ (smoke test 1.5B).

### Phase 2.5 — **Feasibility spike: NPU 4-bit + SpinQuant** (1 ngày) ⚠️ BẮT BUỘC

Mục đích: xác nhận bằng thực nghiệm các kết luận ở mục 1 trước khi đầu tư.

1. Thử `rkllm.build(quantized_dtype='w4a16'/'w4a8', target_platform='rk3588')` trên Qwen2.5-0.5B ở Kaggle → kỳ vọng lỗi `not supported`. Ghi log làm bằng chứng.
2. Thử `rkllm.build(quantized_dtype='w8a8', target_platform='rk3588')` Qwen2.5-7B → đo file size (kỳ vọng ~8GB) → kết luận không vừa RAM 8GB.
3. Build `rk-llama.cpp` (`cmake .. -DLLAMA_RKNPU2=ON`, `ulimit -n 65536`) trên board + chạy Q4_0 model nhỏ (1.5B) → xác nhận NPU pipeline thực tế (mặc định `W4A4_HADAMARD_LOWRANK`, hoặc override qua `RKNPU_HYBRID`), đo NPU load + chất lượng sơ bộ.
4. Thử `QwenSpinQuant` rotation trên Qwen2.5-1.5B (W4A8) → xác nhận pipeline chạy; ghi thời gian/RAM; đánh giá khả năng lên 7B.
5. **Test giới hạn 4GB/IOMMU domain:** chạy thử model 7B Q4_0 trên NPU với `RKNPU_DOMAINS` (ví dụ `"0-3"`) → xác nhận có load được model ~4.3GB hay không; ghi log OOM/panic nếu có.
6. Đo sơ bộ thời gian convert 7B trên Kaggle GPU (ước tính để lên lịch).

**Checkpoint:** cập nhật mục 1 với bằng chứng thực đo → **chốt Hướng A/B/C với sếp** trước Phase 5.

### Phase 3 — Baseline FP16 (Kaggle GPU, 1–1.5 ngày)

Với model đã chọn (và có thể chạy sơ bộ 2 ứng viên còn lại nếu thời gian cho phép):

- **Quality:** chạy đủ 7 tiêu chí trên FP16 (transformers + lm-eval; VMLU qua runner riêng):
  ```bash
  lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=float16 \
    --tasks mmlu,arc_challenge,hellaswag,gsm8k,ifeval --limit 250 --batch_size 8 \
    --output_path data/results/baseline_fp16/
  ./llama.cpp/build/bin/llama-perplexity -m qwen2.5-7b-f16.gguf \
    -f data/corpus/wikitext-2-raw-val.txt --chunks 200 > data/results/ppl_fp16.txt
  python scripts/eval_vmlu.py --model Qwen/Qwen2.5-7B-Instruct --out data/results/vmlu_fp16.json
  ```
- **Lưu ý:** WikiText-2 PPL nên đo bằng cùng tool (llama.cpp) cho mọi biến thể GGUF; với FP16 HF có thể dùng llama.cpp F16 GGUF để đồng nhất.

**Checklist:** có JSON đủ 7 metrics baseline; ghi commit hash model.

### Phase 4 — Calibration data (0.5 ngày)

- Calibration tiếng Việt 100–200 mẫu chat đa chủ đề (format RKLLM `{input,target}`) — dùng cho RKLLM (nếu Hướng C) và để tham chiếu.
- SpinQuant dùng dataset riêng của repo (WikiText-2 calibration cho GPTQ) — **không dùng calibration VN cho bước rotation** trừ khi đổi code; ghi rõ trong báo cáo.

**Checklist:** JSON hợp lệ; ghi nguồn dữ liệu.

### Phase 5 — Quantize (Kaggle GPU + CPU, 2–3 ngày)

**5a. SpinQuant W4A8 (Hướng A — accuracy artifact):**
- Chạy rotation optimization + GPTQ/RTN cho Qwen2.5-7B (`scripts/spinquant_w4a8.sh`).
- Kỳ vọng: rotation ~1–2h, GPTQ ~1–3h trên T4 (nếu thiếu VRAM → dùng `--w_rtn` RTN hoặc thuê A100/2×T4).
- **5b. Đánh giá W4A8 artifact:** chạy đủ 7 quality metrics trên checkpoint W4A8 (fake-quant hoặc GPTQ) — so với baseline.
- **5c. Export GGUF:** F16 → Q4_0, Q4_K_M, Q8_0 (llama.cpp).
- **5d. Đưa file ra:** upload HF private repo (dataset) → laptop → board; ghi SHA256.

**Checklist:**
- [ ] `SpinQuant W4A8` checkpoint + log rotation (loss, thời gian)
- [ ] 7 quality metrics của W4A8 (dùng `--limit` hợp lý)
- [ ] GGUF Q4_0/Q4_K_M/Q8_0, kích thước hợp lý
- [ ] File lên board, SHA256 khớp

### Phase 6 — Deploy board (1 ngày)

- **CPU (mặc định Hướng A):** build llama.cpp native trên board; chạy Q4_K_M/Q4_0.
- **NPU (Hướng B, nếu qua spike):** build `rk-llama.cpp` với `-DLLAMA_RKNPU2=ON`; chạy Q4_0 (mặc định `Q4_0 → W4A4_HADAMARD_LOWRANK`):
  ```bash
  export RKNPU_HYBRID=W4A4_HADAMARD   # optional; hoặc W8A8_STANDARD (cần ~2× RAM)
  export RKNPU_DOMAINS="0-3"          # trải model qua nhiều domain (>4GB/domain)
  # export RKNPU_CORES="0,1,2"        # giới hạn core NPU nếu cần
  ulimit -n 65536
  taskset -c 4-7 ./build/bin/llama-cli -m qwen2.5-7b-Q4_0.gguf -ngl 99 -t 4 -c 4096 -fa on
  ```
- **RKLLM (chỉ nếu Hướng C):** `llm_demo` + `librkllmrt.so` — dùng đúng release đã chốt ở Phase 2 (cân nhắc `release-v1.2.3` khớp driver 0.9.8 vs `release-v1.3.1` mới nhất).
- Checkpoint go/no-go: đo RAM thực (`free -m`, VmHWM); nếu >6.5GB hoặc OOM → giảm context 2048, chọn artifact khác.

**Checklist:** cả artifact chạy được; RAM đạt; quyết định backend chính.

### Phase 7 — Performance benchmark trên board (1–1.5 ngày)

- Khóa điều kiện: `scaling_governor=performance`, chạy script đặt tần số trong thư mục `scripts/` của RKLLM SDK, board idle, ghi nhiệt phòng.
- **Đo bằng tool chính thức của RKLLM (nếu dùng Hướng C):** `export RKLLM_LOG_LEVEL=1` để log prefill/decode + memory; dùng `scripts/eval_perf_watch_cpu.sh` và `scripts/eval_perf_watch_npu.sh` để đo CPU/NPU utilization.
- Đo theo mục 7.2: model size, peak RAM, load time, TTFT (W1/W2/W3), prefill, decode, power.
- Power: `scripts/power_monitor.py` đọc power meter (serial/BLE tùy thiết bị) hoặc nhập tay từ màn hình meter mỗi 1s; đồng bộ log với workload (idle 5 phút → prefill → decode 5 phút).
- 3 backend (nếu có): CPU Q4_K_M, CPU/NPU Q4_0, (tham chiếu host FP16 không đo công suất).

**Checklist:** CSV/JSON đủ 7 metrics × backend, median + p95, ≥5 runs.

### Phase 8 — Quality comparison & report (1 ngày)

- Bảng so sánh: `FP16 baseline` vs `SpinQuant W4A8` (host) vs `GGUF Q4_K_M` (board) vs `Q4_0/NPU` (nếu có).
- Ghi rõ mỗi con số đo ở đâu, tool nào, limit bao nhiêu.
- `make report` → `BAO_CAO_SO_SANH.md`.

### Phase 9 — Serving & ổn định (1 ngày)

- `llama-server` (OpenAI-compatible) hoặc `rk-llama.cpp` llama-server; systemd `MemoryMax=7G`.
- Soak 45 phút với prompt 2000–4000 token (W4): không OOM, nhiệt <85°C, công suất ổn định, tok/s không tụt >10%.

---

## 7. Hệ metrics đánh giá (đúng biên bản họp)

### 7.1 Nhóm 1 — Model Quality Evaluation

| # | Tiêu chí | Benchmark | Tool | Metric | Ghi chú |
|---|---|---|---|---|---|
| Q1 | Language modeling | WikiText-2 (PPL) | `llama-perplexity` (FP16/Q8/Q4 GGUF); với SpinQuant W4A8 dùng eval script của repo | PPL (thấp hơn tốt) | Cùng corpus `wikitext-2-raw-val`, cùng `--chunks 200` cho mọi biến thể |
| Q2 | Knowledge | MMLU | `lm-eval-harness` | Accuracy (0-shot, subset 250/task) | Baseline FP16 trên HF; GGUF qua backend `gguf`/llama-cpp-python |
| Q3 | Reasoning | ARC-Challenge | `lm-eval-harness` | Accuracy | `arc_challenge` |
| Q4 | Commonsense | HellaSwag | `lm-eval-harness` | Accuracy | `hellaswag` |
| Q5 | Mathematics | GSM8K | `lm-eval-harness` | Exact match | `gsm8k`; subset để giới hạn thời gian; ghi rõ limit |
| Q6 | Instruction Following | IFEval | `lm-eval-harness` | Prompt/instruction accuracy | `ifeval` |
| Q7 | Vietnamese | VMLU | `ZaloAI-Jaist/VMLU` (runner riêng `scripts/eval_vmlu.py`) | Accuracy (58 môn, 10.880 câu) | Có thể lấy subset theo môn để tiết kiệm thời gian; ghi rõ subset |

**Cách tính độ suy giảm:** `Δ = quantized − baseline` (điểm %) với accuracy; `ΔPPL% = (PPL_q − PPL_base)/PPL_base`.
**Ngưỡng tham khảo (đề xuất nghiệm thu):**

| Metric | Ngưỡng chấp nhận |
|---|---|
| WikiText-2 PPL | Δ ≤ +5% (W4A8) / Δ ≤ +10% (W4A4) |
| MMLU, ARC-C, HellaSwag, VMLU | Δ ≥ −2 điểm % |
| GSM8K | Δ ≥ −5 điểm % (subset nhỏ dao động cao) |
| IFEval | Δ ≥ −3 điểm % |

> Nếu một ngưỡng bị vượt: ghi nhận trung thực + đề xuất (tăng group size, dùng SpinQuant thay RTN, hybrid quantization, hoặc Hướng C).

### 7.2 Nhóm 2 — Model Performance Evaluation

| # | Tiêu chí | Đơn vị | Tool / cách đo | Ghi chú |
|---|---|---|---|---|
| P1 | Model size | GB | `du -h` file artifact (FP16, Q4_0, Q4_K_M) | Kèm tỷ lệ nén vs FP16 |
| P2 | Peak RAM | GB | `/proc/<pid>/status` VmHWM; `free -m` delta; RKLLM memory log | Đo ở context 4096 |
| P3 | Model loading time | Seconds | timestamp trước/sau init (llama.cpp log / RKLLM init) | Cold start (drop page cache: `echo 3 > /proc/sys/vm/drop_caches`) |
| P4 | TTFT (Time to first token) | Seconds | log benchmark: W1 (128), W2 (~2048), W3 (~4096) | Báo cả 3 mức prompt |
| P5 | Prefill speed | Token/s | `llama-bench -p 512/2048`; RKLLM perf prefill | Tách khỏi decode |
| P6 | Decode speed | Token/s | `llama-bench -n 256`; đếm token/s qua callback | Median + p95, ≥5 runs |
| P7 | Power consumption | W | USB-C PD power meter (inline) + đồng bộ log; tham chiếu ADC điện áp | Đo idle, prefill, decode; báo trung bình decode và peak |

**Quy trình đo công suất (P7):**

1. Chuẩn bị nguồn USB-C PD ≥65W + power meter inline (đo V/A/W).
2. Board idle 5 phút (không service nền) → ghi `P_idle`.
3. Chạy workload prefill W2 → ghi `P_prefill` (trung bình trong lúc prefill).
4. Chạy decode 5 phút → ghi `P_decode` (trung bình) + peak.
5. Nhiệt phòng ghi lại; lặp 3 lần, lấy trung bình.
6. (Tùy chọn) đối chiếu điện áp USB-C qua ADC `/sys/devices/platform/fec10000.saradc/iio:device0/in_voltage2_raw` — chỉ có V, không có I, nên cần meter để có W.

**Điều kiện công bằng:** cùng board, cùng nguồn, cùng governor performance, cùng thread (`-t 4` A76), cùng context 4096, cùng prompt set, board idle trước mỗi run.

**Ngưỡng nghiệm thu đề xuất cho 7B 4-bit (cần sếp chốt):**

| Metric | CPU (Q4_K_M) | NPU (Q4_0/W4A4) | Ghi chú |
|---|---|---|---|
| Model size | ≤4.7GB | ≤4.5GB | |
| Peak RAM | ≤6.5GB | ≤7.0GB | kèm KV 4096 |
| Load time (cold) | ≤30s | ≤60s | |
| TTFT @2048 | ≤120s | ≤90s | đo thực tế trước, chỉnh sau |
| TTFT @4096 | ≤240s | ≤180s | |
| Prefill | ≥8 tok/s | ≥12 tok/s | tham chiếu Turing Pi ~11.4 tok/s |
| Decode | ≥4 tok/s | ≥6 tok/s | tham chiếu Q4_K_M 5.44 tok/s |
| Power | báo cáo | báo cáo | idle/prefill/decode; không đặt ngưỡng cứng |

### 7.3 Bổ sung tùy chọn (ngoài yêu cầu họp — nếu còn thời gian)

- NPU load % (`/sys/kernel/debug/rknpu/load`) và CPU usage cho Hướng B.
- Nhiệt độ + throttle qua soak 45 phút.
- LLM-as-judge so 100 prompt VN/EN (chất lượng thực dụng ngoài benchmark).
- ROUGE trên mẫu có gold answer.

---

## 8. Rủi ro & cách xử lý

| # | Rủi ro | Khả năng | Xử lý |
|---|---|---|---|
| 8.1 | `QwenSpinQuant` lỗi Hadamard non-power-of-2 với Qwen2.5-7B: `intermediate_size=18944 = 128×148` (cần Hadamard cấp 148) **và** `hidden_size=3584 = 512×7` | **Cao** | Fork đã xử lý một số trường hợp (0.5B có 4864=256×19); nếu 7B vẫn lỗi → dùng Llama-3.1-8B (SpinQuant gốc hỗ trợ đầy đủ) hoặc thử AIMET R1/R2 |
| 8.2 | W4A8 không chạy NPU RK3588 (đã xác nhận bằng tài liệu) | **Chắc chắn** | Chốt Hướng A với sếp; ghi rõ trong báo cáo; accuracy study trên host |
| 8.3 | NPU W4A4 (rk-llama.cpp) tụt chất lượng mạnh | Cao | Quality gate: nếu Δ vượt ngưỡng → không dùng NPU cho LLM; dùng CPU GGUF |
| 8.4 | 7B Vm trên board OOM (OS + artifact + KV) | TB | Q4_K_M ~4.7GB + KV 0.5GB + OS ~1GB ≈ 6.2–7GB; giảm `-c` 2048; dùng zram/swap chống OOM |
| 8.5 | Kaggle GPU T4 không đủ VRAM/time cho SpinQuant 7B (12h/session) | TB | `--w_rtn` (RTN) giảm bộ nhớ; 2×T4; hoặc thuê A100 (vài giờ); checkpoint ra HF |
| 8.6 | lm-eval hỗ trợ GGUF/W4A8 checkpoint hạn chế | TB | FP16 + GGUF Q8/Q4 qua llama-cpp-python; W4A8 fake-quant chạy eval script của QwenSpinQuant (PPL) + lm-eval qua HF custom wrapper |
| 8.7 | VMLU không có task sẵn trong lm-eval | TB | Dùng runner riêng `eval_vmlu.py` (ZaloAI) hoặc dataset HF `tridm/VMLU` + harness tối giản |
| 8.8 | Đo công suất thiếu thiết bị/không đồng bộ | TB | Mua USB-C PD meter (~300–700k VND); hoặc INA219/INA226 trên I2C; ghi V từ ADC làm tham chiếu |
| 8.9 | Driver 0.9.8 / toolkit-runtime mismatch | Thấp | Ghim `release-v1.2.3` cả hai đầu; encode release vào tên file + log |
| 8.10 | So sánh PPL không cùng tool giữa các biến thể | TB | Chuẩn hóa: mọi PPL dùng llama.cpp GGUF cùng corpus/chunks; W4A8 ghi rõ là tool khác và không so trực tiếp tuyệt đối |
| 8.11 | **License SpinQuant/QwenSpinQuant là CC-BY-NC 4.0 (phi thương mại)** | **Cao** | Xin ý kiến legal trước khi dùng; nếu không được duyệt → QuaRot/AIMET/LLMC (kiểm tra license) hoặc bỏ SpinQuant, chỉ PTQ W4A8 bằng GPTQ/RTN. Kiểm tra cả license VMLU và model. |
| 8.12 | Model 7B 4-bit (~4.3GB) vượt **giới hạn 4GB/IOMMU domain** của NPU | Cao | Test `RKNPU_DOMAINS` ở Phase 2.5; nếu không load được → NPU 7B bất khả thi, dùng CPU hoặc Hướng C. Tránh chạy đa process song song trên NPU (nguy cơ kernel panic). |
| 8.13 | Lệch phiên bản RKLLM toolkit/runtime/driver (v1.3.1 vs driver 0.9.8 của v1.2.3) | TB | Đối chiếu `rknpu-driver` từng release trước khi chọn; giữ toolkit/runtime/driver cùng release; ghi version vào tên file + log |

---

## 9. Timeline đề xuất (~9–11,5 ngày làm việc)

| Mốc | Việc | Thời gian |
|---|---|---|
| Phase 0–1 | Nền tảng + scaffold repo (mock chạy được) | 1 ngày |
| Phase 2 + 2.5 | Môi trường + **feasibility spike, chốt Hướng A/B/C** | 1.5 ngày |
| Phase 3 | Baseline FP16 đủ 7 quality metrics | 1–1.5 ngày |
| Phase 4 | Calibration data | 0.5 ngày |
| Phase 5 | SpinQuant W4A8 + eval W4A8 + export GGUF | 2–3 ngày |
| Phase 6 | Deploy board (CPU + NPU nếu có) | 1 ngày |
| Phase 7 | Performance benchmark (kể cả công suất) | 1–1.5 ngày |
| Phase 8–9 | So sánh + báo cáo + serving/soak | 1–1.5 ngày |

---

## 10. Deliverables

- [ ] Báo cáo **feasibility** (mục 1) với bằng chứng thực nghiệm từ Phase 2.5 — trình sếp chốt hướng
- [ ] Model đã chọn + baseline FP16: đủ **7 quality metrics** + JSON
- [ ] Artifact `SpinQuant W4A8` (checkpoint + log) + điểm **7 quality metrics** của W4A8
- [ ] Artifact triển khai trên board: `Q4_K_M.gguf` (CPU) và/hoặc `Q4_0.gguf` (NPU rk-llama.cpp)
- [ ] Bảng **7 performance metrics** (size, peak RAM, load time, TTFT, prefill, decode, power) cho từng artifact — median + p95, ≥5 runs
- [ ] `BAO_CAO_SO_SANH.md` đối chiếu baseline vs quantized trên đúng 2 nhóm tiêu chí họp
- [ ] `power_monitor.py` + dữ liệu CSV công suất (idle/prefill/decode)
- [ ] API server + systemd + hướng dẫn vận hành
- [ ] File lock phiên bản cho mọi môi trường (Kaggle, board)
- [ ] README đầy đủ

---

## 11. Câu hỏi cần chốt (với sếp) trước Phase 5

1. **Chấp nhận Hướng A** (W4A8/SpinQuant là nghiên cứu trên host; deploy 4-bit trên board bằng CPU GGUF; NPU chỉ nếu W4A4 đạt quality gate) — hay ưu tiên NPU bằng mọi giá (Hướng B), hay đổi model ≤3B (Hướng C)?
2. Nếu W4A8 không deploy được lên NPU, báo cáo chỉ cần **số liệu chất lượng W4A8 trên host** + **số liệu hiệu năng 4-bit trên board** — có đáp ứng kỳ vọng không?
3. Chấp nhận **W4A4 trên NPU** (khác W4A8) nếu W4A4 đạt ngưỡng chất lượng?
4. **Ngân sách compute** cho Kaggle/GPU thuê (nếu T4 không đủ cho SpinQuant 7B) — có được phép thuê A100 vài giờ (~$2–4/h) không?
5. Ngưỡng chất lượng nghiệm thu (mục 7.1) có cần siết/nới không?
6. Mốc deadline cụ thể để chốt phạm vi (đủ 7 metrics vs subset)?
7. **Legal xác nhận license SpinQuant/QwenSpinQuant (CC-BY-NC 4.0)** — nếu không được dùng cho sản phẩm, chọn QuaRot/AIMET/LLMC hay bỏ bước SpinQuant?
8. Nếu NPU không load nổi 7B vì giới hạn 4GB/IOMMU domain, chấp nhận chạy CPU cho LLM hay giảm model (Hướng C)?
9. Chấp nhận **nâng driver NPU** để dùng RKLLM v1.3.1, hay giữ driver 0.9.8 + pin `release-v1.2.3`?

---

## 12. Nguồn tham khảo

- SpinQuant (paper): https://arxiv.org/abs/2405.16406 — code: https://github.com/facebookresearch/SpinQuant
- QwenSpinQuant (fork hỗ trợ Qwen2/Qwen2.5, tối ưu W4A8) — **license CC-BY-NC 4.0, phi thương mại**: https://github.com/shijiew/QwenSpinQuant (issue: https://github.com/facebookresearch/SpinQuant/issues/33)
- W4A8 paper được dẫn (LiquidGEMM/LiquidQuant — kernel CUDA, **không dùng cho NPU Rockchip**): https://arxiv.org/abs/2509.01229
- AIMET SpinQuant (R1/R2 Hadamard, fuse rotation): https://qualcomm.github.io/aimet-pages/releases/latest/ptq_techniques/spinquant.html
- RKLLM SDK 1.2.3 — danh sách dtype theo platform (RK3588 chỉ w8a8*): tài liệu Rockchip (bản dịch tham khảo https://blog.armsom.org/articles/096-rockchip-rkllm-sdk/); repo (bản mới nhất v1.3.1): https://github.com/airockchip/rknn-llm
- RKLLM — phương pháp đo hiệu năng chính thức (`RKLLM_LOG_LEVEL=1`, `eval_perf_watch_cpu.sh`, `eval_perf_watch_npu.sh`): https://github.com/airockchip/rknn-llm#performance-testing-methods
- RKLLM giới hạn W4A16 trên RK3588: https://github.com/airockchip/rknn-llm/issues/186 , https://github.com/airockchip/rknn-llm/issues/535
- rk-llama.cpp — NPU RK3588 backend, bảng pipeline W4A4/W8A8 (không có W4A8), mapping GGUF mặc định, build `-DLLAMA_RKNPU2=ON`, giới hạn **4GB/IOMMU domain**: https://github.com/invisiofficial/rk-llama.cpp/blob/rknpu2/ggml/src/ggml-rknpu2/README.md
- RKNN-Toolkit2 CHANGELOG (W4A16 chỉ RK3576; w4a8/w4a16 MatMul chỉ RK3562): https://github.com/airockchip/rknn-toolkit2/blob/master/CHANGELOG.md
- VMLU (ZaloAI) — 58 môn, 10.880 câu: https://github.com/ZaloAI-Jaist/VMLU , https://vmlu.ai/ , dataset: https://huggingface.co/datasets/tridm/VMLU
- NanoPi R6C specs (RK3588S, USB-C PD 5–20V, ADC kênh 2 đọc điện áp USB-C): https://www.friendlyelec.com/index.php?product_id=291 , https://wiki.friendlyelec.com/wiki/index.php/NanoPi_R6C
- Turing Pi — benchmark llama.cpp RK3588 (Qwen2.5-7B Q4_K_M: 5.44 tok/s, peak RSS): https://turingpi.com/llm-inference-benchmarks-rk3588-gguf-quantization/
- Repo hiện có: `../edge_ai_r6c` (`rkllm_bindings.py`, `export_slm.py`, `benchmark.py`, `deploy_board.sh`, `KE_HOACH_QUANTIZE_LLM_NANOPI_R6C.md`)

*Cập nhật lần cuối: 2026-11-12*
