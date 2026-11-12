# Evaluation assets

- `prompts_vi.jsonl`, `prompts_en.jsonl`: prompt thật cho `bench_board.py --prompts-file`
  và kiểm tra định tính. Mỗi dòng có `id`, `category`, `max_new_tokens`, `prompt`.
- `gold/`: đáp án tham chiếu / spot-check thủ công (không commit dữ liệu có bản quyền).
- `vmlu/`: đặt dataset VMLU đã tải (JSON/JSONL) tại đây nếu không dùng HuggingFace id.

## VMLU

```bash
python scripts/eval_vmlu.py --dataset eval/vmlu/vmlu.json --limit 500
python scripts/eval_vmlu.py --dataset tridm/VMLU --split validation --limit 200
```

Loader chấp nhận các cột linh hoạt: `question|prompt|query`, `choices|options`,
`answer|label|target`, `subject|category`. Không có dataset thì script dừng với hướng dẫn.

## Chất lượng

- `eval_quality.py` gọi `lm-eval` cho MMLU/ARC-C/HellaSwag/GSM8K/IFEval.
- `eval_ppl.sh` gọi `llama-perplexity` trên WikiText-2.
- So sánh baseline FP16 với 4-bit theo ngưỡng trong README/plan.
