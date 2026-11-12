# Corpus

Đặt file WikiText-2 raw validation tại đây với tên:

```
data/corpus/wikitext-2-raw-val.txt
```

Ví dụ (trên host có mạng):

```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
with open("data/corpus/wikitext-2-raw-val.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(ds["text"]))
PY
```

Sau đó chạy `bash scripts/eval_ppl.sh --model <model.gguf>`.
Không commit corpus/dataset có bản quyền vào repo.
