#!/usr/bin/env python3
"""Tải và chuẩn hóa bộ dataset VMLU cho scripts/eval_vmlu.py.

Nguồn: mirror công khai ``tridm/VMLU`` trên Hugging Face (bản sao của bộ
chính thức tại https://github.com/ZaloAI-Jaist/VMLU / https://vmlu.ai).

Lưu ý quan trọng:
  * ``test.jsonl`` (9,833 câu) KHÔNG có cột ``answer`` -> không dùng để eval.
  * ``valid.jsonl`` (744 câu) có đáp án A-D -> dùng mặc định ở đây.
  * ``train.jsonl`` (303 câu) có đáp án A-E (5 lựa chọn) -> eval_vmlu.py chỉ
    trích A-D nên không khuyến khích.
  * Ở bản gốc, mỗi ``choices[i]`` đã có prefix "A. ", "B. "... Trong khi
    ``format_mcq()`` của eval_vmlu.py tự thêm "A. ", "B. "... nên phải bỏ
    prefix cũ đi, nếu không prompt sẽ bị "A. A. ...".
  * Trường ``subject`` được tạo từ 2 chữ số đầu của ``id`` theo bảng 58 môn
    của VMLU, giúp báo cáo theo từng môn có ý nghĩa.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_BASE = "https://huggingface.co/datasets/tridm/VMLU/resolve/main"

# id -> (tên môn, nhóm), lấy từ README chính thức của VMLU
SUBJECTS = {
    "01": ("Elementary Mathematics", "STEM"),
    "02": ("Elementary Science", "STEM"),
    "03": ("Middle School Biology", "STEM"),
    "04": ("Middle School Chemistry", "STEM"),
    "05": ("Middle School Mathematics", "STEM"),
    "06": ("Middle School Physics", "STEM"),
    "07": ("High School Biology", "STEM"),
    "08": ("High School Chemistry", "STEM"),
    "09": ("High School Mathematics", "STEM"),
    "10": ("High School Physics", "STEM"),
    "11": ("Applied Informatics", "STEM"),
    "12": ("Computer Architecture", "STEM"),
    "13": ("Computer Network", "STEM"),
    "14": ("Discrete Mathematics", "STEM"),
    "15": ("Electrical Engineering", "STEM"),
    "16": ("Introduction to Chemistry", "STEM"),
    "17": ("Introduction to Physics", "STEM"),
    "18": ("Introduction to Programming", "STEM"),
    "19": ("Metrology Engineer", "STEM"),
    "20": ("Operating System", "STEM"),
    "21": ("Statistics and Probability", "STEM"),
    "22": ("Middle School Civil Education", "Social Science"),
    "23": ("Middle School Geography", "Social Science"),
    "24": ("High School Civil Education", "Social Science"),
    "25": ("High School Geography", "Social Science"),
    "26": ("Business Administration", "Social Science"),
    "27": ("Ho Chi Minh Ideology", "Social Science"),
    "28": ("Macroeconomics", "Social Science"),
    "29": ("Microeconomics", "Social Science"),
    "30": ("Principles of Marxism and Leninism", "Social Science"),
    "31": ("Sociology", "Social Science"),
    "32": ("Elementary History", "Humanity"),
    "33": ("Middle School History", "Humanity"),
    "34": ("Middle School Literature", "Humanity"),
    "35": ("High School History", "Humanity"),
    "36": ("High School Literature", "Humanity"),
    "37": ("Administrative Law", "Humanity"),
    "38": ("Business Law", "Humanity"),
    "39": ("Civil Law", "Humanity"),
    "40": ("Criminal Law", "Humanity"),
    "41": ("Economic Law", "Humanity"),
    "42": ("Education Law", "Humanity"),
    "43": ("History of World Civilization", "Humanity"),
    "44": ("Idealogical and Moral Cultivation", "Humanity"),
    "45": ("Introduction to Laws", "Humanity"),
    "46": ("Introduction to Vietnam Culture", "Humanity"),
    "47": ("Logic", "Humanity"),
    "48": ("Revolutionary Policy of the Vietnamese Communist Party", "Humanity"),
    "49": ("Vietnamese Language and Literature", "Humanity"),
    "50": ("Accountant", "Other"),
    "51": ("Clinical Pharmacology", "Other"),
    "52": ("Environmental Engineering", "Other"),
    "53": ("Internal Basic Medicine", "Other"),
    "54": ("Preschool Pedagogy", "Other"),
    "55": ("Tax Accountant", "Other"),
    "56": ("Tax Civil Servant", "Other"),
    "57": ("Civil Servant", "Other"),
    "58": ("Driving License Certificate", "Other"),
}

PREFIX_RE = re.compile(r"^\s*[A-Ea-e][.)]\s+")


def strip_prefix(text: str) -> str:
    """Bỏ prefix 'A. ', 'B)' ... khỏi một lựa chọn."""
    return PREFIX_RE.sub("", str(text)).strip()


def download(split: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{split}.jsonl")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"dùng lại file đã có: {dest}")
        return dest
    url = f"{HF_BASE}/{split}.jsonl"
    print(f"đang tải {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"đã lưu: {dest} ({os.path.getsize(dest)} bytes)")
    return dest


def convert(src: str, out: str) -> int:
    rows = []
    with open(src, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    converted = []
    skipped = 0
    for row in rows:
        if "answer" not in row:
            skipped += 1
            continue  # test set không có đáp án
        qid = str(row.get("id", ""))
        sid = qid.split("-")[0]
        name, category = SUBJECTS.get(sid, (f"subject_{sid}", "unknown"))
        choices = [strip_prefix(c) for c in row.get("choices", [])]
        answer = str(row.get("answer", "")).strip().upper()
        # Chỉ giữ câu hỏi 3-4 lựa chọn với đáp án nằm trong A-D (eval_vmlu.py
        # extract_choice() chỉ nhận A-D).
        if not row.get("question") or answer not in "ABCD" or not 2 <= len(choices) <= 4:
            skipped += 1
            continue
        converted.append(
            {
                "id": qid,
                "question": str(row["question"]),
                "choices": choices,
                "answer": answer,
                "subject": f"{sid}_{name}".replace(" ", "_"),
                "category": category,
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for item in converted:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_subjects = len({item["subject"] for item in converted})
    print(f"đã ghi {len(converted)} câu ({n_subjects} môn) -> {out}")
    if skipped:
        print(f"bỏ qua {skipped} dòng (không có đáp án / không hợp lệ)")
    if not converted:
        print("ERROR: không có câu nào hợp lệ để eval", file=sys.stderr)
        return 2
    sample = converted[0]
    print("mẫu đầu tiên:")
    print(json.dumps(sample, ensure_ascii=False, indent=2)[:600])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tải + chuẩn hóa VMLU cho eval_vmlu.py")
    parser.add_argument("--split", default="valid", choices=["valid", "train", "test"])
    parser.add_argument("--raw-dir", default=os.path.join(ROOT, "data", "vmlu"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--force", action="store_true", help="tải lại ngay cả khi file đã có")
    args = parser.parse_args()

    out = args.out or os.path.join(ROOT, "data", "vmlu", f"vmlu_{args.split}.jsonl")
    raw_dir = os.path.abspath(args.raw_dir)
    raw = os.path.join(raw_dir, f"{args.split}.jsonl")
    if args.force and os.path.exists(raw):
        os.remove(raw)
    src = download(args.split, raw_dir)
    return convert(src, os.path.abspath(out))


if __name__ == "__main__":
    sys.exit(main())
