#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

TEMPLATES: List[Dict[str, str]] = [
    {
        "input": "<|im_start|>system\nBạn là trợ lý AI tiếng Việt.<|im_end|>\n<|im_start|>user\nXin chào, bạn là ai?<|im_end|>\n<|im_start|>assistant\n",
        "target": "Tôi là trợ lý AI chạy cục bộ trên NanoPi-R6C, hỗ trợ trả lời câu hỏi bằng tiếng Việt.",
    },
    {
        "input": "<|im_start|>system\nBạn là trợ lý AI tiếng Việt.<|im_end|>\n<|im_start|>user\nTóm tắt đoạn văn sau: NanoPi-R6C dùng SoC RK3588S, NPU 6 TOPS và 8 GB RAM. Thiết bị phù hợp cho suy luận AI tại biên.<|im_end|>\n<|im_start|>assistant\n",
        "target": "NanoPi-R6C là thiết bị biên dùng RK3588S, NPU 6 TOPS, RAM 8 GB, phù hợp suy luận AI cục bộ.",
    },
    {
        "input": "<|im_start|>system\nBạn là trợ lý AI tiếng Việt.<|im_end|>\n<|im_start|>user\nGiải thích ngắn gọn lượng tử hoá W8A8 là gì.<|im_end|>\n<|im_start|>assistant\n",
        "target": "W8A8 là lượng tử hoá trọng số và kích hoạt xuống 8-bit, giúp giảm bộ nhớ và tăng tốc suy luận với ít mất mát chất lượng.",
    },
    {
        "input": "<|im_start|>system\nBạn là trợ lý AI tiếng Việt.<|im_end|>\n<|im_start|>user\nViết cảnh báo an ninh khi phát hiện 3 người xâm nhập vùng cấm A trong 12 giây.<|im_end|>\n<|im_start|>assistant\n",
        "target": "CẢNH BÁO: Phát hiện 3 người xâm nhập vùng cấm A trong 12 giây. Đề nghị kiểm tra ngay.",
    },
    {
        "input": "<|im_start|>system\nBạn là trợ lý AI tiếng Việt.<|im_end|>\n<|im_start|>user\nTTFT và decode speed khác nhau thế nào?<|im_end|>\n<|im_start|>assistant\n",
        "target": "TTFT là thời gian tới token đầu tiên, còn decode speed là số token sinh ra mỗi giây sau token đầu.",
    },
    {
        "input": "<|im_start|>system\nBạn là trợ lý AI tiếng Việt.<|im_end|>\n<|im_start|>user\nVì sao cần đánh giá cả chất lượng và hiệu năng khi lượng tử hoá?<|im_end|>\n<|im_start|>assistant\n",
        "target": "Lượng tử hoá có thể giảm độ chính xác, nên cần đo chất lượng (PPL, MMLU, VMLU) và hiệu năng (TTFT, decode, RAM, công suất) để chọn cấu hình cân bằng.",
    },
    {
        "input": "<|im_start|>system\nBạn là trợ lý AI tiếng Việt.<|im_end|>\n<|im_start|>user\nLiệt kê 3 chỉ số hiệu năng quan trọng khi chạy LLM trên thiết bị biên.<|im_end|>\n<|im_start|>assistant\n",
        "target": "Ba chỉ số quan trọng là TTFT, tốc độ decode (token/giây) và mức tiêu thụ bộ nhớ RAM.",
    },
    {
        "input": "<|im_start|>system\nBạn là trợ lý AI tiếng Việt.<|im_end|>\n<|im_start|>user\nHãy viết một câu trả lời lịch sự từ chối yêu cầu ngoài phạm vi.<|im_end|>\n<|im_start|>assistant\n",
        "target": "Xin lỗi, yêu cầu này nằm ngoài phạm vi hỗ trợ của tôi. Bạn có thể hỏi về tài liệu hoặc chức năng của hệ thống.",
    },
    {
        "input": "<|im_start|>system\nBạn là trợ lý AI tiếng Việt.<|im_end|>\n<|im_start|>user\nDịch sang tiếng Anh: Thiết bị đang chạy ổn định.<|im_end|>\n<|im_start|>assistant\n",
        "target": "The device is running stably.",
    },
    {
        "input": "<|im_start|>system\nBạn là trợ lý AI tiếng Việt.<|im_end|>\n<|im_start|>user\nTính 12 nhân 7 bằng bao nhiêu?<|im_end|>\n<|im_start|>assistant\n",
        "target": "12 nhân 7 bằng 84.",
    },
]


def load_extra(path: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not path or not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and ("input" in obj or "prompt" in obj):
                    items.append(
                        {
                            "input": str(obj.get("input", obj.get("prompt", ""))),
                            "target": str(obj.get("target", obj.get("response", ""))),
                        }
                    )
            except json.JSONDecodeError:
                items.append({"input": line, "target": ""})
    return items


def load_corpus(path: str, min_chars: int = 120, max_chars: int = 1200) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    if not path or not os.path.exists(path):
        return samples
    with open(path, "r", encoding="utf-8") as fh:
        buffer = []
        size = 0
        for line in fh:
            line = line.strip()
            if not line:
                continue
            buffer.append(line)
            size += len(line)
            if size >= min_chars:
                text = " ".join(buffer)[:max_chars]
                samples.append(
                    {
                        "input": f"<|im_start|>user\nTóm tắt đoạn văn sau:\n{text}<|im_end|>\n<|im_start|>assistant\n",
                        "target": text[:200],
                    }
                )
                buffer, size = [], 0
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate calibration JSON for RKLLM quantization")
    parser.add_argument("--output", default="data/calibration.json")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--extra-file", default="", help="optional JSONL or plain-text samples")
    parser.add_argument("--corpus", default="", help="optional plain-text file to chunk into summaries")
    args = parser.parse_args()

    samples: List[Dict[str, str]] = []
    samples.extend(load_extra(args.extra_file))
    samples.extend(load_corpus(args.corpus))
    while len(samples) < args.count:
        samples.append(TEMPLATES[len(samples) % len(TEMPLATES)])
    samples = samples[: max(1, args.count)]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(samples, fh, ensure_ascii=False, indent=2)
    print(f"OK: wrote {len(samples)} samples to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
