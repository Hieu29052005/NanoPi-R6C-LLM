#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

MODELS = {
    "qwen2.5-7b": {
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "license": "Apache-2.0",
        "note": "Khuyen nghi - Recommend. ChatML, tieng Viet tot, 7B.",
    },
    "llama-3.1-8b": {
        "repo": "meta-llama/Llama-3.1-8B-Instruct",
        "license": "Llama 3.1 Community License (gated)",
        "note": "Can chap nhan dieu khoan tren HuggingFace; SpinQuant ho tro day du.",
    },
    "deepseek-r1-distill-7b": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "license": "MIT (model card) - kiem tra lai truoc khi dung",
        "note": "Kien truc Qwen2, CoT dai co the an het token budget 2k-4k.",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download candidate models from HuggingFace")
    parser.add_argument("--list", action="store_true", help="list candidates and exit")
    parser.add_argument("--model", default="qwen2.5-7b", help="candidate key or full repo id")
    parser.add_argument("--out", default="models", help="destination directory")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--patterns", default="", help="comma-separated allow_patterns")
    args = parser.parse_args()

    if args.list:
        for key, meta in MODELS.items():
            print(f"{key:<26} {meta['repo']:<48} {meta['license']}")
            print(f"{'':<26} {meta['note']}")
        return 0

    entry = MODELS.get(args.model)
    repo = entry["repo"] if entry else args.model
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install -r requirements-host.txt")
        return 2

    if entry:
        print(f"model:   {repo}")
        print(f"license: {entry['license']}")
        print(f"note:    {entry['note']}")
    allow = [item.strip() for item in args.patterns.split(",") if item.strip()] or None
    print(f"target:  {out}")
    path = snapshot_download(
        repo_id=repo,
        revision=args.revision,
        local_dir=os.path.join(out, os.path.basename(repo)),
        allow_patterns=allow,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
