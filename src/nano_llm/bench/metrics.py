from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def percentile(values: Iterable[float], p: float) -> float:
    data = sorted(float(value) for value in values)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    p = min(max(p, 0.0), 100.0)
    rank = (p / 100.0) * (len(data) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return data[low]
    weight = rank - low
    return data[low] * (1.0 - weight) + data[high] * weight


def summarize(values: Iterable[float]) -> Dict[str, float]:
    data = [float(value) for value in values if value is not None]
    if not data:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(data),
        "mean": sum(data) / len(data),
        "median": percentile(data, 50),
        "p95": percentile(data, 95),
        "min": min(data),
        "max": max(data),
    }


def write_json(path: str, payload: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    target = Path(path)
    if not target.exists():
        return items
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                items.append({"prompt": line})
    return items


def find_latest_results(directory: str, pattern: str = "results*.json") -> Optional[str]:
    target = Path(directory)
    if not target.exists():
        return None
    candidates = sorted(
        target.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True
    )
    return str(candidates[0]) if candidates else None
