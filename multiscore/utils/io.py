from __future__ import annotations

import gzip
import json
import os
from typing import Any, Dict, Iterable, Iterator


def _open(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Stream a (optionally gzipped) JSONL file, skipping blank lines."""

    with _open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    count = 0
    with _open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: str, payload: Any, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False, sort_keys=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path
