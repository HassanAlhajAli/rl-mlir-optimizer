"""Manifest integrity and split protection at the measurement boundary."""
import hashlib
import json
from pathlib import Path


def load(path: Path, split: str):
    if split not in ("train", "validation"):
        raise ValueError("held-out performance is sealed until the final-method freeze")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("frozen corpus hash mismatch")
    workloads = json.loads(raw)["workloads"]
    groups = {}
    for item in workloads:
        previous = groups.setdefault(item["group"], item["split"])
        if previous != item["split"]:
            raise ValueError("related shape groups crossed splits")
    return [item for item in workloads if item["split"] == split]
