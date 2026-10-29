"""Audit a complete fixed training grid and select one schedule, without execution."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.corpus import FrozenCorpus
from rlmlir.experiment import measurement_lease, provenance, require_compatible_provenance
from rlmlir.fixed_baseline import ROLE, PROTOCOL, collection_order, digest, prepare, select
from rlmlir.kernels import GENERATORS


def audit_timing(record):
    samples = record["samples"]
    if not 1 <= len(samples) <= 2 or len(samples) != (2 if samples[0]["flagged"] else 1):
        raise ValueError("fixed-grid timing repeat mismatch")
    for index, sample in enumerate(samples):
        if (sample["attempt"] != index or type(sample["count"]) is not int or sample["count"] < 1
                or any(type(sample[k]) not in (int, float) or not math.isfinite(sample[k]) or sample[k] <= 0
                       for k in ("batch_ms", "anchor_before_ms", "anchor_after_ms"))):
            raise ValueError("invalid fixed-grid raw timing")
        drift = abs(sample["anchor_after_ms"] / sample["anchor_before_ms"] - 1)
        flagged = drift > .05 or min(sample[k] for k in ("batch_ms", "anchor_before_ms", "anchor_after_ms")) < 100
        if not math.isclose(drift, sample["drift"]) or sample["flagged"] != flagged:
            raise ValueError("fixed-grid drift flag mismatch")
    if (record["unresolved_drift"] != samples[-1]["flagged"]
            or not math.isclose(record["execution_ms"], statistics.median(s["batch_ms"] / s["count"] for s in samples))):
        raise ValueError("fixed-grid execution summary mismatch")
    errors = record["max_abs_errors"]
    if len(errors) != 2 or any(not math.isfinite(e) or e < 0 for e in errors):
        raise ValueError("missing finite numerical check evidence")


def fit(directory, current):
    """Strict one-to-one join: no silently skipped failures, missing or duplicate rows."""
    corpus = FrozenCorpus()
    meta = json.loads((directory / "run.json").read_text())
    if meta.get("status") != "complete" or meta.get("role") != ROLE:
        raise ValueError("complete fixed-grid collection required")
    require_compatible_provenance(meta, current, context="fixed training measurements")
    if meta["corpus_sha256"] != corpus.sha256:
        raise ValueError("fixed-grid corpus mismatch")
    raw = (directory / "config.json").read_bytes()
    config = json.loads(raw)
    if config != meta["config"] or hashlib.sha256(raw).hexdigest() != meta["config_sha256"]:
        raise ValueError("fixed-grid configuration changed")
    workloads = {w["id"]: w for w in prepare(config, corpus)}
    candidates = {c["name"]: c["schedule"] for c in config["candidates"]}
    order = collection_order(config)
    if meta["order"] != order:
        raise ValueError("fixed-grid order changed")
    trials = [json.loads(line) for line in (directory / "trials.jsonl").read_text().splitlines()]
    measurements = [json.loads(line) for line in (directory / "measurements/measurements.jsonl").read_text().splitlines()]
    if len(trials) != 2 * len(order) or len(measurements) != len(order):
        raise ValueError("incomplete fixed-grid evidence")
    cells = []
    for index, (case, record) in enumerate(zip(order, measurements)):
        started, accepted = trials[index * 2:index * 2 + 2]
        if (started != {**case, "index": index, "status": "started"}
                or any(accepted.get(k) != v for k, v in {**case, "index": index, "status": "accepted"}.items())):
            raise ValueError("fixed-grid trial ordering mismatch")
        workload, schedule = workloads[case["workload_id"]], candidates[case["candidate"]]
        corpus.require_development(record["workload"])
        if (record["workload"] != workload or record["actions"] != schedule
                or record.get("status") != "accepted" or record.get("correct") is not True
                or record["binary_hash"] != current["binary_hash"]
                or record["corpus_sha256"] != corpus.sha256 or record["revision"] != meta["revision"]
                or record.get("cache_seed") or record["input_seeds"] != [7, 17]):
            raise ValueError("incorrect or mismatched fixed-grid measurement")
        source = GENERATORS[workload["family"]](*workload["shape"])
        identity = {"source": source, "actions": schedule, "binary": current["binary_hash"],
                    "input_seed": 7, "measurement_protocol": current["measurement_protocol"]}
        if record["key"] != digest(identity) or accepted["measurement_key"] != record["key"]:
            raise ValueError("fixed-grid measurement key mismatch")
        audit_timing(record)
        with gzip.open(directory / "measurements" / (record["key"] + ".json.gz"), "rt") as stream:
            ir = json.load(stream)
        if (ir["source"] != source or set(ir) != {"source", "transformed_ir", "lowered_ir", "llvm_ir"}
                or {k: hashlib.sha256(v.encode()).hexdigest() for k, v in ir.items()} != record["ir_hashes"]):
            raise ValueError("fixed-grid IR hash mismatch")
        cells.append({**case, **{k: record[k] for k in ("compile_ms", "installation_ms", "execution_ms", "unresolved_drift")},
                      "selection_ms": accepted["selection_ms"], "measurement_key": record["key"]})
    counters = meta["counters"]
    if any(counters[k] != v for k, v in {"candidate_requests": len(order), "new_measurements": len(order),
                                        "cache_hits": 0, "reference_requests": 0}.items()):
        raise ValueError("fixed-grid request accounting mismatch")
    files = ("run.json", "config.json", "trials.jsonl", "measurements/measurements.jsonl")
    return {**current, "schema_version": 1, "status": "complete", "protocol": PROTOCOL,
            "role": "one training-selected schedule; no deployment superiority claim",
            "config": config, "config_digest": digest(config), "cells": cells,
            "selection": select(config, cells, corpus), "source_run": str(directory),
            "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new output path required")
    result = fit(args.run, provenance(ROOT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["selection"], indent=2))


if __name__ == "__main__":
    with measurement_lease(ROOT):
        main()
