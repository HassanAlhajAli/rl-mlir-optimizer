"""Read-only integrity audit of a completed development deployment comparison."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.comparison import interleaved_order
from rlmlir.corpus import FrozenCorpus
from rlmlir.kernels import GENERATORS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(directory):
    meta = json.loads((directory / "run.json").read_text())
    require(meta["status"] == "complete", "comparison is not complete")
    config_bytes = (directory / "config.json").read_bytes()
    require(hashlib.sha256(config_bytes).hexdigest() == meta["config_sha256"], "configuration hash mismatch")
    config = json.loads(config_bytes)
    require(config == meta["config"], "configuration changed")
    corpus = FrozenCorpus()
    require(corpus.sha256 == meta["corpus_sha256"], "corpus mismatch")
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    starts = [e for e in events if e["status"] == "started"]
    records = [e for e in events if e["status"] == "accepted"]
    require(len(events) == 2 * len(meta["order"]) == 2 * len(records) == 2 * len(starts), "incomplete case coverage")
    workloads = {e["workload_id"]: e["workload"] for e in records}
    methods = ["reference", "heuristic", *config["policies"]]
    if config["predictor"] is not None:
        methods.append("predictor")
    if config.get("best_fixed") is not None:
        methods.append("best_fixed")
    expected = interleaved_order([workloads[i] for i in config["workload_ids"]], methods,
                                 config["reuse_counts"], config["rounds"], config["order_seed"])
    require(meta["order"] == expected, "seeded order mismatch")
    counters = Counter(meta["initialization_counts"])
    flagged = unresolved = deployments = 0
    for index, case in enumerate(expected):
        start, record = events[2 * index:2 * index + 2]
        require(start["status"] == "started" and record["status"] == "accepted", "bad event ordering")
        require(start["case_index"] == record["case_index"] == index, "case identity mismatch")
        require(all(start[k] == record[k] == v for k, v in case.items()), "grid identity mismatch")
        corpus.require_development(record["workload"])
        require(start["workload"] == record["workload"], "workload changed during case")
        require(all(record[k] == meta[k] for k in ("revision", "binary_hash", "corpus_sha256", "comparison_protocol")), "provenance mismatch")
        require(record["correct"] is True and not record["cleanup_errors"], "incorrect case or failed cleanup")
        if record["method"] == "best_fixed":
            require(record["selection"]["schedule"] == meta["artifacts"]["best_fixed"]["selected"]["schedule"],
                    "best-fixed schedule changed across deployments")
        require(record["reuse_hint"] == record["reuse"] * config["hint_factor"], "reuse hint mismatch")
        require(math.isfinite(record["selection_ms"]) and record["selection_ms"] >= 0, "invalid selection cost")
        with gzip.open(directory / f"case-{index:05d}-ir.json.gz", "rt") as stream:
            ir = json.load(stream)
        require(all(hashlib.sha256(ir[k].encode()).hexdigest() == v for k, v in record["ir_hashes"].items()), "IR hash mismatch")
        workload = record["workload"]
        require(ir["source"] == GENERATORS[workload["family"]](*workload["shape"]), "source differs from workload")
        attempts = record["attempts"]
        require(1 <= len(attempts) <= 2, "repeat limit violated")
        require(len(attempts) == (2 if attempts[0]["flagged"] else 1), "unexpected or missing drift repeat")
        calls = 8 + sum(s["count"] for field in ("calibration", "anchor_calibration") for s in record[field])
        for attempt, sample in enumerate(attempts):
            require(sample["attempt"] == attempt and sample["status"] == "accepted", "incomplete attempt")
            deployment = sample["deployment"]
            require(deployment["correct"] is True and deployment["cache_hit"] is False
                    and deployment["count"] == record["reuse"], "deployment is not fresh, correct and complete")
            require(all(math.isfinite(deployment[k]) and deployment[k] >= 0 for k in
                        ("compile_ms", "installation_ms", "execution_ms", "total_ms")), "bad deployment cost")
            parts = sum(deployment[k] for k in ("compile_ms", "installation_ms", "execution_ms"))
            require(math.isclose(parts, deployment["total_ms"], rel_tol=1e-9, abs_tol=1e-6), "deployment sum mismatch")
            require(math.isclose(sample["total_with_selection_ms"], record["selection_ms"] + deployment["total_ms"], rel_tol=1e-9), "selection cost omitted")
            drift = abs(sample["anchor_after_ms"] / sample["anchor_before_ms"] - 1)
            is_flagged = drift > .05 or min(sample["anchor_before_ms"], sample["anchor_after_ms"], sample["steady_batch_ms"]) < 100
            require(math.isclose(drift, sample["drift"]) and sample["flagged"] == is_flagged, "drift flag mismatch")
            calls += deployment["count"] + 2 * sample["anchor_count"] + sample["steady_count"]
        require(record["unresolved_drift"] == attempts[-1]["flagged"], "unresolved flag mismatch")
        require(record["hardware_counts"]["deploy_requests"] == len(attempts), "deployment count mismatch")
        require(record["hardware_counts"]["actual_invocations"] == calls, "invocation accounting mismatch")
        require(record["hardware_counts"]["release_requests"] == 2, "missing handle release")
        counters.update(record["hardware_counts"])
        flagged += attempts[0]["flagged"]
        unresolved += record["unresolved_drift"]
        deployments += len(attempts)
    require(dict(counters) == meta["worker_counts"], "worker totals differ from initialization plus cases")
    require(meta["totals"] == {"started": len(records), "accepted": len(records), "failed": 0,
                               "unresolved_drift": unresolved}, "summary counter mismatch")
    return {"status": "passed", "scope": "saved evidence; no kernel execution", "revision": meta["revision"],
            "binary_hash": meta["binary_hash"], "corpus_sha256": corpus.sha256, "cases": len(records),
            "fresh_deployments": deployments, "original_flagged_blocks": flagged, "unresolved_drift": unresolved,
            "worker_stage_elapsed_seconds": meta["worker_stage_elapsed_seconds"], "worker_counts": dict(counters),
            "events_sha256": hashlib.sha256((directory / "events.jsonl").read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
