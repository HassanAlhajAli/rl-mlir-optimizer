"""Bounded training-only evidence for the best fixed schedule; never auto-retry."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.evaluator import Evaluator
from rlmlir.experiment import initialize, measurement_lease, provenance
from rlmlir.fixed_baseline import ROLE, collection_order, prepare
from rlmlir.worker import Worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    workloads = {w["id"]: w for w in prepare(config)}
    order = collection_order(config)
    if args.check_only:
        print(json.dumps({"role": "configuration check; no compiler or measurements",
                          "workloads": len(workloads), "candidates": len(config["candidates"]),
                          "measurements": len(order), "heldout_access": False}))
        return
    if args.output is None or args.output.exists() or not math.isfinite(args.hours) or args.hours <= 0:
        raise ValueError("new output and positive finite wall-time limit required")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    meta = {**provenance(ROOT), "status": "running", "role": ROLE, "config": config,
            "config_sha256": hashlib.sha256(raw).hexdigest(), "order": order,
            "command": sys.argv, "controller_pid": os.getpid(), "wall_limit_seconds": args.hours * 3600}
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_bytes(raw)
    def save():
        (args.output / "run.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    def emit(row):
        with (args.output / "trials.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    save()
    began = time.perf_counter()
    worker, evaluator = None, None
    schedules = {c["name"]: c["schedule"] for c in config["candidates"]}
    try:
        with Worker(ROOT / "build/rlmlir-worker") as worker:
            meta.update(worker_pid=worker.process.pid, **initialize(worker))
            save()
            remaining = args.hours * 3600 - (time.perf_counter() - began)
            if remaining <= 0:
                raise TimeoutError("initialization exhausted fixed-grid time budget")
            evaluator = Evaluator(worker, args.output / "measurements", meta["binary_hash"], meta["revision"],
                                  max_new=len(order), max_requests=len(order), max_seconds=remaining)
            for index, case in enumerate(order):
                row = {**case, "index": index, "status": "started"}
                emit(row)
                start = time.perf_counter()
                schedule = copy.deepcopy(schedules[case["candidate"]])
                selection_ms = (time.perf_counter() - start) * 1000
                outcome = evaluator.measure(workloads[case["workload_id"]], schedule)
                emit({**row, "status": "accepted", "selection_ms": selection_ms, "measurement_key": outcome["key"]})
            meta["status"] = "complete"
    except BaseException as error:
        meta.update(status="stopped", reason=repr(error))
        raise
    finally:
        if worker is not None:
            meta["worker_counts"] = dict(worker.counters)
        if evaluator is not None:
            meta["counters"] = evaluator.counters()
        meta["elapsed_seconds"] = time.perf_counter() - began
        save()


if __name__ == "__main__":
    if "--check-only" in sys.argv:
        main()
    else:
        with measurement_lease(ROOT):
            main()
