"""Replay saved training schedules for correctness, never for performance claims."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.experiment import measurement_lease
from rlmlir.kernels import GENERATORS, REFERENCES
from rlmlir.measurement import validate
from rlmlir.worker import Worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1200)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    records = [json.loads(line) for line in args.records.read_text().splitlines()]
    if any(record["workload"]["split"] != "train" for record in records):
        parser.error("this diagnostic accepts training records only")
    args.output.mkdir(parents=True, exist_ok=False)
    executable = ROOT / "build/rlmlir-worker"
    state = {"purpose": "correctness and process stability, not a performance replay",
             "status": "running", "records": str(args.records), "input_count": len(records),
             "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
             "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
             "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "dirty_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff"], cwd=ROOT)).hexdigest(),
             "completed": 0, "limit_seconds": args.seconds}
    started = time.perf_counter()

    def save():
        state["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "run.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    try:
        with Worker(executable) as worker, (args.output / "checks.jsonl").open("w") as log:
            state["worker_pid"] = worker.process.pid
            state["worker_info"] = worker.request("info")
            state["worker_memory_limit_bytes"] = worker.memory_limit_bytes
            save()
            for index, record in enumerate(records):
                if time.perf_counter() - started >= args.seconds:
                    raise TimeoutError("diagnostic total time limit reached")
                state["current_index"] = index
                state["current_key"] = record["key"]
                save()
                workload = record["workload"]
                source = GENERATORS[workload["family"]](*workload["shape"])
                inputs = [REFERENCES[workload["family"]](*workload["shape"], seed=seed)
                          for seed in (7, 17)]
                worker.request("clear_cache")
                check = {"index": index, "key": record["key"], "original_status": record["status"],
                         "workload": workload, "actions": record["actions"], "variants": []}
                # Keep candidate and anchor lifetimes overlapping as in Evaluator.
                handles = []
                try:
                    for actions in (record["actions"], []):
                        worker.request("analyze", source=source, actions=actions)
                        built = worker.request("compile", source=source, actions=actions)
                        handles.append(built["handle"])
                        errors = []
                        for buffers, expected in inputs:
                            output = worker.request("invoke", handle=built["handle"], buffers=buffers, count=3)
                            errors.append(validate(output["output"], expected))
                        check["variants"].append({"actions": actions, "correct": True,
                                                  "max_abs_errors": errors,
                                                  "llvm_sha256": hashlib.sha256(built["llvm_ir"].encode()).hexdigest()})
                finally:
                    if worker.process.poll() is None:
                        for handle in handles:
                            worker.request("release", handle=handle)
                status = Path(f"/proc/{worker.process.pid}/status").read_text()
                check["worker_peak_rss_kib"] = int(next(line.split()[1] for line in status.splitlines()
                                                       if line.startswith("VmHWM:")))
                log.write(json.dumps(check) + "\n")
                log.flush()
                state["completed"] += 1
                save()
                if state["completed"] % 50 == 0:
                    print(json.dumps({"completed": state["completed"], "elapsed_seconds": state["elapsed_seconds"]}), flush=True)
            state["worker_counts"] = dict(worker.counters)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", reason=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    with measurement_lease(ROOT):
        main()
