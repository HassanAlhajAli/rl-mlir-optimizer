"""Bounded stability replay of the stopped PPO worker's saved measurement calls.

This diagnostic checks compiler/process stability, not optimization performance.
It reuses the original request counts rather than treating new timings as results.
"""
import argparse
import hashlib
import json
import os
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


def process_memory(pid):
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except FileNotFoundError:
        return {"process_exited": True}
    wanted = {"VmRSS", "VmHWM", "VmSize", "VmPeak"}
    observed = {}
    for line in lines:
        name, separator, value = line.partition(":")
        if separator and name in wanted:
            observed[name + "_kib"] = int(value.split()[0])
    return observed


def original_plan(records):
    """Check that the saved file can reproduce the evaluator's completed calls."""
    if not records:
        raise ValueError("empty measurement history")
    failed = [index for index, record in enumerate(records) if record.get("status") != "accepted"]
    if failed != [len(records) - 1] or records[-1]["status"] != "failed":
        raise ValueError("expected exactly one final failed measurement")
    for index, record in enumerate(records):
        if record["workload"]["split"] != "train" or record["input_seeds"] != [7, 17]:
            raise ValueError(f"record {index} is outside the saved training/input boundary")
        if index != len(records) - 1:
            for field in ("calibration", "anchor_calibration", "samples", "anchor_compile"):
                if not record.get(field):
                    raise ValueError(f"record {index} lacks {field}")
            if any(sample["count"] <= 0 or sample["anchor_count"] <= 0 for sample in record["samples"]):
                raise ValueError(f"record {index} has an invalid benchmark count")
    return {"records": len(records), "accepted": len(records) - 1,
            "original_failure_key": records[-1]["key"],
            "original_binary_sha256": records[0]["binary_hash"]}


class RequestLog:
    def __init__(self, worker, path, deadline):
        self.worker, self.stream, self.deadline = worker, path.open("w"), deadline
        self.sequence = 0

    def close(self):
        self.stream.close()

    def request(self, index, key, phase, command, *, input_seed=None, **fields):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("overall diagnostic limit reached before next request")
        self.sequence += 1
        context = {"sequence": self.sequence, "record_index": index, "record_key": key,
                   "phase": phase, "command": command, "input_seed": input_seed,
                   "count": fields.get("count"), "handle": fields.get("handle")}
        began = time.perf_counter()
        self.stream.write(json.dumps({**context, "event": "request_start",
                                      "worker_memory": process_memory(self.worker.process.pid)}) + "\n")
        self.stream.flush()
        try:
            result = self.worker.request(command, **fields)
        except BaseException as error:
            self.stream.write(json.dumps({**context, "event": "request_failure", "reason": repr(error),
                                          "elapsed_seconds": time.perf_counter() - began,
                                          "worker_returncode": self.worker.process.poll(),
                                          "worker_memory": process_memory(self.worker.process.pid)}) + "\n")
            self.stream.flush()
            raise
        self.stream.write(json.dumps({**context, "event": "request_complete",
                                      "elapsed_seconds": time.perf_counter() - began,
                                      "cache_hit": result.get("cache_hit"),
                                      "worker_memory": process_memory(self.worker.process.pid)}) + "\n")
        self.stream.flush()
        return result


def replay_record(record, index, requests):
    workload, actions, key = record["workload"], record["actions"], record["key"]
    source = GENERATORS[workload["family"]](*workload["shape"])
    buffers, expected = REFERENCES[workload["family"]](*workload["shape"], seed=7)
    other, other_expected = REFERENCES[workload["family"]](*workload["shape"], seed=17)
    handles = []

    def call(phase, command, *, input_seed=None, **fields):
        return requests.request(index, key, phase, command, input_seed=input_seed, **fields)

    try:
        call("reset", "clear_cache")
        candidate = call("candidate_compile", "compile", source=source, actions=actions)
        handles.append(candidate["handle"])
        if candidate["cache_hit"]:
            raise RuntimeError("candidate unexpectedly hit a warm cache after clear_cache")
        # The original worker crashed at this point. All prior accepted records
        # continue through their saved calibration/benchmark call sequence.
        if record["status"] == "failed":
            return {"status": "reached_original_failure_point", "candidate_cache_hit": False}

        validate(call("candidate_first", "invoke", handle=candidate["handle"],
                      buffers=buffers, count=1, input_seed=7)["output"], expected)
        validate(call("candidate_repeat", "invoke", handle=candidate["handle"],
                      buffers=buffers, count=3, input_seed=7)["output"], expected)
        validate(call("candidate_other_input", "invoke", handle=candidate["handle"],
                      buffers=other, count=1, input_seed=17)["output"], other_expected)
        for sample in record["calibration"]:
            call("candidate_calibration", "benchmark", handle=candidate["handle"],
                 buffers=buffers, count=sample["count"], input_seed=7)
        anchor = call("anchor_compile", "compile", source=source, actions=[])
        handles.append(anchor["handle"])
        for sample in record["anchor_calibration"]:
            call("anchor_calibration", "benchmark", handle=anchor["handle"],
                 buffers=buffers, count=sample["count"], input_seed=7)
        for sample in record["samples"]:
            for phase, handle, count in (("anchor_before", anchor["handle"], sample["anchor_count"]),
                                         ("candidate_sample", candidate["handle"], sample["count"]),
                                         ("anchor_after", anchor["handle"], sample["anchor_count"])):
                result = call(phase, "benchmark", handle=handle, buffers=buffers,
                              count=count, input_seed=7)
                if phase == "candidate_sample":
                    validate(result["output"], expected)
        return {"status": "complete", "candidate_cache_hit": False,
                "anchor_cache_hit": anchor["cache_hit"]}
    finally:
        if requests.worker.process.poll() is None:
            previous_error = sys.exc_info()[0] is not None
            for handle in handles:
                try:
                    call("release", "release", handle=handle)
                except BaseException:
                    if not previous_error:
                        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    raw = args.records.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    plan = original_plan(records)
    if args.check_only:
        print(json.dumps(plan, indent=2))
        return
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, text=True).strip():
        raise ValueError("commit tracked changes before measurement")
    args.output.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    executable = ROOT / "build/rlmlir-worker"
    state = {"schema_version": 1, "purpose": "compiler/process stability at saved request counts; no performance inference",
             "status": "running", "plan": plan, "records": str(args.records),
             "records_sha256": hashlib.sha256(raw).hexdigest(),
             "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
             "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "guest_affinity": sorted(os.sched_getaffinity(0)), "limit_seconds": args.seconds,
             "completed_records": 0, "limitations": [
                 "PPO policy/analyze interleaving and exact machine state were not recorded and cannot be reproduced",
                 "Saved benchmark counts replay the historical request path; new batch times are not research observations",
                 "The original failed record stops after its compile succeeds; prior guarded replay checked its execution"]}
    started = time.perf_counter()

    def save():
        state["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "run.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    requests = None
    try:
        with (args.output / "worker.stderr").open("wb") as stderr, Worker(executable, stderr=stderr) as worker:
            requests = RequestLog(worker, args.output / "requests.jsonl", started + args.seconds)
            state.update(worker_pid=worker.process.pid, worker_memory_limit_bytes=worker.memory_limit_bytes,
                         worker_info=requests.request(-1, "initialization", "info", "info"))
            save()
            try:
                with (args.output / "records.jsonl").open("w") as completed:
                    for index, record in enumerate(records):
                        state.update(current_index=index, current_key=record["key"])
                        save()
                        result = replay_record(record, index, requests)
                        completed.write(json.dumps({"index": index, "key": record["key"],
                                                    "original_status": record["status"], **result}) + "\n")
                        completed.flush()
                        state["completed_records"] += 1
                        save()
                        if state["completed_records"] % 50 == 0:
                            print(json.dumps({"completed_records": state["completed_records"],
                                              "elapsed_seconds": state["elapsed_seconds"]}), flush=True)
                state["worker_counts"] = dict(worker.counters)
                state["worker_peak_memory"] = process_memory(worker.process.pid)
            finally:
                state["worker_counts"] = dict(worker.counters)
                state["worker_returncode"] = worker.process.poll()
                requests.close()
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", reason=repr(error))
        if requests is not None:
            state["request_sequence"] = requests.sequence
        raise
    finally:
        save()


if __name__ == "__main__":
    with measurement_lease(ROOT):
        main()
