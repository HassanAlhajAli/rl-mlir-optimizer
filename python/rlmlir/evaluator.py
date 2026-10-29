"""Serial measured outcomes with provenance, cache accounting and hard limits."""
import gzip
import hashlib
import json
import statistics
import time
from pathlib import Path

from .kernels import GENERATORS, REFERENCES
from .measurement import calibrate, validate


class BudgetReached(RuntimeError):
    pass


class Evaluator:
    def __init__(self, worker, directory: Path, binary_hash: str, revision: str,
                 max_new=4096, max_requests=8192, max_seconds=48*3600):
        self.worker, self.directory = worker, directory
        directory.mkdir(parents=True, exist_ok=True)
        self.records = {}
        self.binary_hash, self.revision = binary_hash, revision
        self.max_new, self.max_requests, self.max_seconds = max_new, max_requests, max_seconds
        self.requests = self.new_measurements = self.cache_hits = 0
        self.reference_requests = 0
        self.started = time.perf_counter()
        self.path = directory / "measurements.jsonl"
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                record = json.loads(line)
                if record.get("status") == "accepted" and record["binary_hash"] == binary_hash:
                    self.records[record["key"]] = record

    def _append(self, record):
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def measure(self, workload, actions, *, reference=False):
        if workload["split"] not in ("train", "validation"):
            raise ValueError("held-out measurements are sealed")
        if time.perf_counter() - self.started >= self.max_seconds:
            raise BudgetReached("wall-time budget reached")
        if reference:
            self.reference_requests += 1
        else:
            if self.requests >= self.max_requests:
                raise BudgetReached("candidate request budget reached")
            self.requests += 1
        source = GENERATORS[workload["family"]](*workload["shape"])
        identity = json.dumps({"source": source, "actions": actions, "binary": self.binary_hash,
                               "input_seed": 7, "measurement_protocol": "paired-training-v2"}, sort_keys=True)
        key = hashlib.sha256(identity.encode()).hexdigest()
        if key in self.records:
            self.cache_hits += 1
            return self.records[key]
        if self.new_measurements >= self.max_new:
            raise BudgetReached("distinct measurement budget reached")
        self.new_measurements += 1
        began = time.perf_counter()
        hardware_before = self.worker.counters.copy()
        handles = []
        record = {"schema_version": 1, "key": key, "binary_hash": self.binary_hash,
                  "revision": self.revision, "workload": workload, "actions": actions,
                  "input_seeds": [7,17], "status": "started"}
        try:
            self.worker.request("clear_cache")
            built = self.worker.request("compile", source=source, actions=actions)
            handle = built["handle"]
            handles.append(handle)
            if built["cache_hit"]:
                raise RuntimeError("unexpected warm compile in a new training measurement")
            record["compile_ms"] = built["compile_ms"]
            record["lowering_ms"], record["codegen_ms"] = built["lowering_ms"], built["codegen_ms"]
            buffers, expected = REFERENCES[workload["family"]](*workload["shape"], seed=7)
            first = self.worker.request("invoke", handle=handle, buffers=buffers, count=1)
            record["first_invocation_ms"] = first["batch_ms"]
            validate(first["output"], expected)
            check = self.worker.request("invoke", handle=handle, buffers=buffers, count=3)
            record["installation_ms"] = check["installation_ms"]
            errors = [validate(check["output"], expected)]
            other, answer = REFERENCES[workload["family"]](*workload["shape"], seed=17)
            errors.append(validate(self.worker.request("invoke", handle=handle, buffers=other)["output"], answer))
            record["max_abs_errors"] = errors
            count, calibration = calibrate(self.worker, handle, buffers, 120)
            anchor = self.worker.request("compile", source=source, actions=[])
            handles.append(anchor["handle"])
            record["anchor_compile"] = {k: anchor[k] for k in ("cache_hit", "lowering_ms", "codegen_ms")}
            anchor_count, anchor_calibration = calibrate(self.worker, anchor["handle"], buffers, 120)
            samples = []
            for attempt in range(2):
                before = self.worker.request("benchmark", handle=anchor["handle"], buffers=buffers, count=anchor_count)
                measured = self.worker.request("benchmark", handle=handle, buffers=buffers, count=count)
                after = self.worker.request("benchmark", handle=anchor["handle"], buffers=buffers, count=anchor_count)
                validate(measured["output"], expected)
                drift = abs(after["batch_ms"] / before["batch_ms"] - 1)
                flagged = drift > .05 or min(before["batch_ms"], measured["batch_ms"], after["batch_ms"]) < 100
                samples.append({"attempt": attempt, "batch_ms": measured["batch_ms"], "count": count,
                                "anchor_before_ms": before["batch_ms"], "anchor_after_ms": after["batch_ms"],
                                "anchor_count": anchor_count, "drift": drift, "flagged": flagged})
                if not flagged:
                    break
            record.update(status="accepted", calibration=calibration, anchor_calibration=anchor_calibration,
                          samples=samples, execution_ms=statistics.median(s["batch_ms"]/s["count"] for s in samples),
                          unresolved_drift=samples[-1]["flagged"], correct=True)
            artifact = {"source": source, "transformed_ir": built["transformed_ir"],
                        "lowered_ir": built["lowered_ir"], "llvm_ir": built["llvm_ir"]}
            record["ir_hashes"] = {name: hashlib.sha256(value.encode()).hexdigest() for name,value in artifact.items()}
            with gzip.open(self.directory / (key + ".json.gz"), "wt") as stream:
                json.dump(artifact, stream)
            self.records[key] = record
        except BaseException as error:
            record.update(status="failed", reason=repr(error))
            raise
        finally:
            record["measurement_elapsed_seconds"] = time.perf_counter() - began
            record["hardware_counts"] = dict(self.worker.counters - hardware_before)
            self._append(record)
            for handle in handles:
                if self.worker.process.poll() is None:
                    self.worker.request("release", handle=handle)
        return record

    def counters(self):
        return {"candidate_requests": self.requests, "reference_requests": self.reference_requests,
                "new_measurements": self.new_measurements, "cache_hits": self.cache_hits,
                "worker_counts": dict(self.worker.counters),
                "elapsed_seconds": time.perf_counter() - self.started}
