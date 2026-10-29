"""Interleaved, fresh deployments for development-only frozen selectors."""
import copy
import hashlib
import math
import random
import time

from .corpus import FrozenCorpus
from .kernels import GENERATORS, REFERENCES
from .measurement import calibrate, validate, check_fresh_deployment


class AnalysisOnlyWorker:
    """Frozen selection can inspect transformations, never query performance."""
    def __init__(self, request):
        self._request = request

    def request(self, command, **fields):
        if command != "analyze":
            raise ValueError("frozen selector may only analyze compiler state")
        return self._request(command, **fields)


def interleaved_order(workloads, methods, reuse_counts, rounds, seed):
    """Round boundaries are explicit; each round visits the entire same grid."""
    if not workloads or not methods or not reuse_counts or type(rounds) is not int or rounds < 1:
        raise ValueError("nonempty comparison grid and positive integer rounds required")
    ids = [w["id"] for w in workloads]
    if len(set(ids)) != len(ids) or len(set(methods)) != len(methods):
        raise ValueError("duplicate workload or method in comparison grid")
    if (len(set(reuse_counts)) != len(reuse_counts)
            or any(type(n) is not int or not 1 <= n <= 10000000 for n in reuse_counts)):
        raise ValueError("unique bounded integer reuse counts required")
    rng = random.Random(seed)
    order = []
    for index in range(rounds):
        cases = [{"round": index, "workload_id": w, "method": m, "reuse": n}
                 for w in ids for m in methods for n in reuse_counts]
        rng.shuffle(cases)
        order.extend(cases)
    return order


def require_timing(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise RuntimeError("invalid timing: " + name)


class DevelopmentComparison:
    """No held-out override. Emits starts and outcomes so interruption stays visible.

    Selector time is measured around the complete callback, not trusted from a
    self-reported field. Research calibration/checks have separate wall/counter
    boundaries. Every accepted deployment is fresh and includes exactly N calls.
    """
    def __init__(self, worker, *, max_seconds, emit, save_ir, clock=time.perf_counter):
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("positive finite comparison limit required")
        self.worker, self.emit, self.save_ir = worker, emit, save_ir
        self.clock = clock
        self.deadline = clock() + max_seconds
        self.corpus = FrozenCorpus()

    def request(self, command, **fields):
        if self.clock() >= self.deadline:
            raise TimeoutError("comparison wall-time budget reached")
        return self.worker.request(command, **fields)

    @staticmethod
    def checked_output(record, phase, output, expected):
        try:
            return validate(output, expected)
        except RuntimeError:
            record["validation_failure"] = {"phase": phase, "output": output, "expected": expected}
            raise

    def run(self, workloads, selectors, reuse_counts, *, rounds=5, seed=7, hint_factor=1.):
        # Validate the entire grid before the first compiler interaction.
        for workload in workloads:
            self.corpus.require_development(workload)
        if not math.isfinite(hint_factor) or hint_factor <= 0:
            raise ValueError("positive finite reuse hint factor required")
        order = interleaved_order(workloads, list(selectors), reuse_counts, rounds, seed)
        by_id = {w["id"]: w for w in workloads}
        for index, case in enumerate(order):
            self._case(index, case, by_id[case["workload_id"]], selectors[case["method"]], hint_factor)
        return order

    def _case(self, index, case, workload, selector, hint_factor):
        started = self.clock()
        before_counts = self.worker.counters.copy()
        handles = []
        record = {"schema_version": 1, "case_index": index, **case,
                  "workload": copy.deepcopy(workload), "reuse_hint": case["reuse"] * hint_factor,
                  "corpus_sha256": self.corpus.sha256, "status": "started", "attempts": []}
        self.emit(copy.deepcopy(record))
        try:
            if started >= self.deadline:
                raise TimeoutError("comparison wall-time budget reached")
            select_start = self.clock()
            selected = selector(AnalysisOnlyWorker(self.request), copy.deepcopy(workload), record["reuse_hint"])
            record["selection_ms"] = (self.clock() - select_start) * 1000
            schedule = selected["schedule"]
            record["selection"] = selected
            source = GENERATORS[workload["family"]](*workload["shape"])
            buffers, expected = REFERENCES[workload["family"]](*workload["shape"], seed=7)
            other, answer = REFERENCES[workload["family"]](*workload["shape"], seed=17)
            self.request("clear_cache")
            built = self.request("compile", source=source, actions=schedule)
            handles.append(built["handle"])
            record["validation_compile_ms"] = built["compile_ms"]
            first = self.request("invoke", handle=built["handle"], buffers=buffers, count=1)
            self.checked_output(record, "first_invocation", first["output"], expected)
            record["first_invocation_ms"] = first["batch_ms"]
            record["max_abs_errors"] = [
                self.checked_output(record, "candidate_seed7", self.request("invoke", handle=built["handle"], buffers=buffers, count=3)["output"], expected),
                self.checked_output(record, "candidate_seed17", self.request("invoke", handle=built["handle"], buffers=other, count=3)["output"], answer)]
            artifact = {"source": source, **{k: built[k] for k in ("transformed_ir", "lowered_ir", "llvm_ir")}}
            record["ir_hashes"] = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in artifact.items()}
            self.save_ir(index, artifact)
            anchor = self.request("compile", source=source, actions=[])
            handles.append(anchor["handle"])
            self.checked_output(record, "anchor", self.request("invoke", handle=anchor["handle"], buffers=buffers, count=1)["output"], expected)
            count, calibration = calibrate(self, built["handle"], buffers, 120)
            anchor_count, anchor_calibration = calibrate(self, anchor["handle"], buffers, 120)
            record.update(calibration=calibration, anchor_calibration=anchor_calibration)
            for attempt in range(2):
                sample = {"attempt": attempt, "status": "started"}
                record["attempts"].append(sample)
                before = self.request("benchmark", handle=anchor["handle"], buffers=buffers, count=anchor_count)
                self.checked_output(record, "anchor_before", before["output"], expected)
                sample["anchor_before_ms"] = before["batch_ms"]
                deployment = self.request("deploy", source=source, actions=schedule, buffers=buffers, count=case["reuse"])
                output = deployment.pop("output")
                sample["deployment"] = {**deployment, "correct": False}
                error = self.checked_output(record, "deployment", output, expected)
                sample["deployment"].update(correct=True, max_abs_error=error)
                check_fresh_deployment(deployment, case["reuse"])
                measured = self.request("benchmark", handle=built["handle"], buffers=buffers, count=count)
                self.checked_output(record, "steady", measured["output"], expected)
                after = self.request("benchmark", handle=anchor["handle"], buffers=buffers, count=anchor_count)
                self.checked_output(record, "anchor_after", after["output"], expected)
                for name, value in (("anchor_before", before["batch_ms"]), ("steady", measured["batch_ms"]),
                                    ("anchor_after", after["batch_ms"])):
                    require_timing(value, name, positive=True)
                drift = abs(after["batch_ms"] / before["batch_ms"] - 1)
                flagged = drift > .05 or min(before["batch_ms"], measured["batch_ms"], after["batch_ms"]) < 100
                sample.update(status="accepted", anchor_after_ms=after["batch_ms"], anchor_count=anchor_count,
                              steady_batch_ms=measured["batch_ms"], steady_count=count, drift=drift, flagged=flagged,
                              total_with_selection_ms=record["selection_ms"] + deployment["total_ms"])
                if not flagged:
                    break
            record.update(status="accepted", correct=True, unresolved_drift=record["attempts"][-1]["flagged"])
        except BaseException as error:
            record.update(status="failed", reason=repr(error))
            raise
        finally:
            cleanup_errors = []
            for handle in handles:
                if self.worker.process.poll() is None:
                    try:
                        # Release must still run after the measurement deadline.
                        self.worker.request("release", handle=handle)
                    except Exception as error:
                        cleanup_errors.append(repr(error))
            record.update(elapsed_seconds=self.clock() - started,
                          hardware_counts=dict(self.worker.counters - before_counts), cleanup_errors=cleanup_errors)
            if cleanup_errors and record["status"] == "accepted":
                record.update(status="failed", reason="handle cleanup failed")
            self.emit(copy.deepcopy(record))
            if cleanup_errors and record.get("reason") == "handle cleanup failed":
                raise RuntimeError("handle cleanup failed")
