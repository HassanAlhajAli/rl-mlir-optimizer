"""The stability diagnostic must retain the call path and stop at a failure."""
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replay_measurement_history as replay


def record(status="accepted"):
    return {"key": "example", "status": status,
            "workload": {"family": "matvec", "shape": [2, 2], "split": "train"},
            "actions": [], "input_seeds": [7, 17], "binary_hash": "old",
            "calibration": [{"count": 1}, {"count": 4}],
            "anchor_calibration": [{"count": 1}, {"count": 3}],
            "anchor_compile": {"cache_hit": False},
            "samples": [{"count": 4, "anchor_count": 3}]}


def test_plan_requires_one_final_failure():
    assert replay.original_plan([record(), record("failed")])["accepted"] == 1
    with pytest.raises(ValueError, match="final failed"):
        replay.original_plan([record("failed"), record()])


def test_replay_uses_saved_benchmark_counts_and_stops_at_original_failure(monkeypatch):
    monkeypatch.setattr(replay, "validate", lambda *_: None)

    class FakeRequests:
        class worker:
            class process:
                @staticmethod
                def poll():
                    return None

        def __init__(self):
            self.calls = []
            self.handles = 0

        def request(self, index, key, phase, command, **fields):
            self.calls.append((phase, command, fields.get("count")))
            if command == "compile":
                self.handles += 1
                return {"handle": self.handles, "cache_hit": False}
            return {"output": []}

    requests = FakeRequests()
    assert replay.replay_record(record(), 0, requests)["status"] == "complete"
    assert requests.calls == [
        ("reset", "clear_cache", None), ("candidate_compile", "compile", None),
        ("candidate_first", "invoke", 1), ("candidate_repeat", "invoke", 3),
        ("candidate_other_input", "invoke", 1),
        ("candidate_calibration", "benchmark", 1), ("candidate_calibration", "benchmark", 4),
        ("anchor_compile", "compile", None),
        ("anchor_calibration", "benchmark", 1), ("anchor_calibration", "benchmark", 3),
        ("anchor_before", "benchmark", 3), ("candidate_sample", "benchmark", 4),
        ("anchor_after", "benchmark", 3), ("release", "release", None), ("release", "release", None)]
    requests.calls.clear()
    assert replay.replay_record(record("failed"), 1, requests)["status"] == "reached_original_failure_point"
    assert requests.calls == [("reset", "clear_cache", None), ("candidate_compile", "compile", None),
                              ("release", "release", None)]


def test_request_failure_records_context_before_and_after(tmp_path):
    class FakeWorker:
        class process:
            pid = 999999999

            @staticmethod
            def poll():
                return -11

        @staticmethod
        def request(*_args, **_kwargs):
            raise RuntimeError("worker stopped unexpectedly (-11)")

    log = replay.RequestLog(FakeWorker(), tmp_path / "requests.jsonl", time.perf_counter() + 5)
    with pytest.raises(RuntimeError, match="-11"):
        log.request(832, "failed-key", "candidate_compile", "compile", source="source", actions=[])
    log.close()
    events = [__import__("json").loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["request_start", "request_failure"]
    assert events[1]["record_key"] == "failed-key" and events[1]["worker_returncode"] == -11
