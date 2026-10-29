"""Failure controls for adaptation evidence; no native performance measurements."""
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_adaptation as runner
from rlmlir.corpus import DEFAULT_CORPUS, load
from rlmlir.kernels import REFERENCES


class FakeEvaluator:
    def __init__(self, *args, **kwargs):
        self.requests = 0
        self.records = {}

    def counters(self):
        return {"candidate_requests": self.requests}


class FakeWorker:
    class Process:
        pid = 12345

    process = Process()

    def __init__(self, workload, override=None):
        self.workload = workload
        self.override = override or {}
        self.calls = []
        self.counters = Counter()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def request(self, command, **fields):
        self.calls.append(command)
        self.counters[command + "_requests"] += 1
        if command == "clear_cache":
            return {}
        assert command == "deploy"
        self.counters["actual_compilations"] += 1
        self.counters["actual_invocations"] += fields["count"]
        _, answer = REFERENCES[self.workload["family"]](*self.workload["shape"], seed=7)
        return {"output": answer, "count": fields["count"], "cache_hit": False, "compile_ms": 10.,
                "installation_ms": 2., "execution_ms": 3., "total_ms": 15., **self.override}


def fake_adapt(worker, evaluator, workload, reuse, policy, *, method, budget, seed, emit):
    requests = []
    for index in range(budget):
        candidate = {"request": index + 1, "schedule": [], "cost_ms": 12.}
        emit({"event": "request_completed", "candidate": candidate})
        requests.append(candidate)
        evaluator.requests += 1
    return {"method": method, "budget": budget, "seed": seed, "workload": workload, "reuse": reuse,
            "initial_policy_digest": "initial", "final_policy_digest": "initial", "requests": requests,
            "groups": [{"unique_schedules": 1, "update": None}], "best": requests[0],
            "tuning_ms": 20., "counters": evaluator.counters()}


def setup(monkeypatch, tmp_path, override=None):
    workload = next(w for w in load(DEFAULT_CORPUS, "validation") if w["id"] == "b492eae50d11c001")
    worker = FakeWorker(workload, override)
    monkeypatch.setattr(runner, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(runner, "adapt", fake_adapt)
    monkeypatch.setattr(runner, "policy_digest", lambda _: "initial")
    metadata = {"binary_hash": "binary", "revision": "revision", "initial_policy_digest": "initial"}
    directory = tmp_path / "case"

    def run(**kwargs):
        return runner.run_case(worker, directory, metadata, object(), workload, 100, "frozen", 8, 7,
                               kwargs.pop("deadline", 100.), clock=kwargs.pop("clock", lambda: 0.), **kwargs)

    return run, worker, directory, workload


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda value: pytest.fail("nonstandard JSON: " + value))


def test_case_charges_fresh_deployment_and_retains_requests(monkeypatch, tmp_path):
    run, worker, directory, _ = setup(monkeypatch, tmp_path)
    result = run()
    saved = read(directory / "result.json")
    assert saved == result and result["status"] == "complete"
    assert result["adapted_total_ms"] == 35.
    assert result["deployment"]["correct"] and "output" not in result["deployment"]
    assert result["deployment_worker_counts"] == {"deploy_requests": 1, "actual_compilations": 1, "actual_invocations": 100}
    assert result["counters"]["candidate_requests"] == 8
    assert len((directory / "events.jsonl").read_text().splitlines()) == 8
    assert worker.calls == ["clear_cache", "deploy"]


@pytest.mark.parametrize("override,reason", [
    ({"cache_hit": True}, "not fresh"), ({"count": 99}, "N calls"),
    ({"count": 100.}, "N calls"), ({"count": True}, "N calls"),
    ({"compile_ms": float("nan")}, "invalid timing"), ({"installation_ms": -1.}, "invalid timing"),
    ({"execution_ms": True}, "invalid timing"), ({"total_ms": 16.}, "sum to total"),
    ({"output": [float("nan")] * 33}, "nonnumeric output"), ({"output": [42.] * 33}, "incorrect output"),
])
def test_rejects_bad_deployment_and_preserves_raw_evidence(monkeypatch, tmp_path, override, reason):
    run, worker, directory, _ = setup(monkeypatch, tmp_path, override)
    with pytest.raises(RuntimeError, match=reason):
        run()
    saved = read(directory / "result.json")
    assert saved["status"] == "failed" and "adapted_total_ms" not in saved
    assert len(saved["requests"]) == 8 and saved["tuning_ms"] == 20.
    response = saved["failed_deployment"]["response"]
    assert "output" in response and len(saved["failed_deployment"]["expected"]) == 33
    if "compile_ms" in override:
        assert response["compile_ms"] == {"invalid_float": "nan"}
    assert saved["deployment_worker_counts"]["actual_invocations"] == 100


@pytest.mark.parametrize("fault", ["budget", "reset", "best", "counters"])
def test_rejects_tuning_contract_errors_before_deploy(monkeypatch, tmp_path, fault):
    run, worker, directory, _ = setup(monkeypatch, tmp_path)
    def corrupt(*args, **kwargs):
        result = fake_adapt(*args, **kwargs)
        if fault == "budget":
            result["requests"].pop()
        elif fault == "reset":
            result["initial_policy_digest"] = "changed"
        elif fault == "counters":
            args[1].requests -= 1
        else:
            result["best"] = {"schedule": ["unobserved"], "cost_ms": 0.}
        return result
    monkeypatch.setattr(runner, "adapt", corrupt)
    with pytest.raises(RuntimeError):
        run()
    assert worker.calls == ["clear_cache"]
    assert read(directory / "result.json")["status"] == "failed"


def test_deadline_after_tuning_preserves_result_without_deploy(monkeypatch, tmp_path):
    run, worker, directory, _ = setup(monkeypatch, tmp_path)
    now = [0.]
    def late(*args, **kwargs):
        result = fake_adapt(*args, **kwargs)
        now[0] = 2.
        return result
    monkeypatch.setattr(runner, "adapt", late)
    with pytest.raises(TimeoutError, match="before deployment"):
        run(deadline=1., clock=lambda: now[0])
    saved = read(directory / "result.json")
    assert saved["status"] == "failed" and len(saved["requests"]) == 8
    assert worker.calls == ["clear_cache"]


@pytest.mark.parametrize("arguments", [["--methods", "frozen", "frozen"], ["--budgets", "8", "8"],
    ["--hours", "nan"], ["--hours", "0"], ["--hours", "inf"], ["--reuse", "10000001"]])
def test_invalid_configuration_stops_before_loading(monkeypatch, tmp_path, arguments):
    monkeypatch.setattr(sys, "argv", ["run_adaptation", "--policy", "missing.pt", "--output", str(tmp_path / "new"),
                                      "--workload-id", "ignored", *arguments])
    monkeypatch.setattr(runner, "load_policy", lambda *_a, **_k: pytest.fail("loaded policy"))
    monkeypatch.setattr(runner, "Worker", lambda *_a: pytest.fail("started worker"))
    with pytest.raises(SystemExit):
        runner.main()
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("fail_second", [False, True])
def test_complete_controller_or_preserved_partial_failure(monkeypatch, tmp_path, fail_second):
    _, worker, _, workload = setup(monkeypatch, tmp_path)
    policy_path = tmp_path / "policy.pt"
    policy_path.write_text("synthetic policy")
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run_adaptation", "--policy", str(policy_path), "--output", str(output),
                                      "--workload-id", workload["id"], "--methods", "frozen", "random", "mutation",
                                      "--budgets", "8"])
    monkeypatch.setattr(runner, "provenance", lambda _: {"binary_hash": "binary", "revision": "revision"})
    monkeypatch.setattr(runner, "load_policy", lambda *_a, **_k: (object(), {}))
    monkeypatch.setattr(runner, "Worker", lambda _: worker)
    monkeypatch.setattr(runner, "initialize", lambda _: {})
    attempts = []
    def controlled(*args, **kwargs):
        # The controller's initial state must already be durable during tuning.
        assert read(output / "run.json")["status"] == "running"
        attempts.append(kwargs["method"])
        if fail_second and len(attempts) == 2:
            kwargs["emit"]({"event": "request_started", "request": 1})
            raise RuntimeError("synthetic compiler interruption")
        return fake_adapt(*args, **kwargs)
    monkeypatch.setattr(runner, "adapt", controlled)
    if fail_second:
        with pytest.raises(RuntimeError, match="compiler interruption"):
            runner.main()
    else:
        runner.main()
    saved = read(output / "run.json")
    assert worker.closed and saved["worker_pid"] == worker.process.pid
    assert saved["model_loading_ms"] >= 0 and saved["worker_stage_elapsed_seconds"] >= 0
    if fail_second:
        assert saved["status"] == "stopped" and len(saved["summaries"]) == 1
        assert [c["status"] for c in saved["cases"]] == ["complete", "failed", "pending"]
        failed = output / f"{attempts[-1]}-8"
        assert read(failed / "result.json")["status"] == "failed"
        assert len((failed / "events.jsonl").read_text().splitlines()) == 1
    else:
        assert saved["status"] == "complete" and len(saved["summaries"]) == 3
        assert all(c["status"] == "complete" for c in saved["cases"])

