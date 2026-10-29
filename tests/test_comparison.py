"""Control/cost tests use a simulated worker, not real performance measurements."""
import copy
import json
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
import compare_deployments
from audit_comparison import audit
from rlmlir.comparison import DevelopmentComparison, interleaved_order
from rlmlir.corpus import DEFAULT_CORPUS, FrozenCorpus, load
from rlmlir.kernels import REFERENCES


class Clock:
    now = 0.

    def __call__(self):
        return self.now


class SimulatedWorker:
    class Process:
        pid = 12345
        def poll(self):
            return None

    process = Process()

    def __init__(self, workload, *, times=None, deployment_override=None, fail_release=False):
        self.counters = Counter()
        self.calls, self.handles = [], set()
        self.times = iter(times or [])
        self.override = deployment_override or {}
        self.fail_release = fail_release
        self.answers = [REFERENCES[workload["family"]](*workload["shape"], seed=s) for s in (7, 17)]

    def request(self, command, **fields):
        self.calls.append((command, copy.deepcopy(fields)))
        self.counters[command + "_requests"] += 1
        if command == "analyze" or command == "clear_cache":
            return {}
        if command == "compile":
            handle = self.counters["compile_requests"]
            self.handles.add(handle)
            self.counters["actual_compilations"] += 1
            return {"handle": handle, "compile_ms": 10., "transformed_ir": "transformed",
                    "lowered_ir": "lowered", "llvm_ir": "llvm", "cache_hit": False}
        if command == "release":
            if self.fail_release:
                raise RuntimeError("simulated release failure")
            self.handles.remove(fields["handle"])
            return {}
        output = next(answer for buffers, answer in self.answers if buffers == fields["buffers"])
        self.counters["actual_invocations"] += fields["count"]
        if command == "deploy":
            self.counters["actual_compilations"] += 1
            return {"compile_ms": 10., "installation_ms": 2., "execution_ms": 3., "total_ms": 15.,
                    "count": fields["count"], "cache_hit": False, "output": output, **self.override}
        return {"batch_ms": next(self.times, 120.) if command == "benchmark" else 1.,
                "count": fields["count"], "output": output}


def setup_case(**kwargs):
    workload = load(DEFAULT_CORPUS, "train")[0]
    worker = SimulatedWorker(workload, **kwargs)
    events, ir = [], {}
    clock = Clock()
    runner = DevelopmentComparison(worker, max_seconds=1., emit=events.append,
                                   save_ir=lambda i, value: ir.update({i: value}), clock=clock)
    return workload, worker, events, ir, clock, runner


def fixed(*_args):
    return {"schedule": []}


def test_interleaving_is_reproducible_and_complete_each_round():
    workloads = load(DEFAULT_CORPUS, "train")[:2]
    order = interleaved_order(workloads, ["reference", "policy"], [1, 100], 5, 7)
    assert order == interleaved_order(workloads, ["reference", "policy"], [1, 100], 5, 7)
    assert order != interleaved_order(workloads, ["reference", "policy"], [1, 100], 5, 17)
    expected = {(w["id"], m, n) for w in workloads for m in ("reference", "policy") for n in (1, 100)}
    assert len(order) == 40
    for index in range(5):
        assert {(c["workload_id"], c["method"], c["reuse"]) for c in order if c["round"] == index} == expected


def test_complete_cost_charges_outer_selection_and_exact_calls():
    workload, worker, events, ir, clock, runner = setup_case()
    def select(analysis, program, hint):
        assert hint == 25.
        analysis.request("analyze", source="synthetic", actions=[])
        clock.now += .25
        return {"schedule": [], "selection_ms": 0.}  # Cannot hide callback cost.
    runner.run([workload], {"policy": select}, [100], rounds=1, hint_factor=.25)
    start, result = events
    assert start["status"] == "started" and not start["attempts"]
    assert result["status"] == "accepted" and result["selection_ms"] == pytest.approx(250.)
    assert result["attempts"][0]["total_with_selection_ms"] == pytest.approx(265.)
    assert result["attempts"][0]["deployment"]["count"] == 100
    assert len([c for c, _ in worker.calls if c == "deploy"]) == 1
    assert len(result["max_abs_errors"]) == 2 and len(ir) == 1
    assert not worker.handles


@pytest.mark.parametrize("still_unstable", [False, True])
def test_drift_retained_and_repeated_at_most_once(still_unstable):
    times = [120., 120., 120., 120., 144., 120., 120., 144. if still_unstable else 120.]
    workload, worker, events, _, _, runner = setup_case(times=times)
    runner.run([workload], {"reference": fixed}, [1], rounds=1)
    result = events[-1]
    assert len(result["attempts"]) == 2 and result["attempts"][0]["flagged"]
    assert result["unresolved_drift"] is still_unstable
    assert worker.counters["deploy_requests"] == 2 and not worker.handles


@pytest.mark.parametrize("override,match", [
    ({"cache_hit": True}, "not fresh"), ({"count": 2}, "N calls"),
    ({"total_ms": 16.}, "sum"), ({"execution_ms": float("nan")}, "invalid timing"),
    ({"output": []}, "length mismatch"),
])
def test_bad_deployment_is_preserved_and_stops_next_case(override, match):
    workload, worker, events, _, _, runner = setup_case(deployment_override=override)
    with pytest.raises(RuntimeError, match=match):
        runner.run([workload], {"first": fixed, "second": fixed}, [1], rounds=1)
    assert [e["status"] for e in events] == ["started", "failed"]
    assert len(events[-1]["attempts"]) == 1 and not worker.handles
    assert worker.counters["deploy_requests"] == 1
    if "output" in override:
        assert events[-1]["validation_failure"]["phase"] == "deployment"


def test_frozen_selector_cannot_request_performance():
    workload, worker, events, _, _, runner = setup_case()
    def cheating(analysis, *_args):
        analysis.request("benchmark", handle=1, buffers=[], count=1)
    with pytest.raises(ValueError, match="only analyze"):
        runner.run([workload], {"policy": cheating}, [1], rounds=1)
    assert not worker.calls and events[-1]["status"] == "failed"


def test_heldout_relabel_rejected_before_any_event_or_worker_call():
    _, worker, events, _, _, runner = setup_case()
    workload = next(w for w in json.loads(DEFAULT_CORPUS.read_text())["workloads"] if w["split"] == "heldout_shape")
    workload["split"] = "train"
    with pytest.raises(ValueError, match="sealed"):
        runner.run([workload], {"reference": fixed}, [1], rounds=1)
    assert not worker.calls and not events


def test_deadline_stops_after_selection_before_compilation():
    workload, worker, events, _, clock, runner = setup_case()
    def slow(*_args):
        clock.now = 2.
        return {"schedule": []}
    with pytest.raises(TimeoutError, match="budget"):
        runner.run([workload], {"policy": slow}, [1], rounds=1)
    assert not worker.calls and events[-1]["status"] == "failed"


def test_cleanup_failure_is_not_reported_as_success():
    workload, _, events, _, _, runner = setup_case(fail_release=True)
    with pytest.raises(RuntimeError, match="cleanup"):
        runner.run([workload], {"reference": fixed}, [1], rounds=1)
    assert events[-1]["status"] == "failed" and len(events[-1]["cleanup_errors"]) == 2


def test_cleanup_does_not_hide_original_correctness_failure():
    workload, _, events, _, _, runner = setup_case(deployment_override={"output": []}, fail_release=True)
    with pytest.raises(RuntimeError, match="length mismatch"):
        runner.run([workload], {"reference": fixed}, [1], rounds=1)
    assert "length mismatch" in events[-1]["reason"] and events[-1]["cleanup_errors"]


def test_check_only_never_loads_models_or_launches_worker(tmp_path, monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("configuration check tried to start an experiment")
    monkeypatch.setattr(compare_deployments, "Worker", forbidden)
    monkeypatch.setattr(compare_deployments, "selectors_for", forbidden)
    monkeypatch.setattr(compare_deployments, "provenance", forbidden)
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(sys, "argv", ["compare_deployments", "--config", str(ROOT / "configs/comparison-development.json"),
                                     "--output", str(output), "--check-only"])
    compare_deployments.main()
    report = json.loads(capsys.readouterr().out)
    assert report["cases"] == 30 and report["rounds"] == 5
    assert not report["heldout_access"] and not report["model_artifacts_checked"]
    assert not output.exists()


def test_cli_saves_stopped_state_and_nonfinite_diagnostic(tmp_path, monkeypatch):
    workload = load(DEFAULT_CORPUS, "train")[0]
    class ContextWorker(SimulatedWorker):
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
    worker = ContextWorker(workload, deployment_override={"total_ms": float("nan")})
    monkeypatch.setattr(compare_deployments, "Worker", lambda *_args: worker)
    monkeypatch.setattr(compare_deployments, "provenance", lambda *_args: {"revision": "test", "binary_hash": "test"})
    monkeypatch.setattr(compare_deployments, "initialize", lambda *_args: {})
    monkeypatch.setattr(compare_deployments.os, "sched_setaffinity", lambda *_args: None)
    monkeypatch.setattr(compare_deployments, "selectors_for", lambda *_args: ({"reference": fixed, "heuristic": fixed}, {}))
    config = json.loads((ROOT / "configs/comparison-development.json").read_text())
    config.update(split="train", workload_ids=[workload["id"]], rounds=1, reuse_counts=[1])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "stopped"
    monkeypatch.setattr(sys, "argv", ["compare_deployments", "--config", str(path), "--output", str(output)])
    with pytest.raises(RuntimeError, match="invalid timing"):
        compare_deployments.main()
    state = json.loads((output / "run.json").read_text())
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert state["status"] == "stopped" and state["totals"]["failed"] == 1
    assert state["worker_counts"]["deploy_requests"] == 1 and not worker.handles
    assert events[-1]["attempts"][0]["deployment"]["total_ms"] == {"invalid_float": "nan"}


def test_comparison_configuration_cannot_unseal_performance():
    config = json.loads((ROOT / "configs/comparison-development.json").read_text())
    config["split"] = "heldout_shape"
    with pytest.raises(ValueError, match="sealed"):
        compare_deployments.prepare(config)


@pytest.mark.parametrize("best_fixed,tamper", [(False, None), (False, "cost"), (False, "counter"),
                                             (False, "missing_case"), (True, None), (True, "fixed_schedule")])
def test_saved_evidence_audit_rejects_accounting_or_coverage_changes(tmp_path, monkeypatch, best_fixed, tamper):
    workload = load(DEFAULT_CORPUS, "train")[0]
    worker = SimulatedWorker(workload)
    monkeypatch.setattr(compare_deployments, "Worker", lambda *_: nullcontext(worker))
    monkeypatch.setattr(compare_deployments, "provenance", lambda *_: {
        "revision": "synthetic-test", "binary_hash": "synthetic-test", "corpus_sha256": FrozenCorpus().sha256})
    monkeypatch.setattr(compare_deployments, "initialize", lambda *_: {"initialization_counts": {}})
    selectors = {"reference": fixed, "heuristic": fixed}
    artifacts = {}
    if best_fixed:
        selectors["best_fixed"] = fixed
        artifacts["best_fixed"] = {"selected": {"name": "reference", "schedule": []}}
    monkeypatch.setattr(compare_deployments, "selectors_for", lambda *_: (selectors, artifacts))
    monkeypatch.setattr(compare_deployments.os, "sched_setaffinity", lambda *_: None)
    config = json.loads((ROOT / "configs/comparison-development.json").read_text())
    config.update(split="train", workload_ids=[workload["id"]], rounds=1, reuse_counts=[1])
    if best_fixed:
        config["best_fixed"] = "synthetic-unused-artifact.json"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", ["compare_deployments", "--config", str(path), "--output", str(output)])
    compare_deployments.main()
    assert audit(output)["cases"] == len(selectors)
    if tamper == "counter":
        state = json.loads((output / "run.json").read_text())
        state["worker_counts"]["actual_invocations"] += 1
        (output / "run.json").write_text(json.dumps(state))
    elif tamper:
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        if tamper == "cost":
            events[1]["attempts"][0]["total_with_selection_ms"] += 1.
        elif tamper == "fixed_schedule":
            record = next(e for e in events if e["status"] == "accepted" and e["method"] == "best_fixed")
            record["selection"]["schedule"] = [{"kind": "TILE", "axis": 0, "size": 8}]
        else:
            events = events[:-2]
        (output / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    if tamper:
        with pytest.raises(ValueError):
            audit(output)
