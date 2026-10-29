"""Known-cost synthetic evidence, leakage controls and fixed-selector integration."""
import copy
from collections import Counter
from contextlib import nullcontext
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
import collect_fixed_grid
import compare_deployments
from fit_fixed_baseline import fit
from rlmlir.corpus import FrozenCorpus
from rlmlir.fixed_baseline import ROLE, collection_order, digest, fixed_selector, load_fixed, prepare, select
from rlmlir.kernels import GENERATORS


def specification():
    training = FrozenCorpus().development("train")
    first = training[0]
    workloads = [first, next(w for w in training if w["family"] == first["family"] and w != first),
                 next(w for w in training if w["family"] != first["family"])]
    config = {"schema_version": 1, "workload_ids": [w["id"] for w in workloads], "reuse_counts": [1, 100, 10000],
              "order_seed": 627, "candidates": [{"name": "reference", "schedule": []},
                                                {"name": "tile", "schedule": [{"kind": "TILE", "axis": 0, "size": 8}]}]}
    cells = [{**case, "compile_ms": 0., "installation_ms": 0., "execution_ms": 1., "selection_ms": 0.,
              "unresolved_drift": False} for case in collection_order(config)]
    return config, workloads, cells


def metadata():
    return {"binary_hash": "synthetic-test-only", "revision": "synthetic-test-only", "corpus_sha256": FrozenCorpus().sha256,
            "compiler": "LLVM/MLIR 18.1.3", "target": "x86-64-v3", "observation_version": 2,
            "measurement_protocol": "paired-training-v2"}


def test_global_selection_balances_families_not_the_number_of_shapes():
    config, workloads, cells = specification()
    for cell in cells:
        if cell["candidate"] == "tile":
            cell["execution_ms"] = 1 / 3 if cell["workload_id"] == workloads[-1]["id"] else 2.
    result = select(config, cells)
    # Two shapes lose 2x, one other-family shape wins 3x. Equal family weighting
    # wins (sqrt(3/2)); treating the three shapes equally would lose (cbrt(3/4)).
    assert result["selected"]["name"] == "tile"
    assert result["leaderboard"][1]["mean_log_ratio"] == pytest.approx(math.log(1.5) / 2)
    assert not result["full_training_coverage"]


def test_compilation_cost_and_reuse_can_change_the_training_winner():
    config, _, cells = specification()
    for cell in cells:
        if cell["candidate"] == "tile":
            cell.update(compile_ms=100., execution_ms=.1)
    config["reuse_counts"] = [1]
    assert select(config, cells)["selected"]["name"] == "reference"
    config["reuse_counts"] = [10000]
    assert select(config, cells)["selected"]["name"] == "tile"


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unknown", "nan", "negative"])
def test_partial_or_invalid_cost_evidence_cannot_choose_a_winner(problem):
    config, _, cells = specification()
    if problem == "missing":
        cells.pop()
    elif problem == "duplicate":
        cells.append(copy.deepcopy(cells[0]))
    elif problem == "unknown":
        cells[0]["candidate"] = "unregistered"
    else:
        cells[0]["execution_ms"] = float("nan") if problem == "nan" else -1.
    with pytest.raises(ValueError):
        select(config, cells)


@pytest.mark.parametrize("problem", ["validation", "duplicate_candidate", "stop", "after_vectorize"])
def test_grid_rejects_contamination_and_ambiguous_actions(problem):
    config, _, _ = specification()
    if problem == "validation":
        config["workload_ids"][0] = FrozenCorpus().development("validation")[0]["id"]
    elif problem == "duplicate_candidate":
        config["candidates"].append(copy.deepcopy(config["candidates"][1]))
    elif problem == "stop":
        config["candidates"][1]["schedule"].append({"kind": "STOP"})
    else:
        config["candidates"][1]["schedule"].insert(0, {"kind": "VECTORIZE"})
    with pytest.raises(ValueError):
        prepare(config)


def test_exact_ties_use_reference_and_collection_order_is_reproducible():
    config, _, cells = specification()
    assert select(config, cells)["selected"]["name"] == "reference"
    assert collection_order(config) == collection_order(copy.deepcopy(config))
    assert len({(c["workload_id"], c["candidate"]) for c in collection_order(config)}) == 6


def write_evidence(directory):
    config, workloads, cells = specification()
    current = metadata()
    by_id = {w["id"]: w for w in workloads}
    schedules = {c["name"]: c["schedule"] for c in config["candidates"]}
    trials, records = [], []
    directory.mkdir()
    (directory / "measurements").mkdir()
    for index, cell in enumerate(cells):
        workload = by_id[cell["workload_id"]]
        source = GENERATORS[workload["family"]](*workload["shape"])
        schedule = schedules[cell["candidate"]]
        key = digest({"source": source, "actions": schedule, "binary": current["binary_hash"],
                      "input_seed": 7, "measurement_protocol": "paired-training-v2"})
        case = {k: cell[k] for k in ("workload_id", "candidate")}
        trials.extend([{**case, "index": index, "status": "started"},
                       {**case, "index": index, "status": "accepted", "selection_ms": 0., "measurement_key": key}])
        ir = {"source": source, "transformed_ir": "synthetic", "lowered_ir": "synthetic", "llvm_ir": "synthetic"}
        with gzip.open(directory / "measurements" / (key + ".json.gz"), "wt") as stream:
            json.dump(ir, stream)
        records.append({**cell, **current, "key": key, "status": "accepted", "correct": True,
                        "workload": workload, "actions": schedule, "input_seeds": [7, 17], "max_abs_errors": [0., 0.],
                        "samples": [{"attempt": 0, "batch_ms": 120., "count": 120, "anchor_before_ms": 120.,
                                     "anchor_after_ms": 120., "drift": 0., "flagged": False}],
                        "ir_hashes": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in ir.items()}})
    raw = json.dumps(config).encode()
    (directory / "config.json").write_bytes(raw)
    state = {**current, "role": ROLE, "status": "complete", "config": config,
             "config_sha256": hashlib.sha256(raw).hexdigest(), "order": collection_order(config),
             "counters": {"candidate_requests": 6, "new_measurements": 6, "cache_hits": 0, "reference_requests": 0}}
    (directory / "run.json").write_text(json.dumps(state))
    (directory / "trials.jsonl").write_text("\n".join(json.dumps(t) for t in trials) + "\n")
    (directory / "measurements/measurements.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return current


@pytest.mark.parametrize("problem", ["stopped", "compiler", "missing", "incorrect", "join", "key", "raw_cost", "flag", "ir", "counter", "relabelled"])
def test_fit_rejects_bad_measurement_provenance_or_incomplete_results(tmp_path, problem):
    directory = tmp_path / "grid"
    current = write_evidence(directory)
    meta = json.loads((directory / "run.json").read_text())
    records = [json.loads(line) for line in (directory / "measurements/measurements.jsonl").read_text().splitlines()]
    if problem == "stopped":
        meta["status"] = "stopped"
    elif problem == "compiler":
        current["binary_hash"] = "another-binary"
    elif problem == "counter":
        meta["counters"]["candidate_requests"] += 1
    elif problem == "missing":
        records.pop()
    elif problem == "incorrect":
        records[0]["correct"] = False
    elif problem == "join":
        records[0]["actions"] = [{"kind": "TILE", "axis": 0, "size": 64}]
    elif problem == "key":
        records[0]["key"] = "different"
    elif problem == "raw_cost":
        records[0]["execution_ms"] *= 2
    elif problem == "flag":
        records[0]["samples"][0]["drift"] = .2
    elif problem == "ir":
        records[0]["ir_hashes"]["llvm_ir"] = "different"
    else:
        records[0]["workload"]["split"] = "validation"
    (directory / "run.json").write_text(json.dumps(meta))
    (directory / "measurements/measurements.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    with pytest.raises(ValueError):
        fit(directory, current)


def test_artifact_load_checks_winner_and_selector_never_uses_program_or_hint(tmp_path):
    current = write_evidence(tmp_path / "grid")
    artifact = fit(tmp_path / "grid", current)
    path = tmp_path / "fixed.json"
    path.write_text(json.dumps(artifact))
    selector = fixed_selector(load_fixed(path, current))
    for hint in (1, 100, 10000):
        result = selector(object(), {"unseen": "program"}, hint)
        assert result == {"schedule": []}
        result["schedule"].append({"kind": "VECTORIZE"})
    artifact["selection"]["selected"] = artifact["config"]["candidates"][1]
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="evidence"):
        load_fixed(path, current)


def test_comparison_registers_and_hashes_fixed_artifact_without_measurement(tmp_path):
    current = write_evidence(tmp_path / "grid")
    path = tmp_path / "fixed.json"
    path.write_text(json.dumps(fit(tmp_path / "grid", current)))
    config = json.loads((ROOT / "configs/comparison-development.json").read_text())
    config["best_fixed"] = str(path)
    _, order = compare_deployments.prepare(config)
    selectors, artifacts = compare_deployments.selectors_for(config, current)
    assert {c["method"] for c in order} == {"reference", "heuristic", "best_fixed"}
    assert selectors["best_fixed"](None, None, 100) == {"schedule": []}
    assert artifacts["best_fixed"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_grid_check_only_never_starts_worker_or_creates_run(tmp_path, monkeypatch, capsys):
    def forbidden(*_):
        raise AssertionError("configuration inspection tried to execute")
    monkeypatch.setattr(collect_fixed_grid, "Worker", forbidden)
    monkeypatch.setattr(collect_fixed_grid, "provenance", forbidden)
    monkeypatch.setattr(sys, "argv", ["collect_fixed_grid", "--config", str(ROOT / "configs/fixed-training-grid.json"),
                                      "--output", str(tmp_path / "unused"), "--check-only"])
    collect_fixed_grid.main()
    output = json.loads(capsys.readouterr().out)
    assert (output["workloads"], output["candidates"], output["measurements"]) == (128, 16, 2048)
    assert not (tmp_path / "unused").exists()


def test_collection_preserves_failure_without_expanding_requests_or_retrying(tmp_path, monkeypatch):
    config, _, _ = specification()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "stopped"
    class WorkerStub:
        counters = Counter()
        class process:
            pid = 12345
    class Failure:
        def __init__(self, *_args, **limits):
            assert limits["max_new"] == limits["max_requests"] == 6
            self.calls = 0
        def measure(self, *_):
            self.calls += 1
            raise RuntimeError("synthetic incorrect output")
        def counters(self):
            return {"candidate_requests": self.calls}
    monkeypatch.setattr(collect_fixed_grid, "Worker", lambda *_: nullcontext(WorkerStub()))
    monkeypatch.setattr(collect_fixed_grid, "initialize", lambda *_: {})
    monkeypatch.setattr(collect_fixed_grid, "provenance", lambda *_: metadata())
    monkeypatch.setattr(collect_fixed_grid, "Evaluator", Failure)
    monkeypatch.setattr(collect_fixed_grid.os, "sched_setaffinity", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["collect_fixed_grid", "--config", str(path), "--output", str(output)])
    with pytest.raises(RuntimeError, match="incorrect output"):
        collect_fixed_grid.main()
    state = json.loads((output / "run.json").read_text())
    trials = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()]
    assert state["status"] == "stopped" and state["counters"]["candidate_requests"] == 1
    assert len(trials) == 1 and trials[0]["status"] == "started"
