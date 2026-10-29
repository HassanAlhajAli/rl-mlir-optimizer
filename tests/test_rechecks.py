"""Synthetic shared-access and failure controls, without native measurements."""
import copy
import json
import math
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
from rlmlir.corpus import DEFAULT_CORPUS, FrozenCorpus, load
from rlmlir.encoding import ACTIONS, OBSERVATION_SIZE
from rlmlir.experience import digest, load_pool, pool_manifest
from rlmlir.kernels import GENERATORS
from rlmlir.predictor import cost_examples
from rlmlir.rechecks import (collect_rechecks, fingerprint, load_rechecks,
                            measurement_access, planned_rechecks, select_demonstrations)
import pretrain
import fit_predictor as predictor_script


def measured(workload, actions, meta, compile_ms):
    identity = {"source": GENERATORS[workload["family"]](*workload["shape"]),
                "actions": actions, "binary": meta["binary_hash"], "input_seed": 7,
                "measurement_protocol": "paired-training-v2"}
    return {"key": digest(json.dumps(identity, sort_keys=True).encode()), "status": "accepted", "correct": True,
            "binary_hash": meta["binary_hash"], "corpus_sha256": meta["corpus_sha256"],
            "workload": workload, "actions": actions, "compile_ms": compile_ms,
            "installation_ms": 1., "execution_ms": .01, "unresolved_drift": False}


def source_fixture(directory):
    meta = {"binary_hash": "binary", "corpus_sha256": FrozenCorpus().sha256, "observation_version": 2,
            "measurement_protocol": "paired-training-v2", "compiler": "LLVM/MLIR 18.1.3",
            "target": "x86-64-v3", "revision": "synthetic", "status": "complete"}
    records, items = [], []
    for split in ("train", "validation"):
        workload = load(DEFAULT_CORPUS, split)[0]
        for index, cost in enumerate((10., 10.4, 10.45, 40.)):
            actions = ([] if index == 0 else [ACTIONS[index + 1]]) + [ACTIONS[0]]
            record = measured(workload, actions[:-1], meta, cost)
            records.append(record)
            steps = [{"observation": [0.] * OBSERVATION_SIZE, "next_observation": [0.] * OBSERVATION_SIZE,
                      "action": ACTIONS.index(action), "mask": [True] * len(ACTIONS),
                      "terminal": i == len(actions) - 1} for i, action in enumerate(actions)]
            items.append({"workload": workload, "reuse": 1, "actions": actions, "transitions": steps,
                          "outcome": {"measurement_key": record["key"], "analysis_ms": .1, "policy_ms": .2,
                                      "cost_ms": cost + 1.31}})
    (directory / "measurements").mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps(meta))
    write_lines(directory / "measurements/measurements.jsonl", records)
    write_lines(directory / "trajectories.jsonl", items)
    pool, sources = load_pool([directory], meta)
    return meta, pool, pool_manifest(pool, sources, meta)


def write_lines(path, items):
    path.write_text("".join(json.dumps(item) + "\n" for item in items))


class FakeEvaluator:
    def __init__(self, directory, meta, fail_at=None):
        self.directory, self.meta, self.fail_at = directory, meta, fail_at
        directory.mkdir(parents=True, exist_ok=True)
        self.requests, self.records = 0, {}

    def measure(self, workload, actions):
        if self.requests == self.fail_at:
            raise RuntimeError("deliberate measurement failure")
        self.requests += 1
        # The tiled leader wins its recheck; the former winner must still be shared.
        record = measured(workload, actions, self.meta, 4. if actions else 20.)
        if record["key"] not in self.records:
            self.records[record["key"]] = record
            with (self.directory / "measurements.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        return self.records[record["key"]]

    def counters(self):
        return {"candidate_requests": self.requests, "reference_requests": 0,
                "new_measurements": len(self.records), "cache_hits": self.requests - len(self.records)}


def bundle_fixture(tmp_path):
    meta, pool, initial = source_fixture(tmp_path / "source")
    directory = tmp_path / "rechecks"
    rows, manifest = collect_rechecks(FakeEvaluator(directory, meta), pool, initial, meta, directory)
    return meta, pool, initial, directory, rows, manifest


def test_all_leaders_shared_and_losing_rechecks_still_available(tmp_path):
    meta, pool, initial, directory, rows, manifest = bundle_fixture(tmp_path)
    assert len(planned_rechecks(pool)) == len(rows) == 6
    assert manifest["split_counts"] == {"train": 3, "validation": 3}
    assert {r["actions"][-1]["kind"] for r in rows} == {"STOP"}
    best = select_demonstrations(rows)
    assert len(best) == 2 and all(len(b["actions"]) == 2 for b in best)
    assert all(len(b["rechecked_candidates"]) == 3 for b in best)
    assert any(r["actions"] == [ACTIONS[0]] for r in rows)
    loaded, again = load_rechecks(directory, pool, initial, meta)
    assert loaded == rows and again == manifest
    assert measurement_access(initial, manifest) != measurement_access(initial)
    relocated = tmp_path / "relocated"
    shutil.copytree(directory, relocated)
    assert load_rechecks(relocated, pool, initial, meta)[1] == manifest


def test_cache_hits_and_reuse_do_not_multiply_timing_weight(tmp_path):
    meta, pool, initial = source_fixture(tmp_path / "source")
    # Simulate a second reuse context in the pinned source, plus repeated cached trajectories.
    path = tmp_path / "source/trajectories.jsonl"
    items = [json.loads(line) for line in path.read_text().splitlines()]
    extra = copy.deepcopy(items[0]); extra["reuse"] = 100
    extra["outcome"]["cost_ms"] += .99
    write_lines(path, items + [extra] + [items[0]] * 5)
    pool, sources = load_pool([path.parent], meta)
    initial = pool_manifest(pool, sources, meta)
    directory = tmp_path / "rechecks"
    rows, manifest = collect_rechecks(FakeEvaluator(directory, meta), pool, initial, meta, directory)
    assert manifest["counts"] == {"candidate_requests": 7, "reference_requests": 0, "new_measurements": 6, "cache_hits": 1}
    examples = cost_examples(pool + rows)
    stop = next(e for e in examples if e["workload_id"] == pool[0]["workload"]["id"])
    assert stop["distinct_measurements"] == 2
    assert math.exp(stop["log_costs"][0]) == pytest.approx(16.)  # median(11,21)
    assert len(examples) == 8


def test_partial_rechecks_preserved_but_not_loadable(tmp_path):
    meta, pool, initial = source_fixture(tmp_path / "source")
    directory = tmp_path / "rechecks"
    with pytest.raises(RuntimeError, match="deliberate"):
        collect_rechecks(FakeEvaluator(directory, meta, fail_at=1), pool, initial, meta, directory)
    assert len((directory / "requests.jsonl").read_text().splitlines()) == 1
    assert (directory / "measurements.jsonl").exists()
    assert not (directory / "shared-rechecks.json").exists()
    with pytest.raises(FileNotFoundError):
        load_rechecks(directory, pool, initial, meta)


@pytest.mark.parametrize("fault", ["bytes", "cost", "indices", "missing", "duplicate", "count",
                                   "compiler", "pool", "failed", "corpus", "drift", "schedule", "split"])
def test_corrupted_recheck_evidence_is_rejected(tmp_path, fault):
    meta, pool, initial, directory, _, manifest = bundle_fixture(tmp_path)
    path = directory / "requests.jsonl"
    requests = [json.loads(line) for line in path.read_text().splitlines()]
    measurements = [json.loads(line) for line in (directory / "measurements.jsonl").read_text().splitlines()]
    if fault == "bytes":
        path.write_text(path.read_text() + " ")
    elif fault == "cost":
        requests[0]["cost_ms"] += 1
    elif fault == "indices":
        requests[0]["source_indices"] = [1]
    elif fault == "missing":
        requests.pop()
    elif fault == "duplicate":
        requests.append(requests[-1])
    elif fault == "count":
        manifest["counts"]["cache_hits"] += 1
    elif fault == "compiler":
        manifest["compatibility"]["binary_hash"] = "other"
    elif fault == "pool":
        manifest["initial_pool_fingerprint"] = "other"
    elif fault == "failed":
        measurements[0]["status"] = "failed"
    elif fault == "corpus":
        del measurements[0]["corpus_sha256"]
    elif fault == "drift":
        del measurements[0]["unresolved_drift"]
    elif fault == "schedule":
        requests[0]["measurement_key"] = requests[1]["measurement_key"]
    elif fault == "split":
        measurements[0]["workload"]["split"] = "validation"
    if fault != "bytes":
        write_lines(path, requests)
        write_lines(directory / "measurements.jsonl", measurements)
        # Re-sign hashes to exercise semantic checks, not only byte mismatch.
        manifest["request_sha256"] = digest(path.read_bytes())
        manifest["measurement_sha256"] = digest((directory / "measurements.jsonl").read_bytes())
        manifest["fingerprint"] = fingerprint(manifest)
        (directory / "shared-rechecks.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, KeyError)):
        load_rechecks(directory, pool, initial, meta)


def test_changed_shared_pool_rejected_before_predictor_creation(tmp_path, monkeypatch):
    meta, _, _, directory, _, _ = bundle_fixture(tmp_path)
    source = tmp_path / "source"
    trajectory = source / "trajectories.jsonl"
    trajectory.write_text(trajectory.read_text() + trajectory.read_text().splitlines()[0] + "\n")
    monkeypatch.setattr(sys, "argv", ["predictor", "--runs", str(source), "--output", str(tmp_path / "out"), "--rechecks", str(directory)])
    monkeypatch.setattr(predictor_script, "provenance", lambda _: copy.deepcopy(meta))
    monkeypatch.setattr(predictor_script, "CostPredictor", lambda: pytest.fail("invalid bundle reached model creation"))
    with pytest.raises(ValueError, match="different initial"):
        predictor_script.main()


def test_both_entrypoints_share_access_and_preserve_split_roles(tmp_path, monkeypatch):
    source = tmp_path / "source"
    meta, _, _ = source_fixture(source)
    output = tmp_path / "pretrain"
    class FakeWorker:
        def __init__(self, *_a): pass
        def __enter__(self): return self
        def __exit__(self, *_a): pass
    monkeypatch.setattr(pretrain, "Worker", FakeWorker)
    monkeypatch.setattr(pretrain, "initialize", lambda _: {})
    monkeypatch.setattr(pretrain, "provenance", lambda _: copy.deepcopy(meta))
    monkeypatch.setattr(pretrain, "Evaluator", lambda _w, path, *_a, **_k: FakeEvaluator(path, meta))
    monkeypatch.setattr(pretrain, "create_policy", lambda _: object())
    def actor(_p, train, validation, **_k):
        assert len(train) == len(validation) == 1
        assert train[0]["workload"]["split"] == "train"
        assert validation[0]["workload"]["split"] == "validation"
        return []
    def critic(_p, train, **_k):
        assert len(train) == 4 and all(t["workload"]["split"] == "train" for t in train)
        return []
    monkeypatch.setattr(pretrain, "fit_actor", actor)
    monkeypatch.setattr(pretrain, "fit_critic", critic)
    monkeypatch.setattr(pretrain, "save_policy", lambda *_a: "synthetic-policy")
    monkeypatch.setattr(sys, "argv", ["pretrain", "--runs", str(source), "--output", str(output)])
    pretrain.main()
    pre_meta = json.loads((output / "run.json").read_text())
    assert pre_meta["status"] == "complete"
    monkeypatch.setattr(predictor_script, "provenance", lambda _: copy.deepcopy(meta))
    def fit(_m, train, validation, **_k):
        assert len(train) == len(validation) == 4
        assert all(t["split"] == "train" for t in train)
        assert all(t["split"] == "validation" for t in validation)
        assert sum(t["distinct_measurements"] for t in train) == 7
        return []
    monkeypatch.setattr(predictor_script, "fit_predictor", fit)
    prediction = tmp_path / "predictor"
    monkeypatch.setattr(sys, "argv", ["predictor", "--runs", str(source), "--output", str(prediction), "--rechecks", str(output / "rechecks")])
    predictor_script.main()
    pred_meta = json.loads((prediction / "run.json").read_text())
    assert pred_meta["measurement_access"] == pre_meta["measurement_access"]
