"""Synthetic checkpoint-selection round trip and ten-run readiness boundaries."""
import copy
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
import validate_ppo
from rlmlir import campaign_readiness
from rlmlir.corpus import FrozenCorpus
from rlmlir.kernels import GENERATORS
from rlmlir.validation import audit, checkpoints_for, sha256


def metadata():
    return {"revision": "synthetic-test", "binary_hash": "synthetic-test", "corpus_sha256": FrozenCorpus().sha256,
            "compiler": "LLVM/MLIR 18.1.3", "target": "x86-64-v3", "observation_version": 2,
            "measurement_protocol": "paired-training-v2"}


def create_training(path, **changes):
    path.mkdir(parents=True)
    trained = {**metadata(), "status": "budget_reached", "objective": "reuse", "seed": 7, **changes}
    (path / "run.json").write_text(json.dumps(trained))
    (path / "behavior-256.zip").write_bytes(b"synthetic initial checkpoint")
    (path / "latest-policy.zip").write_bytes(b"synthetic trained checkpoint")
    return trained


def run_synthetic_validation(tmp_path, monkeypatch, *, failure=None, training_status="budget_reached"):
    training, output = tmp_path / "training", tmp_path / "validation"
    create_training(training, status=training_status)
    class Worker:
        counters = Counter()
        class process:
            pid = 12345
    class Policy:
        def __init__(self, latest):
            self.latest = latest
        def set_training_mode(self, mode):
            assert mode is False
    class Model:
        def __init__(self, path):
            self.policy = Policy(Path(path).name == "latest-policy.zip")
        @classmethod
        def load(cls, path, **_):
            if failure == "changed_checkpoint":
                Path(path).write_bytes(b"changed during validation")
            return cls(path)
    class Evaluator:
        def __init__(self, _worker, directory, binary_hash, revision, **limits):
            assert limits["max_requests"] == 192 and limits["max_new"] == 224
            directory.mkdir(parents=True)
            self.path = directory / "measurements.jsonl"
            self.records, self.counts = {}, Counter()
        def measure(self, workload, actions, reference=False):
            self.counts["reference_requests" if reference else "candidate_requests"] += 1
            if failure == "measurement" and not reference:
                raise RuntimeError("synthetic compiler failure")
            source = GENERATORS[workload["family"]](*workload["shape"])
            key = hashlib.sha256(json.dumps({"source": source, "actions": actions, "binary": "synthetic-test",
                                            "input_seed": 7, "measurement_protocol": "paired-training-v2"}, sort_keys=True).encode()).hexdigest()
            if key in self.records:
                self.counts["cache_hits"] += 1
                return self.records[key]
            record = {**metadata(), "key": key, "workload": workload, "actions": actions, "status": "accepted",
                      "correct": True, "compile_ms": 6. if actions else 10., "installation_ms": 1.,
                      "execution_ms": .5 if actions else 1., "unresolved_drift": False}
            self.records[key] = record
            with self.path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            self.counts["new_measurements"] += 1
            return record
        def counters(self):
            return dict(self.counts)
    def choose(_worker, _workload, _reuse, *, policy, deterministic):
        assert deterministic
        return {"schedule": [{"kind": "TILE", "axis": 0, "size": 8}] if policy.latest else [], "selection_ms": .1}
    def save(_policy, path, _meta):
        path.write_bytes(b"synthetic exported policy")
        return sha256(path)
    monkeypatch.setattr(validate_ppo, "Worker", lambda *_: nullcontext(Worker()))
    monkeypatch.setattr(validate_ppo, "Evaluator", Evaluator)
    monkeypatch.setattr(validate_ppo, "MaskablePPO", Model)
    monkeypatch.setattr(validate_ppo, "sample_schedule", choose)
    monkeypatch.setattr(validate_ppo, "save_policy", save)
    monkeypatch.setattr(validate_ppo, "initialize", lambda *_: {})
    monkeypatch.setattr(validate_ppo, "provenance", lambda *_: metadata())
    monkeypatch.setattr(validate_ppo.os, "sched_setaffinity", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["validate_ppo", "--run", str(training), "--output", str(output), "--behavior-checkpoints", "1"])
    return training, output


def test_complete_validation_round_trip_checks_all_cases_and_winner(tmp_path, monkeypatch):
    _, output = run_synthetic_validation(tmp_path, monkeypatch)
    validate_ppo.main()
    result = audit(output, metadata())
    assert result["status"] == "passed" and result["cases"] == 192 and result["checkpoints"] == 2
    assert Path(result["selected"]["checkpoint"]).name == "latest-policy.zip"


@pytest.mark.parametrize("problem", ["missing", "duplicate", "cost", "wrong_workload", "checkpoint_hash", "winner", "policy", "counts", "training_metadata"])
def test_audit_rejects_partial_or_altered_validation_evidence(tmp_path, monkeypatch, problem):
    training, output = run_synthetic_validation(tmp_path, monkeypatch)
    validate_ppo.main()
    rows = [json.loads(line) for line in (output / "selections.jsonl").read_text().splitlines()]
    meta = json.loads((output / "run.json").read_text())
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif problem == "cost":
        rows[0]["reward"] += 1
    elif problem == "wrong_workload":
        rows[0]["workload"] = FrozenCorpus().development("train")[0]
    elif problem == "checkpoint_hash":
        rows[0]["checkpoint_sha256"] = "different"
    elif problem == "winner":
        meta["selected"] = meta["summaries"][0]
    elif problem == "policy":
        (output / "policy.pt").write_bytes(b"different policy")
    elif problem == "counts":
        meta["counters"]["candidate_requests"] += 1
    else:
        (training / "run.json").write_text((training / "run.json").read_text() + " ")
    (output / "run.json").write_text(json.dumps(meta))
    (output / "selections.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError):
        audit(output, metadata())


@pytest.mark.parametrize("failure", ["measurement", "changed_checkpoint"])
def test_failed_validation_preserves_counters_and_does_not_export_winner(tmp_path, monkeypatch, failure):
    _, output = run_synthetic_validation(tmp_path, monkeypatch, failure=failure)
    with pytest.raises((ValueError, RuntimeError)):
        validate_ppo.main()
    state = json.loads((output / "run.json").read_text())
    assert state["status"] == "stopped" and state["counters"]["candidate_requests"] > 0
    assert not (output / "policy.pt").exists()


def test_stopped_training_is_rejected_before_validation_worker_or_output(tmp_path, monkeypatch):
    _, output = run_synthetic_validation(tmp_path, monkeypatch, training_status="stopped")
    monkeypatch.setattr(validate_ppo, "Worker", lambda *_: pytest.fail("ineligible training started worker"))
    with pytest.raises(ValueError, match="completed bounded"):
        validate_ppo.main()
    assert not output.exists()


def test_checkpoint_subset_is_evenly_spaced_and_hashes_bytes(tmp_path):
    for step in (256, 512, 768, 1024, 1280):
        (tmp_path / f"behavior-{step}.zip").write_bytes(str(step).encode())
    (tmp_path / "latest-policy.zip").write_bytes(b"latest")
    chosen = checkpoints_for(tmp_path, 3)
    assert [Path(c["path"]).name for c in chosen] == ["behavior-256.zip", "behavior-768.zip", "behavior-1280.zip", "latest-policy.zip"]
    assert all(c["sha256"] == sha256(c["path"]) for c in chosen)
    assert len(checkpoints_for(tmp_path, 0)) == 1


def campaign_fixture(tmp_path):
    config = json.loads((ROOT / "configs/ppo-campaign.json").read_text())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    contracts = campaign_readiness.definitions(config["jobs"])
    for name, contract in contracts.items():
        create_training(campaign / name, **{k: v for k, v in contract.items() if k != "name"}, timesteps=256,
                        counters={"candidate_requests": 100, "new_measurements": 90})
    (campaign / "campaign.json").write_text(json.dumps({**metadata(), "config_sha256": sha256(config_path)}))
    queue = {"status": "complete", "elapsed_seconds": 10., "definition": config["jobs"],
             "jobs": [{**job, "status": "complete", "exit_code": 0} for job in config["jobs"]]}
    (campaign / "queue.json").write_text(json.dumps(queue))
    return campaign, config_path, config, queue


@pytest.mark.parametrize("problem", ["missing_seed", "duplicate_seed", "unequal_budget", "renamed_job"])
def test_campaign_cannot_hide_a_seed_or_change_only_one_allocation(tmp_path, problem):
    directory, path, config, _ = campaign_fixture(tmp_path)
    if problem == "missing_seed":
        config["jobs"].pop()
    elif problem == "duplicate_seed":
        config["jobs"][-1] = copy.deepcopy(config["jobs"][0])
    elif problem == "unequal_budget":
        argv = config["jobs"][0]["argv"]
        argv[argv.index("--max-new") + 1] = "1537"
    else:
        config["jobs"][0]["name"] = "renamed"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        campaign_readiness.report(directory, path, metadata())


def test_campaign_reports_unstarted_failed_and_incompatible_runs(tmp_path):
    directory, path, _, queue = campaign_fixture(tmp_path)
    queue["jobs"] = queue["jobs"][:4]
    queue["status"] = "stopped"
    queue["jobs"][-1].update(status="failed", exit_code=1)
    (directory / "queue.json").write_text(json.dumps(queue))
    current = {**metadata(), "binary_hash": "repaired-backend"}
    report = campaign_readiness.report(directory, path, current)
    assert len(report["jobs"]) == 10
    assert Counter(row["queue_status"] for row in report["jobs"]) == {"complete": 3, "failed": 1, "not_started": 6}
    assert all(row["incompatible_fields"] == ["binary_hash"] for row in report["jobs"][:4])
    assert not report["all_ten_selections_accounted_for"] and not report["heldout_access"]


def test_completed_training_alone_is_not_validation_or_full_study_freeze(tmp_path, monkeypatch):
    directory, path, _, _ = campaign_fixture(tmp_path)
    report = campaign_readiness.report(directory, path, metadata())
    assert all(row["validation_status"] == "missing" for row in report["jobs"])
    assert not report["all_ten_selections_accounted_for"]
    validations = {}
    for row in report["jobs"]:
        validation = tmp_path / ("validation-" + row["name"])
        validation.mkdir()
        (validation / "run.json").write_text(json.dumps({"training_run": str(directory / row["name"])}))
        validations[row["name"]] = str(validation)
    # This test exercises the coverage gate. The real validation auditor is
    # separately checked above against complete synthetic measured evidence.
    monkeypatch.setattr(campaign_readiness, "audit_validation", lambda *_: {"status": "passed"})
    ready = campaign_readiness.report(directory, path, metadata(), validations)
    assert ready["all_ten_selections_accounted_for"]
    assert not ready["training_update_evidence_checked"]
    assert not ready["whole_study_frozen"] and not ready["heldout_access"]
    validations[report["jobs"][0]["name"]] = validations[report["jobs"][1]["name"]]
    assert not campaign_readiness.report(directory, path, metadata(), validations)["all_ten_selections_accounted_for"]
