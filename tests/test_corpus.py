"""Split protection and experience joins; no held-out performance measurements."""
import copy
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
from rlmlir.corpus import FrozenCorpus, DEFAULT_CORPUS, load
from rlmlir.evaluator import Evaluator
from pretrain import load_pool
from rlmlir.encoding import ACTIONS, OBSERVATION_SIZE
from rlmlir.experience import check_transitions, pool_manifest, require_manifest
from rlmlir.kernels import GENERATORS
import pretrain
import fit_predictor as predictor_script


class NoCompiler:
    counters = Counter()

    def request(self, *_args, **_kwargs):
        raise AssertionError("invalid workload reached compiler")


def write_corpus(path, workloads):
    data = json.dumps({"schema_version": 1, "workloads": workloads}).encode()
    path.write_bytes(data)
    path.with_suffix(".sha256").write_text(hashlib.sha256(data).hexdigest())
    return path


@pytest.mark.parametrize("field,value", [
    ("id", "unknown"), ("family", "row_reduce"), ("shape", [1, 1, 1]),
    ("group", "wrong-group"), ("split", "validation"),
])
def test_modified_identity_rejected_before_request_or_compile(tmp_path, field, value):
    workload = load(DEFAULT_CORPUS, "train")[0]
    workload[field] = value
    evaluator = Evaluator(NoCompiler(), tmp_path, "binary", "revision")
    with pytest.raises(ValueError, match="corpus"):
        evaluator.measure(workload, [])
    assert evaluator.requests == evaluator.new_measurements == 0
    assert not evaluator.path.exists()


@pytest.mark.parametrize("split", ["heldout_shape", "heldout_structure"])
def test_relabelled_heldout_identity_remains_sealed(tmp_path, split):
    # Read only public shape metadata, never construct native code or measure it.
    workload = next(w for w in json.loads(DEFAULT_CORPUS.read_text())["workloads"] if w["split"] == split)
    workload["split"] = "train"
    evaluator = Evaluator(NoCompiler(), tmp_path, "binary", "revision")
    with pytest.raises(ValueError, match="sealed"):
        evaluator.measure(workload, [])
    assert evaluator.requests == 0 and not evaluator.path.exists()


def test_returned_workload_cannot_mutate_corpus_authority():
    corpus = FrozenCorpus()
    workload = corpus.development("train")[0]
    original = copy.deepcopy(workload)
    workload["shape"][0] += 1
    with pytest.raises(ValueError, match="identity"):
        corpus.require_development(workload)
    corpus.require_development(original)


def test_actual_manifest_bytes_must_match_hash(tmp_path):
    path = write_corpus(tmp_path / "corpus.json", load(DEFAULT_CORPUS, "train")[:2])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        FrozenCorpus(path)


@pytest.mark.parametrize("problem", ["id", "shape", "group"])
def test_manifest_rejects_ambiguous_membership(tmp_path, problem):
    first, second = load(DEFAULT_CORPUS, "train")[:2]
    if problem == "id":
        second["id"] = first["id"]
    elif problem == "shape":
        second["shape"] = first["shape"]
    else:
        second.update(group=first["group"], split="validation")
    path = write_corpus(tmp_path / "corpus.json", [first, second])
    with pytest.raises(ValueError, match="duplicate|crossed"):
        FrozenCorpus(path)


@pytest.mark.parametrize("problem", ["relabelled", "key", "binary", "incorrect"])
def test_seed_cache_cannot_bypass_identity_guard(tmp_path, problem):
    workload = load(DEFAULT_CORPUS, "train")[0]
    evaluator = Evaluator(NoCompiler(), tmp_path / "identity", "binary", "revision")
    from rlmlir.kernels import GENERATORS
    source = GENERATORS[workload["family"]](*workload["shape"])
    record = {"status": "accepted", "correct": True, "binary_hash": "binary",
              "workload": workload, "actions": [], "key": evaluator._key(source, [])}
    if problem == "relabelled":
        record["workload"]["split"] = "validation"
    elif problem == "key":
        record["key"] = "incorrect-key"
    elif problem == "binary":
        record["binary_hash"] = "different-binary"
    else:
        record["correct"] = False
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        Evaluator(NoCompiler(), tmp_path / "resumed", "binary", "revision", seed_measurements=path)
    assert not (tmp_path / "resumed/measurements.jsonl").exists()


def pool_fixture(directory, index=0):
    workload = load(DEFAULT_CORPUS, "train")[index]
    meta = {"binary_hash": "binary", "corpus_sha256": FrozenCorpus().sha256,
            "observation_version": 2, "measurement_protocol": "paired-training-v2",
            "compiler": "LLVM/MLIR 18.1.3", "target": "x86-64-v3", "revision": "source", "status": "complete"}
    source = GENERATORS[workload["family"]](*workload["shape"])
    key = hashlib.sha256(json.dumps({"source":source,"actions":[],"binary":"binary","input_seed":7,
                                    "measurement_protocol":"paired-training-v2"},sort_keys=True).encode()).hexdigest()
    record = {"key": key, "status": "accepted", "correct": True, "corpus_sha256": meta["corpus_sha256"],
              "binary_hash": "binary", "workload": workload, "actions": [],
              "compile_ms": 1., "installation_ms": 2., "execution_ms": .5}
    step = {"observation":[0.] * OBSERVATION_SIZE, "next_observation":[0.] * OBSERVATION_SIZE,
            "action":0, "mask":[True] * len(ACTIONS), "terminal":True}
    trajectory = {"workload": copy.deepcopy(workload), "reuse": 100, "actions": [{"kind": "STOP"}],
                  "transitions":[step], "outcome": {"measurement_key": key, "analysis_ms": .1, "policy_ms": .2, "cost_ms":53.3}}
    (directory / "measurements").mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps(meta))
    return meta, record, trajectory


@pytest.mark.parametrize("problem", [None, "workload", "actions", "compiler", "incorrect", "relabelled"])
def test_experience_requires_same_accepted_workload_and_schedule(tmp_path, problem):
    meta, record, trajectory = pool_fixture(tmp_path)
    if problem == "workload":
        record["workload"] = load(DEFAULT_CORPUS, "train")[1]
    elif problem == "actions":
        record["actions"] = [{"kind": "TILE", "axis": 0, "size": 4}]
    elif problem == "compiler":
        record["binary_hash"] = "different-binary"
    elif problem == "incorrect":
        record["correct"] = False
    elif problem == "relabelled":
        trajectory["workload"]["split"] = "validation"
    (tmp_path / "measurements/measurements.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "trajectories.jsonl").write_text(json.dumps(trajectory) + "\n")
    if problem:
        with pytest.raises(ValueError):
            load_pool([tmp_path], meta)
    else:
        pool, _ = load_pool([tmp_path], meta)
        assert len(pool) == 1 and pool[0]["cost_ms"] == pytest.approx(53.3)


def save_pool_fixture(directory, record, trajectory):
    (directory / "measurements/measurements.jsonl").write_text(json.dumps(record) + "\n")
    (directory / "trajectories.jsonl").write_text(json.dumps(trajectory) + "\n")


@pytest.mark.parametrize("fault", ["key", "duplicate_measurement", "failed_measurement", "failed_source",
    "outcome_cost", "negative_cost", "missing_installation", "zero_execution", "nan_observation", "mask",
    "action", "unterminated", "infinite_probability", "selection_missing"])
def test_experience_corruption_is_rejected_before_training(tmp_path, fault):
    meta, record, item = pool_fixture(tmp_path)
    if fault == "key":
        record["key"] = item["outcome"]["measurement_key"] = "made-up-identity"
    elif fault == "failed_measurement":
        record["status"] = "failed"
    elif fault == "failed_source":
        (tmp_path / "run.json").write_text(json.dumps({**meta, "status":"stopped"}))
    elif fault == "outcome_cost":
        item["outcome"]["cost_ms"] += 1.
    elif fault == "negative_cost":
        record["installation_ms"] = -1.
    elif fault == "missing_installation":
        del record["installation_ms"]
    elif fault == "zero_execution":
        record["execution_ms"] = 0.
    elif fault == "nan_observation":
        item["transitions"][0]["observation"][0] = float("nan")
    elif fault == "mask":
        item["transitions"][0]["mask"][0] = False
    elif fault == "action":
        item["transitions"][0]["action"] = -1
    elif fault == "unterminated":
        item["transitions"][0]["terminal"] = False
    elif fault == "infinite_probability":
        item["transitions"][0]["behavior_log_probability"] = float("inf")
    elif fault == "selection_missing":
        item["outcome"]["analysis_ms"] = None
    save_pool_fixture(tmp_path, record, item)
    if fault == "duplicate_measurement":
        with (tmp_path / "measurements/measurements.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        load_pool([tmp_path], meta)


def test_runtime_trajectory_reconstructs_complete_cost_for_reuse_training(tmp_path):
    meta, record, item = pool_fixture(tmp_path)
    item["objective"] = "runtime"
    item["outcome"]["cost_ms"] = .5
    save_pool_fixture(tmp_path, record, item)
    pool, sources = load_pool([tmp_path], meta)
    assert pool[0]["cost_ms"] == pytest.approx(53.3)
    assert pool[0]["outcome"]["cost_ms"] == .5
    assert {"run_sha256", "measurement_sha256", "trajectory_sha256"} <= sources[0].keys()


def test_legacy_corpus_binding_is_explicit_but_mismatched_hash_rejected(tmp_path):
    meta, record, item = pool_fixture(tmp_path)
    del record["corpus_sha256"]
    save_pool_fixture(tmp_path, record, item)
    _, sources = load_pool([tmp_path], meta)
    assert sources[0]["measurements_without_local_corpus_hash"] == 1
    record["corpus_sha256"] = "incorrect"
    save_pool_fixture(tmp_path, record, item)
    with pytest.raises(ValueError, match="provenance"):
        load_pool([tmp_path], meta)


def test_manifest_pins_measurements_and_is_location_independent(tmp_path):
    first, relocated = tmp_path / "first", tmp_path / "relocated"
    meta, record, item = pool_fixture(first)
    save_pool_fixture(first, record, item)
    pool, sources = load_pool([first], meta)
    manifest = pool_manifest(pool, sources, meta)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    shutil.copytree(first, relocated)
    copied, copied_sources = load_pool([relocated], meta, manifest_path=manifest_path)
    assert pool_manifest(copied, copied_sources, meta)["fingerprint"] == manifest["fingerprint"]
    # A legitimate but changed timing must require a new declared input manifest.
    record["compile_ms"] += 1.
    item["outcome"]["cost_ms"] += 1.
    save_pool_fixture(relocated, record, item)
    with pytest.raises(ValueError, match="pinned manifest"):
        load_pool([relocated], meta, manifest_path=manifest_path)
    corrupted = copy.deepcopy(manifest)
    corrupted["trajectories"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        require_manifest(corrupted, manifest)


def test_duplicate_sources_or_copies_do_not_multiply_training_weight(tmp_path):
    first, copied = tmp_path / "first", tmp_path / "copied"
    meta, record, item = pool_fixture(first)
    save_pool_fixture(first, record, item)
    with pytest.raises(ValueError, match="duplicate experience"):
        load_pool([first, first / "."], meta)
    shutil.copytree(first, copied)
    with pytest.raises(ValueError, match="byte-identical"):
        load_pool([first, copied], meta)


def test_source_order_is_explicit_and_budget_limited_source_is_visible(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    meta, record, item = pool_fixture(first)
    save_pool_fixture(first, record, item)
    other, record2, item2 = pool_fixture(second, index=1)
    other["status"] = "budget_reached"
    (second / "run.json").write_text(json.dumps(other))
    save_pool_fixture(second, record2, item2)
    pool, sources = load_pool([first, second], meta)
    reverse, reversed_sources = load_pool([second, first], meta)
    assert sources[1]["status"] == "budget_reached"
    assert pool_manifest(pool, sources, meta)["fingerprint"] != pool_manifest(reverse, reversed_sources, meta)["fingerprint"]


def test_transition_chain_cannot_jump_states_or_continue_after_stop(tmp_path):
    _, _, item = pool_fixture(tmp_path)
    item["actions"] = [ACTIONS[2], ACTIONS[0]]
    item["transitions"] *= 2
    item["transitions"] = copy.deepcopy(item["transitions"])
    # Break shared list identity before preparing the two independent states.
    item["transitions"][0] = copy.deepcopy(item["transitions"][0])
    item["transitions"][0].update(action=2, terminal=False)
    check_transitions(item)
    item["transitions"][1]["observation"][0] = 1.
    with pytest.raises(ValueError, match="disconnected"):
        check_transitions(item)
    item["transitions"][1]["observation"][0] = 0.
    item["actions"][0] = ACTIONS[0]
    item["transitions"][0]["action"] = 0
    with pytest.raises(ValueError, match="after termination"):
        check_transitions(item)


@pytest.mark.parametrize("learner", [pretrain, predictor_script])
def test_pinned_pool_mismatch_stops_both_learners_before_work(tmp_path, monkeypatch, learner):
    source = tmp_path / "source"
    meta, record, item = pool_fixture(source)
    save_pool_fixture(source, record, item)
    pool, sources = load_pool([source], meta)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(pool_manifest(pool, sources, meta)))
    record["compile_ms"] += 1.
    item["outcome"]["cost_ms"] += 1.
    save_pool_fixture(source, record, item)
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(sys, "argv", ["learner", "--runs", str(source), "--output", str(output),
                                      "--pool-manifest", str(manifest_path)])
    monkeypatch.setattr(learner, "provenance", lambda _: meta)
    def forbidden(*_a, **_k):
        pytest.fail("invalid input reached model training or compiler")
    if learner is pretrain:
        monkeypatch.setattr(learner, "Worker", forbidden)
        monkeypatch.setattr(learner, "fit_actor", forbidden)
    else:
        monkeypatch.setattr(learner, "CostPredictor", forbidden)
    with pytest.raises(ValueError, match="pinned manifest"):
        learner.main()
    assert not output.exists()
