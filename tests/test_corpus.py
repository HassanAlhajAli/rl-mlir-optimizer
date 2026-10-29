"""Split protection and experience joins; no held-out performance measurements."""
import copy
import hashlib
import json
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


def pool_fixture(directory):
    workload = load(DEFAULT_CORPUS, "train")[0]
    meta = {"binary_hash": "binary", "corpus_sha256": FrozenCorpus().sha256,
            "observation_version": 2, "measurement_protocol": "paired-training-v2",
            "compiler": "LLVM/MLIR 18.1.3", "target": "x86-64-v3", "revision": "source"}
    record = {"key": "measurement", "status": "accepted", "correct": True,
              "binary_hash": "binary", "workload": workload, "actions": [],
              "compile_ms": 1., "installation_ms": 2., "execution_ms": .5}
    trajectory = {"workload": copy.deepcopy(workload), "reuse": 100, "actions": [{"kind": "STOP"}],
                  "outcome": {"measurement_key": "measurement", "analysis_ms": .1, "policy_ms": .2}}
    (directory / "measurements").mkdir()
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
