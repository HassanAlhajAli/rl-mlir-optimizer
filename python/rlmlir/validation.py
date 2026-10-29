"""Traceable PPO checkpoint selection on the complete validation split."""
import hashlib
import json
import math
from pathlib import Path
import statistics

from .corpus import FrozenCorpus
from .experiment import require_compatible_provenance
from .kernels import GENERATORS
from .objective import deployment_ms, reward

PROTOCOL = "ppo-validation-v2"
REUSE_COUNTS = (1, 100, 10000)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def eligible_training(trained, current):
    require_compatible_provenance(trained, current, context="validation checkpoint selection")
    require(trained.get("status") in ("timesteps_complete", "budget_reached"),
            "validation requires a completed bounded training run; inspect stopped runs first")
    require(trained.get("objective") in ("runtime", "reuse") and type(trained.get("seed")) is int,
            "invalid training objective or seed")


def checkpoints_for(directory, behavior_count):
    require(type(behavior_count) is int and behavior_count >= 0, "nonnegative behavior checkpoint count required")
    directory = Path(directory).resolve()
    behavior = sorted(directory.glob("behavior-*.zip"), key=lambda p: int(p.stem.split("-")[1]))
    count = min(behavior_count, len(behavior))
    # Match the previous evenly spaced, floor-rounded selection, without numpy.
    indices = [i * (len(behavior) - 1) // (count - 1) if count > 1 else 0 for i in range(count)]
    paths = [behavior[i] for i in indices] + [directory / "latest-policy.zip"]
    return [{"path": str(path), "sha256": sha256(path)} for path in paths]


def summarize(records, checkpoints, objective, measurements, binary_hash, *, corpus=None):
    corpus = corpus or FrozenCorpus()
    require(objective in ("runtime", "reuse"), "invalid validation objective")
    workloads = {w["id"]: w for w in corpus.development("validation")}
    checkpoint_hashes = {c["path"]: c["sha256"] for c in checkpoints}
    require(checkpoints and len(checkpoint_hashes) == len(checkpoints), "duplicate or missing checkpoint")
    expected = {(path, identity, n) for path in checkpoint_hashes for identity in workloads for n in REUSE_COUNTS}
    by_key = {}
    for measurement in measurements:
        corpus.require_development(measurement["workload"])
        require(measurement["workload"]["split"] == "validation", "non-validation performance in selection")
        require(measurement["status"] == "accepted" and measurement.get("correct") is True,
                "unverified validation measurement")
        require(measurement["binary_hash"] == binary_hash and measurement["corpus_sha256"] == corpus.sha256,
                "validation measurement provenance mismatch")
        workload = measurement["workload"]
        identity = {"source": GENERATORS[workload["family"]](*workload["shape"]), "actions": measurement["actions"],
                    "binary": binary_hash, "input_seed": 7, "measurement_protocol": "paired-training-v2"}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        require(measurement["key"] == key and key not in by_key, "duplicate or wrong validation measurement identity")
        by_key[key] = measurement
    seen, used = set(), set()
    scores = {path: {} for path in checkpoint_hashes}
    for record in records:
        workload = record["workload"]
        corpus.require_development(workload)
        key = (record["checkpoint"], workload["id"], record["reuse"])
        require(key in expected and key not in seen and workload == workloads[workload["id"]],
                "duplicate, unexpected or non-validation selection")
        require(record["checkpoint_sha256"] == checkpoint_hashes[record["checkpoint"]], "checkpoint hash mismatch")
        seen.add(key)
        measured = by_key[record["measurement_key"]]
        baseline = by_key[record["reference_key"]]
        used.update((record["measurement_key"], record["reference_key"]))
        require(measured["workload"] == baseline["workload"] == workload
                and measured["actions"] == record["selection"]["schedule"] and baseline["actions"] == [],
                "validation outcome join mismatch")
        cost = deployment_ms(measured["compile_ms"], measured["execution_ms"], record["reuse"],
                             installation_ms=measured["installation_ms"], selection_ms=record["selection"]["selection_ms"])
        reference = deployment_ms(baseline["compile_ms"], baseline["execution_ms"], record["reuse"],
                                  installation_ms=baseline["installation_ms"])
        score = reward(baseline["execution_ms"], measured["execution_ms"]) if objective == "runtime" else reward(reference, cost)
        require(all(math.isclose(record[k], v, rel_tol=1e-9, abs_tol=1e-9)
                    for k, v in (("cost_ms", cost), ("reference_cost_ms", reference), ("reward", score))),
                "validation cost or reward mismatch")
        require(record["unresolved_drift"] == measured["unresolved_drift"], "validation drift flag mismatch")
        scores[record["checkpoint"]].setdefault(workload["family"], []).append(score)
    require(seen == expected and used == set(by_key), "incomplete validation coverage or unjoined measurements")
    summaries = [{"checkpoint": c["path"], "checkpoint_sha256": c["sha256"],
                  "equal_family_log_objective_improvement": statistics.mean(statistics.mean(v) for v in scores[c["path"]].values())}
                 for c in checkpoints]
    return summaries, max(summaries, key=lambda r: r["equal_family_log_objective_improvement"])


def audit(directory, current):
    """No model updates or native execution. Verify the saved selection evidence."""
    directory = Path(directory)
    meta = json.loads((directory / "run.json").read_text())
    require(meta.get("status") == "complete" and meta.get("selection_protocol") == PROTOCOL,
            "complete version-2 validation selection required")
    require_compatible_provenance(meta, current, context="validation audit")
    training = Path(meta["training_run"])
    require(sha256(training / "run.json") == meta["training_run_sha256"], "training run metadata changed")
    trained = json.loads((training / "run.json").read_text())
    eligible_training(trained, current)
    require(all(meta[k] == trained[k] for k in ("seed", "objective")), "validation belongs to another training run")
    require(meta["checkpoints"] == checkpoints_for(training, meta["behavior_checkpoint_count"]),
            "checkpoint plan or bytes changed")
    records = [json.loads(line) for line in (directory / "selections.jsonl").read_text().splitlines()]
    measurements = [json.loads(line) for line in (directory / "measurements/measurements.jsonl").read_text().splitlines()]
    require(all(m["revision"] == meta["revision"] for m in measurements), "validation measurement revision mismatch")
    summaries, winner = summarize(records, meta["checkpoints"], meta["objective"], measurements, meta["binary_hash"])
    require(meta["summaries"] == summaries and meta["selected"] == winner, "selected checkpoint or aggregate changed")
    require(meta["policy_sha256"] == sha256(directory / "policy.pt"), "exported policy changed")
    require(meta["selections_sha256"] == sha256(directory / "selections.jsonl")
            and meta["measurements_sha256"] == sha256(directory / "measurements/measurements.jsonl"),
            "validation evidence hash mismatch")
    require(meta["counters"]["candidate_requests"] == len(records)
            and meta["counters"]["new_measurements"] == len(measurements)
            and meta["counters"]["reference_requests"] == len(meta["checkpoints"]) * len(FrozenCorpus().development("validation")),
            "validation request accounting mismatch")
    return {"status": "passed", "scope": "saved checkpoint-selection coverage, costs, hashes and provenance",
            "objective": meta["objective"], "seed": meta["seed"], "cases": len(records),
            "checkpoints": len(meta["checkpoints"]), "selected": winner, "policy_sha256": meta["policy_sha256"]}
