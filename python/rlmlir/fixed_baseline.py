"""One global schedule, selected on a complete, training-only comparison grid."""
import copy
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .corpus import FrozenCorpus
from .encoding import ACTION_INDEX
from .experiment import require_compatible_provenance
from .objective import deployment_ms, reward

ROLE = "fixed-schedule training grid"
PROTOCOL = "family-balanced-fixed-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare(config, corpus=None):
    corpus = corpus or FrozenCorpus()
    fields = {"schema_version", "workload_ids", "candidates", "reuse_counts", "order_seed"}
    if set(config) != fields or config["schema_version"] != 1:
        raise ValueError("unsupported fixed-grid configuration")
    available = {w["id"]: w for w in corpus.development("train")}
    ids = config["workload_ids"]
    if not ids or len(ids) != len(set(ids)) or any(i not in available for i in ids):
        raise ValueError("unique canonical training workload IDs required")
    reuse = config["reuse_counts"]
    if (not reuse or any(type(n) is not int or not 1 <= n <= 10000000 for n in reuse)
            or len(set(reuse)) != len(reuse) or type(config["order_seed"]) is not int):
        raise ValueError("unique bounded reuse counts and integer order seed required")
    candidates = config["candidates"]
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("at least two fixed candidates required")
    names, schedules = set(), set()
    for candidate in candidates:
        if set(candidate) != {"name", "schedule"} or not isinstance(candidate["name"], str) or not candidate["name"]:
            raise ValueError("invalid fixed candidate")
        schedule = candidate["schedule"]
        if not isinstance(schedule, list) or len(schedule) > 6:
            raise ValueError("invalid schedule length")
        for index, action in enumerate(schedule):
            if (json.dumps(action, sort_keys=True) not in ACTION_INDEX
                    or action["kind"] == "STOP"
                    or (action["kind"] == "VECTORIZE" and index != len(schedule) - 1)):
                raise ValueError("invalid fixed action sequence")
        identity = digest(schedule)
        if candidate["name"] in names or identity in schedules:
            raise ValueError("duplicate fixed candidate name or schedule")
        names.add(candidate["name"])
        schedules.add(identity)
    if candidates[0] != {"name": "reference", "schedule": []}:
        raise ValueError("the first fixed candidate must be the untransformed reference")
    return [available[i] for i in ids]


def collection_order(config):
    order = [{"workload_id": i, "candidate": c["name"]}
             for i in config["workload_ids"] for c in config["candidates"]]
    random.Random(config["order_seed"]).shuffle(order)
    return order


def select(config, cells, corpus=None):
    """Equal weight per family, then program, then reuse; no incomplete-cell fallback.

    The cells contain one measured C/T pair per workload/schedule. Reuse levels
    share that evidence rather than pretending cached copies are new samples.
    """
    corpus = corpus or FrozenCorpus()
    workloads = {w["id"]: w for w in prepare(config, corpus)}
    expected = {(w, c["name"]) for w in workloads for c in config["candidates"]}
    indexed = {}
    for cell in cells:
        key = (cell["workload_id"], cell["candidate"])
        if key not in expected or key in indexed:
            raise ValueError("unexpected or duplicate fixed-grid cell")
        for field in ("compile_ms", "installation_ms", "execution_ms", "selection_ms"):
            if type(cell[field]) not in (int, float) or not math.isfinite(cell[field]) or cell[field] < 0:
                raise ValueError("invalid fixed-grid timing")
        if type(cell["unresolved_drift"]) is not bool:
            raise ValueError("missing fixed-grid timing flag")
        indexed[key] = cell
    if set(indexed) != expected:
        raise ValueError("incomplete fixed grid; every candidate needs every training workload")
    def cost(cell, reuse):
        return deployment_ms(cell["compile_ms"], cell["execution_ms"], reuse,
                             selection_ms=cell["selection_ms"], installation_ms=cell["installation_ms"])
    leaderboard = []
    for candidate in config["candidates"]:
        families = defaultdict(list)
        for identity, workload in workloads.items():
            reference, chosen = indexed[identity, "reference"], indexed[identity, candidate["name"]]
            value = statistics.mean(reward(cost(reference, n), cost(chosen, n)) for n in config["reuse_counts"])
            families[workload["family"]].append(value)
        means = {family: statistics.mean(values) for family, values in sorted(families.items())}
        leaderboard.append({"name": candidate["name"], "mean_log_ratio": statistics.mean(means.values()),
                            "family_mean_log_ratios": means})
    # Exact ties follow the predeclared candidate order, with reference first.
    winner = max(leaderboard, key=lambda row: row["mean_log_ratio"])
    return {"selected": copy.deepcopy(next(c for c in config["candidates"] if c["name"] == winner["name"])),
            "leaderboard": leaderboard, "training_workloads": len(workloads),
            "full_training_coverage": set(workloads) == {w["id"] for w in corpus.development("train")},
            "unresolved_drift_cells": sum(c["unresolved_drift"] for c in cells)}


def load_fixed(path, current):
    artifact = json.loads(Path(path).read_text())
    if (artifact.get("schema_version") != 1 or artifact.get("status") != "complete"
            or artifact.get("protocol") != PROTOCOL):
        raise ValueError("incomplete or unsupported best-fixed artifact")
    require_compatible_provenance(artifact, current, context="best-fixed artifact")
    corpus = FrozenCorpus()
    if artifact["corpus_sha256"] != corpus.sha256 or artifact["config_digest"] != digest(artifact["config"]):
        raise ValueError("best-fixed corpus or configuration mismatch")
    if artifact["selection"] != select(artifact["config"], artifact["cells"], corpus):
        raise ValueError("best-fixed selection does not match its evidence")
    return artifact


def fixed_selector(artifact):
    # Deliberately ignore the program and reuse hint. Copy to prevent one caller
    # mutating the global schedule used by subsequent programs.
    schedule = copy.deepcopy(artifact["selection"]["selected"]["schedule"])
    return lambda *_: {"schedule": copy.deepcopy(schedule)}
