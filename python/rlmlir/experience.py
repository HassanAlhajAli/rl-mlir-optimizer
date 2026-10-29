"""Verify and fingerprint the initial experience shared by learning baselines."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .corpus import FrozenCorpus
from .encoding import ACTIONS, OBSERVATION_SIZE
from .experiment import COMPATIBILITY_FIELDS, require_compatible_provenance
from .kernels import GENERATORS
from .objective import deployment_ms

PROTOCOL = "initial-experience-v2"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def finite(value, name, *, nonnegative=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (nonnegative and value < 0)):
        raise ValueError("invalid experience value: " + name)
    return value


def check_transitions(item):
    actions, steps = item["actions"], item["transitions"]
    if not 1 <= len(steps) <= 6 or len(actions) != len(steps):
        raise ValueError("incomplete or excessive trajectory")
    for index, (action, step) in enumerate(zip(actions, steps)):
        selected = step["action"]
        if type(selected) is not int or not 0 <= selected < len(ACTIONS) or ACTIONS[selected] != action:
            raise ValueError("trajectory action encoding mismatch")
        mask = step["mask"]
        if len(mask) != len(ACTIONS) or any(type(v) is not bool for v in mask) or not mask[0] or not mask[selected]:
            raise ValueError("trajectory action mask is invalid")
        for key in ("observation", "next_observation"):
            if len(step[key]) != OBSERVATION_SIZE:
                raise ValueError("trajectory observation shape mismatch")
            for value in step[key]:
                finite(value, key)
        if index and steps[index - 1]["next_observation"] != step["observation"]:
            raise ValueError("disconnected trajectory states")
        if step["terminal"] is not (index == len(steps) - 1):
            raise ValueError("trajectory terminal placement mismatch")
        if action["kind"] in ("STOP", "VECTORIZE") and not step["terminal"]:
            raise ValueError("actions continue after termination")
        for key in ("behavior_log_probability", "behavior_value"):
            if step.get(key) is not None:
                finite(step[key], key)
    if actions[-1]["kind"] not in ("STOP", "VECTORIZE") and len(actions) != 6:
        raise ValueError("trajectory ends before a terminal action or action limit")


def pool_manifest(pool, sources, expected_provenance):
    manifest = {"schema_version": 1, "protocol": PROTOCOL,
                "compatibility": {k: expected_provenance[k] for k in COMPATIBILITY_FIELDS},
                "sources": sources, "trajectories": len(pool),
                "contexts": len({(p["workload"]["id"], p["reuse"]) for p in pool}),
                "split_counts": dict(sorted(Counter(p["workload"]["split"] for p in pool).items())),
                "scope": "initial source access only; excludes additional demonstration rechecks"}
    manifest["fingerprint"] = manifest_fingerprint(manifest)
    return manifest


def manifest_fingerprint(manifest):
    identity = {k: v for k, v in manifest.items() if k != "fingerprint"}
    identity["sources"] = [{k: v for k, v in source.items() if k != "run"} for source in manifest["sources"]]
    return digest(json.dumps(identity, sort_keys=True, allow_nan=False).encode())


def require_manifest(saved, current):
    if saved.get("fingerprint") != manifest_fingerprint(saved):
        raise ValueError("initial experience manifest fingerprint is invalid")
    if saved["fingerprint"] != current["fingerprint"]:
        raise ValueError("initial experience differs from the pinned manifest")


def measurement_records(raw, expected_provenance, *, allow_parent_corpus=False):
    """Validate measured identities without executing native code again."""
    corpus = FrozenCorpus()
    if expected_provenance.get("corpus_sha256") != corpus.sha256:
        raise ValueError("experience pool does not match the frozen corpus")
    data = {}
    for line in raw.splitlines():
        record = json.loads(line)
        if record["key"] in data:
            raise ValueError("duplicate measurement identity in experience source")
        if record["status"] != "accepted" or record.get("correct") is not True:
            raise ValueError("experience source contains an unaccepted measurement")
        corpus.require_development(record["workload"])
        corpus_hash = record.get("corpus_sha256", corpus.sha256 if allow_parent_corpus else None)
        if record["binary_hash"] != expected_provenance["binary_hash"] or corpus_hash != corpus.sha256:
            raise ValueError("experience measurement provenance mismatch")
        workload = record["workload"]
        source = GENERATORS[workload["family"]](*workload["shape"])
        key = digest(json.dumps({"source": source, "actions": record["actions"], "binary": record["binary_hash"],
                                "input_seed": 7, "measurement_protocol": "paired-training-v2"}, sort_keys=True).encode())
        if record["key"] != key:
            raise ValueError("experience measurement key mismatch")
        for name in ("compile_ms", "installation_ms", "execution_ms"):
            finite(record.get(name), name, nonnegative=True)
        if record["execution_ms"] <= 0 or record["compile_ms"] + record["installation_ms"] <= 0:
            raise ValueError("experience costs must be positive for log-cost fitting")
        data[key] = record
    if not data:
        raise ValueError("empty experience measurements")
    return data


def load_pool(runs, expected_provenance, *, manifest_path=None):
    corpus = FrozenCorpus()
    if expected_provenance.get("corpus_sha256") != corpus.sha256:
        raise ValueError("experience pool does not match the frozen corpus")
    pool, sources, directories, source_copies = [], [], set(), set()
    for supplied in runs:
        directory = Path(supplied)
        resolved = directory.resolve()
        if resolved in directories:
            raise ValueError("duplicate experience source directory")
        directories.add(resolved)
        # Hash precisely the bytes that are parsed, not a second read after loading.
        raw = {"run": (directory / "run.json").read_bytes(),
               "measurement": (directory / "measurements/measurements.jsonl").read_bytes(),
               "trajectory": (directory / "trajectories.jsonl").read_bytes()}
        hashes = {key + "_sha256": digest(value) for key, value in raw.items()}
        identity = (hashes["measurement_sha256"], hashes["trajectory_sha256"])
        if identity in source_copies:
            raise ValueError("byte-identical experience sources would duplicate training weight")
        source_copies.add(identity)
        meta = json.loads(raw["run"])
        require_compatible_provenance(meta, expected_provenance, context="experience pool")
        if meta.get("status") not in ("complete", "budget_reached", "timesteps_complete"):
            raise ValueError("experience source is failed, running or lacks a completed status")
        # Earlier records bind corpus identity through compatible run metadata.
        # This legacy gap is counted below, never silently backfilled on disk.
        data = measurement_records(raw["measurement"], meta, allow_parent_corpus=True)
        items = []
        for line in raw["trajectory"].splitlines():
            item = json.loads(line)
            corpus.require_development(item["workload"])
            check_transitions(item)
            outcome = item["outcome"]
            record = data.get(outcome["measurement_key"])
            if (record is None or record["workload"] != item["workload"]
                    or record["actions"] != [a for a in item["actions"] if a["kind"] != "STOP"]):
                raise ValueError("trajectory does not match its accepted measurement")
            selection = sum(finite(outcome[k], k, nonnegative=True) for k in ("analysis_ms", "policy_ms"))
            cost = deployment_ms(record["compile_ms"], record["execution_ms"], item["reuse"],
                                 selection_ms=selection, installation_ms=record["installation_ms"])
            if item.get("objective") not in (None, "reuse", "runtime"):
                raise ValueError("unknown trajectory objective")
            original_cost = record["execution_ms"] if item.get("objective") == "runtime" else cost
            if not math.isclose(finite(outcome["cost_ms"], "recorded cost", nonnegative=True), original_cost, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("trajectory outcome cost disagrees with measured costs")
            item.update(selection_ms=selection, compile_install_ms=record["compile_ms"] + record["installation_ms"],
                        execution_ms=record["execution_ms"], cost_ms=cost, origin=str(directory),
                        measurement_identity=[hashes["measurement_sha256"], record["key"]])
            items.append(item)
        if not items or not data:
            raise ValueError("empty experience source")
        sources.append({"run": str(directory), "revision": meta["revision"], **hashes,
                        "status": meta["status"], "measurements": len(data), "trajectories": len(items),
                        "measurements_without_local_corpus_hash": sum("corpus_sha256" not in r for r in data.values()),
                        "split_counts": dict(sorted(Counter(i["workload"]["split"] for i in items).items()))})
        pool.extend(items)
    if not pool:
        raise ValueError("experience pool is empty")
    if manifest_path is not None:
        require_manifest(json.loads(Path(manifest_path).read_text()), pool_manifest(pool, sources, expected_provenance))
    return pool, sources
