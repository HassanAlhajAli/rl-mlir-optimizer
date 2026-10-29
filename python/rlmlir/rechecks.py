"""Complete, shared evidence from imitation's additional leader measurements."""
import copy
import json
import statistics
from collections import Counter
from pathlib import Path

from .experience import digest, measurement_records, require_manifest
from .experiment import COMPATIBILITY_FIELDS, require_compatible_provenance
from .objective import deployment_ms

PROTOCOL = "shared-leader-rechecks-v1"


def planned_rechecks(pool):
    """Indices refer to the ordered, fingerprinted initial pool, never new states."""
    contexts = {}
    for index, item in enumerate(pool):
        histories = contexts.setdefault((item["workload"]["id"], item["reuse"]), {})
        histories.setdefault(json.dumps(item["actions"], sort_keys=True), []).append(index)
    plan = []
    for histories in contexts.values():
        candidates = []
        for indices in histories.values():
            item = pool[indices[0]]
            candidates.append({"source_indices": indices, "workload_id": item["workload"]["id"],
                               "reuse": item["reuse"], "actions": item["actions"],
                               "initial_cost_ms": statistics.median(pool[i]["cost_ms"] for i in indices),
                               "selection_ms": statistics.median(pool[i]["selection_ms"] for i in indices)})
        candidates.sort(key=lambda c: c["initial_cost_ms"])
        plan.extend(c for i, c in enumerate(candidates)
                    if i < 2 or c["initial_cost_ms"] <= candidates[0]["initial_cost_ms"] * 1.05)
    return plan


def request_record(index, planned, measured):
    if measured.get("status") != "accepted" or measured.get("correct") is not True:
        raise ValueError("recheck requires an accepted correct measurement")
    if type(measured.get("unresolved_drift")) is not bool:
        raise ValueError("recheck must preserve its drift flag")
    return {"request_index": index, **planned, "measurement_key": measured["key"],
            "cost_ms": deployment_ms(measured["compile_ms"], measured["execution_ms"], planned["reuse"],
                                     selection_ms=planned["selection_ms"], installation_ms=measured["installation_ms"]),
            "unresolved_drift": measured["unresolved_drift"]}


def fingerprint(manifest):
    return digest(json.dumps({k: v for k, v in manifest.items() if k != "fingerprint"},
                             sort_keys=True, allow_nan=False).encode())


def measurement_access(initial_manifest, recheck_manifest=None):
    """Access identity is independent of the fitting algorithm and local paths."""
    identity = {"protocol": "measurement-access-v1", "initial_pool_fingerprint": initial_manifest["fingerprint"],
                "recheck_fingerprint": None if recheck_manifest is None else recheck_manifest["fingerprint"],
                "scope": "initial_only" if recheck_manifest is None else "initial_and_all_leader_rechecks"}
    return {**identity, "fingerprint": fingerprint(identity)}


def _verify(directory, pool, initial_manifest, expected_provenance, counters):
    require_manifest(initial_manifest, initial_manifest)
    raw_measurements = (directory / "measurements.jsonl").read_bytes()
    measurement_hash = digest(raw_measurements)
    raw_requests = (directory / "requests.jsonl").read_bytes()
    records = measurement_records(raw_measurements, expected_provenance)
    requests = [json.loads(line) for line in raw_requests.splitlines()]
    plan = planned_rechecks(pool)
    if len(requests) != len(plan):
        raise ValueError("incomplete recheck leader coverage")
    result, used = [], set()
    for index, (planned, request) in enumerate(zip(plan, requests)):
        record = records.get(request["measurement_key"])
        original = pool[planned["source_indices"][0]]
        if (record is None or record["workload"] != original["workload"]
                or record["actions"] != [a for a in planned["actions"] if a["kind"] != "STOP"]):
            raise ValueError("recheck does not match its initial trajectory")
        if request != request_record(index, planned, record):
            raise ValueError("recheck request differs from the complete planned evidence")
        used.add(record["key"])
        item = copy.deepcopy(original)
        item.update(original_cost_ms=planned["initial_cost_ms"], selection_ms=planned["selection_ms"],
                    cost_ms=request["cost_ms"], recheck_key=record["key"],
                    recheck_unresolved_drift=record["unresolved_drift"],
                    compile_install_ms=record["compile_ms"] + record["installation_ms"],
                    execution_ms=record["execution_ms"],
                    measurement_identity=[measurement_hash, record["key"]])
        result.append(item)
    if used != set(records):
        raise ValueError("recheck measurements are missing from request evidence")
    expected_counts = {"candidate_requests": len(plan), "reference_requests": 0,
                       "new_measurements": len(records), "cache_hits": len(plan) - len(records)}
    if any(type(counters.get(k)) is not int or counters[k] != v for k, v in expected_counts.items()):
        raise ValueError("recheck request/cache counts disagree with saved evidence")
    manifest = {"schema_version": 1, "protocol": PROTOCOL, "status": "complete",
                "initial_pool_fingerprint": initial_manifest["fingerprint"],
                "compatibility": {k: expected_provenance[k] for k in COMPATIBILITY_FIELDS},
                "measurement_sha256": measurement_hash, "request_sha256": digest(raw_requests),
                "counts": expected_counts,
                "split_counts": dict(sorted(Counter(i["workload"]["split"] for i in result).items()))}
    manifest["fingerprint"] = fingerprint(manifest)
    return result, manifest


def collect_rechecks(evaluator, pool, initial_manifest, provenance, directory):
    """Retain partial requests on failure; only a fully verified bundle is usable."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "requests.jsonl"
    if path.exists() or (directory / "shared-rechecks.json").exists():
        raise ValueError("recheck export already exists")
    with path.open("x") as stream:
        for index, planned in enumerate(planned_rechecks(pool)):
            original = pool[planned["source_indices"][0]]
            record = evaluator.measure(original["workload"], [a for a in planned["actions"] if a["kind"] != "STOP"])
            stream.write(json.dumps(request_record(index, planned, record), allow_nan=False) + "\n")
            stream.flush()
    result, manifest = _verify(directory, pool, initial_manifest, provenance, evaluator.counters())
    (directory / "shared-rechecks.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result, manifest


def load_rechecks(directory, pool, initial_manifest, expected_provenance):
    directory = Path(directory)
    saved = json.loads((directory / "shared-rechecks.json").read_bytes())
    if saved.get("protocol") != PROTOCOL or saved.get("status") != "complete" or saved.get("fingerprint") != fingerprint(saved):
        raise ValueError("invalid or incomplete shared recheck manifest")
    if saved.get("initial_pool_fingerprint") != initial_manifest["fingerprint"]:
        raise ValueError("rechecks belong to a different initial experience pool")
    require_compatible_provenance(saved["compatibility"], expected_provenance, context="shared rechecks")
    result, current = _verify(directory, pool, initial_manifest, expected_provenance, saved["counts"])
    if current != saved:
        raise ValueError("shared recheck evidence changed after completion")
    return result, current


def select_demonstrations(rechecked):
    contexts = {}
    for item in rechecked:
        contexts.setdefault((item["workload"]["id"], item["reuse"]), []).append(item)
    best = []
    for candidates in contexts.values():
        leaders = sorted(candidates, key=lambda c: c["cost_ms"])
        chosen = copy.deepcopy(leaders[0])
        chosen["near_tie"] = len(leaders) > 1 and leaders[1]["cost_ms"] <= chosen["cost_ms"] * 1.05
        chosen["rechecked_candidates"] = [{"cost_ms": c["cost_ms"], "key": c["recheck_key"]} for c in leaders]
        best.append(chosen)
    return best
