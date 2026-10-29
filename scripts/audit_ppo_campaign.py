"""Audit completed and interrupted PPO jobs without running generated code."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.policies import policy_digest
from rlmlir.corpus import FrozenCorpus
from rlmlir.campaign_readiness import report as readiness_report
from rlmlir.experiment import provenance


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(directory):
    torch.set_num_threads(1)
    queue = read(directory / "queue.json")
    require(queue["status"] != "running", "do not audit an active campaign")
    corpus = FrozenCorpus()
    actual = {job["name"]: job for job in queue["jobs"]}
    summary = {"queue_status": queue["status"], "queue_elapsed_seconds": queue["elapsed_seconds"],
               "corpus_sha256": corpus.sha256,
               "jobs": {}, "scope": "saved evidence; no new execution or performance validation"}
    for definition in queue["definition"]:
        name = definition["name"]
        if name not in actual:
            summary["jobs"][name] = {"status": "not_started"}
            continue
        run = directory / name
        meta = read(run / "run.json")
        require(meta["corpus_sha256"] == corpus.sha256, "training corpus hash mismatch")
        counts = meta["counters"]
        require(counts["new_measurements"] <= meta["max_new"], "distinct budget exceeded")
        require(counts["candidate_requests"] <= meta["max_requests"], "request budget exceeded")
        require(meta["timesteps"] <= meta["requested_timesteps"] + 255, "rollout limit exceeded")
        records = lines(run / "measurements/measurements.jsonl")
        require(len(records) == counts["new_measurements"], "measurement counter mismatch")
        by_key = {r["key"]: r for r in records}
        require(len(by_key) == len(records), "duplicate saved measurements")
        failures = []
        for record in records:
            corpus.require_development(record["workload"])
            require(record["workload"]["split"] == "train", "non-training outcome")
            require(record["binary_hash"] == meta["binary_hash"] and record["revision"] == meta["revision"], "provenance mismatch")
            if record["status"] == "failed":
                failures.append({"key": record["key"], "reason": record["reason"]})
                continue
            require(record["status"] == "accepted" and record["correct"], "unverified outcome")
            require(all(math.isfinite(record[k]) and record[k] >= 0
                        for k in ("compile_ms", "execution_ms", "installation_ms")), "invalid timing")
            with gzip.open(run / "measurements" / (record["key"] + ".json.gz"), "rt") as stream:
                artifact = json.load(stream)
            require(all(hashlib.sha256(artifact[key].encode()).hexdigest() == value
                        for key, value in record["ir_hashes"].items()), "IR hash mismatch")
        trajectories = lines(run / "trajectories.jsonl")
        require(len(trajectories) == meta["episodes"], "episode counter mismatch")
        for item in trajectories:
            outcome = item["outcome"]
            measured, baseline = by_key[outcome["measurement_key"]], by_key[outcome["reference_key"]]
            require(measured["status"] == baseline["status"] == "accepted", "failed outcome used for learning")
            require(item["workload"] == measured["workload"] == baseline["workload"], "trajectory workload mismatch")
            require(measured["actions"] == [a for a in item["actions"] if a["kind"] != "STOP"], "trajectory schedule mismatch")
            require(all(step["mask"][step["action"]] for step in item["transitions"]), "illegal action")
            require(item["transitions"][-1]["terminal"], "unfinished saved trajectory")
            cost, reference = measured["execution_ms"], baseline["execution_ms"]
            if item["objective"] == "reuse":
                cost = measured["compile_ms"] + measured["installation_ms"] + item["reuse"] * cost + outcome["analysis_ms"] + outcome["policy_ms"]
                reference = baseline["compile_ms"] + baseline["installation_ms"] + item["reuse"] * reference
            require(math.isclose(cost, outcome["cost_ms"], rel_tol=1e-9, abs_tol=1e-9), "candidate cost mismatch")
            require(math.isclose(reference, outcome["reference_cost_ms"], rel_tol=1e-9, abs_tol=1e-9), "reference cost mismatch")
            require(math.isclose(math.log(reference/cost), outcome["reward"], rel_tol=1e-9, abs_tol=1e-9), "reward mismatch")
        initial = MaskablePPO.load(run / "initial-policy.zip", device="cpu")
        latest = MaskablePPO.load(run / "latest-policy.zip", device="cpu")
        require(all(torch.isfinite(value).all().item() for value in latest.policy.state_dict().values()), "nonfinite checkpoint")
        require(latest._n_updates > 0 and policy_digest(initial.policy) != policy_digest(latest.policy), "missing policy update")
        summary["jobs"][name] = {
            "status": actual[name]["status"], "objective": meta["objective"], "seed": meta["seed"],
            "termination_reason": meta.get("reason"), "accepted": len(records)-len(failures), "failures": failures,
            "candidate_requests": counts["candidate_requests"], "new_measurements": counts["new_measurements"],
            "cache_hits": counts["cache_hits"], "episodes": len(trajectories), "timesteps": meta["timesteps"],
            "optimizer_epochs": latest._n_updates, "initial_policy_digest": policy_digest(initial.policy),
            "final_policy_digest": policy_digest(latest.policy),
            "elapsed_seconds": actual[name]["elapsed_seconds"],
            "original_drift_flags": sum(r["samples"][0]["flagged"] for r in records if r["status"] == "accepted"),
            "unresolved_drift": sum(r["unresolved_drift"] for r in records if r["status"] == "accepted"),
            "split_counts": dict(Counter(r["workload"]["split"] for r in records))}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readiness-only", action="store_true", help="check all ten runs and validation eligibility without loading models")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ppo-campaign.json")
    parser.add_argument("--validations", type=Path, help="JSON mapping from each job name to its validation directory")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("audit output exists; preserve previous evidence")
    if args.readiness_only:
        validations = json.loads(args.validations.read_text()) if args.validations else None
        result = readiness_report(args.directory, args.config, provenance(ROOT), validations)
    else:
        result = audit(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
