"""Read-only audit of the development queue; never execute or retime a kernel."""
import argparse
from collections import Counter
import hashlib
import gzip
import json
import math
from pathlib import Path
import sys

import torch
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.policies import policy_digest


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def audit(directory):
    torch.set_num_threads(1)
    queue = read(directory / "queue.json")
    check(queue["status"] == "complete", "queue is not complete")
    check(all(j["status"] == "complete" and j["exit_code"] == 0 for j in queue["jobs"]), "failed stage")
    corpus = {w["id"]: w for w in read(ROOT / "configs/corpus.json")["workloads"]}
    tables, summaries = {}, {}
    for path in sorted(directory.rglob("measurements.jsonl")):
        records = lines(path)
        for record in records:
            check(record["status"] == "accepted" and record["correct"], f"failed measurement: {path}")
            check(record["workload"] == corpus[record["workload"]["id"]], "changed workload")
            check(record["workload"]["split"] in ("train", "validation"), "held-out measurement")
            check(all(math.isfinite(record[k]) and record[k] >= 0 for k in ("compile_ms", "execution_ms", "installation_ms")), "bad cost")
            with gzip.open(path.parent / (record["key"] + ".json.gz"), "rt") as stream:
                artifact = json.load(stream)
            check(all(hashlib.sha256(artifact[k].encode()).hexdigest() == v for k,v in record["ir_hashes"].items()), "IR artifact hash mismatch")
        tables[path.parent] = {r["key"]: r for r in records}
        check(len(tables[path.parent]) == len(records), "duplicate measured record")
        summaries[str(path.parent.relative_to(directory))] = {
            "accepted": len(records),
            "flagged_original": sum(r["samples"][0]["flagged"] for r in records),
            "unresolved_drift": sum(r["unresolved_drift"] for r in records),
            "splits": dict(Counter(r["workload"]["split"] for r in records)),
        }
    trajectory_counts, ppo = {}, {}
    for path in sorted(directory.glob("*/trajectories.jsonl")):
        trajectories, records = lines(path), tables[path.parent / "measurements"]
        trajectory_counts[path.parent.name] = len(trajectories)
        for item in trajectories:
            outcome = item["outcome"]
            measured = records[outcome["measurement_key"]]
            check(item["workload"] == measured["workload"], "trajectory/measurement mismatch")
            check(all(s["mask"][s["action"]] for s in item["transitions"]), "illegal recorded action")
            check(item["transitions"][-1]["terminal"], "unfinished saved trajectory")
            if "reward" in outcome:
                baseline = records[outcome["reference_key"]]
                cost = measured["execution_ms"]
                reference = baseline["execution_ms"]
                if item["objective"] == "reuse":
                    cost = measured["compile_ms"] + measured["installation_ms"] + item["reuse"] * cost + outcome["analysis_ms"] + outcome["policy_ms"]
                    reference = baseline["compile_ms"] + baseline["installation_ms"] + item["reuse"] * reference
                check(close(cost, outcome["cost_ms"]) and close(reference, outcome["reference_cost_ms"]), "incorrect trajectory cost")
                check(close(math.log(reference / cost), outcome["reward"]), "incorrect reward")
    for name in ("ppo-reuse", "ppo-runtime"):
        run = directory / name
        metadata = read(run / "run.json")
        check(metadata["counters"]["new_measurements"] <= metadata["max_new"], "PPO measurement cap exceeded")
        check(metadata["counters"]["candidate_requests"] <= metadata["max_requests"], "PPO request cap exceeded")
        initial = MaskablePPO.load(run / "initial-policy.zip", device="cpu")
        latest = MaskablePPO.load(run / "latest-policy.zip", device="cpu")
        check(policy_digest(initial.policy) != policy_digest(latest.policy), "PPO weights unchanged")
        check(latest._n_updates == 20, "expected two ten-epoch PPO updates")
        ppo[name] = {"optimizer_epochs": latest._n_updates,
                     "initial_policy_digest": policy_digest(initial.policy),
                     "final_policy_digest": policy_digest(latest.policy)}
    pretrain = read(directory / "pretrain/run.json")
    for source in pretrain["sources"]:
        # Recorded absolute paths are provenance; resolve by stage name for portability.
        path = directory / Path(source["run"]).name / "trajectories.jsonl"
        check(hashlib.sha256(path.read_bytes()).hexdigest() == source["trajectory_sha256"], "source hash mismatch")
    for name, field in (("demonstrations.json", "demonstration_sha256"), ("policy.pt", "policy_sha256")):
        check(hashlib.sha256((directory / "pretrain" / name).read_bytes()).hexdigest() == pretrain[field], "pretraining artifact hash mismatch")
    demonstrations = read(directory / "pretrain/demonstrations.json")["demonstrations"]
    for demo in demonstrations:
        check(demo["workload"]["split"] in ("train", "validation"), "held-out demonstration")
        check(demo["recheck_key"] in tables[directory / "pretrain/rechecks"], "missing demonstration recheck")
    adaptation = {}
    for path in sorted((directory / "adaptation").glob("*/result.json")):
        result = read(path)
        budget, method = result["budget"], result["method"]
        check(len(result["requests"]) == budget == result["counters"]["candidate_requests"], "candidate budget mismatch")
        check([r["request"] for r in result["requests"]] == list(range(1, budget + 1)), "request numbering")
        records = tables[path.parent / "measurements"]
        for request in result["requests"]:
            record = records[request["measurement_key"]]
            cost = record["compile_ms"] + record["installation_ms"] + request["selection_ms"] + result["reuse"] * record["execution_ms"]
            check(close(cost, request["cost_ms"]) and close(-math.log(cost), request["reward"]), "adaptation cost/reward mismatch")
        updates = [g["update"] for g in result["groups"] if g["update"] is not None]
        check(len(updates) == (budget // 8 - 1 if method in ("grpo", "ppo") else 0), "update budget mismatch")
        check(all(u["epochs"] == 4 and all(math.isfinite(m["gradient_norm"]) and m["gradient_norm"] > 0 for m in u["metrics"]) for u in updates), "missing gradient evidence")
        deployment = result["deployment"]
        check(deployment["correct"] and not deployment["cache_hit"] and deployment["count"] == result["reuse"], "deployment is not fresh/correct")
        check(close(deployment["total_ms"], deployment["compile_ms"] + deployment["installation_ms"] + deployment["execution_ms"]), "deployment accounting")
        check(close(result["adapted_total_ms"], result["tuning_ms"] + deployment["total_ms"]), "tuning accounting")
        adaptation[path.parent.name] = result
    controls = {}
    for budget in (8, 16):
        sequences = [[r["actions"] for r in adaptation[f"{method}-{budget}"]["requests"]] for method in ("frozen", "grpo", "ppo")]
        check(sequences[0][:8] == sequences[1][:8] == sequences[2][:8], "reset/first-group control differs")
        controls[str(budget)] = {"first_eight_identical": True, "all_candidates_identical": sequences[0] == sequences[1] == sequences[2]}
    return {"schema_version": 1, "status": "passed", "queue_elapsed_seconds": queue["elapsed_seconds"],
            "measurements": summaries, "trajectory_counts": trajectory_counts, "ppo_updates": ppo,
            "demonstration_splits": dict(Counter(d["workload"]["split"] for d in demonstrations)),
            "demonstration_near_ties": sum(d["near_tie"] for d in demonstrations),
            "demonstration_unresolved_drift": sum(d["recheck_unresolved_drift"] for d in demonstrations),
            "adaptation_controls": controls,
            "historical_counter_scope": "integration-003 worker_counts are worker lifetime totals; do not sum method totals",
            "limitations": "Development only; one validation adaptation context; no held-out performance or superiority test."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
