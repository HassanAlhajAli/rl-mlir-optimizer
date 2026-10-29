"""Recheck leading demonstrations, then fit a masked actor and training critic."""
import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from rlmlir.evaluator import Evaluator
from rlmlir.experiment import initialize, provenance, measurement_lease
from rlmlir.experience import load_pool, pool_manifest
from rlmlir.imitation import fit_actor, fit_critic
from rlmlir.objective import deployment_ms
from rlmlir.policies import create_policy, save_policy
from rlmlir.worker import Worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--max-rechecks",type=int,default=4096)
    parser.add_argument("--hours",type=float,default=4)
    parser.add_argument("--pool-manifest",type=Path,help="pin the initial experience bytes before training")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists")
    torch.set_num_threads(1)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    meta = provenance(ROOT)
    pool, sources = load_pool(args.runs,meta,manifest_path=args.pool_manifest)
    if {t["workload"]["split"] for t in pool} != {"train","validation"}:
        raise SystemExit("both training and validation experience are required")
    args.output.mkdir(parents=True)
    manifest = pool_manifest(pool,sources,meta)
    (args.output/"initial-pool-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    meta["initial_pool_fingerprint"] = manifest["fingerprint"]
    meta.update(schema_version=1,status="running",role="development imitation pretraining",seed=args.seed,sources=sources)
    grouped = {}
    for item in pool:
        key = (item["workload"]["id"],item["reuse"])
        grouped.setdefault(key,[]).append(item)
    best = []
    try:
        with Worker(ROOT/"build/rlmlir-worker") as worker:
            meta.update(initialize(worker))
            evaluator = Evaluator(worker,args.output/"rechecks",meta["binary_hash"],meta["revision"],
                                  max_new=args.max_rechecks,max_requests=args.max_rechecks*4,max_seconds=args.hours*3600)
            for items in grouped.values():
                # Collapse identical action histories by median observed selection
                # cost, so a frequently sampled history cannot win via one lucky timer.
                histories = {}
                for item in items:
                    histories.setdefault(json.dumps(item["actions"],sort_keys=True),[]).append(item)
                candidates = []
                for copies in histories.values():
                    item = dict(copies[0])
                    item["cost_ms"] = statistics.median(c["cost_ms"] for c in copies)
                    item["selection_ms"] = statistics.median(c["selection_ms"] for c in copies)
                    candidates.append(item)
                candidates.sort(key=lambda c:c["cost_ms"])
                leaders = [c for i,c in enumerate(candidates) if i < 2 or c["cost_ms"] <= candidates[0]["cost_ms"]*1.05]
                for item in leaders:
                    actions = [a for a in item["actions"] if a["kind"] != "STOP"]
                    checked = evaluator.measure(item["workload"],actions)
                    item["original_cost_ms"] = item["cost_ms"]
                    item["cost_ms"] = deployment_ms(checked["compile_ms"],checked["execution_ms"],item["reuse"],
                                                    selection_ms=item["selection_ms"],installation_ms=checked["installation_ms"])
                    item["recheck_key"] = checked["key"]
                    item["recheck_unresolved_drift"] = checked["unresolved_drift"]
                leaders.sort(key=lambda c:c["cost_ms"])
                chosen = leaders[0]
                chosen["near_tie"] = len(leaders)>1 and leaders[1]["cost_ms"] <= chosen["cost_ms"]*1.05
                chosen["rechecked_candidates"] = [{"cost_ms":c["cost_ms"],"key":c["recheck_key"]} for c in leaders]
                best.append(chosen)
            meta["recheck_counters"] = evaluator.counters()
        training = [t for t in best if t["workload"]["split"] == "train"]
        validation = [t for t in best if t["workload"]["split"] == "validation"]
        dataset = {"schema_version":1,"sources":sources,"demonstrations":best}
        (args.output/"demonstrations.json").write_text(json.dumps(dataset)+"\n")
        policy = create_policy(args.seed)
        actor_history = fit_actor(policy,training,validation,seed=args.seed,max_epochs=args.epochs)
        critic_history = fit_critic(policy,[t for t in pool if t["workload"]["split"] == "train"],seed=args.seed)
        meta.update(status="complete",critic_fitted=True,critic_target="negative log total cost in ms",
                    training_contexts=len(training),validation_contexts=len(validation),
                    actor_history=actor_history,critic_history=critic_history,
                    demonstration_sha256=hashlib.sha256((args.output/"demonstrations.json").read_bytes()).hexdigest())
        meta["policy_sha256"] = save_policy(policy,args.output/"policy.pt",meta)
    except BaseException as error:
        meta.update(status="stopped",reason=repr(error))
        raise
    finally:
        (args.output/"run.json").write_text(json.dumps(meta,indent=2)+"\n")
        print(json.dumps({k:v for k,v in meta.items() if k not in ("actor_history","critic_history","sources")},indent=2))


if __name__ == "__main__":
    with measurement_lease(ROOT):
        main()
