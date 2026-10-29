"""Select each PPO run's checkpoint on validation, never the held-out corpus."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from rlmlir.corpus import load
from rlmlir.evaluator import Evaluator
from rlmlir.experiment import initialize,provenance,measurement_lease,require_compatible_provenance
from rlmlir.objective import deployment_ms,reward
from rlmlir.policies import save_policy
from rlmlir.selection import sample_schedule
from rlmlir.worker import Worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--behavior-checkpoints",type=int,default=4)
    parser.add_argument("--hours",type=float,default=1)
    args = parser.parse_args()
    if args.output.exists() or args.behavior_checkpoints < 0:
        raise SystemExit("new output and nonnegative checkpoint count required")
    torch.set_num_threads(1)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    meta = provenance(ROOT)
    trained = json.loads((args.run/"run.json").read_text())
    require_compatible_provenance(trained,meta,context="validation checkpoint selection")
    behavior = sorted(args.run.glob("behavior-*.zip"),key=lambda p:int(p.stem.split("-")[1]))
    indices = np.linspace(0,len(behavior)-1,min(args.behavior_checkpoints,len(behavior)),dtype=int)
    checkpoints = [behavior[i] for i in indices] + [args.run/"latest-policy.zip"]
    workloads = load(ROOT/"configs/corpus.json","validation")
    args.output.mkdir(parents=True)
    meta.update(status="running",role="validation checkpoint selection",training_run=str(args.run),
                seed=trained["seed"],objective=trained["objective"],checkpoints=[str(p) for p in checkpoints])
    summaries = []
    try:
        with Worker(ROOT/"build/rlmlir-worker") as worker:
            meta.update(initialize(worker))
            evaluator = Evaluator(worker,args.output/"measurements",meta["binary_hash"],meta["revision"],
                                  max_new=len(checkpoints)*96+32,max_requests=len(checkpoints)*96,max_seconds=args.hours*3600)
            for path in checkpoints:
                model = MaskablePPO.load(path,device="cpu")
                family_scores = {}
                for workload in workloads:
                    baseline = evaluator.measure(workload,[],reference=True)
                    for reuse in (1,100,10000):
                        selected = sample_schedule(worker,workload,reuse,policy=model.policy,deterministic=True)
                        measured = evaluator.measure(workload,selected["schedule"])
                        baseline_cost = deployment_ms(baseline["compile_ms"],baseline["execution_ms"],reuse,
                                                      installation_ms=baseline["installation_ms"])
                        cost = deployment_ms(measured["compile_ms"],measured["execution_ms"],reuse,
                                              selection_ms=selected["selection_ms"],installation_ms=measured["installation_ms"])
                        score = (reward(baseline["execution_ms"],measured["execution_ms"])
                                 if trained["objective"] == "runtime" else reward(baseline_cost,cost))
                        family_scores.setdefault(workload["family"],[]).append(score)
                        record = {"checkpoint":str(path),"workload":workload,"reuse":reuse,"cost_ms":cost,
                                  "reference_cost_ms":baseline_cost,"reward":score,"measurement_key":measured["key"],
                                  "selection":selected,"unresolved_drift":measured["unresolved_drift"]}
                        with (args.output/"selections.jsonl").open("a") as stream:
                            stream.write(json.dumps(record)+"\n")
                aggregate = float(np.mean([np.mean(v) for v in family_scores.values()]))
                summaries.append({"checkpoint":str(path),"equal_family_log_objective_improvement":aggregate})
                print(json.dumps(summaries[-1]),flush=True)
            winner = max(summaries,key=lambda r:r["equal_family_log_objective_improvement"])
            policy = MaskablePPO.load(winner["checkpoint"],device="cpu").policy
            meta.update(status="complete",selected=winner,summaries=summaries,counters=evaluator.counters(),
                        critic_fitted=False,critic_target="reference-relative PPO return")
            meta["policy_sha256"] = save_policy(policy,args.output/"policy.pt",meta)
    except BaseException as error:
        meta.update(status="stopped",reason=repr(error),summaries=summaries)
        raise
    finally:
        (args.output/"run.json").write_text(json.dumps(meta,indent=2)+"\n")


if __name__ == "__main__":
    with measurement_lease(ROOT):
        main()
