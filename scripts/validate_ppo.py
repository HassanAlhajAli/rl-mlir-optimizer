"""Select each PPO run's checkpoint on validation, never the held-out corpus."""
import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
import torch
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from rlmlir.corpus import load
from rlmlir.evaluator import Evaluator
from rlmlir.experiment import initialize,provenance,measurement_lease
from rlmlir.objective import deployment_ms,reward
from rlmlir.policies import save_policy
from rlmlir.selection import sample_schedule
from rlmlir.worker import Worker
from rlmlir.validation import PROTOCOL, REUSE_COUNTS, checkpoints_for, eligible_training, sha256, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--behavior-checkpoints",type=int,default=4)
    parser.add_argument("--hours",type=float,default=1)
    args = parser.parse_args()
    if args.output.exists() or args.behavior_checkpoints < 0 or not math.isfinite(args.hours) or args.hours <= 0:
        raise SystemExit("new output, nonnegative checkpoint count and positive finite time limit required")
    torch.set_num_threads(1)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    meta = provenance(ROOT)
    trained = json.loads((args.run/"run.json").read_text())
    eligible_training(trained,meta)
    checkpoints = checkpoints_for(args.run,args.behavior_checkpoints)
    workloads = load(ROOT/"configs/corpus.json","validation")
    args.output.mkdir(parents=True)
    meta.update(status="running",role="validation checkpoint selection",training_run=str(args.run.resolve()),
                training_run_sha256=sha256(args.run/"run.json"), selection_protocol=PROTOCOL,
                seed=trained["seed"],objective=trained["objective"],checkpoints=checkpoints,
                behavior_checkpoint_count=args.behavior_checkpoints, controller_pid=os.getpid(),
                wall_limit_seconds=args.hours*3600)
    summaries = []
    records = []
    evaluator, worker = None, None
    began = time.perf_counter()
    def save():
        (args.output/"run.json").write_text(json.dumps(meta,indent=2,allow_nan=False)+"\n")
    save()
    try:
        with Worker(ROOT/"build/rlmlir-worker") as worker:
            meta.update(worker_pid=worker.process.pid, **initialize(worker))
            save()
            remaining = args.hours*3600-(time.perf_counter()-began)
            if remaining <= 0:
                raise TimeoutError("initialization exhausted validation budget")
            case_count = len(checkpoints)*len(workloads)*len(REUSE_COUNTS)
            evaluator = Evaluator(worker,args.output/"measurements",meta["binary_hash"],meta["revision"],
                                  max_new=case_count+len(workloads),max_requests=case_count,max_seconds=remaining)
            for checkpoint in checkpoints:
                path = checkpoint["path"]
                if sha256(path) != checkpoint["sha256"]:
                    raise ValueError("checkpoint changed after validation plan was saved")
                model = MaskablePPO.load(path,device="cpu")
                model.policy.set_training_mode(False)
                family_scores = {}
                for workload in workloads:
                    baseline = evaluator.measure(workload,[],reference=True)
                    for reuse in REUSE_COUNTS:
                        selected = sample_schedule(worker,workload,reuse,policy=model.policy,deterministic=True)
                        measured = evaluator.measure(workload,selected["schedule"])
                        baseline_cost = deployment_ms(baseline["compile_ms"],baseline["execution_ms"],reuse,
                                                      installation_ms=baseline["installation_ms"])
                        cost = deployment_ms(measured["compile_ms"],measured["execution_ms"],reuse,
                                              selection_ms=selected["selection_ms"],installation_ms=measured["installation_ms"])
                        score = (reward(baseline["execution_ms"],measured["execution_ms"])
                                 if trained["objective"] == "runtime" else reward(baseline_cost,cost))
                        family_scores.setdefault(workload["family"],[]).append(score)
                        record = {"checkpoint":str(path),"checkpoint_sha256":checkpoint["sha256"],
                                  "workload":workload,"reuse":reuse,"cost_ms":cost,
                                  "reference_cost_ms":baseline_cost,"reward":score,"measurement_key":measured["key"],
                                  "reference_key":baseline["key"],
                                  "selection":selected,"unresolved_drift":measured["unresolved_drift"]}
                        records.append(record)
                        with (args.output/"selections.jsonl").open("a") as stream:
                            stream.write(json.dumps(record)+"\n")
                aggregate = statistics.mean(statistics.mean(v) for v in family_scores.values())
                summaries.append({"checkpoint":str(path),"equal_family_log_objective_improvement":aggregate})
                print(json.dumps(summaries[-1]),flush=True)
            # Full coverage and independently recomputed costs are required
            # before a partial or interrupted comparison can export a winner.
            summaries,winner = summarize(records,checkpoints,trained["objective"],
                                         list(evaluator.records.values()),meta["binary_hash"])
            if checkpoints_for(args.run,args.behavior_checkpoints) != checkpoints or sha256(args.run/"run.json") != meta["training_run_sha256"]:
                raise ValueError("training artifacts changed during validation")
            policy = MaskablePPO.load(winner["checkpoint"],device="cpu").policy
            meta.update(status="complete",selected=winner,summaries=summaries,counters=evaluator.counters(),
                        critic_fitted=False,critic_target="reference-relative PPO return",
                        selections_sha256=sha256(args.output/"selections.jsonl"),
                        measurements_sha256=sha256(args.output/"measurements/measurements.jsonl"))
            meta["policy_sha256"] = save_policy(policy,args.output/"policy.pt",meta)
    except BaseException as error:
        meta.update(status="stopped",reason=repr(error),summaries=summaries)
        raise
    finally:
        if evaluator is not None:
            meta["counters"] = evaluator.counters()
        if worker is not None:
            meta["worker_counts"] = dict(worker.counters)
        meta["elapsed_seconds"] = time.perf_counter()-began
        save()


if __name__ == "__main__":
    with measurement_lease(ROOT):
        main()
