"""Measured development comparison with a fresh cache and actor per context."""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from rlmlir.adaptation import adapt
from rlmlir.corpus import load
from rlmlir.evaluator import Evaluator
from rlmlir.experiment import initialize, provenance, measurement_lease
from rlmlir.kernels import GENERATORS, REFERENCES
from rlmlir.measurement import validate
from rlmlir.objective import adapted_deployment_ms
from rlmlir.policies import load_policy
from rlmlir.worker import Worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workload-id",required=True)
    parser.add_argument("--split",choices=["train","validation"],default="validation")
    parser.add_argument("--reuse",type=int,default=10000)
    parser.add_argument("--budgets",type=int,nargs="+",choices=[8,16,32],default=[8,16,32])
    parser.add_argument("--methods",nargs="+",choices=["frozen","grpo","ppo","random","mutation"],
                        default=["frozen","grpo","ppo","random","mutation"])
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--hours",type=float,default=2)
    args = parser.parse_args()
    if args.output.exists() or args.reuse < 1:
        raise SystemExit("new output and positive reuse required")
    torch.set_num_threads(1)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    metadata = provenance(ROOT)
    policy, policy_metadata = load_policy(args.policy, expected_provenance=metadata)
    if "ppo" in args.methods and not policy_metadata.get("critic_fitted"):
        raise SystemExit("warm PPO requires a critic fitted to absolute log cost on training data")
    workloads = [w for w in load(ROOT/"configs/corpus.json",args.split) if w["id"] == args.workload_id]
    if len(workloads) != 1:
        raise SystemExit("workload ID is not unique in the requested development split")
    workload = workloads[0]
    args.output.mkdir(parents=True)
    metadata.update(status="running",role="development adaptation; not final evaluation",workload=workload,
                    policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),seed=args.seed,
                    reuse=args.reuse,methods=args.methods,budgets=args.budgets)
    order = [(m,b) for m in args.methods for b in args.budgets]
    random.Random(args.seed).shuffle(order)
    metadata["method_order"] = order
    summaries = []
    import time
    deadline = time.perf_counter()+args.hours*3600
    try:
        with Worker(ROOT/"build/rlmlir-worker") as worker:
            metadata.update(initialize(worker))
            for method,budget in order:
                remaining = deadline-time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("comparison wall-time limit reached")
                directory = args.output/f"{method}-{budget}"
                directory.mkdir()
                worker.request("clear_cache")
                evaluator = Evaluator(worker,directory/"measurements",metadata["binary_hash"],metadata["revision"],
                                      max_new=budget,max_requests=budget,max_seconds=remaining)
                result = adapt(worker,evaluator,workload,args.reuse,policy,method=method,budget=budget,seed=args.seed)
                buffers, expected = REFERENCES[workload["family"]](*workload["shape"],seed=7)
                before_deploy = worker.counters.copy()
                deployment = worker.request("deploy",source=GENERATORS[workload["family"]](*workload["shape"]),
                                            actions=result["best"]["schedule"],buffers=buffers,count=args.reuse)
                error = validate(deployment.pop("output"),expected)
                result["deployment"] = {**deployment,"max_abs_error":error,"correct":True}
                result["deployment_worker_counts"] = dict(worker.counters-before_deploy)
                result["adapted_total_ms"] = adapted_deployment_ms(result["tuning_ms"],deployment["total_ms"])
                (directory/"result.json").write_text(json.dumps(result)+"\n")
                summaries.append({"method":method,"budget":budget,"tuning_ms":result["tuning_ms"],
                                  "deployment_ms":deployment["total_ms"],"adapted_total_ms":result["adapted_total_ms"],
                                  "unique_schedules":result["groups"][-1]["unique_schedules"],
                                  "updates":sum(g["update"] is not None and not g["update"]["skipped"] for g in result["groups"]),
                                  "schedule":result["best"]["schedule"],"counters":result["counters"]})
                print(json.dumps(summaries[-1]),flush=True)
        metadata["status"] = "complete"
    except BaseException as error:
        metadata.update(status="stopped",reason=repr(error))
        raise
    finally:
        metadata["summaries"] = summaries
        (args.output/"run.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__ == "__main__":
    with measurement_lease(ROOT):
        main()
