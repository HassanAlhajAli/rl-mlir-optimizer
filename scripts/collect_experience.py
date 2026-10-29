"""Verified fixed/heuristic/random trajectories for training or validation only."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from rlmlir.corpus import load
from rlmlir.evaluator import Evaluator, BudgetReached
from rlmlir.objective import deployment_ms
from rlmlir.selection import heuristic, sample_schedule
from rlmlir.worker import Worker
from rlmlir.experiment import initialize, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--split",choices=["train","validation"],default="train")
    parser.add_argument("--per-family",type=int,default=32)
    parser.add_argument("--random-per-context",type=int,default=2)
    parser.add_argument("--max-new",type=int,default=4096)
    parser.add_argument("--hours",type=float,default=4)
    parser.add_argument("--seed",type=int,default=17)
    args = parser.parse_args()
    if args.output.exists() or args.per_family < 1 or args.random_per_context < 0:
        raise SystemExit("new output and nonnegative/positive counts required")
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip():
        raise SystemExit("commit tracked changes before measurement")
    args.output.mkdir(parents=True)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    workloads = load(ROOT/"configs/corpus.json",args.split)
    selected = []
    for family in sorted({w["family"] for w in workloads}):
        selected.extend([w for w in workloads if w["family"] == family][:args.per_family])
    rng = np.random.default_rng(args.seed)
    binary = ROOT/"build/rlmlir-worker"
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    metadata = {"schema_version":1,"role":"development trajectory collection","status":"running",
                "revision":revision,"binary_hash":hashlib.sha256(binary.read_bytes()).hexdigest(),
                "seed":args.seed,"workloads":selected,"split":args.split,
                "random_per_context":args.random_per_context,"reuse_counts":[1,100,10000]}
    (args.output/"run.json").write_text(json.dumps(metadata,indent=2)+"\n")
    with Worker(binary) as worker:
        metadata.update(provenance(ROOT), **initialize(worker))
        evaluator = Evaluator(worker,args.output/"measurements",metadata["binary_hash"],revision,
                              max_new=args.max_new,max_requests=args.max_new*4,max_seconds=args.hours*3600)
        try:
            for workload in selected:
                for reuse in (1,100,10000):
                    methods = ["fixed","heuristic"] + ["random"]*args.random_per_context
                    rng.shuffle(methods)
                    for method in methods:
                        start = time.perf_counter()
                        forced = ([{"kind":"STOP"}] if method == "fixed" else
                                  heuristic(worker,workload) if method == "heuristic" else None)
                        trajectory = sample_schedule(worker,workload,reuse,forced=forced,rng=rng)
                        trajectory["selection_ms"] = (time.perf_counter()-start)*1000
                        measurement = evaluator.measure(workload,trajectory["schedule"])
                        cost = deployment_ms(measurement["compile_ms"],measurement["execution_ms"],reuse,
                                             selection_ms=trajectory["selection_ms"],
                                             installation_ms=measurement["installation_ms"])
                        trajectory.update(schema_version=1,workload=workload,reuse=reuse,
                                          policy={"source":method,"seed":args.seed},
                                          outcome={"cost_ms":cost,"measurement_key":measurement["key"],
                                                   "analysis_ms":trajectory["selection_ms"],"policy_ms":0.})
                        with (args.output/"trajectories.jsonl").open("a") as stream:
                            stream.write(json.dumps(trajectory)+"\n")
            metadata["status"] = "complete"
        except BudgetReached as error:
            metadata.update(status="budget_reached",reason=str(error))
        except BaseException as error:
            metadata.update(status="stopped",reason=repr(error))
            raise
        finally:
            metadata["counters"] = evaluator.counters()
            (args.output/"run.json").write_text(json.dumps(metadata,indent=2)+"\n")
            print(json.dumps(metadata["counters"],indent=2))


if __name__ == "__main__":
    main()
