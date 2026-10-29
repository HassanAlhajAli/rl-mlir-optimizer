"""Measured development comparison with a fresh cache and actor per context."""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.adaptation import adapt
from rlmlir.corpus import load
from rlmlir.evaluator import Evaluator
from rlmlir.experiment import initialize, provenance, measurement_lease
from rlmlir.kernels import GENERATORS, REFERENCES
from rlmlir.measurement import validate, check_fresh_deployment
from rlmlir.objective import adapted_deployment_ms
from rlmlir.policies import load_policy, policy_digest
from rlmlir.records import json_safe
from rlmlir.worker import Worker

PROTOCOL = "development-adaptation-v2"


def save(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(json_safe(record), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_tuning(result, workload, reuse, method, budget, seed, initial_digest):
    expected = {"workload": workload, "reuse": reuse, "method": method,
                "budget": budget, "seed": seed, "initial_policy_digest": initial_digest}
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("adaptation context or initial policy mismatch")
    requests = result["requests"]
    if (len(requests) != budget or [r["request"] for r in requests] != list(range(1, budget + 1))
            or result["counters"]["candidate_requests"] != budget):
        raise RuntimeError("adaptation did not consume exactly the declared request budget")
    if result["best"] not in requests or result["best"] != min(requests, key=lambda r: r["cost_ms"]):
        raise RuntimeError("returned schedule is not the best observed candidate")


def run_case(worker, directory, metadata, policy, workload, reuse, method, budget, seed,
             deadline, *, clock=time.perf_counter):
    directory.mkdir()
    result = {"protocol": PROTOCOL, "status": "started", "method": method, "budget": budget,
              "seed": seed, "workload": workload, "reuse": reuse}
    save(directory / "result.json", result)
    started, before = clock(), worker.counters.copy()
    evaluator = None
    deployment = None

    def emit(event):
        # Close each append before the next trial. Logging during tuning is charged.
        with (directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(json_safe(event), allow_nan=False) + "\n")

    try:
        if clock() >= deadline:
            raise TimeoutError("comparison wall-time limit reached")
        worker.request("clear_cache")
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("comparison wall-time limit reached")
        evaluator = Evaluator(worker, directory / "measurements", metadata["binary_hash"], metadata["revision"],
                              max_new=budget, max_requests=budget, max_seconds=remaining)
        result.update(adapt(worker, evaluator, workload, reuse, policy,
                            method=method, budget=budget, seed=seed, emit=emit))
        result["status"] = "tuned"
        save(directory / "result.json", result)
        if result["counters"] != evaluator.counters():
            raise RuntimeError("reported adaptation counters disagree with the evaluator")
        check_tuning(result, workload, reuse, method, budget, seed, metadata["initial_policy_digest"])
        if policy_digest(policy) != metadata["initial_policy_digest"]:
            raise RuntimeError("adaptation mutated the shared initial policy")
        buffers, expected = REFERENCES[workload["family"]](*workload["shape"], seed=7)
        source = GENERATORS[workload["family"]](*workload["shape"])
        if clock() >= deadline:
            raise TimeoutError("comparison wall-time limit reached before deployment")
        before_deploy = worker.counters.copy()
        try:
            deployment = worker.request("deploy", source=source, actions=result["best"]["schedule"],
                                        buffers=buffers, count=reuse)
        finally:
            result["deployment_worker_counts"] = dict(worker.counters - before_deploy)
        # Keep the original response, including output, until all checks have passed.
        error = validate(deployment["output"], expected)
        check_fresh_deployment(deployment, reuse)
        result["adapted_total_ms"] = adapted_deployment_ms(result["tuning_ms"], deployment["total_ms"])
        result["deployment"] = {**{k: v for k, v in deployment.items() if k != "output"},
                                "max_abs_error": error, "correct": True}
        result["status"] = "complete"
        return result
    except BaseException as error:
        result.update(status="failed", reason=repr(error))
        if deployment is not None:
            result["failed_deployment"] = {"response": deployment, "expected": expected}
        raise
    finally:
        if evaluator is not None:
            result["counters"] = evaluator.counters()
        result.update(case_elapsed_seconds=clock() - started, hardware_counts=dict(worker.counters - before))
        save(directory / "result.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--reuse", type=int, default=10000)
    parser.add_argument("--budgets", type=int, nargs="+", choices=[8, 16, 32], default=[8, 16, 32])
    parser.add_argument("--methods", nargs="+", choices=["frozen", "grpo", "ppo", "random", "mutation"],
                        default=["frozen", "grpo", "ppo", "random", "mutation"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hours", type=float, default=2)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.reuse <= 10000000:
        raise SystemExit("new output and bounded positive reuse required")
    if len(set(args.methods)) != len(args.methods) or len(set(args.budgets)) != len(args.budgets):
        raise SystemExit("methods and budgets must be unique")
    if not math.isfinite(args.hours) or args.hours <= 0:
        raise SystemExit("positive finite time limit required")
    workloads = [w for w in load(ROOT / "configs/corpus.json", args.split) if w["id"] == args.workload_id]
    if len(workloads) != 1:
        raise SystemExit("workload ID is not unique in the requested development split")
    workload = workloads[0]
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    metadata = provenance(ROOT)
    order = [(m, b) for m in args.methods for b in args.budgets]
    random.Random(args.seed).shuffle(order)
    metadata.update(status="running", protocol=PROTOCOL, role="development adaptation; not final evaluation",
                    workload=workload, policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                    seed=args.seed, reuse=args.reuse, methods=args.methods, budgets=args.budgets,
                    method_order=order, max_seconds=args.hours * 3600, command=sys.argv, controller_pid=os.getpid(),
                    summaries=[], cases=[{"method": m, "budget": b, "status": "pending"} for m, b in order])
    args.output.mkdir(parents=True)
    save(args.output / "run.json", metadata)
    started = time.perf_counter()
    deadline = started + args.hours * 3600
    worker = None
    worker_start = None
    try:
        loading = time.perf_counter()
        policy, policy_metadata = load_policy(args.policy, expected_provenance=metadata)
        metadata["model_loading_ms"] = (time.perf_counter() - loading) * 1000
        if hashlib.sha256(args.policy.read_bytes()).hexdigest() != metadata["policy_sha256"]:
            raise RuntimeError("policy changed while loading")
        if "ppo" in args.methods and not policy_metadata.get("critic_fitted"):
            raise ValueError("warm PPO requires a critic fitted to absolute log cost on training data")
        metadata["initial_policy_digest"] = policy_digest(policy)
        if time.perf_counter() >= deadline:
            raise TimeoutError("comparison wall-time limit reached before worker startup")
        worker_start = time.perf_counter()
        with Worker(ROOT / "build/rlmlir-worker") as worker:
            metadata["worker_pid"] = worker.process.pid
            save(args.output / "run.json", metadata)
            metadata.update(initialize(worker))
            for case in metadata["cases"]:
                method, budget = case["method"], case["budget"]
                case["status"] = "running"
                save(args.output / "run.json", metadata)
                try:
                    result = run_case(worker, args.output / f"{method}-{budget}", metadata, policy,
                                      workload, args.reuse, method, budget, args.seed, deadline)
                except BaseException as error:
                    case.update(status="failed", reason=repr(error))
                    raise
                case["status"] = "complete"
                summary = {"method": method, "budget": budget, "tuning_ms": result["tuning_ms"],
                           "deployment_ms": result["deployment"]["total_ms"],
                           "adapted_total_ms": result["adapted_total_ms"],
                           "unique_schedules": result["groups"][-1]["unique_schedules"],
                           "updates": sum(g["update"] is not None and not g["update"]["skipped"] for g in result["groups"]),
                           "schedule": result["best"]["schedule"], "counters": result["counters"]}
                metadata["summaries"].append(summary)
                save(args.output / "run.json", metadata)
                print(json.dumps(summary, allow_nan=False), flush=True)
        metadata["status"] = "complete"
    except BaseException as error:
        metadata.update(status="stopped", reason=repr(error))
        raise
    finally:
        if worker is not None:
            metadata["hardware_counts"] = dict(worker.counters)
        if worker_start is not None:
            metadata["worker_stage_elapsed_seconds"] = time.perf_counter() - worker_start
        metadata["elapsed_seconds"] = time.perf_counter() - started
        save(args.output / "run.json", metadata)


if __name__ == "__main__":
    with measurement_lease(ROOT):
        main()
