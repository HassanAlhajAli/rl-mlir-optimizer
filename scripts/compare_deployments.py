"""Development comparison of frozen selectors; held-out access is unavailable."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.comparison import DevelopmentComparison, interleaved_order
from rlmlir.corpus import load
from rlmlir.experiment import initialize, measurement_lease, provenance, require_compatible_provenance
from rlmlir.worker import Worker
from rlmlir.records import json_safe


def prepare(config):
    fields = {"schema_version", "split", "workload_ids", "reuse_counts", "rounds",
              "order_seed", "hint_factor", "policies", "predictor"}
    if set(config) - {"best_fixed"} != fields or config["schema_version"] != 1:
        raise ValueError("unsupported or incomplete comparison configuration")
    if not isinstance(config["policies"], dict) or {"reference", "heuristic", "predictor", "best_fixed"} & config["policies"].keys():
        raise ValueError("policy names must be distinct from built-in selectors")
    if not math.isfinite(config["hint_factor"]) or config["hint_factor"] <= 0:
        raise ValueError("positive finite hint factor required")
    available = {w["id"]: w for w in load(ROOT / "configs/corpus.json", config["split"])}
    if any(identity not in available for identity in config["workload_ids"]):
        raise ValueError("workload ID is absent from the requested development split")
    workloads = [available[identity] for identity in config["workload_ids"]]
    methods = ["reference", "heuristic", *config["policies"]]
    if config["predictor"] is not None:
        methods.append("predictor")
    if config.get("best_fixed") is not None:
        methods.append("best_fixed")
    order = interleaved_order(workloads, methods, config["reuse_counts"], config["rounds"], config["order_seed"])
    return workloads, order


def selectors_for(config, metadata):
    import torch
    from rlmlir.policies import load_policy
    from rlmlir.predictor import CostPredictor, rank_schedules
    from rlmlir.selection import heuristic, sample_schedule
    from rlmlir.fixed_baseline import fixed_selector, load_fixed

    selectors = {
        "reference": lambda *_: {"schedule": []},
        "heuristic": lambda w, p, n: sample_schedule(w, p, n, forced=heuristic(w, p)),
    }
    artifacts = {}
    for name, value in config["policies"].items():
        path = ROOT / value
        actor, saved = load_policy(path, expected_provenance=metadata)
        actor.set_training_mode(False)
        selectors[name] = lambda w, p, n, model=actor: sample_schedule(w, p, n, policy=model, deterministic=True)
        artifacts[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "seed": saved.get("seed"), "objective": saved.get("objective"),
                           "role": saved.get("role")}
    if config["predictor"] is not None:
        path = ROOT / config["predictor"]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        require_compatible_provenance(saved["metadata"], metadata, context="deployment predictor")
        model = CostPredictor()
        model.load_state_dict(saved["state_dict"])
        model.eval()
        selectors["predictor"] = lambda w, p, n: rank_schedules(w, p, n, model)
        artifacts["predictor"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "seed": saved["metadata"]["seed"]}
    if config.get("best_fixed") is not None:
        path = ROOT / config["best_fixed"]
        saved = load_fixed(path, metadata)
        selectors["best_fixed"] = fixed_selector(saved)
        artifacts["best_fixed"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "selected": saved["selection"]["selected"],
                                   "full_training_coverage": saved["selection"]["full_training_coverage"],
                                   "protocol": saved["protocol"]}
    return selectors, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument("--check-only", action="store_true", help="inspect the development grid; no models, worker or output run")
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    workloads, order = prepare(config)
    if args.check_only:
        print(json.dumps({"role": "configuration check only; not compiler readiness or a method freeze",
                          "cases": len(order), "rounds": config["rounds"], "workloads": len(workloads),
                          "methods": sorted({c["method"] for c in order}),
                          "model_artifacts_checked": False, "heldout_access": False}))
        return
    if args.output is None or args.output.exists() or not math.isfinite(args.hours) or args.hours <= 0:
        raise SystemExit("new output path and positive finite time limit required")
    import torch
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    metadata = provenance(ROOT)
    began = time.perf_counter()
    selectors, artifacts = selectors_for(config, metadata)
    metadata.update(model_loading_ms=(time.perf_counter() - began) * 1000,
                    status="running", role="development frozen-selector comparison; not final evaluation",
                    comparison_protocol="interleaved-development-v1", artifacts=artifacts,
                    controller_pid=os.getpid(), command=sys.argv,
                    config_sha256=hashlib.sha256(raw).hexdigest(), config=config, order=order,
                    worker_stage_limit_seconds=args.hours * 3600,
                    boundary="selection + fresh compile/install/N calls; research validation/calibration recorded separately")
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_bytes(raw)
    events_path = args.output / "events.jsonl"
    totals = {"started": 0, "accepted": 0, "failed": 0, "unresolved_drift": 0}
    def save_metadata():
        (args.output / "run.json").write_text(json.dumps(json_safe(metadata), indent=2, allow_nan=False) + "\n")
    def emit(record):
        record.update(revision=metadata["revision"], binary_hash=metadata["binary_hash"],
                      comparison_protocol=metadata["comparison_protocol"])
        with events_path.open("a") as stream:
            stream.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
        totals[record["status"]] += 1
        totals["unresolved_drift"] += bool(record.get("unresolved_drift"))
    def save_ir(index, artifact):
        with gzip.open(args.output / f"case-{index:05d}-ir.json.gz", "wt") as stream:
            json.dump(artifact, stream)
    save_metadata()
    stage_started = time.perf_counter()
    worker = None
    try:
        with Worker(ROOT / "build/rlmlir-worker") as worker:
            metadata["worker_pid"] = worker.process.pid
            metadata.update(initialize(worker))
            remaining = args.hours * 3600 - (time.perf_counter() - stage_started)
            if remaining <= 0:
                raise TimeoutError("initialization exhausted comparison time limit")
            runner = DevelopmentComparison(worker, max_seconds=remaining, emit=emit, save_ir=save_ir)
            runner.run(workloads, selectors, config["reuse_counts"], rounds=config["rounds"],
                       seed=config["order_seed"], hint_factor=config["hint_factor"])
            metadata["worker_counts"] = dict(worker.counters)
        metadata["status"] = "complete"
    except BaseException as error:
        metadata.update(status="stopped", reason=repr(error))
        raise
    finally:
        if worker is not None:
            metadata["worker_counts"] = dict(worker.counters)
        metadata.update(totals=totals, worker_stage_elapsed_seconds=time.perf_counter() - stage_started)
        save_metadata()


if __name__ == "__main__":
    # A configuration-only inspection cannot start a worker or acquire timing resources.
    if "--check-only" in sys.argv:
        main()
    else:
        with measurement_lease(ROOT):
            main()
