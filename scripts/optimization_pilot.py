"""Pretraining-only, 100-candidate pilot with retained drift blocks and provenance."""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.kernels import matmul, matmul_case, matvec, matvec_case
from rlmlir.measurement import calibrate, schedule, validate
from rlmlir.worker import Worker


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot.json")
    parser.add_argument("--worker", type=Path, default=ROOT / "build/rlmlir-worker")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing campaign")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise SystemExit("commit tracked changes before measuring")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_bytes(config_bytes)
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    rng = random.Random(config["seed"])
    workloads = [(family, dims) for family in ("matmul", "matvec") for dims in config[f"{family}_shapes"]]
    rng.shuffle(workloads)
    metadata = {"schema_version": 1, "status": "running", "kind": "training-only-optimization-pilot",
                "source_revision": revision, "worker_sha256": digest(args.worker.read_bytes()),
                "protocol_sha256": digest(config_bytes), "guest_cpu": cpu, "os": platform.platform(),
                "cpu": next(s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")),
                "packages": subprocess.check_output(["dpkg-query", "-W", "llvm-18-dev", "libmlir-18-dev", "clang-18", "mlir-18-tools"], text=True),
                "completed_workloads": 0, "distinct_candidates": 0,
                "warning": "Exploratory training-only measurements; no learned-method comparison or final performance claim."}
    def save_metadata():
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save_metadata()
    started = time.perf_counter()
    with (args.output / "records.jsonl").open("w") as records:
        def record(event):
            records.write(json.dumps({"schema_version": 1, **event}, allow_nan=False) + "\n")
            records.flush()
        try:
            with Worker(args.worker) as worker:
                metadata["worker"] = worker.request("info")
                # Record one-time compiler initialization; deployed service starts afterwards.
                warm_start = time.perf_counter()
                warm = worker.request("compile", source=matmul(1, 1, 1))
                worker.request("invoke", handle=warm["handle"], buffers=[[2], [3], [0]])
                worker.request("release", handle=warm["handle"])
                worker.request("clear_cache")
                metadata["initialization_ms"] = (time.perf_counter() - warm_start) * 1000
                for family, dims in workloads:
                    workload = family + "-" + "x".join(map(str, dims))
                    generator = matmul if family == "matmul" else matvec
                    make_case = matmul_case if family == "matmul" else matvec_case
                    source = generator(*dims)
                    buffers, expected = make_case(*dims, seed=7)
                    codes, counts = {}, {}
                    names = list(config["schedules"])
                    rng.shuffle(names)
                    for name in names:
                        actions = schedule(name, family)
                        target = args.output / "artifacts" / workload / name
                        target.mkdir(parents=True)
                        tick = time.perf_counter()
                        built = worker.request("compile", source=source, actions=actions)
                        request_ms = (time.perf_counter() - tick) * 1000
                        if built["cache_hit"]:
                            raise RuntimeError("pilot expected a distinct uncached candidate")
                        handle = built["handle"]
                        codes[name] = handle
                        first = worker.request("invoke", handle=handle, buffers=buffers)
                        errors = [validate(first["output"], expected)]
                        for seed in (17, 27):
                            additional, answer = make_case(*dims, seed=seed)
                            check = worker.request("invoke", handle=handle, buffers=additional, count=3)
                            errors.append(validate(check["output"], answer))
                        for file, text in [("source.mlir", source), ("transformed.mlir", built["transformed_ir"]),
                                           ("lowered.mlir", built["lowered_ir"]), ("optimized.ll", built["llvm_ir"])]:
                            (target / file).write_text(text)
                        worker.request("dump_object", handle=handle, path=str((target / "kernel.o").resolve()))
                        assembly = subprocess.check_output(["llvm-objdump-18", "-d", "--no-show-raw-insn", str(target / "kernel.o")], text=True)
                        # Remove local directory paths, keeping the complete disassembly body.
                        assembly = assembly.replace(str(target / "kernel.o"), "kernel.o")
                        (target / "assembly.txt").write_text(assembly)
                        record({"event": "compiled", "workload": workload, "family": family, "shape": dims,
                                "schedule": name, "actions": actions, "correct": True, "input_seeds": [7,17,27],
                                "max_abs_errors": errors, "compile_request_ms": request_ms,
                                "lowering_ms": built["lowering_ms"], "codegen_ms": built["codegen_ms"],
                                "first_invocation_ms": first["batch_ms"], "source_sha256": digest(source.encode()),
                                "transformed_sha256": digest(built["transformed_ir"].encode()),
                                "llvm_sha256": digest(built["llvm_ir"].encode()),
                                "object_sha256": digest((target / "kernel.o").read_bytes()), "cache_hit": False})
                        metadata["distinct_candidates"] += 1
                    # All compilation and disassembly finish before timed blocks begin.
                    for name in config["schedules"]:
                        counts[name], samples = calibrate(worker, codes[name], buffers, config["batch_target_ms"])
                        record({"event": "calibration", "workload": workload, "schedule": name, "samples": samples})
                    def timing(name):
                        result = worker.request("benchmark", handle=codes[name], buffers=buffers, count=counts[name])
                        validate(result["output"], expected)
                        return {"schedule": name, "count": counts[name], "batch_ms": result["batch_ms"],
                                "per_invocation_ms": result["batch_ms"] / counts[name]}
                    for round_index in range(config["rounds"]):
                        order = list(config["schedules"])
                        rng.shuffle(order)
                        for attempt in range(2):
                            before = timing("baseline")
                            samples = [timing(name) for name in order]
                            after = timing("baseline")
                            drift = abs(after["per_invocation_ms"] / before["per_invocation_ms"] - 1)
                            too_short = any(s["batch_ms"] < config["minimum_batch_ms"] for s in samples + [before, after])
                            flagged = drift > config["drift_limit"] or too_short
                            record({"event": "block", "workload": workload, "round": round_index, "attempt": attempt,
                                    "anchor_before": before, "anchor_after": after, "anchor_drift": drift,
                                    "below_minimum": too_short, "flagged": flagged, "samples": samples})
                            if not flagged:
                                break
                    for handle in codes.values():
                        worker.request("release", handle=handle)
                    worker.request("clear_cache")
                    metadata["completed_workloads"] += 1
                    metadata["elapsed_seconds"] = time.perf_counter() - started
                    save_metadata()
                    print(json.dumps({"completed": metadata["completed_workloads"], "workload": workload,
                                      "candidates": metadata["distinct_candidates"]}), flush=True)
            metadata["status"] = "complete"
        except BaseException as error:
            metadata["status"] = "stopped"
            metadata["reason"] = repr(error)
            record({"event": "stopped", "reason": repr(error)})
            raise
        finally:
            metadata["elapsed_seconds"] = time.perf_counter() - started
            save_metadata()


if __name__ == "__main__":
    main()
