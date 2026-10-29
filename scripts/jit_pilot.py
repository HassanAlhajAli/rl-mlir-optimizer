"""Serial environment pilot. These development workloads are not held-out data."""
import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.kernels import matmul, matmul_case
from rlmlir.worker import Worker


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, default=ROOT / "build/rlmlir-worker")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-ms", type=float, default=1000)
    args = parser.parse_args()
    if not 100 <= args.batch_ms <= 5000:
        raise SystemExit("batch duration must be between 100 and 5000 ms")
    if args.output.exists():
        raise SystemExit("refusing to overwrite a previous pilot")
    # The child inherits guest affinity. WSL may still move the underlying vCPU.
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    source = matmul(32, 32, 32)
    buffers, expected = matmul_case(32, 32, 32)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
    if dirty:
        raise SystemExit("commit tracked source changes before recording a pilot")
    packages = subprocess.check_output(["dpkg-query", "-W", "llvm-18-dev", "libmlir-18-dev", "clang-18", "mlir-18-tools"], text=True)
    cpu_model = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    record = {"schema_version": 1, "kind": "environment-pilot", "source_revision": revision,
              "worker_sha256": hashlib.sha256(args.worker.read_bytes()).hexdigest(),
              "source_ir_sha256": sha(source), "shape": [32, 32, 32], "input_seed": 7,
              "cpu": cpu_model, "guest_cpu": cpu, "os": platform.platform(), "minimum_batch_ms": args.batch_ms,
              "packages": packages, "target": "x86-64-v3", "rounds": [],
              "limitations": ["development kernel only", "WSL guest affinity is not physical isolation",
                              "no optimization or learning comparison", "JSON and input allocation excluded from execution"]}
    campaign_start = time.perf_counter()
    args.output.mkdir(parents=True)
    (args.output / "source.mlir").write_text(source)
    with Worker(args.worker) as worker:
        record["worker"] = worker.request("info")
        for round_index in range(5):
            start = time.perf_counter()
            built = worker.request("compile", source=source)
            compile_request_ms = (time.perf_counter() - start) * 1000
            handle = built["handle"]
            first = worker.request("invoke", handle=handle, buffers=buffers)
            if len(first["output"]) != len(expected) or any(
                not math.isfinite(a) or abs(a-b) > 1e-10 + 1e-9*abs(b)
                for a,b in zip(first["output"], expected)
            ):
                raise RuntimeError("correctness failed; no accepted timing")
            count = 1
            while True:
                timing = worker.request("benchmark", handle=handle, buffers=buffers, count=count)
                if timing["batch_ms"] >= args.batch_ms:
                    break
                count = min(10000000, max(count + 1, math.ceil(count * args.batch_ms * 1.2 / max(timing["batch_ms"], .001))))
                if count == 10000000:
                    raise RuntimeError("could not calibrate a bounded timing batch")
            batch = worker.request("benchmark", handle=handle, buffers=buffers, count=count)
            anchor = worker.request("benchmark", handle=handle, buffers=buffers, count=count)
            drift = abs(anchor["batch_ms"] - batch["batch_ms"]) / batch["batch_ms"]
            repeat = None
            if drift > .05 or batch["batch_ms"] < args.batch_ms:
                # Retain the original block and repeat once. Never select the faster result.
                repeat = worker.request("benchmark", handle=handle, buffers=buffers, count=count)
            entry = {"round": round_index, "correct": True, "compile_request_ms": compile_request_ms,
                     "lowering_ms": built["lowering_ms"], "codegen_ms": built["codegen_ms"],
                     "first_invocation_ms": first["batch_ms"], "count": count,
                     "batch_ms": batch["batch_ms"], "per_invocation_ms": batch["batch_ms"] / count,
                     "anchor_batch_ms": anchor["batch_ms"], "anchor_drift": drift,
                     "flagged": drift > .05 or batch["batch_ms"] < args.batch_ms,
                     "repeat_batch_ms": None if repeat is None else repeat["batch_ms"],
                     "lowered_ir_sha256": sha(built["lowered_ir"])}
            record["rounds"].append(entry)
            (args.output / "lowered.mlir").write_text(built["lowered_ir"])
            worker.request("release", handle=handle)
            worker.request("clear_cache")
            (args.output / "measurements.json").write_text(json.dumps(record, indent=2) + "\n")
    record["elapsed_seconds"] = time.perf_counter() - campaign_start
    samples = [r["per_invocation_ms"] for r in record["rounds"]]
    record["median_invocation_ms"] = statistics.median(samples)
    record["relative_range"] = (max(samples) - min(samples)) / statistics.median(samples)
    record["status"] = "complete"
    (args.output / "measurements.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("status", "median_invocation_ms", "relative_range", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
