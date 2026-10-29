"""Fresh complete deployments for two explicitly selected development examples."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.kernels import matmul, matmul_case, matvec, matvec_case
from rlmlir.measurement import validate, schedule
from rlmlir.worker import Worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite prior confirmation")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise SystemExit("commit changes before measuring")
    config_bytes = (ROOT / "configs/confirmation.json").read_bytes()
    config = json.loads(config_bytes)
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_bytes(config_bytes)
    executable = ROOT / "build/rlmlir-worker"
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    metadata = {"schema_version": 1, "status": "running", "guest_cpu": cpu,
                "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "worker_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "protocol_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "selection": config["selection_rule"], "count": 0,
                "boundary": "fresh compile + installation/owned buffer setup + N invocations; transport, compiler-service startup, independent validation and teardown excluded"}
    started = time.perf_counter()
    rng = random.Random(config["seed"])
    try:
        with Worker(executable) as worker, (args.output / "records.jsonl").open("w") as records:
            warm = worker.request("compile", source=matmul(1, 1, 1))
            worker.request("release", handle=warm["handle"])
            worker.request("clear_cache")
            for round_index in range(config["rounds"]):
                cases = [(item, name, n) for item in config["workloads"] for name in config["schedules"] for n in config["reuse_counts"]]
                rng.shuffle(cases)
                for item, name, n in cases:
                    family, dims = item["family"], item["shape"]
                    source = (matmul if family == "matmul" else matvec)(*dims)
                    buffers, expected = (matmul_case if family == "matmul" else matvec_case)(*dims, seed=7)
                    result = worker.request("deploy", source=source, actions=schedule(name, family), buffers=buffers, count=n)
                    max_error = validate(result.pop("output"), expected)
                    records.write(json.dumps({"schema_version": 1, "round": round_index, "family": family,
                                               "shape": dims, "schedule": name, "reuse": n,
                                               "correct": True, "input_seed": 7, "max_abs_error": max_error, **result}) + "\n")
                    records.flush()
                    metadata["count"] += 1
                print(json.dumps({"round_completed": round_index, "deployments": metadata["count"]}), flush=True)
        metadata["status"] = "complete"
    except BaseException as error:
        metadata["status"] = "stopped"
        metadata["reason"] = repr(error)
        raise
    finally:
        metadata["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
