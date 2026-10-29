"""Run an inspected fixed campaign; never change code while it is running."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from rlmlir.queue import run_jobs
from rlmlir.experiment import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--hours",type=float,default=8)
    args = parser.parse_args()
    if not 0 < args.hours <= 48:
        raise SystemExit("queue limit must be within the48-hour approved campaign budget")
    os.chdir(ROOT)
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT/"runs"):
        raise SystemExit("campaign output must stay inside this repository's runs folder")
    meta = provenance(ROOT)
    jobs = json.loads(args.config.read_text())["jobs"]
    for job in jobs:
        job["argv"] = [arg.replace("{python}",sys.executable).replace("{output}",str(args.output)) for arg in job["argv"]]
    args.output.mkdir(parents=True,exist_ok=True)
    manifest = args.output/"campaign.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["revision"] != meta["revision"] or previous["binary_hash"] != meta["binary_hash"]:
            raise SystemExit("campaign revision changed; review it before starting a new campaign")
    else:
        manifest.write_text(json.dumps({**meta,"config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
                                        "max_hours":args.hours},indent=2)+"\n")
    def guard():
        current = provenance(ROOT)
        if any(current[key] != meta[key] for key in ("revision","binary_hash","corpus_sha256")):
            raise RuntimeError("code, compiler or corpus changed during campaign")
    result = run_jobs(jobs,args.output,max_seconds=args.hours*3600,guard=guard)
    print(json.dumps({"status":result["status"],"elapsed_seconds":result["elapsed_seconds"]}))
    if result["status"] == "stopped":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
