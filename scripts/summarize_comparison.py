"""Audit and summarize saved development results without native execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from audit_comparison import audit
from rlmlir.analysis import summarize
from rlmlir.corpus import load


def analyze(directory, *, baseline="reference", draws=2000, seed=20261029):
    integrity = audit(directory)
    metadata = json.loads((directory / "run.json").read_text())
    config = metadata["config"]
    available = {w["id"]: w for w in load(ROOT / "configs/corpus.json", config["split"])}
    workloads = [available[key] for key in config["workload_ids"]]
    methods = ["reference", "heuristic", *config["policies"]]
    if config["predictor"] is not None:
        methods.append("predictor")
    if config.get("best_fixed") is not None:
        methods.append("best_fixed")
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    report = summarize([e for e in events if e["status"] == "accepted"], workloads, methods,
                       config["reuse_counts"], config["rounds"], baseline=baseline, draws=draws, seed=seed)
    report.update(integrity_audit=integrity, numerical_output_rechecked=False, native_execution=False,
                  source_hashes={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                 for name in ("run.json", "config.json", "events.jsonl")},
                  analysis_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="reference")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261029)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("analysis output must be new")
    report = analyze(args.directory, baseline=args.baseline, draws=args.bootstrap_replicates, seed=args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"coverage": report["coverage"], "aggregate": report["aggregate"]}, indent=2))
