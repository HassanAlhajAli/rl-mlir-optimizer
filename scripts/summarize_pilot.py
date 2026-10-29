"""Inspect every pilot workload; never select only the best timing attempts."""
import argparse
import collections
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.run / "metadata.json").read_text())
    if metadata["status"] != "complete":
        raise SystemExit("campaign incomplete; preserve its evidence and investigate first")
    config = json.loads((args.run / "protocol.json").read_text())
    records = [json.loads(line) for line in (args.run / "records.jsonl").read_text().splitlines()]
    compiled = {(r["workload"], r["schedule"]): r for r in records if r["event"] == "compiled"}
    blocks = [r for r in records if r["event"] == "block"]
    samples = collections.defaultdict(list)
    for block in blocks:
        if block["attempt"] == 0:
            for sample in block["samples"]:
                samples[block["workload"], sample["schedule"]].append(sample["per_invocation_ms"])
    if len(compiled) != 100 or any(len(v) != config["rounds"] for v in samples.values()):
        raise RuntimeError("pilot cardinality mismatch")
    medians = {key: statistics.median(value) for key, value in samples.items()}
    original = [b for b in blocks if b["attempt"] == 0]
    repeats = [b for b in blocks if b["attempt"] == 1]
    summary = {"source_revision": metadata["source_revision"], "candidates": len(compiled),
               "original_blocks": len(original), "flagged_original_blocks": sum(b["flagged"] for b in original),
               "repeat_blocks": len(repeats), "flagged_repeat_blocks": sum(b["flagged"] for b in repeats),
               "elapsed_seconds": metadata["elapsed_seconds"], "workloads": []}
    rows = ["# Training-only pilot observations", "", "Exploratory measurements on one Intel/WSL2 laptop. No policy was trained.", "",
            f"All {len(compiled)} candidates passed independent output checks. "
            f"{summary['flagged_original_blocks']}/{len(original)} original timing blocks were flagged; "
            f"{summary['flagged_repeat_blocks']}/{len(repeats)} repeat blocks remained flagged.", "",
            "The table uses medians of all five original blocks, including flagged blocks. Repeats remain in raw records. "
            "Best-observed schedules are chosen on these development observations, so their ratios are descriptive and selection-biased, not confirmatory gains.", "",
            "| Workload | Baseline us | Lowest median among five | Descriptive ratio | Estimated choices N=1/100/10000 |",
            "|---|---:|---|---:|---|"]
    for workload in sorted({key[0] for key in compiled}):
        reference = medians[workload, "baseline"]
        winner = min(config["schedules"], key=lambda name: medians[workload, name])
        choices = {}
        for n in config["reuse_counts"]:
            choices[n] = min(config["schedules"], key=lambda name: compiled[workload, name]["lowering_ms"] + compiled[workload, name]["codegen_ms"] + n * medians[workload, name])
        entry = {"workload": workload, "median_ms": {name: medians[workload, name] for name in config["schedules"]},
                 "observed_lowest_runtime": winner, "descriptive_runtime_ratio": reference / medians[workload, winner],
                 "estimated_cost_choices": choices,
                 "distinct_object_hashes": len({compiled[workload, name]["object_sha256"] for name in config["schedules"]})}
        summary["workloads"].append(entry)
        rows.append(f"| {workload} | {reference * 1000:.3f} | {winner} | {entry['descriptive_runtime_ratio']:.2f}x | {' / '.join(choices.values())} |")
    summary["workloads_with_reuse_dependent_estimated_choice"] = sum(len(set(w["estimated_cost_choices"].values())) > 1 for w in summary["workloads"])
    rows += ["", "The C+N*T choice estimates include measured transformations/backend compilation but no learned policy selection, which does not exist yet. "
             "Compile cost was measured once per candidate, and complete deployment batches were not measured. These estimates cannot establish H1 yet.", "",
             "Reproduce with `python3 scripts/summarize_pilot.py <run-directory>`. Source revision, protocol hash, executable hash, package versions, affinity, all calibration samples and every block are retained beside this report."]
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.run / "observations.md").write_text("\n".join(rows) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "workloads"}, indent=2))


if __name__ == "__main__":
    main()
