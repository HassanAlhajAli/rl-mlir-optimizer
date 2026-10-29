"""Verify the long-lived diagnostic's request trail without making performance claims."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(directory):
    run = json.loads((directory / "run.json").read_text())
    require(run["status"] == "complete", "diagnostic did not complete")
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    require(len(records) == run["plan"]["records"] == run["completed_records"], "record count mismatch")
    require(all(row["index"] == index and row["status"] == "complete"
                for index, row in enumerate(records[:-1])), "accepted history incomplete")
    require(records[-1]["key"] == run["plan"]["original_failure_key"] and
            records[-1]["status"] == "reached_original_failure_point", "original failure point missing")

    commands, peak, count_invocations = Counter(), Counter(), 0
    pending = None
    with (directory / "requests.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            sequence = event["sequence"]
            for name in ("VmRSS_kib", "VmHWM_kib", "VmSize_kib", "VmPeak_kib"):
                peak[name] = max(peak[name], event["worker_memory"].get(name, 0))
            if event["event"] == "request_start":
                require(pending is None and sequence == sum(commands.values()) + 1,
                        "nonconsecutive or overlapping request")
                pending = event
                continue
            require(event["event"] == "request_complete", "failed or unknown request event")
            require(pending is not None and all(event[field] == pending[field]
                    for field in ("sequence", "record_index", "record_key", "phase", "command", "count")),
                    "request completion does not match its start")
            commands[event["command"]] += 1
            if event["command"] in ("invoke", "benchmark"):
                count_invocations += event["count"]
            pending = None
    require(pending is None, "unfinished last request")
    require(sum(commands.values()) == run["worker_counts"]["worker_requests"], "worker request count mismatch")
    for command, count in commands.items():
        require(count == run["worker_counts"][command + "_requests"], f"{command} count mismatch")
    require(count_invocations == run["worker_counts"]["actual_invocations"], "invocation count mismatch")
    require(peak["VmHWM_kib"] < run["worker_memory_limit_bytes"] / 1024,
            "observed RSS reached address-space ceiling")
    stderr = (directory / "worker.stderr").read_bytes()
    return {"status": "passed", "run_revision": run["revision"],
            "binary_sha256": run["binary_sha256"], "records_sha256": run["records_sha256"],
            "records": len(records), "request_counts": dict(commands),
            "total_requests": sum(commands.values()), "total_invocations": count_invocations,
            "peak_worker_memory_kib": dict(peak), "stderr_bytes": len(stderr),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "elapsed_seconds": run["elapsed_seconds"], "limit_seconds": run["limit_seconds"],
            "original_failure_point_reached": True,
            "limitations": run["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
