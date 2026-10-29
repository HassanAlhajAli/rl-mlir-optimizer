"""Read-only PPO coverage report. It cannot unseal final performance access."""
import json
import math
from pathlib import Path

from .experiment import COMPATIBILITY_FIELDS
from .validation import audit as audit_validation, require, sha256

SEEDS = (7, 17, 27, 37, 47)
OBJECTIVES = ("reuse", "runtime")


def job_contract(job):
    argv = job["argv"]
    require(len(argv) >= 2 and argv[1] == "scripts/train_ppo.py", "PPO job must invoke the training entry point")
    tail = argv[2:]
    flags = ("--output", "--objective", "--seed", "--max-new", "--max-requests", "--hours", "--timesteps")
    require(len(tail) == 2 * len(flags) and set(tail[::2]) == set(flags), "missing, duplicate or unexpected PPO argument")
    args = dict(zip(tail[::2], tail[1::2]))
    seed, objective = int(args["--seed"]), args["--objective"]
    name = f"ppo-{objective}-{seed}"
    require(seed in SEEDS and objective in OBJECTIVES and job["name"] == name
            and Path(args["--output"]).name == name, "PPO job identity/output mismatch")
    limits = {"max_new": int(args["--max-new"]), "max_requests": int(args["--max-requests"]),
              "wall_limit_hours": float(args["--hours"]), "requested_timesteps": int(args["--timesteps"])}
    require(all(math.isfinite(v) and v > 0 for v in limits.values()) and limits["max_new"] <= 4096,
            "invalid PPO allocation")
    return {"name": name, "seed": seed, "objective": objective, **limits}


def definitions(jobs):
    contracts = [job_contract(job) for job in jobs]
    names = {c["name"] for c in contracts}
    expected = {f"ppo-{objective}-{seed}" for seed in SEEDS for objective in OBJECTIVES}
    require(names == expected and len(contracts) == len(expected), "both objectives and all five seeds must appear exactly once")
    allocations = {tuple(c[k] for k in ("max_new", "max_requests", "wall_limit_hours", "requested_timesteps")) for c in contracts}
    require(len(allocations) == 1, "PPO allocations differ across seeds/objectives")
    return {c["name"]: c for c in contracts}


def report(directory, config_path, current, validation_runs=None):
    directory, config_path = Path(directory), Path(config_path)
    contracts = definitions(json.loads(config_path.read_text())["jobs"])
    manifest = json.loads((directory / "campaign.json").read_text())
    queue = json.loads((directory / "queue.json").read_text())
    require(queue["status"] != "running", "do not audit an active campaign")
    require(manifest["config_sha256"] == sha256(config_path), "campaign configuration changed")
    require(definitions(queue["definition"]) == contracts, "resolved campaign definitions differ from configuration")
    actual = {job["name"]: job for job in queue["jobs"]}
    require(len(actual) == len(queue["jobs"]) and set(actual) <= set(contracts), "duplicate or unregistered campaign job")
    validation_runs = validation_runs or {}
    require(set(validation_runs) <= set(contracts), "validation mapping contains an unknown PPO run")
    rows = []
    for name, expected in contracts.items():
        job = actual.get(name)
        row = {**expected, "queue_status": job["status"] if job else "not_started",
               "validation_status": "missing", "blockers": []}
        blockers = row["blockers"]
        if job is None:
            blockers.append("training was not started")
        else:
            require(job_contract(job) == expected, "executed job arguments changed")
            path = directory / name / "run.json"
            row["training_metadata_sha256"] = sha256(path)
            trained = json.loads(path.read_text())
            row["training_status"] = trained["status"]
            row["training_binary_hash"] = trained["binary_hash"]
            row["termination_reason"] = trained.get("reason")
            if job["status"] != "complete" or job.get("exit_code") != 0 or trained["status"] not in ("timesteps_complete", "budget_reached"):
                blockers.append("training did not finish successfully within its bounded protocol")
            for field in ("seed", "objective", "max_new", "max_requests", "wall_limit_hours", "requested_timesteps"):
                if trained.get(field) != expected[field]:
                    blockers.append("training metadata differs from declared " + field)
            mismatches = [field for field in COMPATIBILITY_FIELDS if current.get(field) is None
                          or trained.get(field) != current[field] or manifest.get(field) != trained.get(field)]
            row["incompatible_fields"] = mismatches
            if mismatches:
                blockers.append("incompatible provenance: " + ", ".join(mismatches))
            counts = trained.get("cumulative_counters", trained.get("counters", {}))
            row["candidate_requests"] = counts.get("candidate_requests")
            row["new_measurements"] = counts.get("new_measurements")
            for field, limit in (("candidate_requests", "max_requests"), ("new_measurements", "max_new")):
                if type(counts.get(field)) is not int or not 0 <= counts[field] <= expected[limit]:
                    blockers.append("missing or exceeded cumulative " + field)
            if type(trained.get("timesteps")) is not int or not 0 <= trained["timesteps"] <= expected["requested_timesteps"] + 255:
                blockers.append("missing or exceeded timestep count")
        if name not in validation_runs:
            blockers.append("separate audited validation selection is missing")
        elif blockers:
            row["validation_status"] = "not_checked_because_training_is_ineligible"
        else:
            try:
                selected_path = Path(validation_runs[name])
                selected_meta = json.loads((selected_path / "run.json").read_text())
                require(Path(selected_meta["training_run"]).resolve() == (directory / name).resolve(),
                        "validation points to another training run")
                row["validation"] = audit_validation(selected_path, current)
                row["validation_metadata_sha256"] = sha256(selected_path / "run.json")
                row["validation_status"] = "audited"
            except (ValueError, KeyError, OSError, TypeError) as error:
                row["validation_status"] = "invalid"
                blockers.append("validation audit failed: " + str(error))
        rows.append(row)
    return {"schema_version": 1, "scope": "training metadata and checkpoint-selection coverage only; not a full method freeze",
            "queue_status": queue["status"], "queue_elapsed_seconds": queue["elapsed_seconds"],
            "current_contract": {k: current[k] for k in COMPATIBILITY_FIELDS},
            "campaign_sha256": sha256(directory / "campaign.json"), "queue_sha256": sha256(directory / "queue.json"),
            "config_sha256": sha256(config_path), "jobs": rows,
            "all_ten_selections_accounted_for": queue["status"] == "complete" and all(not row["blockers"] for row in rows),
            "training_update_evidence_checked": False,
            "whole_study_frozen": False, "heldout_access": False}
