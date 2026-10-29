"""Read saved deployment contracts, without model loading or native execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.measurement import check_fresh_deployment
from rlmlir.objective import adapted_deployment_ms


def audit(directories):
    campaigns = []
    for directory in directories:
        path = directory / "run.json"
        metadata = json.loads(path.read_text())
        if metadata["status"] != "complete":
            raise ValueError("saved adaptation campaign is incomplete")
        paths = sorted(directory.glob("*/result.json"))
        results = [json.loads(path.read_text()) for path in paths]
        pairs = [(r["method"], r["budget"]) for r in results]
        expected = {(m, b) for m in metadata["methods"] for b in metadata["budgets"]}
        if len(pairs) != len(expected) or set(pairs) != expected:
            raise ValueError("saved method/budget grid is incomplete or duplicated")
        cases, digests = [], set()
        for path, result in zip(paths, results):
            if result.get("status", "complete") != "complete":
                raise ValueError("saved case failed")
            requests, budget = result["requests"], result["budget"]
            if (len(requests) != budget or result["counters"]["candidate_requests"] != budget
                    or [r["request"] for r in requests] != list(range(1, budget + 1))):
                raise ValueError("saved logical request budget mismatch")
            if result["best"] not in requests or result["best"] != min(requests, key=lambda r: r["cost_ms"]):
                raise ValueError("saved best schedule was not the best observed candidate")
            deployment = result["deployment"]
            if deployment["correct"] is not True:
                raise ValueError("saved deployment was not recorded correct")
            check_fresh_deployment(deployment, result["reuse"])
            total = adapted_deployment_ms(result["tuning_ms"], deployment["total_ms"])
            if not math.isclose(total, result["adapted_total_ms"], rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError("saved adapted total mismatch")
            digest = result.get("initial_policy_digest")
            if digest is not None:
                digests.add(digest)
                if (result["method"] in ("frozen", "random", "mutation") or budget == 8) and result["final_policy_digest"] != digest:
                    raise ValueError("saved no-update control changed policy")
            counts = result.get("deployment_worker_counts")
            if counts is not None and (counts.get("actual_compilations") != 1 or counts.get("actual_invocations") != result["reuse"]):
                raise ValueError("saved deployment hardware counters mismatch")
            cases.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "method": result["method"], "budget": budget, "deployment_contract_passed": True,
                          "reset_digest_recorded": digest is not None, "deployment_counters_recorded": counts is not None,
                          "event_log_present": (path.parent / "events.jsonl").exists()})
        if len(digests) > 1:
            raise ValueError("saved contexts have different initial policies")
        campaigns.append({"path": directory.relative_to(ROOT).as_posix(), "run_sha256": hashlib.sha256((directory / "run.json").read_bytes()).hexdigest(),
                          "binary_hash": metadata["binary_hash"], "cases": cases})
    return {"schema_version": 1, "status": "passed", "audit_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "native_execution": False, "numerical_output_rechecked": False,
            "scope": "Saved correctness flags, deployment cost/count/cache contracts, logical budgets, best-observed selection, available reset identities and counters only.",
            "limitations": "Missing historical events/digests/counters are reported, not reconstructed. This is not a current-backend performance or full learning-provenance audit.",
            "campaigns": campaigns}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output must be new")
    report = audit([path.resolve() for path in args.directories])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "cases": sum(len(c["cases"]) for c in report["campaigns"])}))
