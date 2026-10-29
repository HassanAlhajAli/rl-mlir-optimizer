"""Balanced, paired development summaries with shape-group uncertainty."""
import math
from collections import defaultdict
from statistics import fmean

import numpy as np

from .comparison import interleaved_order
from .corpus import FrozenCorpus
from .measurement import check_fresh_deployment

PROTOCOL = "development-comparison-analysis-v1"


def positive(value, name, *, allow_zero=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (value == 0 and not allow_zero)):
        raise ValueError("invalid analysis value: " + name)
    return value


def case_cost(record):
    """One deployment estimate per case, retaining every flagged attempt."""
    selection = positive(record["selection_ms"], "selection_ms", allow_zero=True)
    attempts = record["attempts"]
    if (not 1 <= len(attempts) <= 2 or len(attempts) != (2 if attempts[0]["flagged"] else 1)
            or record["unresolved_drift"] != attempts[-1]["flagged"]):
        raise ValueError("invalid drift repeat accounting")
    samples = []
    for index, attempt in enumerate(attempts):
        if attempt["attempt"] != index or attempt["status"] != "accepted":
            raise ValueError("incomplete deployment attempt")
        deployment = attempt["deployment"]
        if deployment["correct"] is not True:
            raise ValueError("incorrect deployment")
        check_fresh_deployment(deployment, record["reuse"])
        total = positive(attempt["total_with_selection_ms"], "complete deployment")
        if not math.isclose(total, selection + deployment["total_ms"], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("complete deployment cost mismatch")
        if type(attempt["steady_count"]) is not int or attempt["steady_count"] < 1:
            raise ValueError("invalid steady invocation count")
        samples.append({"selection_ms": selection, **{k: deployment[k] for k in
                        ("compile_ms", "installation_ms", "execution_ms")},
                        "total_ms": total, "steady_ms_per_call": positive(attempt["steady_batch_ms"], "steady batch") / attempt["steady_count"]})
    return {"mean_costs": {k: fmean(s[k] for s in samples) for k in samples[0]},
            "attempt_costs": samples, "attempts": len(samples),
            "flagged_attempts": sum(bool(a["flagged"]) for a in attempts),
            "original_flagged": bool(attempts[0]["flagged"]),
            "unresolved_drift": bool(record["unresolved_drift"])}


def effect(log_ratio):
    return {"mean_log_speedup": log_ratio, "geometric_mean_speedup": math.exp(log_ratio),
            "latency_reduction_percent": -100 * math.expm1(-log_ratio)}


def clustered_interval(rows, draws, seed):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["family"]][row["group"]].append(row["mean_log_speedup"])
    counts = {family: len(items) for family, items in sorted(groups.items())}
    base = {"unit": "frozen corpus group within family", "groups_per_family": counts,
            "replicates": draws, "seed": seed, "level": .95,
            "small_cluster_warning": any(n < 10 for n in counts.values())}
    if any(n < 2 for n in counts.values()):
        return {**base, "speedup_interval": None, "latency_reduction_interval_percent": None,
                "reason": "At least one family has fewer than two independent shape groups."}
    # Reuse the same seeded group draws for each method on an identical grid.
    rng = np.random.default_rng(seed)
    family_samples = []
    for family in sorted(groups):
        values = [groups[family][group] for group in sorted(groups[family])]
        sums = np.array([sum(v) for v in values])
        sizes = np.array([len(v) for v in values])
        indices = rng.integers(len(values), size=(draws, len(values)))
        # All shapes in a sampled group travel together, including repeated groups.
        family_samples.append(sums[indices].sum(axis=1) / sizes[indices].sum(axis=1))
    lower, upper = np.quantile(np.mean(family_samples, axis=0), [.025, .975])
    return {**base, "speedup_interval": [math.exp(lower), math.exp(upper)],
            "latency_reduction_interval_percent": [-100 * math.expm1(-lower), -100 * math.expm1(-upper)],
            "reason": "Conditional workload-group bootstrap; not training-seed or machine uncertainty."}


def summarize(records, workloads, methods, reuse_counts, rounds, *, baseline="reference", draws=2000, seed=20261029):
    """Inputs must be saved-evidence audited by the caller; recheck coverage here."""
    if baseline not in methods or len(methods) < 2:
        raise ValueError("a declared baseline and another method are required")
    if type(draws) is not int or draws < 100 or type(seed) is not int or seed < 0:
        raise ValueError("at least 100 bootstrap replicates and a nonnegative integer seed required")
    corpus = FrozenCorpus()
    for workload in workloads:
        corpus.require_development(workload)
    expected = interleaved_order(workloads, methods, reuse_counts, rounds, 0)
    keys = {(c["workload_id"], c["method"], c["reuse"], c["round"]) for c in expected}
    known = {w["id"]: w for w in workloads}
    table = {}
    for record in records:
        key = (record["workload_id"], record["method"], record["reuse"], record["round"])
        if key not in keys or key in table:
            raise ValueError("unexpected or duplicate comparison case")
        if record["workload"] != known[record["workload_id"]]:
            raise ValueError("comparison workload identity mismatch")
        if record["status"] != "accepted" or record["correct"] is not True:
            raise ValueError("failed or incorrect case cannot be aggregated")
        table[key] = case_cost(record)
    if set(table) != keys:
        raise ValueError("incomplete comparison grid; no successful-only aggregate")
    by_workload, by_family, aggregate = [], [], []
    for method in methods:
        if method == baseline:
            continue
        for reuse in reuse_counts:
            rows = []
            for workload in workloads:
                paired = []
                for index in range(rounds):
                    candidate = table[workload["id"], method, reuse, index]
                    control = table[workload["id"], baseline, reuse, index]
                    paired.append({"round": index, "method": candidate, "baseline": control,
                                   "log_speedup": math.log(control["mean_costs"]["total_ms"]) - math.log(candidate["mean_costs"]["total_ms"]),
                                   "log_steady_speedup": math.log(control["mean_costs"]["steady_ms_per_call"]) - math.log(candidate["mean_costs"]["steady_ms_per_call"])})
                score = fmean(p["log_speedup"] for p in paired)
                row = {"workload_id": workload["id"], "family": workload["family"], "group": workload["group"],
                       "shape": workload["shape"], "method": method, "reuse": reuse, **effect(score),
                       "regression": score < 0, "steady_geometric_mean_speedup": math.exp(fmean(p["log_steady_speedup"] for p in paired)),
                       "paired_rounds": paired,
                       "mean_method_costs": {k: fmean(p["method"]["mean_costs"][k] for p in paired) for k in paired[0]["method"]["mean_costs"]},
                       "mean_baseline_costs": {k: fmean(p["baseline"]["mean_costs"][k] for p in paired) for k in paired[0]["baseline"]["mean_costs"]},
                       "warning_free_paired_rounds": sum(not p["method"]["original_flagged"] and not p["baseline"]["original_flagged"] for p in paired),
                       "unresolved_paired_rounds": sum(p["method"]["unresolved_drift"] or p["baseline"]["unresolved_drift"] for p in paired)}
                rows.append(row)
            families = []
            for family in sorted({r["family"] for r in rows}):
                members = [r for r in rows if r["family"] == family]
                families.append({"family": family, "method": method, "reuse": reuse,
                                 "workloads": len(members), "groups": len({r["group"] for r in members}),
                                 "regressions": sum(r["regression"] for r in members),
                                 **effect(fmean(r["mean_log_speedup"] for r in members))})
            aggregate.append({"method": method, "reuse": reuse, "families": [f["family"] for f in families],
                              "workloads": len(rows), "regressions": sum(r["regression"] for r in rows),
                              **effect(fmean(f["mean_log_speedup"] for f in families)),
                              "uncertainty": clustered_interval(rows, draws, seed)})
            by_workload.extend(rows)
            by_family.extend(families)
    return {"schema_version": 1, "analysis_protocol": PROTOCOL, "baseline": baseline,
            "scope": "exploratory development frozen-selector comparison", "heldout_access": False,
            "superiority_claim_authorized": False, "bootstrap_replicates": draws, "bootstrap_seed": seed,
            "rounds": rounds, "fewer_than_five_rounds": rounds < 5,
            "weighting": "equal workloads within family, then equal represented families; reuse counts separate",
            "repeat_rule": "arithmetic mean of all attempts in a case, including flagged attempts",
            "coverage": {"cases": len(table), "attempts": sum(c["attempts"] for c in table.values()),
                         "original_flagged_cases": sum(c["original_flagged"] for c in table.values()),
                         "unresolved_cases": sum(c["unresolved_drift"] for c in table.values())},
            "per_workload": by_workload, "per_family": by_family, "aggregate": aggregate}
