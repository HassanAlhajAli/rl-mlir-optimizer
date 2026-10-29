"""Known-answer statistics and failure controls, using synthetic timings only."""
import copy
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.analysis import case_cost, clustered_interval, summarize
from rlmlir.corpus import DEFAULT_CORPUS, load


def workload(family="matmul", index=0):
    return [w for w in load(DEFAULT_CORPUS, "train") if w["family"] == family][index]


def record(program, method, cost, *, reuse=100, round=0):
    return {"workload_id": program["id"], "workload": program, "method": method, "reuse": reuse,
            "round": round, "status": "accepted", "correct": True, "selection_ms": 0., "unresolved_drift": False,
            "attempts": [{"attempt": 0, "status": "accepted", "flagged": False,
                          "total_with_selection_ms": cost, "steady_batch_ms": 120., "steady_count": 10,
                          "deployment": {"correct": True, "cache_hit": False, "count": reuse,
                                         "compile_ms": cost / 2, "installation_ms": 0.,
                                         "execution_ms": cost / 2, "total_ms": cost}}]}


def report(records, programs, *, rounds=1, reuse=(100,), **kwargs):
    return summarize(records, programs, ["reference", "heuristic"], list(reuse), rounds, **kwargs)


def test_equal_families_do_not_let_shape_count_hide_regression():
    programs = [workload(index=i) for i in range(3)] + [workload("matvec")]
    records = []
    for program in programs:
        records += [record(program, "reference", 10.), record(program, "heuristic", 5. if program["family"] == "matmul" else 20.)]
    summary = report(records, programs)
    overall = summary["aggregate"][0]
    assert overall["geometric_mean_speedup"] == pytest.approx(1.)
    assert overall["regressions"] == 1 and overall["workloads"] == 4
    assert summary["per_family"][0]["geometric_mean_speedup"] == pytest.approx(2.)
    assert summary["per_family"][1]["latency_reduction_percent"] == pytest.approx(-100.)
    assert not summary["superiority_claim_authorized"] and not summary["heldout_access"]


def test_paired_round_log_ratios_not_ratio_of_means():
    program = workload()
    records = [record(program, "reference", 4., round=i) for i in range(2)]
    records += [record(program, "heuristic", value, round=i) for i, value in enumerate([1., 16.])]
    summary = report(records, [program], rounds=2)
    row = summary["per_workload"][0]
    assert row["geometric_mean_speedup"] == pytest.approx(1.)
    assert row["mean_method_costs"]["total_ms"] == 8.5
    assert [math.exp(p["log_speedup"]) for p in row["paired_rounds"]] == pytest.approx([4., .25])


def test_drift_repeat_keeps_slow_original_and_slow_repeat():
    program = workload()
    candidate = record(program, "heuristic", 2.)
    candidate["attempts"][0]["flagged"] = True
    repeated = record(program, "heuristic", 8.)["attempts"][0]
    repeated.update(attempt=1, flagged=True)
    candidate["attempts"].append(repeated)
    candidate["unresolved_drift"] = True
    summary = report([record(program, "reference", 10.), candidate], [program])
    assert summary["aggregate"][0]["geometric_mean_speedup"] == pytest.approx(2.)
    assert summary["coverage"] == {"cases": 2, "attempts": 3, "original_flagged_cases": 1, "unresolved_cases": 1}
    row = summary["per_workload"][0]
    assert row["warning_free_paired_rounds"] == 0 and row["unresolved_paired_rounds"] == 1
    assert [c["total_ms"] for c in row["paired_rounds"][0]["method"]["attempt_costs"]] == [2., 8.]


def test_reuse_levels_stay_separate_and_explicit_baseline_can_reverse():
    program = workload()
    records = []
    for reuse, cost in [(1, 20.), (10000, 5.)]:
        records += [record(program, "reference", 10., reuse=reuse), record(program, "heuristic", cost, reuse=reuse)]
    summary = report(records, [program], reuse=(1, 10000))
    assert [r["geometric_mean_speedup"] for r in summary["aggregate"]] == pytest.approx([.5, 2.])
    reversed_summary = report(records, [program], reuse=(1, 10000), baseline="heuristic")
    assert [r["geometric_mean_speedup"] for r in reversed_summary["aggregate"]] == pytest.approx([2., .5])


def test_cluster_bootstrap_keeps_related_shapes_together_and_is_reproducible():
    rows = [{"family": "matmul", "group": "a", "mean_log_speedup": math.log(.5)},
            {"family": "matmul", "group": "b", "mean_log_speedup": math.log(2.)}]
    answer = clustered_interval(rows, 2000, 20261029)
    assert answer == clustered_interval(rows * 10, 2000, 20261029)
    assert answer["speedup_interval"] == pytest.approx([.5, 2.])
    assert answer["latency_reduction_interval_percent"] == pytest.approx([-100., 50.])
    assert answer["small_cluster_warning"]


def test_many_rounds_of_one_program_do_not_create_generalization_interval():
    program = workload()
    records = [record(program, method, cost, round=i) for i in range(30)
               for method, cost in [("reference", 10.), ("heuristic", 5.)]]
    summary = report(records, [program], rounds=30)
    uncertainty = summary["aggregate"][0]["uncertainty"]
    assert uncertainty["speedup_interval"] is None and "fewer than two" in uncertainty["reason"]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "failed", "wrong_identity", "bad_cost", "warm", "wrong_count", "nan"])
def test_no_incomplete_successful_only_or_corrupt_aggregate(fault):
    program = workload()
    records = [record(program, "reference", 10.), record(program, "heuristic", 5.)]
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        records.append(copy.deepcopy(records[-1]))
    elif fault == "failed":
        records[-1]["status"] = "failed"
    elif fault == "wrong_identity":
        records[-1]["workload"] = workload(index=1)
    elif fault == "bad_cost":
        records[-1]["attempts"][0]["total_with_selection_ms"] += 1.
    elif fault == "warm":
        records[-1]["attempts"][0]["deployment"]["cache_hit"] = True
    elif fault == "wrong_count":
        records[-1]["attempts"][0]["deployment"]["count"] = 99
    else:
        records[-1]["selection_ms"] = float("nan")
    with pytest.raises((ValueError, RuntimeError)):
        report(records, [program])


def test_selection_cost_is_charged_and_zero_total_rejected():
    program = workload()
    candidate = record(program, "heuristic", 5.)
    candidate["selection_ms"] = 5.
    candidate["attempts"][0]["total_with_selection_ms"] = 10.
    assert case_cost(candidate)["mean_costs"]["total_ms"] == 10.
    with pytest.raises(ValueError):
        case_cost(record(program, "heuristic", 0.))


def test_actual_groups_are_used_and_unknown_baseline_is_rejected():
    programs = [workload(index=i) for i in range(3)]
    records = [record(p, m, cost) for p in programs for m, cost in [("reference", 10.), ("heuristic", 5.)]]
    summary = report(records, programs)
    assert summary["aggregate"][0]["uncertainty"]["groups_per_family"] == {"matmul": 3}
    assert summary["aggregate"][0]["uncertainty"]["speedup_interval"] == pytest.approx([2., 2.])
    with pytest.raises(ValueError, match="baseline"):
        report(records, programs, baseline="absent")
