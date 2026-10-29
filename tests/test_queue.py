import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from rlmlir.queue import run_jobs
from rlmlir.experiment import measurement_lease


def test_completed_stages_are_not_repeated_and_failure_stops(tmp_path):
    marker = tmp_path/"marker.txt"
    jobs = [{"name":"first","argv":[sys.executable,"-c",f"from pathlib import Path; p=Path({str(marker)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x')"]},
            {"name":"failure","argv":[sys.executable,"-c","raise SystemExit(3)"]},
            {"name":"never","argv":[sys.executable,"-c","raise AssertionError('must not run')"]}]
    state = run_jobs(jobs,tmp_path/"queue",max_seconds=5)
    assert state["status"] == "stopped" and len(state["jobs"]) == 2
    with pytest.raises(RuntimeError,match="explicit review"):
        run_jobs(jobs,tmp_path/"queue",max_seconds=5)
    assert marker.read_text() == "x"


def test_exclusive_measurement_lease(tmp_path):
    with measurement_lease(tmp_path):
        with pytest.raises(RuntimeError,match="another project"):
            with measurement_lease(tmp_path):
                pass
    with measurement_lease(tmp_path):
        pass


def test_no_job_starts_after_budget(tmp_path):
    result = run_jobs([{"name":"never","argv":["nonexistent-test-executable"]}],tmp_path,max_seconds=0)
    assert result["status"] == "budget_reached" and not result["jobs"]
