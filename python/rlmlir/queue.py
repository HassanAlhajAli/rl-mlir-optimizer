"""Serial resumable stage queue with explicit revision and elapsed-time limits."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def write_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state,indent=2)+"\n")
    temporary.replace(path)


def run_jobs(jobs, directory: Path, *, max_seconds, guard=lambda: None):
    """Completed stages survive restart. Failed/partial stages need inspection."""
    import fcntl
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/"queue.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_file = directory/"queue.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {
            "schema_version":1,"status":"ready","elapsed_seconds":0.,"jobs":[]}
        if state.get("definition") not in (None,jobs):
            raise ValueError("queue definition changed; use a new campaign directory")
        state["definition"] = jobs
        state["controller_pid"] = os.getpid()
        for job in jobs:
            existing = next((j for j in state["jobs"] if j["name"] == job["name"]),None)
            if existing:
                if existing["status"] == "complete":
                    continue
                raise RuntimeError("partial/failed stage requires explicit review and continuation: "+job["name"])
            remaining = max_seconds-state["elapsed_seconds"]
            if remaining <= 0:
                state["status"] = "budget_reached"
                write_state(state_file,state)
                return state
            guard()
            record = {"name":job["name"],"status":"running","argv":job["argv"]}
            state["jobs"].append(record)
            state["status"] = "running"
            started = time.perf_counter()
            with (directory/(job["name"]+".log")).open("a") as log:
                child = subprocess.Popen(job["argv"],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                record["pid"] = child.pid
                write_state(state_file,state)
                try:
                    code = child.wait(timeout=remaining)
                    record.update(status="complete" if code == 0 else "failed",exit_code=code)
                except (subprocess.TimeoutExpired,KeyboardInterrupt):
                    # Let training save its latest checkpoint before terminating
                    # the whole process group, including a stuck compiler child.
                    os.kill(child.pid,signal.SIGINT)
                    try:
                        child.wait(timeout=150)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGTERM)
                        child.wait(timeout=10)
                    record.update(status="interrupted",reason="controller deadline or interruption")
                finally:
                    record["elapsed_seconds"] = time.perf_counter()-started
                    state["elapsed_seconds"] += record["elapsed_seconds"]
                    state["status"] = "ready" if record["status"] == "complete" else "stopped"
                    write_state(state_file,state)
            if record["status"] != "complete":
                return state
        state["status"] = "complete"
        write_state(state_file,state)
        return state
