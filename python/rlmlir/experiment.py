"""Common provenance and startup boundary for measured experiments."""
import hashlib
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from .kernels import matmul


def provenance(root: Path):
    from .encoding import OBSERVATION_VERSION
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=root,text=True).strip():
        raise ValueError("commit tracked changes before measurement")
    executable = root/"build/rlmlir-worker"
    return {"revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
            "binary_hash":hashlib.sha256(executable.read_bytes()).hexdigest(),
            "corpus_sha256":(root/"configs/corpus.sha256").read_text().strip(),
            "python":platform.python_version(),"os":platform.platform(),
            "guest_affinity":sorted(os.sched_getaffinity(0)),
            "compiler":"LLVM/MLIR 18.1.3","target":"x86-64-v3",
            "observation_version":OBSERVATION_VERSION,
            "measurement_protocol":"paired-training-v2"}


def initialize(worker):
    before = worker.counters.copy()
    start = time.perf_counter()
    warm = worker.request("compile",source=matmul(1,1,1))
    worker.request("release",handle=warm["handle"])
    worker.request("clear_cache")
    info = worker.request("info")
    return {"compiler_initialization_ms":(time.perf_counter()-start)*1000,
            "initialization_counts":dict(worker.counters-before),"worker":info}


@contextmanager
def measurement_lease(root: Path):
    """An OS-released lease prevents overlapping benchmark scripts after crashes."""
    import fcntl
    (root/"runs").mkdir(exist_ok=True)
    with (root/"runs/measurement.lock").open("a+") as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another project measurement process is running") from error
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid())+"\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)
