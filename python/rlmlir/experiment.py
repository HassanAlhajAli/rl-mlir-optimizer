"""Common provenance and startup boundary for measured experiments."""
import hashlib
import os
import platform
import subprocess
import time
from pathlib import Path

from .kernels import matmul


def provenance(root: Path):
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=root,text=True).strip():
        raise ValueError("commit tracked changes before measurement")
    executable = root/"build/rlmlir-worker"
    return {"revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
            "binary_hash":hashlib.sha256(executable.read_bytes()).hexdigest(),
            "corpus_sha256":(root/"configs/corpus.sha256").read_text().strip(),
            "python":platform.python_version(),"os":platform.platform(),
            "guest_affinity":sorted(os.sched_getaffinity(0)),
            "compiler":"LLVM/MLIR 18.1.3","target":"x86-64-v3",
            "measurement_protocol":"paired-training-v2"}


def initialize(worker):
    start = time.perf_counter()
    warm = worker.request("compile",source=matmul(1,1,1))
    worker.request("release",handle=warm["handle"])
    worker.request("clear_cache")
    return {"compiler_initialization_ms":(time.perf_counter()-start)*1000,
            "worker":worker.request("info")}
