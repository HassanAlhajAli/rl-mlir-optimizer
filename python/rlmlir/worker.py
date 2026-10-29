"""Synchronous local worker; stdout is reserved for JSON responses."""
import json
import subprocess
import selectors
import resource
from collections import Counter
from pathlib import Path


class Worker:
    def __init__(self, executable: str | Path, timeout: float = 120, memory_limit_bytes: int = 4 * 1024**3):
        if timeout <= 0 or memory_limit_bytes <= 0:
            raise ValueError("worker time and memory limits must be positive")
        self.process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        self.memory_limit_bytes = memory_limit_bytes
        # Limit the child before sending any compiler request. prlimit avoids
        # Python preexec hooks after a potentially multithreaded trainer forks.
        try:
            resource.prlimit(self.process.pid, resource.RLIMIT_AS,
                             (memory_limit_bytes, memory_limit_bytes))
        except BaseException:
            self.process.kill()
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            raise
        self.timeout = timeout
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.counters = Counter()

    def request(self, command: str, **fields):
        self.counters["attempted_requests"] += 1
        self.counters["attempted_" + command] += 1
        self.process.stdin.write(json.dumps({"command": command, **fields}, allow_nan=False) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(timeout=self.timeout):
            self.process.kill()
            self.process.wait()
            raise TimeoutError(f"worker exceeded {self.timeout} seconds for {command}")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"worker stopped unexpectedly ({self.process.wait(timeout=5)})")
        response = json.loads(line)
        if not response.get("ok"):
            raise ValueError(response.get("error", "unspecified worker error"))
        self.counters["worker_requests"] += 1
        self.counters[command + "_requests"] += 1
        if command in ("compile", "deploy") and not response.get("cache_hit", False):
            self.counters["actual_compilations"] += 1
        if command in ("invoke", "benchmark", "deploy"):
            self.counters["actual_invocations"] += response["count"]
        return response

    def close(self):
        self.selector.close()
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
