"""Synchronous local worker; stdout is reserved for JSON responses."""
import json
import subprocess
from pathlib import Path


class Worker:
    def __init__(self, executable: str | Path):
        self.process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)

    def request(self, command: str, **fields):
        self.process.stdin.write(json.dumps({"command": command, **fields}, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"worker stopped unexpectedly ({self.process.poll()})")
        response = json.loads(line)
        if not response.get("ok"):
            raise ValueError(response.get("error", "unspecified worker error"))
        return response

    def close(self):
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
