"""Executed JIT correctness, repeated-output and structured-error checks."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from rlmlir.kernels import matmul, matmul_case
from rlmlir.worker import Worker


def main(executable):
    with Worker(executable) as worker:
        assert worker.request("info")["llvm_version"] == "18.1.3"
        for m, n, k in [(1, 1, 1), (3, 5, 7), (16, 16, 16), (0, 3, 2), (2, 3, 0)]:
            built = worker.request("compile", source=matmul(m, n, k))
            assert "llvm.func" in built["lowered_ir"]
            for seed in (7, 17):
                buffers, expected = matmul_case(m, n, k, seed)
                for count in (1, 3):
                    result = worker.request("invoke", handle=built["handle"], buffers=buffers, count=count)
                    assert len(result["output"]) == len(expected)
                    for actual, target in zip(result["output"], expected):
                        assert math.isfinite(actual) and abs(actual - target) <= 1e-10 + 1e-9 * abs(target), (m, n, k, actual, target)
            worker.request("release", handle=built["handle"])
            try:
                worker.request("invoke", handle=built["handle"], buffers=buffers)
            except ValueError as error:
                assert "unknown executable" in str(error)
            else:
                raise AssertionError("released handle remained executable")
        try:
            worker.request("compile", source="not MLIR")
        except ValueError:
            pass
        else:
            raise AssertionError("malformed MLIR accepted")
    print("JIT reference, repeat, empty-shape, release and malformed-input checks passed")


if __name__ == "__main__":
    main(sys.argv[1])
