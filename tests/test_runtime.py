"""Eviction must not release active code; malformed inputs must be rejected."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from rlmlir.kernels import matmul, matmul_case
from rlmlir.worker import Worker


def rejects(worker, command, **fields):
    try:
        worker.request(command, **fields)
    except ValueError:
        return
    raise AssertionError(f"invalid {command} request accepted")


def main(executable):
    with Worker(executable) as worker:
        source = matmul(1, 1, 1)
        first = worker.request("compile", source=source)
        second = worker.request("compile", source=source)
        assert not first["cache_hit"] and second["cache_hit"]
        assert second["lowering_ms"] == second["codegen_ms"] == 0
        worker.request("release", handle=second["handle"])
        for k in range(2, 42):
            built = worker.request("compile", source=matmul(1, 1, k))
            worker.request("release", handle=built["handle"])
        assert worker.request("info")["cache_entries"] == 32
        assert worker.request("invoke", handle=first["handle"], buffers=[[2], [3], [99]])["output"] == [6]
        for alignment in ("start", "end"):
            guarded = worker.request("check_memory", handle=first["handle"], buffers=[[2],[3],[99]], count=3, alignment=alignment)
            assert guarded["memory_check_passed"] and guarded["output"] == [6]
            assert "batch_ms" not in guarded and "installation_ms" not in guarded
        rejects(worker, "check_memory", handle=first["handle"], buffers=[[2],[3],[0]], count=1001, alignment="end")
        rejects(worker, "check_memory", handle=first["handle"], buffers=[[2],[3],[0]], count=1, alignment="unknown")
        assert worker.request("clear_cache")["remaining_entries"] == 1
        rejects(worker, "invoke", handle=first["handle"], buffers=[[2, 4], [3], [0]])
        rejects(worker, "invoke", handle=first["handle"], buffers=[[2], [3], [0]], count=-1)
        rejects(worker, "invoke", handle=first["handle"], buffers=[[2], [3], [0]], count=1.5)
        rejects(worker, "compile", source=source, actions="VECTORIZE")
        rejects(worker, "compile", source=source.replace("arith.mulf %x, %y", "arith.mulf %x, %y fastmath<fast>"))
        worker.request("release", handle=first["handle"])
        assert worker.request("clear_cache")["remaining_entries"] == 0
        deployed = worker.request("deploy", source=source, actions=[], buffers=[[2], [3], [99]], count=5)
        assert deployed["output"] == [6] and not deployed["cache_hit"]
        assert abs(deployed["total_ms"] - sum(deployed[k] for k in ["compile_ms", "installation_ms", "execution_ms"])) < 1e-6
        # Capacity pressure is explicit, rather than evicting a live function pointer.
        handles = [worker.request("compile", source=matmul(1, 1, k))["handle"] for k in range(1, 33)]
        rejects(worker, "compile", source=matmul(1, 1, 40))
        worker.request("release", handle=handles.pop())
        extra = worker.request("compile", source=matmul(1, 1, 40))
        worker.request("release", handle=extra["handle"])
        for handle in handles:
            worker.request("release", handle=handle)
    print("cache hits, eviction, pinned lifetimes, capacity and invalid-input checks passed")


if __name__ == "__main__":
    main(sys.argv[1])
