"""Transformation legality, rollback and actual generated-code checks."""
import itertools
import math
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from rlmlir.kernels import matmul, matmul_case, matvec, matvec_case
from rlmlir.worker import Worker


def check(worker, source, buffers, expected, actions):
    try:
        built = worker.request("compile", source=source, actions=actions)
    except (ValueError, RuntimeError) as error:
        raise AssertionError(f"schedule failed: {actions}") from error
    result = worker.request("invoke", handle=built["handle"], buffers=buffers, count=3)
    assert "llvm.fma." not in built["llvm_ir"] and "llvm.fmuladd." not in built["llvm_ir"]
    assert len(result["output"]) == len(expected)
    assert all(math.isfinite(a) and abs(a-b) <= 1e-10 + 1e-9*abs(b)
               for a,b in zip(result["output"], expected)), actions
    worker.request("release", handle=built["handle"])
    return built


def main(executable):
    with Worker(executable) as worker:
        schedules = [[], [{"kind": "INTERCHANGE", "a": 0, "b": 1}],
                     [{"kind": "TILE", "axis": 0, "size": 4}],
                     [{"kind": "TILE", "axis": 2, "size": 4}],
                     [{"kind": "TILE", "axis": 0, "size": 8}, {"kind": "TILE", "axis": 0, "size": 4}],
                     [{"kind": "VECTORIZE"}],
                     [{"kind": "TILE", "axis": 0, "size": 4}, {"kind": "TILE", "axis": 1, "size": 4}, {"kind": "VECTORIZE"}],
                     [{"kind": "INTERCHANGE", "a": 0, "b": 1}, {"kind": "TILE", "axis": 0, "size": 4}, {"kind": "VECTORIZE"}]]
        for dims in [(1, 1, 1), (3, 5, 7), (8, 8, 8), (17, 13, 9)]:
            source = matmul(*dims)
            buffers, expected = matmul_case(*dims)
            for actions in schedules:
                built = check(worker, source, buffers, expected, actions)
                if any(a["kind"] == "VECTORIZE" for a in actions):
                    assert "vector." in built["transformed_ir"]
                elif any(a["kind"] == "TILE" for a in actions):
                    assert "scf.for" in built["transformed_ir"]
        for dims in [(1, 1), (5, 7), (32, 17)]:
            source = matvec(*dims)
            buffers, expected = matvec_case(*dims)
            for actions in [[], [{"kind": "VECTORIZE"}], [{"kind": "TILE", "axis": 0, "size": 4}, {"kind": "VECTORIZE"}]]:
                check(worker, source, buffers, expected, actions)
        source = matmul(4, 4, 4)
        before = worker.request("analyze", source=source)
        illegal = [[{"kind": "INTERCHANGE", "a": 0, "b": 2}],
                   [{"kind": "TILE", "axis": 9, "size": 4}],
                   [{"kind": "TILE", "axis": 0, "size": 3}],
                   [{"kind": "TILE", "axis": 0, "size": 4}, {"kind": "TILE", "axis": 0, "size": 8}],
                   [{"kind": "STOP"}, {"kind": "VECTORIZE"}]]
        for actions in illegal:
            try:
                worker.request("analyze", source=source, actions=actions)
            except ValueError:
                pass
            else:
                raise AssertionError(f"illegal schedule accepted: {actions}")
        assert worker.request("analyze", source=source) == before
        # Exercise action masks after actual sequential transformations, not a mirrored model.
        rng = random.Random(90210)
        source = matmul(17, 13, 9)
        buffers, expected = matmul_case(17, 13, 9)
        for _ in range(40):
            actions = []
            while True:
                state = worker.request("analyze", source=source, actions=actions)
                if state["terminal"]:
                    break
                action = rng.choice(state["legal_actions"])
                actions.append(action)
                if action["kind"] in ("STOP", "VECTORIZE"):
                    break
            check(worker, source, buffers, expected, actions)
    print("81 scheduled kernels matched references; masks, strict math and rollback checks passed")


if __name__ == "__main__":
    main(sys.argv[1])
