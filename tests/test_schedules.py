"""Transformation legality, rollback and actual generated-code checks."""
import itertools
import math
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from rlmlir.kernels import matmul, matmul_case, matvec, matvec_case, GENERATORS, REFERENCES
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
        # The first measured PPO trial found stack growth in vector-to-SCF
        # temporaries inside nested reduction tiles. These sizes exceeded the
        # default stack before buffer-loop-hoisting was added to the backend.
        regression = [{"kind":"TILE","axis":1,"size":4},
                      {"kind":"TILE","axis":2,"size":16},
                      {"kind":"TILE","axis":2,"size":8}, {"kind":"VECTORIZE"}]
        for dims in [(96,78,98), (95,79,97), (64,65,67)]:
            buffers, expected = matmul_case(*dims, seed=17)
            check(worker, matmul(*dims), buffers, expected, regression)
        # Main PPO exploration exposed a wide-vector shuffle expansion inside
        # LLVM InstCombine. The former backend exceeded 3 GiB on this schedule.
        wide = [{"kind":"TILE","axis":0,"size":16},
                {"kind":"TILE","axis":0,"size":8},
                {"kind":"TILE","axis":2,"size":32},
                {"kind":"TILE","axis":2,"size":16}, {"kind":"VECTORIZE"}]
        for dims in [(32,128,8), (17,127,9), (16,64,7)]:
            buffers, expected = matmul_case(*dims, seed=17)
            check(worker, matmul(*dims), buffers, expected, wide)
        # Preserve the original later SIGSEGV input as well. It did not fail
        # in isolation, so this test is coverage rather than proof of its cause.
        row_actions = [{"kind":"TILE","axis":1,"size":8},
                       {"kind":"TILE","axis":0,"size":64},
                       {"kind":"TILE","axis":0,"size":16}]
        buffers, expected = REFERENCES["row_reduce"](336,205,seed=17)
        check(worker, GENERATORS["row_reduce"](336,205), buffers, expected, row_actions)
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
    print("88 scheduled kernels matched references; masks, strict math, stack/wide-vector regressions and rollback passed")


if __name__ == "__main__":
    main(sys.argv[1])
