"""Correctness-only coverage for every family; no held-out performance is read."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from rlmlir.kernels import GENERATORS, REFERENCES
from rlmlir.measurement import validate
from rlmlir.worker import Worker


def main(executable):
    rng = random.Random(63)
    cases = {"row_reduce": [(1,1),(5,7),(0,7),(3,0)], "elementwise": [(1,1),(5,7),(0,7)],
             "stencil": [(1,1),(5,7),(0,7)], "conv1d": [(1,1),(9,5),(0,3)]}
    total = 0
    with Worker(executable) as worker:
        for family, shapes in cases.items():
            for dims in shapes:
                source = GENERATORS[family](*dims)
                for _ in range(6):
                    actions = []
                    while True:
                        state = worker.request("analyze", source=source, actions=actions)
                        if state["terminal"]:
                            break
                        action = rng.choice(state["legal_actions"])
                        actions.append(action)
                        if action["kind"] in ("STOP", "VECTORIZE"):
                            break
                    if family in ("stencil", "conv1d"):
                        assert not any(a["kind"] == "VECTORIZE" for a in actions)
                    built = worker.request("compile", source=source, actions=actions)
                    for seed in (7,17):
                        buffers, expected = REFERENCES[family](*dims, seed=seed)
                        actual = worker.request("invoke", handle=built["handle"], buffers=buffers, count=3)
                        validate(actual["output"], expected)
                        for alignment in ("start", "end"):
                            checked = worker.request("check_memory", handle=built["handle"], buffers=buffers,
                                                     count=3, alignment=alignment)
                            assert checked["memory_check_passed"]
                            validate(checked["output"], expected)
                    worker.request("release", handle=built["handle"])
                    total += 1
        # An overflowing affine access may verify as MLIR but is outside our safe subset.
        unsafe = GENERATORS["conv1d"](9,5).replace('(i+k)', '(i+k+100)')
        try:
            worker.request("analyze", source=unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-bounds indexing accepted")
    print(f"{total} additional-family schedules matched independent references; indexing rejection passed")


if __name__ == "__main__":
    main(sys.argv[1])
