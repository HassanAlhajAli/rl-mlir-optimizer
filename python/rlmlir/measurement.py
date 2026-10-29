"""Timing and validation primitives; callers retain every block, including failures."""
import math


def validate(output, expected):
    if len(output) != len(expected):
        raise RuntimeError("output length mismatch")
    maximum = 0.0
    for actual, reference in zip(output, expected):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise RuntimeError("nonfinite or nonnumeric output")
        error = abs(actual - reference)
        maximum = max(maximum, error)
        if error > 1e-10 + 1e-9 * abs(reference):
            raise RuntimeError(f"incorrect output: {actual} != {reference}")
    return maximum


def calibrate(worker, handle, buffers, target_ms):
    count = 1
    samples = []
    for _ in range(12):
        result = worker.request("benchmark", handle=handle, buffers=buffers, count=count)
        samples.append({"count": count, "batch_ms": result["batch_ms"]})
        if result["batch_ms"] >= target_ms:
            return count, samples
        if count == 10000000:
            raise RuntimeError("calibration reached the invocation limit")
        count = min(10000000, max(count + 1, math.ceil(count * target_ms * 1.2 / max(result["batch_ms"], .001))))
    raise RuntimeError("calibration did not converge within 12 batches")


def schedule(name, family):
    if name == "baseline":
        return []
    if name in ("row_tile4", "row_tile16"):
        return [{"kind": "TILE", "axis": 0, "size": int(name.removeprefix("row_tile"))}]
    if name in ("parallel_tile8_vector", "parallel_tile16_vector"):
        size = int(name.removeprefix("parallel_tile").removesuffix("_vector"))
        parallel_axes = range(2) if family == "matmul" else range(1)
        return [{"kind": "TILE", "axis": axis, "size": size} for axis in parallel_axes] + [{"kind": "VECTORIZE"}]
    raise ValueError(f"unknown pilot schedule {name}")
