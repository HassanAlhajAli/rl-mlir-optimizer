"""Shared units and cost accounting for selection and adaptation."""
import math


def deployment_ms(compile_ms, execution_ms, reuse, selection_ms=0., installation_ms=0.):
    values = (compile_ms, execution_ms, selection_ms, installation_ms)
    if any(not math.isfinite(v) or v < 0 for v in values) or type(reuse) is not int or reuse < 1:
        raise ValueError("costs must be finite/nonnegative and reuse a positive integer")
    return compile_ms + selection_ms + installation_ms + reuse * execution_ms


def reward(reference_ms, candidate_ms):
    if not (math.isfinite(reference_ms) and math.isfinite(candidate_ms) and reference_ms > 0 and candidate_ms > 0):
        raise ValueError("reward requires positive finite costs")
    return math.log(reference_ms / candidate_ms)


def adapted_deployment_ms(tuning_ms, fresh_deployment_ms):
    if any(not math.isfinite(v) or v < 0 for v in (tuning_ms, fresh_deployment_ms)):
        raise ValueError("adaptation costs must be finite/nonnegative")
    return tuning_ms + fresh_deployment_ms
