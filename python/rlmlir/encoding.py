"""Fixed numerical observation/action contract. No workload identifiers or labels."""
import json
import math
import numpy as np

ACTIONS = [{"kind":"STOP"}, {"kind":"VECTORIZE"}]
ACTIONS += [{"kind":"TILE", "axis":axis, "size":size} for axis in range(4) for size in (4,8,16,32,64)]
ACTIONS += [{"kind":"INTERCHANGE", "a":a, "b":b} for a in range(4) for b in range(a+1,4)]
ACTION_INDEX = {json.dumps(a, sort_keys=True): i for i,a in enumerate(ACTIONS)}
OBSERVATION_SIZE = 235


def action_mask(state):
    mask = np.zeros(len(ACTIONS), dtype=bool)
    for action in state["legal_actions"]:
        mask[ACTION_INDEX[json.dumps(action, sort_keys=True)]] = True
    if not mask[0]:
        raise ValueError("compiler state lacks STOP")
    return mask


def log_feature(value):
    return min(1., math.log2(1 + value) / 24.)


def encode(state, actions, reuse):
    result = []
    dims = state["dimensions"]
    for i in range(4):
        if i < len(dims):
            d = dims[i]
            result.extend([log_feature(d["bound"]), log_feature(d["tile_bound"]), float(d["parallel"]), len(state["tiles"][i])/2.])
        else:
            result.extend([0.] * 4)
    for position in range(4):
        result.extend(float(position < len(state["order"]) and state["order"][position] == axis) for axis in range(4))
    for strides in state["access_strides"]:
        result.extend(log_feature(strides[i]) if i < len(strides) else 0. for i in range(4))
    for shape in state["buffer_shapes"]:
        result.append(len(shape)/4.)
        result.extend(log_feature(shape[i]) if i < len(shape) else 0. for i in range(4))
    result.extend(log_feature(c) for c in state["arithmetic_counts"])
    # Nominal structured-op memory accesses: two inputs and one output.
    result.extend([2./4., 1./4., float(state["vectorized"]), state["steps"]/6., min(1., math.log10(1+reuse)/6.)])
    for step in range(6):
        selected = ACTION_INDEX[json.dumps(actions[step], sort_keys=True)] if step < len(actions) else -1
        result.extend(float(index == selected) for index in range(len(ACTIONS)))
    array = np.asarray(result, dtype=np.float32)
    if array.shape != (OBSERVATION_SIZE,) or not np.isfinite(array).all():
        raise ValueError(f"observation contract violated: {array.shape}")
    return array
