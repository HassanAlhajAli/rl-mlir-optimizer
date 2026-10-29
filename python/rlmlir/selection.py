"""One bounded transformation sequence, with no compiler performance queries."""
import time
import numpy as np
import torch

from .encoding import ACTIONS, action_mask, encode
from .kernels import GENERATORS


def sample_schedule(worker, workload, reuse, *, policy=None, rng=None,
                    deterministic=False, forced=None, prefix=None):
    """None policy means uniformly random legal actions, not random schedules."""
    rng = rng or np.random.default_rng(7)
    source = GENERATORS[workload["family"]](*workload["shape"])
    actions, steps = [], []
    start = time.perf_counter()
    state = worker.request("analyze", source=source, actions=[])
    while not state["terminal"]:
        obs, mask = encode(state, actions, reuse), action_mask(state)
        value = None
        if prefix is not None and len(actions) < len(prefix):
            index = ACTIONS.index(prefix[len(actions)])
            if not mask[index]:
                raise ValueError("mutation prefix is illegal in this compiler state")
            log_probability = None
        elif forced is not None:
            action = forced[len(actions)] if len(actions) < len(forced) else {"kind":"STOP"}
            index = ACTIONS.index(action)
            if not mask[index]:
                raise ValueError(f"forced schedule contains an illegal action: {action}")
            log_probability = None
        elif policy is None:
            legal = np.flatnonzero(mask)
            index = int(rng.choice(legal))
            log_probability = float(-np.log(len(legal)))
        else:
            with torch.no_grad():
                tensor = torch.as_tensor(obs[None])
                distribution = policy.get_distribution(tensor, action_masks=mask[None])
                selected = distribution.get_actions(deterministic=deterministic)
                index = int(selected.item())
                log_probability = float(distribution.log_prob(selected).item())
                value = float(policy.predict_values(tensor).item())
        actions.append(ACTIONS[index])
        following = worker.request("analyze", source=source, actions=actions)
        steps.append({"observation":obs.tolist(), "mask":mask.tolist(), "action":index,
                      "behavior_log_probability":log_probability, "behavior_value":value,
                      "next_observation":encode(following, actions, reuse).tolist(),
                      "terminal":following["terminal"]})
        state = following
    if forced is not None and len(forced) > len(actions):
        raise ValueError("forced actions continue after termination")
    return {"actions":actions, "transitions":steps,
            "schedule":[a for a in actions if a["kind"] != "STOP"],
            "selection_ms":(time.perf_counter()-start)*1000}


def heuristic(worker, workload):
    """Tile parallel axes to eight, then vectorize only when the compiler allows."""
    source = GENERATORS[workload["family"]](*workload["shape"])
    state = worker.request("analyze", source=source, actions=[])
    actions = []
    for axis, dimension in enumerate(state["dimensions"]):
        action = {"kind":"TILE", "axis":axis, "size":8}
        if dimension["parallel"] and action in state["legal_actions"]:
            actions.append(action)
            state = worker.request("analyze", source=source, actions=actions)
    vectorize = {"kind":"VECTORIZE"}
    actions.append(vectorize if vectorize in state["legal_actions"] else {"kind":"STOP"})
    return actions
