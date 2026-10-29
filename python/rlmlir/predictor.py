"""Supervised compilation/execution-cost baseline, independent of RL updates."""
import copy
import json
import math
import statistics
import numpy as np
import torch

from .encoding import OBSERVATION_SIZE
from .selection import heuristic, sample_schedule

REUSE_FEATURE_INDEX = 66


def cost_examples(pool):
    """Each measured timing votes once, even if many trajectories reuse it."""
    unique = {}
    for item in pool:
        key = (item["workload"]["id"], json.dumps(item["actions"], sort_keys=True))
        vector = features(item["transitions"][-1]["next_observation"])
        sample = unique.setdefault(key, {"workload_id": item["workload"]["id"],
                                         "split": item["workload"]["split"], "features": vector, "samples": {}})
        if sample["split"] != item["workload"]["split"] or not np.array_equal(sample["features"], vector):
            raise ValueError("inconsistent predictor features for one program/history")
        identity = tuple(item["measurement_identity"])
        costs = [item["compile_install_ms"], item["execution_ms"]]
        if identity in sample["samples"] and sample["samples"][identity] != costs:
            raise ValueError("one measured identity has conflicting costs")
        sample["samples"][identity] = costs
    for sample in unique.values():
        measurements = list(sample.pop("samples").values())
        sample["distinct_measurements"] = len(measurements)
        sample["log_costs"] = [math.log(statistics.median(v[i] for v in measurements)) for i in range(2)]
    return list(unique.values())


def features(observation):
    vector = np.asarray(observation,dtype=np.float32).copy()
    if vector.shape != (OBSERVATION_SIZE,):
        raise ValueError("invalid predictor observation")
    # Predict intrinsic C and T. Apply the reuse hint only when ranking C+N*T.
    vector[REUSE_FEATURE_INDEX] = math.log10(2)/6.
    return vector


class CostPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(OBSERVATION_SIZE,128),torch.nn.Tanh(),
                                            torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,2))
        self.register_buffer("mean",torch.zeros(2))
        self.register_buffer("scale",torch.ones(2))

    def forward(self,x):
        return self.network(x)*self.scale + self.mean

    def milliseconds(self,x):
        return self(x).clamp(-30,30).exp()


def fit_predictor(model, training, validation, *, seed=7, epochs=100):
    if not training or not validation:
        raise ValueError("training and validation examples required")
    if any(e["split"] != "train" for e in training) or any(e["split"] != "validation" for e in validation):
        raise ValueError("predictor data crossed split boundaries")
    rng = np.random.default_rng(seed)
    grouped = {}
    for item in training:
        grouped.setdefault(item["workload_id"],[]).append(item)
    contexts = list(grouped.values())
    y = torch.tensor([e["log_costs"] for e in training],dtype=torch.float32)
    model.mean.copy_(y.mean(0))
    model.scale.copy_(y.std(0,unbiased=False).clamp_min(.1))
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    best, score, stale, history = None,math.inf,0,[]
    for epoch in range(epochs):
        model.train()
        for _ in range(max(1,math.ceil(len(training)/128))):
            batch = []
            for _ in range(128):
                context = contexts[int(rng.integers(len(contexts)))]
                batch.append(context[int(rng.integers(len(context)))])
            x = torch.tensor(np.asarray([e["features"] for e in batch]),dtype=torch.float32)
            target = torch.tensor([e["log_costs"] for e in batch],dtype=torch.float32)
            loss = ((model(x)-target)/model.scale).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            x = torch.tensor(np.asarray([e["features"] for e in validation]),dtype=torch.float32)
            target = torch.tensor([e["log_costs"] for e in validation],dtype=torch.float32)
            errors = ((model(x)-target)/model.scale).square().mean(-1).tolist()
        per_program = {}
        for item,error in zip(validation,errors):
            per_program.setdefault(item["workload_id"],[]).append(error)
        current = float(np.mean([np.mean(v) for v in per_program.values()]))
        history.append({"epoch":epoch+1,"validation_standardized_log_mse":current})
        if current < score-1e-6:
            best,score,stale = copy.deepcopy(model.state_dict()),current,0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best)
    return history


def rank_schedules(worker,workload,reuse,model,*,seed=314159,size=256):
    """Deterministic candidate pool. No test-program performance query occurs."""
    import time
    began = time.perf_counter()
    rng = np.random.default_rng(seed)
    candidates,seen = [],set()
    forced = [[{"kind":"STOP"}],heuristic(worker,workload)]
    for attempt in range(max(2,size*4)):
        candidate = sample_schedule(worker,workload,reuse,rng=rng,
                                    forced=forced[attempt] if attempt < len(forced) else None)
        identity = tuple(tuple(sorted(a.items())) for a in candidate["schedule"])
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(candidate)
        if len(candidates) == size:
            break
    x = torch.tensor(np.asarray([features(c["transitions"][-1]["next_observation"]) for c in candidates]))
    model.eval()
    with torch.no_grad():
        predicted = model.milliseconds(x)
        totals = predicted[:,0] + reuse*predicted[:,1]
        winner = int(totals.argmin())
    return {"schedule":candidates[winner]["schedule"],"candidate_count":len(candidates),
            "predicted_cost_ms":float(totals[winner]),
            "selection_ms":(time.perf_counter()-began)*1000}
