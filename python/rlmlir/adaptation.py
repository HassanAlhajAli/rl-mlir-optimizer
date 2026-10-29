"""Group-budgeted adaptation; no generic learn() call may add hidden trials."""
import copy
import json
import math
import time

import numpy as np
import torch

from .objective import deployment_ms
from .policies import actor_parameters, tensors, policy_digest
from .selection import heuristic, sample_schedule


def _advantages_and_returns(steps, score):
    values = [float(s["behavior_value"]) for s in steps] + [0.]
    advantages, advantage = [], 0.
    for i in range(len(steps)-1, -1, -1):
        terminal_reward = score if i == len(steps)-1 else 0.
        delta = terminal_reward + values[i+1] - values[i]
        advantage = delta + .95 * advantage
        advantages.append(advantage)
    advantages.reverse()
    return advantages, [a+v for a,v in zip(advantages,values)]


def update_group(policy, reference, trajectories, optimizer, method):
    """Update on exactly these complete trajectories, never collect more here."""
    if method not in ("grpo", "ppo") or len(trajectories) != 8:
        raise ValueError("updates require eight trajectories and grpo or ppo")
    scores = np.asarray([t["reward"] for t in trajectories], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite group reward")
    spread = float(scores.std())
    if method == "grpo" and spread < 1e-8:
        return {"skipped":True,"reason":"zero_reward_variance","reward_std":spread,"epochs":0}
    start = time.perf_counter()
    steps = [s for t in trajectories for s in t["transitions"]]
    observations, masks, actions = tensors(steps)
    if not masks[torch.arange(len(steps)),actions].all():
        raise ValueError("trajectory contains an illegal action")
    old_log = torch.tensor([s["behavior_log_probability"] for s in steps])
    lengths = [len(t["transitions"]) for t in trajectories]
    with torch.no_grad():
        reference_dist = reference.get_distribution(observations, action_masks=masks).distribution
        reference_log = reference_dist.logits.detach()
    if method == "grpo":
        normalized = (scores-scores.mean())/(spread+1e-8)
        advantages = torch.tensor(np.repeat(normalized, lengths), dtype=torch.float32)
    else:
        gae = [_advantages_and_returns(t["transitions"],t["reward"]) for t in trajectories]
        advantages = torch.tensor([a for group,_ in gae for a in group])
        returns = torch.tensor([r for _,group in gae for r in group])
        advantages = (advantages-advantages.mean()) / (advantages.std(unbiased=False)+1e-8)
    metrics = []
    policy.set_training_mode(True)
    for _ in range(4):
        distribution = policy.get_distribution(observations, action_masks=masks)
        log_probability = distribution.log_prob(actions)
        ratio = (log_probability-old_log).exp()
        surrogate = torch.minimum(ratio*advantages, ratio.clamp(.8,1.2)*advantages)
        categorical = distribution.distribution
        kl = (categorical.probs * (categorical.logits-reference_log)).sum(-1)
        if method == "grpo":
            losses = (-surrogate + .02*kl).split(lengths)
            loss = torch.stack([item.mean() for item in losses]).mean()
        else:
            predicted = policy.predict_values(observations).flatten()
            loss = -surrogate.mean() + .5*(predicted-returns).square().mean() - .01*distribution.entropy().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite adaptation loss")
        optimizer.zero_grad()
        loss.backward()
        parameters = actor_parameters(policy) if method == "grpo" else list(policy.parameters())
        norm = torch.nn.utils.clip_grad_norm_(parameters,.5)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite adaptation gradient")
        optimizer.step()
        metrics.append({"loss":float(loss.detach()),"kl":float(kl.mean().detach()),"gradient_norm":float(norm)})
    policy.set_training_mode(False)
    return {"skipped":False,"reward_std":spread,"epochs":4,
            "update_ms":(time.perf_counter()-start)*1000,"metrics":metrics}


def adapt(worker, evaluator, workload, reuse, initial_policy, *, method, budget, seed=7):
    if budget not in (8,16,32) or method not in ("frozen","grpo","ppo","random","mutation"):
        raise ValueError("unsupported method or candidate budget")
    if evaluator.requests or evaluator.records:
        raise ValueError("each adaptation context requires an independent empty measurement cache")
    start = time.perf_counter()
    initial_digest = policy_digest(initial_policy)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    policy, reference = copy.deepcopy(initial_policy), copy.deepcopy(initial_policy)
    policy.set_training_mode(False)
    reference.set_training_mode(False)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    parameters = actor_parameters(policy) if method == "grpo" else list(policy.parameters())
    optimizer = torch.optim.Adam(parameters,lr=1e-4)
    requests, groups, seen, best = [], [], set(), None
    for offset in range(0,budget,8):
        group = []
        for index in range(offset,offset+8):
            selection_start = time.perf_counter()
            kwargs = {"policy":None if method == "random" else policy,"rng":rng}
            if method == "mutation":
                if index == 0:
                    kwargs["forced"] = heuristic(worker,workload)
                elif index > 1:
                    prior = best["schedule"]
                    kwargs["prefix"] = prior[:int(rng.integers(len(prior)+1))]
            candidate = sample_schedule(worker,workload,reuse,**kwargs)
            # Includes heuristic construction and mutation overhead as well as sampling.
            candidate["selection_ms"] = (time.perf_counter()-selection_start)*1000
            identity = json.dumps(candidate["schedule"],sort_keys=True)
            candidate["duplicate"] = identity in seen
            seen.add(identity)
            measurement = evaluator.measure(workload,candidate["schedule"])
            cost = deployment_ms(measurement["compile_ms"],measurement["execution_ms"],reuse,
                                 selection_ms=candidate["selection_ms"],
                                 installation_ms=measurement.get("installation_ms",0.))
            candidate.update(request=index+1, cost_ms=cost, reward=-math.log(cost),
                             measurement_key=measurement["key"],
                             unresolved_drift=measurement["unresolved_drift"])
            group.append(candidate)
            requests.append(candidate)
            if best is None or cost < best["cost_ms"]:
                best = candidate
        update = None
        if method in ("grpo","ppo") and offset+8 < budget:
            update = update_group(policy,reference,group,optimizer,method)
        groups.append({"requests":offset+8,"unique_schedules":len(seen),"update":update,
                       "best_cost_ms":best["cost_ms"],"elapsed_ms":(time.perf_counter()-start)*1000})
    final_digest = policy_digest(policy)
    if policy_digest(initial_policy) != initial_digest:
        raise RuntimeError("adaptation mutated the shared initial policy")
    return {"schema_version":1,"method":method,"budget":budget,"seed":seed,
            "initial_policy_digest":initial_digest,"final_policy_digest":final_digest,
            "workload":workload,"reuse":reuse,"requests":requests,"groups":groups,
            "best":best,"tuning_ms":(time.perf_counter()-start)*1000,
            "counters":evaluator.counters()}
