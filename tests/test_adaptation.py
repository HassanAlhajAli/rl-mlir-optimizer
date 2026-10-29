"""Exact budgets and real gradient mechanics, with deterministic synthetic costs."""
import copy
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from rlmlir.adaptation import adapt, update_group
from rlmlir.encoding import ACTIONS, OBSERVATION_SIZE
from rlmlir.imitation import fit_actor, fit_critic
from rlmlir.policies import create_policy, actor_parameters, load_policy, save_policy
from rlmlir.selection import sample_schedule
from rlmlir.worker import Worker


def group(policy, equal=False):
    obs = torch.zeros((1,OBSERVATION_SIZE))
    mask = np.zeros((1,len(ACTIONS)),dtype=bool)
    mask[0,:2] = True
    with torch.no_grad():
        dist = policy.get_distribution(obs,action_masks=mask)
        value = float(policy.predict_values(obs).item())
    items = []
    for i in range(8):
        action = i % 2
        step = {"observation":obs[0].tolist(),"mask":mask[0].tolist(),"action":action,
                "behavior_log_probability":float(dist.log_prob(torch.tensor([action])).item()),
                "behavior_value":value,"terminal":True}
        items.append({"transitions":[step],"reward":1. if equal or action == 1 else -1.})
    return items


@pytest.mark.parametrize("method",["grpo","ppo"])
def test_gradient_favors_better_action_and_leaves_reference(method):
    torch.set_num_threads(1)
    policy = create_policy(7)
    reference = copy.deepcopy(policy)
    original = copy.deepcopy(reference.state_dict())
    critic = copy.deepcopy(policy.value_net.state_dict())
    trajectories = group(policy)
    obs = torch.zeros((1,OBSERVATION_SIZE))
    mask = np.array([trajectories[0]["transitions"][0]["mask"]])
    before = float(policy.get_distribution(obs,action_masks=mask).distribution.probs[0,1].detach())
    parameters = actor_parameters(policy) if method == "grpo" else policy.parameters()
    optimizer = torch.optim.Adam(parameters,lr=1e-4)
    result = update_group(policy,reference,trajectories,optimizer,method)
    after = float(policy.get_distribution(obs,action_masks=mask).distribution.probs[0,1].detach())
    assert after > before and result["epochs"] == 4
    assert all(torch.equal(v,reference.state_dict()[k]) for k,v in original.items())
    if method == "grpo":
        assert all(torch.equal(v,policy.value_net.state_dict()[k]) for k,v in critic.items())


def test_zero_variance_skips_without_mutation():
    policy = create_policy(7)
    original = copy.deepcopy(policy.state_dict())
    result = update_group(policy,copy.deepcopy(policy),group(policy,equal=True),
                          torch.optim.Adam(actor_parameters(policy),lr=1e-4),"grpo")
    assert result["skipped"] and result["epochs"] == 0
    assert all(torch.equal(v,policy.state_dict()[k]) for k,v in original.items())


class DeterministicEvaluator:
    def __init__(self):
        self.requests, self.records = 0, {}
    def measure(self,workload,actions):
        self.requests += 1
        return {"compile_ms":10.,"execution_ms":1.,"key":str(actions),"unresolved_drift":False}
    def counters(self):
        return {"candidate_requests":self.requests}


def test_exact_budget_reset_and_first_group_control():
    torch.set_num_threads(1)
    policy = create_policy(7)
    original = copy.deepcopy(policy.state_dict())
    workload = {"id":"test","family":"matvec","shape":[3,5],"split":"train"}
    with Worker(ROOT/"build/rlmlir-worker") as worker:
        first = []
        for method in ("frozen","grpo","ppo"):
            result = adapt(worker,DeterministicEvaluator(),workload,100,policy,method=method,budget=8,seed=7)
            assert result["counters"]["candidate_requests"] == 8
            assert all(g["update"] is None for g in result["groups"])
            first.append([r["actions"] for r in result["requests"]])
        assert first[0] == first[1] == first[2]
        for budget in (16,32):
            result = adapt(worker,DeterministicEvaluator(),workload,100,policy,method="grpo",budget=budget,seed=7)
            assert result["counters"]["candidate_requests"] == budget
            assert sum(g["update"] is not None for g in result["groups"]) == budget//8-1
        for method in ("random","mutation"):
            result = adapt(worker,DeterministicEvaluator(),workload,100,policy,method=method,budget=8,seed=7)
            assert result["counters"]["candidate_requests"] == 8
    assert all(torch.equal(v,policy.state_dict()[k]) for k,v in original.items())


def test_imitation_critic_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    policy = create_policy(7)
    steps = group(policy)[1]["transitions"]
    training = [{"workload":{"split":"train"},"transitions":steps,"cost_ms":2.}]
    validation = [{"workload":{"split":"validation"},"transitions":steps,"cost_ms":2.}]
    history = fit_actor(policy,training,validation,max_epochs=4)
    assert history[-1]["validation_loss"] < history[0]["validation_loss"]
    before = [p.detach().clone() for p in actor_parameters(policy)]
    fit_critic(policy,training,epochs=4)
    assert all(torch.equal(a,b) for a,b in zip(before,actor_parameters(policy)))
    path = tmp_path/"policy.pt"
    save_policy(policy,path,{"critic_fitted":True})
    restored, meta = load_policy(path)
    assert meta["critic_fitted"]
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in policy.state_dict().items())
