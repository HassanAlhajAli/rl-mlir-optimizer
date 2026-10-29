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
from rlmlir.experiment import COMPATIBILITY_FIELDS, require_compatible_provenance
from rlmlir.imitation import fit_actor, fit_critic
from rlmlir.policies import create_policy, actor_parameters, load_policy, save_policy
from rlmlir.selection import sample_schedule
from rlmlir.worker import Worker
import rlmlir.adaptation as adaptation_module


@pytest.mark.parametrize("field", COMPATIBILITY_FIELDS)
def test_policy_rejects_incompatible_or_missing_provenance(tmp_path, field):
    expected = {"binary_hash":"binary", "corpus_sha256":"corpus", "observation_version":2,
                "measurement_protocol":"paired-training-v2", "compiler":"LLVM/MLIR 18.1.3",
                "target":"x86-64-v3"}
    path = tmp_path/"policy.pt"
    save_policy(create_policy(7),path,expected)
    changed = {**expected,field:"different"}
    with pytest.raises(ValueError,match=field):
        load_policy(path,expected_provenance=changed)
    missing = {k:v for k,v in expected.items() if k != field}
    with pytest.raises(ValueError,match=field):
        require_compatible_provenance(missing,expected,context="experience pool")


def test_policy_accepts_same_contract_across_documentation_revisions(tmp_path):
    expected = {"binary_hash":"binary", "corpus_sha256":"corpus", "observation_version":2,
                "measurement_protocol":"paired-training-v2", "compiler":"LLVM/MLIR 18.1.3",
                "target":"x86-64-v3"}
    policy = create_policy(7)
    path = tmp_path/"policy.pt"
    save_policy(policy,path,{**expected,"revision":"earlier-documentation"})
    restored,_ = load_policy(path,expected_provenance={**expected,"revision":"new-documentation"})
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in policy.state_dict().items())


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
        return {"compile_ms":10.,"installation_ms":0.,"execution_ms":1.,"key":str(actions),"unresolved_drift":False}
    def counters(self):
        return {"candidate_requests":self.requests}


@pytest.mark.parametrize("failure", [None, "request", "update"])
def test_events_preserve_partial_history_and_update_order(monkeypatch, failure):
    policy = create_policy(7)
    evaluator = DeterministicEvaluator()
    workload = {"id": "synthetic", "family": "matvec", "shape": [3, 5], "split": "train"}
    events = []
    monkeypatch.setattr(adaptation_module, "sample_schedule", lambda *_a, **_k: {"schedule": [], "transitions": []})
    measure = evaluator.measure
    def measured(*args):
        if failure == "request" and evaluator.requests == 2:
            evaluator.requests += 1
            raise RuntimeError("third request failed")
        return measure(*args)
    evaluator.measure = measured
    def updated(*_args):
        if failure == "update":
            raise RuntimeError("policy update failed")
        return {"skipped": False, "epochs": 4}
    monkeypatch.setattr(adaptation_module, "update_group", updated)
    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            adapt(None, evaluator, workload, 100, policy, method="grpo", budget=16, emit=events.append)
        assert events[-1]["event"] == "context_failed"
        assert not any(e["event"] == "context_completed" for e in events)
        completed = [e for e in events if e["event"] == "request_completed"]
        assert len(completed) == (2 if failure == "request" else 8)
    else:
        result = adapt(None, evaluator, workload, 100, policy, method="grpo", budget=16, emit=events.append)
        assert events[-1]["event"] == "context_completed"
        position = next(i for i, e in enumerate(events) if e["event"] == "update_started")
        assert events[position - 1]["request"] == 8
        assert events[position + 1]["event"] == "group_completed"
        assert events[position + 2]["request"] == 9
        assert len([e for e in events if e["event"] == "update_started"]) == 1
        assert result["counters"]["candidate_requests"] == 16
    # Completed candidates are mutated in place by adaptation, but start events are snapshots.
    assert all("cost_ms" not in e["candidate"] for e in events if e["event"] == "request_started")
    assert all(e["workload"] == workload and e["budget"] == 16 for e in events)


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
            assert result["initial_policy_digest"] == result["final_policy_digest"]
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
