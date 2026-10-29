"""Meaningful learning-contract tests, without hardware performance comparisons."""
import hashlib
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.corpus import load
from rlmlir.encoding import ACTIONS, OBSERVATION_SIZE, action_mask, encode
from rlmlir.environment import CompilerEnv
from rlmlir.objective import deployment_ms, reward, adapted_deployment_ms
from rlmlir.worker import Worker
from rlmlir.evaluator import Evaluator, BudgetReached


def test_costs_and_sealed_split():
    assert deployment_ms(10., 2., 4, selection_ms=1., installation_ms=3.) == 22.
    assert reward(8.,2.) == pytest.approx(math.log(4.))
    assert adapted_deployment_ms(100.,22.) == 122.
    with pytest.raises(ValueError):
        deployment_ms(1.,2.,0)
    with pytest.raises(ValueError, match="sealed"):
        load(ROOT / "configs/corpus.json", "heldout_shape")
    assert len(load(ROOT / "configs/corpus.json", "train")) == 128
    assert len(load(ROOT / "configs/corpus.json", "validation")) == 32


class FakeEvaluator:
    def measure(self, workload, actions, **kwargs):
        return {"compile_ms":10.,"execution_ms":2. if not actions else 1.,
                "key":str(actions),"unresolved_drift":False}


def test_real_compiler_environment():
    with Worker(ROOT / "build/rlmlir-worker") as worker:
        workloads = [{"family":"matmul", "shape":[3,5,7], "split":"train"}]
        env = CompilerEnv(worker, FakeEvaluator(), workloads, reuse_counts=(100,))
        obs, _ = env.reset(seed=7)
        assert obs.shape == (OBSERVATION_SIZE,) and env.observation_space.contains(obs)
        assert env.action_masks()[0]
        # Logical axis2 is a reduction, so the parallel interchange must be masked out.
        illegal = next(i for i,a in enumerate(ACTIONS) if a == {"kind":"INTERCHANGE","a":0,"b":2})
        with pytest.raises(ValueError):
            env.step(illegal)
        _, score, terminated, truncated, info = env.step(0)
        assert terminated and not truncated and math.isfinite(score)
        assert info["reference_cost_ms"] == 210.
        with pytest.raises(ValueError):
            env.step(0)


def test_budget_stops_before_compilation(tmp_path):
    class NoCompiler:
        def request(self, *_args, **_kwargs):
            raise AssertionError("budget exhaustion reached the compiler")
    evaluator = Evaluator(NoCompiler(),tmp_path,"test-binary","test-revision",max_new=0,max_requests=1)
    workload = {"family":"matmul","shape":[1,1,1],"split":"train"}
    with pytest.raises(BudgetReached,match="distinct"):
        evaluator.measure(workload,[])
    assert evaluator.new_measurements == 0 and evaluator.requests == 1
    with pytest.raises(BudgetReached,match="request"):
        evaluator.measure(workload,[])
    assert evaluator.requests == 1


def test_seeded_measurements_do_not_reset_request_limit(tmp_path):
    # A resumed cache may save hardware work, but a logical request still costs
    # a request slot. The measurement identity is obtained from a real zero-
    # hardware failure record's key through the same public evaluator boundary.
    from collections import Counter
    class RejectCompiler:
        counters = Counter()
        class Process:
            def poll(self): return 1
        process = Process()
        def request(self,*args,**kwargs):
            raise RuntimeError("expected diagnostic stop")
    workload = {"family":"matmul","shape":[1,1,1],"split":"train"}
    first = Evaluator(RejectCompiler(),tmp_path/"first","binary","revision",max_new=1)
    with pytest.raises(RuntimeError,match="diagnostic"):
        first.measure(workload,[])
    record = json.loads(first.path.read_text())
    record.update(status="accepted",compile_ms=1.,execution_ms=1.,correct=True)
    first.path.write_text(json.dumps(record)+"\n")
    resumed = Evaluator(RejectCompiler(),tmp_path/"resumed","binary","revision",max_new=0,
                        max_requests=1,seed_measurements=first.path)
    assert resumed.measure(workload,[])["key"] == record["key"]
    assert resumed.cache_hits == 1 and resumed.new_measurements == 0
    with pytest.raises(BudgetReached,match="request"):
        resumed.measure(workload,[])


def test_evaluator_counters_exclude_previous_worker_activity(tmp_path):
    with Worker(ROOT/"build/rlmlir-worker") as worker:
        worker.request("info")
        first = Evaluator(worker,tmp_path/"first","binary","revision")
        worker.request("info")
        assert first.counters()["worker_counts"]["info_requests"] == 1
        second = Evaluator(worker,tmp_path/"second","binary","revision")
        assert second.counters()["worker_counts"] == {}
        worker.request("info")
        assert second.counters()["worker_counts"]["info_requests"] == 1
        assert worker.counters["info_requests"] == 3


class KnownChoice(gym.Env):
    """Two contexts, different correct actions, and one always-illegal action."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(0.,1.,(2,),dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.choice = int(self.np_random.integers(2))
        return np.eye(2,dtype=np.float32)[self.choice], {}

    def action_masks(self):
        return np.array([True,True,False])

    def step(self, action):
        if action == 2:
            raise AssertionError("masked action reached environment")
        return np.eye(2,dtype=np.float32)[self.choice], (1. if action == self.choice else -1.), True, False, {}


def test_ppo_learns_and_restores(tmp_path):
    torch.set_num_threads(1)
    env = KnownChoice()
    model = MaskablePPO("MlpPolicy", env, seed=7, device="cpu", n_steps=256, batch_size=64,
                        gamma=1., learning_rate=3e-4, ent_coef=.01,
                        policy_kwargs={"net_arch":{"pi":[128,128],"vf":[128,128]}})
    model.learn(total_timesteps=2048)
    for choice in range(2):
        obs = np.eye(2,dtype=np.float32)[choice]
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        assert int(action) == choice
        distribution = model.policy.get_distribution(torch.tensor(obs[None]), action_masks=env.action_masks()[None])
        assert distribution.distribution.probs[0,2].item() == 0.
    model.save(tmp_path / "diagnostic")
    restored = MaskablePPO.load(tmp_path / "diagnostic", env=env, device="cpu")
    for name, weights in model.policy.state_dict().items():
        assert torch.equal(weights, restored.policy.state_dict()[name])
