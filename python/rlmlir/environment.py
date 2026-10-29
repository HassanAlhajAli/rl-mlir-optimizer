"""Sequential legal compiler decisions with terminal measured reward."""
import json
import time
import gymnasium as gym
import numpy as np

from .encoding import ACTIONS, OBSERVATION_SIZE, action_mask, encode
from .kernels import GENERATORS
from .objective import deployment_ms, reward


class CompilerEnv(gym.Env):
    def __init__(self, worker, evaluator, workloads, *, objective="reuse", reuse_counts=(1,100,10000), trajectory_path=None):
        super().__init__()
        if objective not in ("reuse", "runtime"):
            raise ValueError("unknown objective")
        self.worker, self.evaluator, self.workloads = worker, evaluator, workloads
        self.objective, self.reuse_counts, self.trajectory_path = objective, reuse_counts, trajectory_path
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(0.,1.,(OBSERVATION_SIZE,),dtype=np.float32)
        self.finished = True
        self.episodes = 0
        self.policy_ms = 0.
        self.last_log_probability = None
        self.policy_metadata = {}

    def charge_policy_time(self, milliseconds):
        self.policy_ms += milliseconds

    def _analyze(self):
        start = time.perf_counter()
        self.state = self.worker.request("analyze", source=self.source, actions=self.actions)
        self.analysis_ms += (time.perf_counter() - start) * 1000
        self.observation = encode(self.state, self.actions, self.reuse)
        return self.observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.workload = self.workloads[int(self.np_random.integers(len(self.workloads)))]
        self.reuse = int(self.np_random.choice(self.reuse_counts))
        self.source = GENERATORS[self.workload["family"]](*self.workload["shape"])
        self.actions, self.transitions = [], []
        self.analysis_ms = self.policy_ms = 0.
        self.finished = False
        return self._analyze(), {}

    def action_masks(self):
        return action_mask(self.state)

    def step(self, index):
        index = int(index)
        if self.finished or not 0 <= index < len(ACTIONS) or not self.action_masks()[index]:
            raise ValueError("step after termination or illegal compiler action")
        transition = {"observation": self.observation.tolist(), "mask": self.action_masks().tolist(), "action": index,
                      "behavior_log_probability": self.last_log_probability}
        self.last_log_probability = None
        self.actions.append(ACTIONS[index])
        observation = self._analyze()
        self.finished = self.state["terminal"]
        transition.update(next_observation=observation.tolist(), terminal=self.finished)
        self.transitions.append(transition)
        if not self.finished:
            return observation, 0., False, False, {}
        # STOP does not change code, so remove it from the compile identity.
        schedule = [action for action in self.actions if action["kind"] != "STOP"]
        baseline = self.evaluator.measure(self.workload, [], reference=True)
        measured = self.evaluator.measure(self.workload, schedule)
        if self.objective == "runtime":
            reference_cost, cost = baseline["execution_ms"], measured["execution_ms"]
        else:
            reference_cost = deployment_ms(baseline["compile_ms"], baseline["execution_ms"], self.reuse)
            cost = deployment_ms(measured["compile_ms"], measured["execution_ms"], self.reuse,
                                 selection_ms=self.analysis_ms + self.policy_ms)
        score = reward(reference_cost, cost)
        self.episodes += 1
        info = {"reward": score, "cost_ms": cost, "reference_cost_ms": reference_cost,
                "measurement_key": measured["key"], "reference_key": baseline["key"],
                "analysis_ms": self.analysis_ms, "policy_ms": self.policy_ms,
                "unresolved_drift": measured["unresolved_drift"], "episodes": self.episodes}
        if self.trajectory_path:
            with self.trajectory_path.open("a") as stream:
                stream.write(json.dumps({"schema_version":1, "workload":self.workload, "reuse":self.reuse,
                                         "objective":self.objective, "transitions":self.transitions, "actions":self.actions,
                                         "outcome":info, "policy":self.policy_metadata}) + "\n")
        return observation, score, True, False, info
