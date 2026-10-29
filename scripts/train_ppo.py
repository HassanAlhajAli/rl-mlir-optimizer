"""Bounded, checkpointed measured PPO training; held-out evaluation is unavailable."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.corpus import load
from rlmlir.environment import CompilerEnv
from rlmlir.evaluator import BudgetReached, Evaluator
from rlmlir.worker import Worker
from rlmlir.experiment import initialize, provenance, measurement_lease


class Checkpoints(BaseCallback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
    def _on_step(self):
        return True
    def _on_rollout_end(self):
        self.model.save(self.directory / f"behavior-{self.num_timesteps}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=["reuse","runtime"], default="reuse")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new", type=int, default=4096)
    parser.add_argument("--max-requests", type=int, default=8192)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--timesteps", type=int, default=49152)
    parser.add_argument("--resume",type=Path,help="previous run; continue into a new output without resetting total limits")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists; explicit checkpoint continuation is required")
    if min(args.max_new,args.max_requests,args.hours,args.timesteps) <= 0:
        raise SystemExit("budgets must be positive")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise SystemExit("commit tracked changes before training")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    executable = ROOT / "build/rlmlir-worker"
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    binary_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    meta = {"schema_version":1,"status":"running","revision":revision,"binary_hash":binary_hash,
            "objective":args.objective,"seed":args.seed,"max_new":args.max_new,"max_requests":args.max_requests,
            "wall_limit_hours":args.hours,"requested_timesteps":args.timesteps,
            "corpus_sha256":(ROOT/"configs/corpus.sha256").read_text().strip(),
            "role":"development run; not selected on validation and not a final result"}
    meta.update(provenance(ROOT))
    previous = json.loads((args.resume/"run.json").read_text()) if args.resume else None
    prior = previous.get("cumulative_counters",previous["counters"]) if previous else {}
    if previous:
        for field in ("binary_hash","corpus_sha256","objective","seed","observation_version"):
            if previous.get(field) != meta.get(field):
                raise SystemExit(f"cannot resume with a changed {field}")
        meta["resumed_from"] = str(args.resume)
        meta["resume_boundary"] = "weights/optimizer and measurement cache preserved; partial episode/rollout discarded"
    remaining_new = args.max_new-prior.get("new_measurements",0)
    remaining_requests = args.max_requests-prior.get("candidate_requests",0)
    remaining_seconds = args.hours*3600-prior.get("elapsed_seconds",0)
    remaining_steps = args.timesteps-(previous["timesteps"] if previous else 0)
    if min(remaining_new,remaining_requests,remaining_seconds,remaining_steps) <= 0:
        raise SystemExit("cumulative budget already exhausted; do not silently enlarge it")
    (args.output/"run.json").write_text(json.dumps(meta,indent=2)+"\n")
    with Worker(executable) as worker:
        meta.update(initialize(worker))
        evaluator = Evaluator(worker,args.output/"measurements",binary_hash,revision,
                              max_new=remaining_new,max_requests=remaining_requests,max_seconds=remaining_seconds,
                              seed_measurements=args.resume/"measurements/measurements.jsonl" if args.resume else None)
        env = CompilerEnv(worker,evaluator,load(ROOT/"configs/corpus.json","train"),objective=args.objective,
                          trajectory_path=args.output/"trajectories.jsonl")
        model = MaskablePPO("MlpPolicy",env,seed=args.seed,device="cpu",learning_rate=3e-4,
                            n_steps=256,batch_size=64,n_epochs=10,gamma=1.,gae_lambda=.95,clip_range=.2,
                            ent_coef=.01,max_grad_norm=.5,verbose=1,
                            policy_kwargs={"net_arch":{"pi":[128,128],"vf":[128,128]}})
        if args.resume:
            model = MaskablePPO.load(args.resume/"latest-policy.zip",env=env,device="cpu",force_reset=True)
        selection_started = [0.]
        def before(module, inputs):
            selection_started[0] = time.perf_counter()
        def after(module, inputs, outputs):
            env.charge_policy_time((time.perf_counter()-selection_started[0])*1000)
            env.last_log_probability = float(outputs[2].detach().cpu().item())
            env.policy_metadata = {"seed":args.seed,"objective":args.objective,"timesteps":model.num_timesteps}
        pre_hook = model.policy.register_forward_pre_hook(before)
        post_hook = model.policy.register_forward_hook(after)
        model.save(args.output/"initial-policy")
        try:
            model.learn(total_timesteps=remaining_steps,callback=Checkpoints(args.output),reset_num_timesteps=not bool(previous))
            meta["status"] = "timesteps_complete"
        except BudgetReached as error:
            meta.update(status="budget_reached",reason=str(error))
        except BaseException as error:
            meta.update(status="stopped",reason=repr(error))
            raise
        finally:
            pre_hook.remove()
            post_hook.remove()
            model.save(args.output/"latest-policy")
            meta.update(timesteps=model.num_timesteps,episodes=env.episodes,counters=evaluator.counters())
            current = meta["counters"]
            meta["cumulative_counters"] = {key:prior.get(key,0)+current[key] for key in
                ("candidate_requests","reference_requests","new_measurements","cache_hits","elapsed_seconds")}
            meta["cumulative_counters"]["worker_counts"] = {
                key:prior.get("worker_counts",{}).get(key,0)+current["worker_counts"].get(key,0)
                for key in set(prior.get("worker_counts",{}))|set(current["worker_counts"])}
            (args.output/"run.json").write_text(json.dumps(meta,indent=2)+"\n")
            print(json.dumps(meta,indent=2),flush=True)


if __name__ == "__main__":
    with measurement_lease(ROOT):
        main()
