"""Shared masked actor architecture and explicit portable policy checkpoints."""
import hashlib
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from .encoding import ACTIONS, OBSERVATION_SIZE, OBSERVATION_VERSION


def create_policy(seed=7):
    torch.manual_seed(seed)
    policy = MaskableActorCriticPolicy(
        gym.spaces.Box(0., 1., (OBSERVATION_SIZE,), dtype=np.float32),
        gym.spaces.Discrete(len(ACTIONS)), lambda _: 3e-4,
        net_arch={"pi": [128, 128], "vf": [128, 128]},
    )
    policy.set_training_mode(False)
    return policy


def actor_parameters(policy):
    # Features are a parameter-free flatten operation. Keep the critic out of
    # imitation and GRPO updates even though the checkpoint stores both heads.
    return list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())


def policy_digest(policy):
    digest = hashlib.sha256()
    for name,value in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_policy(policy, path: Path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "observation_version":metadata.get("observation_version",OBSERVATION_VERSION),
                "state_dict": policy.state_dict(), "metadata": metadata}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy(path: Path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["schema_version"] != 1:
        raise ValueError("unsupported policy checkpoint schema")
    if saved.get("observation_version",1) != OBSERVATION_VERSION:
        raise ValueError("policy uses an incompatible observation version; use its recorded source revision")
    policy = create_policy()
    policy.load_state_dict(saved["state_dict"])
    return policy, saved["metadata"]


def tensors(steps):
    observations = torch.as_tensor(np.asarray([s["observation"] for s in steps]), dtype=torch.float32)
    masks = torch.as_tensor(np.asarray([s["mask"] for s in steps]), dtype=torch.bool)
    actions = torch.as_tensor([s["action"] for s in steps], dtype=torch.long)
    return observations, masks, actions
