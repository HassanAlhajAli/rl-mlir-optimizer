"""Balanced supervised imitation, followed by a separately fitted value head."""
import copy
import math
import numpy as np
import torch

from .policies import actor_parameters, tensors


def _loss(policy, demonstrations):
    # Equal context weight, then equal action weight within each demonstration.
    losses = []
    for item in demonstrations:
        obs, masks, actions = tensors(item["transitions"])
        distribution = policy.get_distribution(obs,action_masks=masks)
        losses.append(-distribution.log_prob(actions).mean())
    return torch.stack(losses).mean()


def fit_actor(policy, training, validation, *, seed=7, max_epochs=100, patience=10):
    if not training or not validation:
        raise ValueError("separate nonempty training and validation demonstrations are required")
    if any(t["workload"]["split"] != "train" for t in training):
        raise ValueError("actor training received a non-training workload")
    if any(t["workload"]["split"] != "validation" for t in validation):
        raise ValueError("actor selection received an invalid validation workload")
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(actor_parameters(policy),lr=3e-4)
    best, best_loss, stale, history = None, math.inf, 0, []
    for epoch in range(max_epochs):
        policy.set_training_mode(True)
        losses = []
        # One best demonstration per program/reuse context. Random ordering
        # visits every context once, avoiding frequency and length domination.
        indices = rng.permutation(len(training))
        for start in range(0,len(indices),128):
            batch = [training[i] for i in indices[start:start+128]]
            loss = _loss(policy,batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters(policy),.5)
            optimizer.step()
            losses.append(float(loss.detach()))
        policy.set_training_mode(False)
        with torch.no_grad():
            validation_loss = float(_loss(policy,validation))
        history.append({"epoch":epoch+1,"training_loss":float(np.mean(losses)),"validation_loss":validation_loss})
        if validation_loss < best_loss-1e-6:
            best, best_loss, stale = copy.deepcopy(policy.state_dict()), validation_loss, 0
        else:
            stale += 1
        if stale >= patience:
            break
    policy.load_state_dict(best)
    return history


def fit_critic(policy, training, *, seed=7, epochs=50):
    """Training-only terminal return targets; actor weights remain untouched."""
    if not training or any(t["workload"]["split"] != "train" for t in training):
        raise ValueError("critic requires training-only trajectories")
    observations, targets = [], []
    for item in training:
        target = -math.log(item["cost_ms"])
        # Contexts have equal total weight via a deterministic one-state draw
        # per context on each epoch, not by flattening long trajectories.
        observations.append([s["observation"] for s in item["transitions"]])
        targets.append(target)
    parameters = list(policy.mlp_extractor.value_net.parameters()) + list(policy.value_net.parameters())
    optimizer = torch.optim.Adam(parameters,lr=3e-4)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        indices = rng.permutation(len(training))
        losses = []
        for start in range(0,len(indices),128):
            selected = indices[start:start+128]
            obs = torch.tensor(np.asarray([observations[i][rng.integers(len(observations[i]))] for i in selected]),dtype=torch.float32)
            target = torch.tensor([targets[i] for i in selected],dtype=torch.float32)
            loss = (policy.predict_values(obs).flatten()-target).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,.5)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch":epoch+1,"mse":float(np.mean(losses))})
    policy.set_training_mode(False)
    return history
