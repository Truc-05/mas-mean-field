from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mfg_llm.control.mf_rl.env import MeanFieldDebateEnv
from mfg_llm.metrics.stats import summarize


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu_head = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.zeros(2) - 0.5)
        self.register_buffer("action_scale", torch.tensor([2.0, 2.0]))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        mean = torch.sigmoid(self.mu_head(h)) * self.action_scale
        std = torch.exp(self.log_std)
        return mean, std


def train_policy_gradient(env: MeanFieldDebateEnv, n_episodes: int = 200, lr: float = 1e-3,
                           gamma: float = 0.99, seed: int = 0) -> tuple[PolicyNet, list[float]]:
    torch.manual_seed(seed)
    policy = PolicyNet(obs_dim=env.obs_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    episode_returns = []

    for ep in range(n_episodes):
        _, obs = env.reset()
        log_probs, rewards = [], []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            mean, std = policy(obs_t)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            u = action.clamp(min=0.01).detach().numpy()
            _, obs, reward, done, info = env.step(u)
            log_probs.append(log_prob)
            rewards.append(reward)

        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        loss = -torch.stack([lp * g for lp, g in zip(log_probs, returns)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_returns.append(float(sum(rewards)))

    return policy, episode_returns


def rollout_policy(env: MeanFieldDebateEnv, policy: PolicyNet) -> dict:
    _, obs = env.reset()
    accs, traces = [], []
    done = False
    with torch.no_grad():
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            mean, _ = policy(obs_t)
            u = mean.clamp(min=0.01).numpy()
            _, obs, reward, done, info = env.step(u)
            accs.append(info["accuracy"])
            traces.append(info["trace_cov"])
    return {"accuracy_traj": np.array(accs), "trace_traj": np.array(traces)}


def train_replicates(env_factory: Callable[[int], MeanFieldDebateEnv], n_seeds: int = 5,
                      n_episodes: int = 200, lr: float = 1e-3, gamma: float = 0.99,
                      base_seed: int = 0) -> dict:
    """Train over multiple seeds; report final accuracy and return as mean +/- std."""
    final_accs, final_returns, policies = [], [], []
    for i in range(n_seeds):
        seed = base_seed + i
        env = env_factory(seed)
        policy, ep_returns = train_policy_gradient(env, n_episodes=n_episodes, lr=lr,
                                                    gamma=gamma, seed=seed)
        roll = rollout_policy(env_factory(seed + 10_000), policy)
        final_accs.append(float(roll["accuracy_traj"][-1]))
        final_returns.append(ep_returns[-1])
        policies.append(policy)
    return {
        "final_accuracy": summarize(final_accs),
        "final_return": summarize(final_returns),
        "policies": policies,
    }
