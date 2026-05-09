"""Shared A2C loss + advantage utilities used by both trainers.

The loss function expects any model that implements the Metagross
forward contract:

    forward(states: Tensor[B, S]) -> (logits: Tensor[B, A], values: Tensor[B])

Sync trainer (train_mvp0) calls a2c_loss with `advantages=None` and lets
it compute (returns - V).detach() inside, matching the prior
optimize_after_battle behavior. Async trainer (train_a3c) computes GAE
externally, normalizes, and passes them in.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def a2c_loss(
    model: nn.Module,
    states: torch.Tensor,        # [T, S]
    actions: torch.Tensor,       # [T] long
    masks: torch.Tensor,         # [T, A] bool, True = legal
    returns: torch.Tensor,       # [T]
    advantages: Optional[torch.Tensor] = None,  # [T]; computed from V if None
    *,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """One training pass of A2C losses against `model`.

    Returns a dict with detached scalars under the standard keys
    (`policy_loss`, `value_loss`, `entropy`) plus a `total` tensor
    that has gradients attached for backprop.
    """
    logits, values = model(states)
    if advantages is None:
        advantages = (returns - values).detach()

    masked = logits.masked_fill(~masks, float("-inf"))
    log_probs = F.log_softmax(masked, dim=-1)
    log_probs_taken = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    policy_loss = -(log_probs_taken * advantages).mean()

    value_loss = F.mse_loss(values, returns)

    probs = torch.softmax(masked, dim=-1).clamp_min(1e-8)
    entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()

    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return {
        "total": total,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        # Detached extras for per-step CSV logging:
        "advantages": advantages.detach(),
        "values": values.detach(),
        "probs": probs.detach(),
    }


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation. Returns (advantages, returns)."""
    T = len(rewards)
    adv = [0.0] * T
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        done = dones[t]
        delta = rewards[t] + (0.0 if done else gamma * next_value) - values[t]
        gae = delta + (0.0 if done else gamma * lam) * gae
        adv[t] = gae
        next_value = values[t]
    returns = [a + v for a, v in zip(adv, values)]
    return (
        torch.tensor(adv, dtype=torch.float32),
        torch.tensor(returns, dtype=torch.float32),
    )


def discounted_returns(rewards: List[float], gamma: float = 1.0) -> List[float]:
    R = 0.0
    out: List[float] = []
    for r in reversed(rewards):
        R = r + gamma * R
        out.append(R)
    out.reverse()
    return out


def legal_mask_from_lists(legal_sets: List[List[int]], n_actions: int) -> torch.Tensor:
    """Build a [T, A] bool tensor (True = legal) from per-step legal-index lists."""
    T = len(legal_sets)
    mask = torch.zeros(T, n_actions, dtype=torch.bool)
    for t, legal in enumerate(legal_sets):
        for j in legal or ():
            jj = int(j)
            if 0 <= jj < n_actions:
                mask[t, jj] = True
    return mask
