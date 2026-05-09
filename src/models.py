"""Model registry for Metagross.

All models implement a single forward signature so the agent, the sync
trainer, and the async A3C trainer can use them interchangeably:

    forward(states: Tensor[B, state_dim]) -> (
        logits: Tensor[B, n_actions],
        values: Tensor[B],
    )

To add a new architecture: subclass nn.Module with that contract, then
add an entry to _REGISTRY at the bottom of this file. Both trainers
expose --arch <name> and resolve via make_model(name, ...).
"""

from __future__ import annotations
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int, hidden_layers: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class ActorCriticStateAction(nn.Module):
    """Actor scores [state ⊕ one-hot(action)] -> scalar logit per slot.
    Critic maps state -> scalar value.

    State-dict compatible with the prior agent2.ActorCritic and
    train_a3c.GlobalAC at hidden_layers=2.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128, hidden_layers: int = 2):
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.actor = _mlp(state_dim + n_actions, hidden, 1, hidden_layers)
        self.critic = _mlp(state_dim, hidden, 1, hidden_layers)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = states.shape[0]
        A = self.n_actions
        s_exp = states.unsqueeze(1).expand(B, A, self.state_dim)
        eye = torch.eye(A, device=states.device, dtype=states.dtype).unsqueeze(0).expand(B, A, A)
        sa = torch.cat([s_exp, eye], dim=-1).reshape(B * A, self.state_dim + A)
        logits = self.actor(sa).reshape(B, A)
        values = self.critic(states).squeeze(-1)
        return logits, values


class ActorCriticDirect(nn.Module):
    """Standard form: actor maps state -> [n_actions] logits directly.

    Faster than ActorCriticStateAction (one forward instead of B*A), but
    state_dicts are not compatible — the actor's output dim differs.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128, hidden_layers: int = 2):
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.actor = _mlp(state_dim, hidden, n_actions, hidden_layers)
        self.critic = _mlp(state_dim, hidden, 1, hidden_layers)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(states), self.critic(states).squeeze(-1)


_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "mlp_2h":     lambda **kw: ActorCriticStateAction(hidden_layers=2, **kw),
    "mlp_1h":     lambda **kw: ActorCriticStateAction(hidden_layers=1, **kw),
    "mlp_direct": lambda **kw: ActorCriticDirect(hidden_layers=2, **kw),
}


def make_model(name: str, state_dim: int, n_actions: int, hidden: int = 128) -> nn.Module:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model arch: {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](state_dim=state_dim, n_actions=n_actions, hidden=hidden)


def list_models() -> list[str]:
    return sorted(_REGISTRY)
