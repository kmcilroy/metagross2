from __future__ import annotations
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical
from .config import SOFTMAX_TEMPERATURE, ENTROPY_BETA


from poke_env.player import Player
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from poke_env.environment import AbstractBattle  # type: ignore



from .encoder import encode_battle, ENCODER_DIM
from .utils.poke_helpers import (
    enumerate_legal_indices,
    order_from_index,
    encode_action_index,
    MAX_ACTIONS,
)

class ActionPolicy(nn.Module):
    """Scores (state, action_onehot) -> scalar logit."""
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + MAX_ACTIONS, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, sa_batch: torch.Tensor) -> torch.Tensor:
        return self.net(sa_batch).squeeze(-1)  # [B]



class LearningPlayer(Player):
    def __init__(self, epsilon: float, lr: float, hidden: int, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

        # Delay building the model until we see the first state vector (to get state_dim right)
        self.model = None              # type: ignore
        self.optimizer = None          # type: ignore
        self._hidden = int(hidden)
        self._lr = float(lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Trajectory buffers
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._legal_sets: List[List[int]] = []

        # Track HP and fainted deltas for reward shaping (filled by driver)
        self.prev_you_hp = 1.0
        self.prev_opp_hp = 1.0
        self.prev_you_fainted = 0
        self.prev_opp_fainted = 0


    def choose_move(self, battle: "AbstractBattle"):
        # 1) Legal action indices
        legal = enumerate_legal_indices(battle)

        # 2) Encode state
        s_np = encode_battle(battle)
        self._states.append(s_np)
        self._legal_sets.append(list(legal))

        # Lazily build the model once we know state_dim
        if self.model is None:
            state_dim = int(len(s_np))
            self.model = ActionPolicy(state_dim=state_dim, hidden=self._hidden).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self._lr)

        # 3) ε-mixed masked softmax sampling
        if np.random.rand() < self.epsilon:
            a_idx = int(np.random.choice(legal))
        else:
            with torch.no_grad():
                sa_rows = []
                for j in legal:
                    a_one = encode_action_index(j)
                    sa_rows.append(np.concatenate([s_np, a_one], axis=0))
                sa = torch.tensor(np.stack(sa_rows, axis=0), dtype=torch.float32, device=self.device)  # [L, D]
                logits = self.model(sa)  # [L]
                probs = torch.softmax(logits / max(1e-6, SOFTMAX_TEMPERATURE), dim=0)
                dist = Categorical(probs=probs)
                choice = int(dist.sample().item())
                a_idx = legal[choice]

        self._actions.append(a_idx)
        return order_from_index(self, battle, a_idx)

    def learn_step(self, reward: float):
        self._rewards.append(float(reward))

    def optimize_after_battle(self, gamma: float = 1.0):
        # Nothing to do?
        if not self._states or not self._actions or not self._legal_sets:
            self._states.clear(); self._actions.clear(); self._rewards.clear(); self._legal_sets.clear()
            return

        # 1) Discounted returns
        R = 0.0
        returns = []
        for r in reversed(self._rewards):
            R = r + gamma * R
            returns.append(R)
        returns = list(reversed(returns))

        # 2) Align lengths to number of decisions
        T = min(len(self._states), len(self._actions), len(self._legal_sets))
        if len(returns) < T:
            returns = [0.0] * (T - len(returns)) + returns
        elif len(returns) > T:
            returns = returns[-T:]

        if T == 0:
            self._states.clear(); self._actions.clear(); self._rewards.clear(); self._legal_sets.clear()
            return

        # Per-episode baseline to reduce variance
        baseline = float(np.mean(returns)) if returns else 0.0
        advantages = [r - baseline for r in returns]

        # 3) Policy gradient with masked softmax + entropy bonus
        losses = []
        entropies = []
        for t in range(T):
            s_np = self._states[t]
            a_idx = self._actions[t]
            legal = self._legal_sets[t]

            sa_rows = []
            for j in legal:
                a_one = encode_action_index(j)
                sa_rows.append(np.concatenate([s_np, a_one], axis=0))
            sa = torch.tensor(np.stack(sa_rows, axis=0), dtype=torch.float32, device=self.device)  # [L, D]

            if self.model is None:
                state_dim = int(len(s_np))
                self.model = ActionPolicy(state_dim=state_dim, hidden=self._hidden).to(self.device)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self._lr)

            logits = self.model(sa)  # [L]
            log_probs = logits - torch.logsumexp(logits, dim=0)
            probs = torch.softmax(logits, dim=0).clamp_min(1e-8)

            chosen_pos = legal.index(a_idx) if a_idx in legal else 0
            logp = log_probs[chosen_pos]

            # entropy of the masked policy at this step
            entropy = -(probs * log_probs).sum()
            entropies.append(entropy)

            adv = torch.tensor(advantages[t], dtype=torch.float32, device=self.device)
            losses.append(-logp * adv)

        pg_loss = torch.stack(losses).mean()
        ent_loss = -ENTROPY_BETA * torch.stack(entropies).mean() if ENTROPY_BETA > 0 else torch.tensor(0.0, device=self.device)
        loss = pg_loss + ent_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Save episode return for logging and clear buffers
        self.last_episode_return = float(sum(self._rewards))  # type: ignore[attr-defined]
        self._states.clear(); self._actions.clear(); self._rewards.clear(); self._legal_sets.clear()
