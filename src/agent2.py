from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from poke_env.player import Player

if TYPE_CHECKING:
    from poke_env.environment import AbstractBattle  # type: ignore

from .encoder import encode_battle
from .utils.poke_helpers import (
    encode_action_index,
    enumerate_legal_indices,
    MAX_ACTIONS,
    order_from_index,
)
from .config import (
    ENTROPY_BETA,
    LOG_DIR,
    LOG_TURN_BY_TURN,
    MAX_GRAD_NORM,
    SOFTMAX_TEMPERATURE,
    VALUE_COEF,
)


# ------------------------------
# Model: Actor (state⊕action_onehot → logit) + Critic (state → value)
# ------------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim + MAX_ACTIONS, hidden),
            nn.ReLU(),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def actor_logits(self, sa_batch: torch.Tensor) -> torch.Tensor:
        # sa_batch: [B, state_dim + MAX_ACTIONS] -> [B]
        return self.actor(sa_batch).squeeze(-1)

    def value(self, s_batch: torch.Tensor) -> torch.Tensor:
        # s_batch: [T, state_dim] -> [T]
        return self.critic(s_batch).squeeze(-1)




def _compute_total_grad_norm(parameters, norm_type: float = 2.0) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total += float(param_norm ** norm_type)
    return float(total ** (1.0 / norm_type))

class LearningPlayerAC(Player):
    def __init__(self, epsilon: float, lr: float, hidden: int, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

        # Lazy model setup
        self.model = None  # type: ignore
        self.optimizer = None  # type: ignore
        self._hidden = int(hidden)
        self._lr = float(lr)

        # Require CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required but not available")
        torch.relu(torch.randn(1, device="cuda")).item()
        self.device = torch.device("cuda")

        # Buffers
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._legal_sets: List[List[int]] = []
        self._step_logs: List[Dict[str, Any]] = []

        # Reward helpers (used by your driver)
        self.prev_you_hp = 1.0
        self.prev_opp_hp = 1.0
        self.prev_you_fainted = 0
        self.prev_opp_fainted = 0

        self.episode_idx: int = 0

        self.entropy_beta = float(ENTROPY_BETA)
        self.value_coef = float(VALUE_COEF)
        self.max_grad_norm = float(MAX_GRAD_NORM)

        # last episode stats (for external logging if desired)
        self.last_episode_return: float = 0.0
        self.last_losses: Dict[str, float] = {}

    # ---------- Acting ----------
    def choose_move(self, battle: "AbstractBattle"):
        legal = enumerate_legal_indices(battle)

        # Encode state
        s_np = encode_battle(battle)
        self._states.append(s_np)
        self._legal_sets.append(list(legal))

        # Lazy model init
        if self.model is None:
            state_dim = int(len(s_np))
            self.model = ActorCritic(state_dim=state_dim, hidden=self._hidden).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self._lr)

        # Build SA for legal set
        sa_rows = [np.concatenate([s_np, encode_action_index(j)], axis=0) for j in legal]
        sa = torch.tensor(np.stack(sa_rows, axis=0), dtype=torch.float32, device=self.device)  # [L, D]

        # Critic V(s)
        with torch.no_grad():
            s_t = torch.tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            v_est = self.model.value(s_t).item()

        # Actor sample with ε‑mix
        probs_list = None
        entropy_val = None
        if np.random.rand() < self.epsilon:
            choice_pos = int(np.random.randint(len(legal)))
        else:
            with torch.no_grad():
                logits = self.model.actor_logits(sa)  # [L]
                probs = torch.softmax(logits / max(1e-6, float(SOFTMAX_TEMPERATURE)), dim=0)
                dist = Categorical(probs=probs)
                choice_pos = int(dist.sample().item())
                probs_list = probs.detach().cpu().tolist()
                log_probs = torch.log(probs.clamp_min(1e-8))
                entropy_val = float(-(probs * log_probs).sum().item())

        a_idx = legal[choice_pos]
        self._actions.append(a_idx)

        # Human label for the action
        action_label = self._describe_action(battle, a_idx)

        self._step_logs.append({
            "step": len(self._step_logs),
            "epsilon": float(self.epsilon),
            "value": float(v_est),
            "legal": list(legal),
            "action_idx": int(a_idx),
            "action_label": action_label,
            "probs": probs_list,    # None if ε-branch
            "entropy": entropy_val, # None if ε-branch
        })

        return order_from_index(self, battle, a_idx)

    @staticmethod
    def _describe_action(battle: "AbstractBattle", idx: int) -> str:
        try:
            if 0 <= idx <= 3:
                moves = battle.available_moves or []
                if idx < len(moves):
                    mv = moves[idx]
                    name = getattr(mv, "id", None) or getattr(mv, "move_id", None) or "move"
                    return f"move:{name}"
                return "move:NA"
            elif 4 <= idx <= 9:
                sw_idx = idx - 4
                sws = battle.available_switches or []
                if 0 <= sw_idx < len(sws):
                    species = getattr(sws[sw_idx], "species", "switch")
                    return f"switch:{species}"
                return "switch:NA"
        except Exception:
            pass
        return "unknown"

    # Driver calls this each env step with shaped reward
    def learn_step(self, reward: float):
        r = float(reward)
        self._rewards.append(r)
        if self._step_logs:
            self._step_logs[-1]["reward"] = r

    # ---------- Learning ----------
    def optimize_after_battle(self, gamma: float = 1.0) -> Dict[str, float]:
        """
        Returns a metrics dict for logging:
          { 'policy_loss', 'value_loss', 'entropy', 'grad_norm' }
        """
        if not self._states or not self._actions or not self._legal_sets:
            self._clear_buffers()
            return {}

        # 1) discounted returns
        R = 0.0
        returns = []
        for r in reversed(self._rewards):
            R = r + gamma * R
            returns.append(R)
        returns.reverse()

        # 2) align sequence length
        T = min(len(self._states), len(self._actions), len(self._legal_sets))
        if len(returns) < T:
            returns = [0.0] * (T - len(returns)) + returns
        elif len(returns) > T:
            returns = returns[-T:]
        if T == 0:
            self._clear_buffers()
            return {}

        # 3) tensors
        S = torch.tensor(np.stack(self._states[:T], axis=0), dtype=torch.float32, device=self.device)
        G = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # lazy ensure model/opt
        if self.model is None:
            state_dim = int(S.shape[1])
            self.model = ActorCritic(state_dim=state_dim, hidden=self._hidden).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self._lr)

        V = self.model.value(S)  # [T]
        advantages = (G - V.detach())

        policy_losses = []
        entropies = []
        for t in range(T):
            s_np = self._states[t]
            a_idx = self._actions[t]
            legal = self._legal_sets[t]

            sa_rows = [np.concatenate([s_np, encode_action_index(j)], axis=0) for j in legal]
            sa = torch.tensor(np.stack(sa_rows, axis=0), dtype=torch.float32, device=self.device)  # [L, D]

            logits = self.model.actor_logits(sa)  # [L]
            log_probs = logits - torch.logsumexp(logits, dim=0)
            probs = torch.softmax(logits, dim=0).clamp_min(1e-8)

            chosen_pos = legal.index(a_idx) if a_idx in legal else 0
            logp = log_probs[chosen_pos]
            policy_losses.append(-logp * advantages[t])

            entropy = -(probs * log_probs).sum()
            entropies.append(entropy)

            # enrich per-step logs
            try:
                row = self._step_logs[t]
                row["return"] = float(G[t].item())
                row["advantage"] = float(advantages[t].item())
                row["probs_train"] = [float(x) for x in probs.detach().cpu().tolist()]
            except Exception:
                pass

        policy_loss = torch.stack(policy_losses).mean()
        entropy_mean = torch.stack(entropies).mean()
        entropy_loss = -self.entropy_beta * entropy_mean  # negative term to maximize entropy
        value_loss = F.mse_loss(V, G)

        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss

        # optimize
        self.optimizer.zero_grad()  # type: ignore[union-attr]
        total_loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)  # type: ignore[arg-type]
        self.optimizer.step()  # type: ignore[union-attr]

        # metrics
        try:
            grad_norm = _compute_total_grad_norm(self.model.parameters())
        except Exception:
            grad_norm = 0.0

        metrics = {
            "policy_loss": float(policy_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "entropy": float(entropy_mean.detach().item()),  # report +entropy
            "grad_norm": float(grad_norm),
        }

        # episode aggregates & cleanup
        self.last_episode_return = float(sum(self._rewards))
        self.last_losses = dict(metrics)

        self._flush_step_logs()
        self._clear_buffers()
        return metrics
    
    def _flush_step_logs(self) -> None:
        """Write per-turn logs to CSV if LOG_TURN_BY_TURN is enabled."""
        if not self._step_logs or not LOG_TURN_BY_TURN:
            return

        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

        ep = int(getattr(self, "episode_idx", 0) or 0)
        fname = f"episode_{ep:04d}.csv" if ep > 0 else "episode.csv"
        fpath = Path(LOG_DIR) / fname

        # Determine columns (union of keys across rows, consistent order)
        keys = []
        seen = set()
        for row in self._step_logs:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)

        # write
        with fpath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self._step_logs:
                # Convert any non-serializable entries safely
                out = {}
                for k, v in row.items():
                    try:
                        # cast tensors/ndarrays to python scalars/lists
                        if hasattr(v, "detach"):
                            v = v.detach().cpu().item()
                        elif hasattr(v, "tolist"):
                            v = v.tolist()
                    except Exception:
                        pass
                    out[k] = v
                writer.writerow(out)

    def _clear_buffers(self) -> None:
        """Reset episodic buffers after optimize."""
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._legal_sets.clear()
        self._step_logs.clear()
