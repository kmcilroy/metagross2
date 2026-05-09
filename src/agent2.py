from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from poke_env.player import Player

if TYPE_CHECKING:
    from poke_env.environment import AbstractBattle  # type: ignore

from .encoder import encode_battle
from .models import make_model
from .training import a2c_loss, discounted_returns, legal_mask_from_lists
from .utils.poke_helpers import (
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


class LearningPlayerAC(Player):
    """Pokémon Showdown player wrapping an actor-critic policy.

    Two consumption paths for its rollout buffers:
      - Sync trainer drives shaped rewards via learn_step() and calls
        optimize_after_battle() at episode end.
      - Async A3C trainer ignores learn_step, calls pop_rollout() to
        drain the buffers, and runs the loss against its own local
        copy of the model.
    """

    def __init__(
        self,
        epsilon: float,
        lr: float,
        hidden: int,
        arch: str = "mlp_2h",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

        # Lazy model setup. choose_move builds it once state_dim is known
        # (sync path); ensure_model() builds eagerly (async path).
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self._hidden = int(hidden)
        self._lr = float(lr)
        self._arch = str(arch)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required but not available")
        self.device = torch.device("cuda")

        # Episodic rollout buffers
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._legal_sets: List[List[int]] = []
        self._step_logs: List[Dict[str, Any]] = []

        # Reward-shaping trackers updated by the sync trainer's hook.
        self.prev_you_hp = 1.0
        self.prev_opp_hp = 1.0
        self.prev_you_fainted = 0
        self.prev_opp_fainted = 0

        self.episode_idx: int = 0

        self.entropy_beta = float(ENTROPY_BETA)
        self.value_coef = float(VALUE_COEF)
        self.max_grad_norm = float(MAX_GRAD_NORM)

        self.last_episode_return: float = 0.0
        self.last_losses: Dict[str, float] = {}

    # ---------- Model lifecycle ----------
    def ensure_model(self, state_dim: int) -> None:
        """Eagerly build the actor-critic so external state_dicts (e.g. the
        async global model, or a frozen-opponent checkpoint) can be loaded
        before the first choose_move. No-op if already built."""
        if self.model is None:
            self.model = make_model(
                self._arch,
                state_dim=state_dim,
                n_actions=MAX_ACTIONS,
                hidden=self._hidden,
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self._lr)

    def pop_rollout(self) -> List[Dict[str, Any]]:
        """Drain buffered (state, action, legal) into a trajectory list:
        [{state, action, reward, done, mask}, ...].

        Per-step `reward` is taken from `_rewards` if learn_step was called
        (sync path); otherwise zero — the async trainer fills the terminal
        reward externally. Buffers are NOT cleared here; the caller decides
        via _clear_buffers."""
        T = min(len(self._states), len(self._actions), len(self._legal_sets))
        out: List[Dict[str, Any]] = []
        for t in range(T):
            mask = [False] * MAX_ACTIONS
            for j in self._legal_sets[t] or []:
                jj = int(j)
                if 0 <= jj < MAX_ACTIONS:
                    mask[jj] = True
            out.append({
                "state": np.asarray(self._states[t], dtype=np.float32),
                "action": int(self._actions[t]),
                "reward": float(self._rewards[t]) if t < len(self._rewards) else 0.0,
                "done": False,
                "mask": mask,
            })
        return out

    # ---------- Acting ----------
    def choose_move(self, battle: "AbstractBattle"):
        legal = enumerate_legal_indices(battle)

        s_np = encode_battle(battle)
        self._states.append(s_np)
        self._legal_sets.append(list(legal))

        # Lazy build (sync path enters here on first move).
        if self.model is None:
            self.ensure_model(int(len(s_np)))

        s_t = torch.tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, values = self.model(s_t)        # [1, A], [1]
            v_est = float(values.item())
            logits_a = logits[0]                    # [A]

            mask = torch.zeros(MAX_ACTIONS, dtype=torch.bool, device=self.device)
            for j in legal:
                if 0 <= j < MAX_ACTIONS:
                    mask[j] = True
            masked_logits = logits_a.masked_fill(~mask, float("-inf"))
            temp = max(1e-6, float(SOFTMAX_TEMPERATURE))
            probs_a = torch.softmax(masked_logits / temp, dim=0)  # [A], 0 on illegal
            log_probs_a = torch.log(probs_a.clamp_min(1e-8))
            entropy_val = float(-(probs_a * log_probs_a).sum().item())
            probs_legal_only = [float(probs_a[j].item()) for j in legal]

        if np.random.rand() < self.epsilon:
            a_idx = int(np.random.choice(legal))
            probs_list_log: Optional[List[float]] = None
            entropy_log: Optional[float] = None
        else:
            dist = Categorical(probs=probs_a)
            a_idx = int(dist.sample().item())
            # Fall back to a uniform legal pick on the (rare) edge case
            # where the sampled action came out illegal due to fp noise.
            if a_idx not in legal:
                a_idx = int(np.random.choice(legal))
            probs_list_log = probs_legal_only
            entropy_log = entropy_val

        self._actions.append(a_idx)

        self._step_logs.append({
            "step": len(self._step_logs),
            "epsilon": float(self.epsilon),
            "value": v_est,
            "legal": list(legal),
            "action_idx": int(a_idx),
            "action_label": self._describe_action(battle, a_idx),
            "probs": probs_list_log,
            "entropy": entropy_log,
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
            if 4 <= idx <= 9:
                sw_idx = idx - 4
                sws = battle.available_switches or []
                if 0 <= sw_idx < len(sws):
                    species = getattr(sws[sw_idx], "species", "switch")
                    return f"switch:{species}"
                return "switch:NA"
        except Exception:
            pass
        return "unknown"

    def learn_step(self, reward: float) -> None:
        """Sync trainer's per-turn reward push. Async path doesn't call this."""
        r = float(reward)
        self._rewards.append(r)
        if self._step_logs:
            self._step_logs[-1]["reward"] = r

    # ---------- Learning ----------
    def optimize_after_battle(self, gamma: float = 1.0) -> Dict[str, float]:
        """One A2C step over the buffered trajectory. Returns metrics:
        {policy_loss, value_loss, entropy, grad_norm}."""
        T = min(len(self._states), len(self._actions), len(self._legal_sets))
        if T == 0:
            self._clear_buffers()
            return {}

        returns = discounted_returns(self._rewards, gamma=gamma)
        if len(returns) < T:
            returns = [0.0] * (T - len(returns)) + returns
        elif len(returns) > T:
            returns = returns[-T:]

        S = torch.tensor(np.stack(self._states[:T], axis=0), dtype=torch.float32, device=self.device)
        A = torch.tensor(self._actions[:T], dtype=torch.long, device=self.device)
        G = torch.tensor(returns, dtype=torch.float32, device=self.device)
        masks = legal_mask_from_lists(self._legal_sets[:T], MAX_ACTIONS).to(self.device)

        if self.model is None:
            self.ensure_model(int(S.shape[1]))

        losses = a2c_loss(
            self.model,
            S, A, masks, G,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_beta,
        )

        # Per-step CSV enrichment before optimizer.step (uses pre-update probs/values)
        for t in range(T):
            try:
                self._step_logs[t]["return"] = float(G[t].item())
                self._step_logs[t]["advantage"] = float(losses["advantages"][t].item())
                self._step_logs[t]["probs_train"] = [
                    float(x) for x in losses["probs"][t].tolist()
                ]
            except IndexError:
                pass

        self.optimizer.zero_grad()
        losses["total"].backward()
        max_norm = self.max_grad_norm if (self.max_grad_norm and self.max_grad_norm > 0) else float("inf")
        grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), max_norm))
        self.optimizer.step()

        metrics = {
            "policy_loss": float(losses["policy_loss"].item()),
            "value_loss": float(losses["value_loss"].item()),
            "entropy": float(losses["entropy"].item()),
            "loss_total": float(losses["total"].detach().item()),
            "grad_norm": grad_norm,
        }
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

        keys: List[str] = []
        seen: set[str] = set()
        for row in self._step_logs:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)

        with fpath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self._step_logs:
                out: Dict[str, Any] = {}
                for k, v in row.items():
                    try:
                        if hasattr(v, "detach"):
                            v = v.detach().cpu().item()
                        elif hasattr(v, "tolist"):
                            v = v.tolist()
                    except Exception:
                        pass
                    out[k] = v
                writer.writerow(out)

    def _clear_buffers(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._legal_sets.clear()
        self._step_logs.clear()
