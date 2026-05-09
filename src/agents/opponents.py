"""Opponent factories driven by a spec string from --opponent.

Both trainers parse `--opponent` once at startup. The factory caches
any expensive setup (e.g., reading a checkpoint from disk) and then
mints a fresh poke-env Player per battle (Showdown rejects duplicate
usernames within a session, so the player object can't be reused).

Spec format:
    "random"        -> uniform-random baseline
    "ac:PATH.pt"    -> a frozen LearningPlayerAC loaded from PATH.
                       The checkpoint dict's "arch" and "hidden" keys
                       (saved by both trainers since the refactor) are
                       used to rebuild the matching architecture.
"""

from __future__ import annotations
from typing import Any, Dict, Protocol

import torch
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration


class OpponentFactory(Protocol):
    def make_player(self, *, battle_format: str, team: str,
                    max_concurrent_battles: int, state_dim: int,
                    server_configuration=LocalhostServerConfiguration) -> Any:
        ...


class RandomOpponent:
    def make_player(self, *, battle_format, team, max_concurrent_battles, state_dim,
                    server_configuration=LocalhostServerConfiguration):
        return RandomPlayer(
            battle_format=battle_format,
            server_configuration=server_configuration,
            team=team,
            max_concurrent_battles=max_concurrent_battles,
            log_level=30,
        )


class FrozenAcOpponent:
    """LearningPlayerAC with weights loaded from a checkpoint, ε=0, no learning.

    The checkpoint is read once at construction; each make_player() call
    builds a fresh player with a unique auto-generated username and copies
    the cached state_dict in.
    """

    def __init__(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.arch: str = ckpt.get("arch", "mlp_2h") if isinstance(ckpt, dict) else "mlp_2h"
        self.hidden: int = ckpt.get("hidden", 128) if isinstance(ckpt, dict) else 128
        self.state_dict: Dict[str, Any] = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        self.checkpoint_path = checkpoint_path

    def make_player(self, *, battle_format, team, max_concurrent_battles, state_dim,
                    server_configuration=LocalhostServerConfiguration):
        # Late import to avoid circular dependency between agent2 and this module.
        from ..agent2 import LearningPlayerAC

        opp = LearningPlayerAC(
            epsilon=0.0,
            lr=0.0,           # not used
            hidden=self.hidden,
            arch=self.arch,
            battle_format=battle_format,
            server_configuration=server_configuration,
            team=team,
            max_concurrent_battles=max_concurrent_battles,
            log_level=30,
        )
        opp.ensure_model(state_dim)
        opp.model.load_state_dict(self.state_dict)
        opp.model.eval()
        return opp


def make_opponent_factory(spec: str) -> OpponentFactory:
    """Parse --opponent and return a factory.

    Errors are raised at parse time so a bad path/spec fails before any
    Showdown connection or worker fork.
    """
    if spec == "random":
        return RandomOpponent()
    if spec.startswith("ac:"):
        return FrozenAcOpponent(spec[len("ac:"):])
    raise ValueError(
        f"Unknown opponent spec: {spec!r}. Expected 'random' or 'ac:PATH.pt'."
    )
