# src/agents/registry.py
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple, Type
from poke_env.player.baselines import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

# Your AC agent
from src.agent2 import LearningPlayerAC  # uses your existing class

@dataclass
class AgentSpec:
    key: str                      # e.g., "ac", "random"
    player_cls: Type             # class to instantiate
    # optional default kwargs function (per episode)
    default_kwargs_fn: Optional[Callable[[dict], dict]] = None

_REGISTRY: Dict[str, AgentSpec] = {}

def register(spec: AgentSpec):
    _REGISTRY[spec.key] = spec

# ---- Built-ins ----
register(AgentSpec(
    key="ac",
    player_cls=LearningPlayerAC,
    default_kwargs_fn=lambda ctx: {
        "epsilon": ctx.get("epsilon", 0.1),
        "lr": ctx.get("lr", 3e-4),
        "hidden": ctx.get("hidden", 256),
    },
))

register(AgentSpec(
    key="random",
    player_cls=RandomPlayer,
))

def make_player(
    key: str,
    *,
    username: str,
    team: str,
    battle_format: str,
    server=LocalhostServerConfiguration,
    max_concurrent_battles: int = 1,
    extra: Optional[dict] = None,
):
    if key not in _REGISTRY:
        raise KeyError(f"Unknown agent type: {key}")
    spec = _REGISTRY[key]

    ctx = extra or {}
    kwargs = spec.default_kwargs_fn(ctx) if spec.default_kwargs_fn else {}
    # Common kwargs (account/server/team/format)
    base = dict(
        account_configuration=AccountConfiguration(username, None),
        server_configuration=server,
        battle_format=battle_format,
        max_concurrent_battles=max_concurrent_battles,
        team=team,
    )
    base.update(kwargs)
    return spec.player_cls(**base)
