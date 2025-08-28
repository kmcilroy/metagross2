# src/psio/accounts.py
from __future__ import annotations
import time, uuid
from typing import Tuple, Type
from poke_env.exceptions import ShowdownException
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

def fresh_username(prefix: str, wid: int, ep: int) -> str:
    return (f"{prefix}{wid}{ep}{uuid.uuid4().hex[:6]}")[:18]

async def create_players_with_retry(
    learner_cls: Type,
    opponent_cls: Type,
    *,
    wid: int,
    ep: int,
    team1: str,
    team2: str,
    fmt: str,
    max_concurrent_battles: int,
    device: str = "cpu",
    extra_learner_kwargs: dict | None = None,
    extra_opponent_kwargs: dict | None = None,
    server=LocalhostServerConfiguration,
    max_tries: int = 5,
) -> Tuple[object, object]:
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            ac_name = fresh_username("mga", wid, ep)
            rn_name = fresh_username("mgr", wid, ep)

            ac = learner_cls(
                account_configuration=("AccountConfiguration", ac_name, None),  # placeholder for late binding
                # we pass these as kwargs so registry can massage them
            )
            rn = opponent_cls(
                account_configuration=("AccountConfiguration", rn_name, None),
            )
            # Rebuild with explicit kwargs (works with make_player from registry)
            from src.agents.registry import make_player
            ac = make_player("ac",
                username=ac_name, team=team1, battle_format=fmt, server=server,
                max_concurrent_battles=max_concurrent_battles, extra=extra_learner_kwargs or {}
            )
            rn = make_player("random",
                username=rn_name, team=team2, battle_format=fmt, server=server,
                max_concurrent_battles=max_concurrent_battles, extra=extra_opponent_kwargs or {}
            )
            # Optional device push for learner internals
            if device == "cuda":
                try:
                    import torch, torch.nn as nn
                    if hasattr(ac, 'to_device'):
                        ac.to_device('cuda')
                    else:
                        for attr in ('model', 'policy', 'value'):
                            m = getattr(ac, attr, None)
                            if isinstance(m, nn.Module):
                                m.to('cuda')
                except Exception:
                    pass
            return ac, rn
        except ShowdownException as e:
            last_err = e
            if "|nametaken|" in str(e):
                time.sleep(0.2 * attempt)
                continue
            raise
    raise last_err
