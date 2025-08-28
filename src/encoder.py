from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from poke_env.environment import AbstractBattle  # type: ignore

# For MVP0 we keep this deliberately simple & stable across gens:
# - Active HP frac (you + opp)
# - Known/unknown opp HP flag
# - Count fainted (you + opp)
# - TWave/Para/Sleep on actives (you + opp)
# - Move availability mask (size 4) for your active
# - Switch availability count (0..5)

ENCODER_DIM = 1 + 1 + 1 + 1 + 2 + 2 + 4 + 1  # 13


def encode_battle(battle: "AbstractBattle") -> np.ndarray:
    you_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon

    # HP fractions
    you_hp = float(you_active.current_hp_fraction if you_active else 0.0)
    if opp_active and opp_active.current_hp is not None:
        opp_hp = float(opp_active.current_hp_fraction)
        opp_hp_unknown = 0.0
    else:
        opp_hp = 0.0
        opp_hp_unknown = 1.0

    # Fainted counts
    you_fainted = sum(1 for p in battle.team.values() if p.fainted)
    opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)

    # Status flags on actives
    def status_flags(p):
        if p is None:
            return (0.0, 0.0)
        s = getattr(p, "status", None)
        if s is None:
            return (0.0, 0.0)
        # handle both Enum and string cases
        name = getattr(s, "name", str(s)).lower()
        par = 1.0 if "par" in name else 0.0
        slp = 1.0 if ("slp" in name or "sleep" in name) else 0.0
        return (par, slp)

    you_par, you_slp = status_flags(you_active)
    opp_par, opp_slp = status_flags(opp_active)

    # Move availability (size 4)
    move_mask = [0.0, 0.0, 0.0, 0.0]
    if battle.available_moves:
        for i in range(min(4, len(battle.available_moves))):
            move_mask[i] = 1.0

    # Switch availability count
    switch_count = float(len(battle.available_switches or []))

    vec = [
        you_hp, opp_hp, opp_hp_unknown,
        float(you_fainted), float(opp_fainted),
        you_par, you_slp, opp_par, opp_slp,
        *move_mask,
        switch_count,
    ]
    return np.asarray(vec, dtype=np.float32)
