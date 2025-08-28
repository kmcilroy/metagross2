from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Only for type checkers; not executed at runtime
    from poke_env.environment.abstract_battle import AbstractBattle  # type: ignore


# Shaped rewards (per your spec):
# +10 per opponent KO
# +0.3 per 10% HP damage dealt  (i.e., +3.0 per full 100% HP)
# +0.3 per 10% HP recovered
# -10 per your KO
# -0.3 per 10% HP you lose
# Terminal: +100 win, -100 loss, 0 tie

def step_reward(prev_you_hp: float, prev_opp_hp: float, battle: "AbstractBattle") -> float:
    # Current HP fractions (actives only; MVP0)
    you = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    you_hp = float(you.current_hp_fraction if you else 0.0)
    opp_hp = float(opp.current_hp_fraction if (opp and opp.current_hp is not None) else prev_opp_hp)

    # ΔHP terms (+ for opp damage, - for your damage taken)
    dmg_dealt = max(0.0, prev_opp_hp - opp_hp)
    dmg_taken = max(0.0, prev_you_hp - you_hp)

    r = 0.0
    r += 3.0 * dmg_dealt            # 0.3 per 10% = 3.0 per full HP
    r -= 3.0 * dmg_taken

    # Recovery terms (rare in Gen1 OU but included)
    heal_you = max(0.0, you_hp - prev_you_hp)
    heal_opp = max(0.0, opp_hp - prev_opp_hp)
    r += 3.0 * heal_you
    r -= 3.0 * heal_opp

    # KOs (count total fainted on each side)
    # (For MVP0, approximate via team fainted deltas is fine; handled in train loop if desired.)
    return r

def terminal_reward(battle: "AbstractBattle") -> float:
    if battle.won:
        return 100.0
    if battle.lost:
        return -100.0
    return 0.0
