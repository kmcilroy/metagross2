from __future__ import annotations
from typing import List
from poke_env.player.player import Player

# Fixed 10-slot action space:
# [0..3] -> moves by index into battle.available_moves
# [4..9] -> switches by index into battle.available_switches

MAX_ACTIONS = 10

def enumerate_legal_indices(battle) -> List[int]:
    legal: List[int] = []

    # Moves (0..3)
    if battle.available_moves:
        for i in range(min(4, len(battle.available_moves))):
            legal.append(i)

    # Switches (4..9)
    if battle.available_switches:
        for i in range(min(6, len(battle.available_switches))):
            legal.append(4 + i)

    if not legal:
        legal.append(-1)  # safe fallback
    return legal


def order_from_index(player: Player, battle, idx: int):
    """Map our fixed index to a proper BattleOrder via Player.create_order()."""
    # Fallback: just do something legal
    if idx == -1:
        return player.choose_random_move(battle)

    # Moves
    if 0 <= idx <= 3:
        moves = battle.available_moves or []
        if idx < len(moves):
            # Pass the Move object, not an int slot
            return player.create_order(moves[idx])
        # if the mapped move isn't available, fall back
        return player.choose_random_move(battle)

    # Switches
    sw_idx = idx - 4
    switches = battle.available_switches or []
    if 0 <= sw_idx < len(switches):
        # Pass the Pokemon object to switch to
        return player.create_order(switches[sw_idx])

    # Final fallback
    return player.choose_random_move(battle)


def encode_action_index(idx: int):
    import numpy as np
    one = np.zeros(MAX_ACTIONS, dtype=np.float32)
    if 0 <= idx < MAX_ACTIONS:
        one[idx] = 1.0
    return one
