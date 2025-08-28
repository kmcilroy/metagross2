import asyncio
import random
import numpy as np
import torch

from pathlib import Path
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from .agent2 import LearningPlayerAC

from .config import *
# from .agent import LearningPlayer
from .rewards import step_reward, terminal_reward


def load_team(path: str) -> str:
    p = (Path(__file__).resolve().parent / path).resolve()
    txt = p.read_text(encoding="utf-8").strip()
    return txt + ("\n" if not txt.endswith("\n") else "")

def fainted_counts(battle) -> tuple[int, int]:
    you = sum(1 for p in battle.team.values() if p.fainted)
    opp = sum(1 for p in battle.opponent_team.values() if p.fainted)
    return you, opp

def _extract_battle_from_args(*args, **kwargs):
    if args:
        return args[0]
    return kwargs.get("battle", None)

def _get_any_battle(player):
    """Fallback: pull a battle from player's registries after battle_against returns."""
    battles = getattr(player, "battles", None) or getattr(player, "_battles", {})
    if not battles:
        return None
    vals = list(battles.values())
    finished = [b for b in vals if getattr(b, "finished", False)]
    if finished:
        return finished[-1]
    return vals[-1]

def _carry_model(from_player: LearningPlayerAC, to_player: LearningPlayerAC):
    """Move model/optimizer/device and lazy-init knobs between learner instances."""
    if getattr(from_player, "model", None) is None:
        return
    to_player.model = from_player.model
    to_player.optimizer = from_player.optimizer
    to_player.device = from_player.device
    to_player._hidden = from_player._hidden
    to_player._lr = from_player._lr

async def train_once():
    # seeds
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    team_a = load_team(TEAM_A_PATH)
    team_b = load_team(TEAM_B_PATH)

    # epsilon schedule
    def eps_for_battle(bi: int):
        if bi >= EPSILON_DECAY_BATTLES:
            return EPSILON_END
        t = bi / max(1, EPSILON_DECAY_BATTLES)
        return EPSILON_START * (1 - t) + EPSILON_END * t

    wins = 0
    last_learner: LearningPlayerAC | None = None

    for bi in range(TOTAL_BATTLES):
        # Fresh instances each battle -> unique auto-generated usernames (no popup)
        # learner = LearningPlayer(
        #     battle_format=FORMAT_ID,
        #     server_configuration=LocalhostServerConfiguration,
        #     team=team_a,
        #     epsilon=eps_for_battle(bi),
        #     lr=LR,
        #     hidden=HIDDEN_DIM,
        #     log_level=30,
        #     max_concurrent_battles=MAX_CONCURRENT_BATTLES,
        # )

        

                
        learner = LearningPlayerAC(
            battle_format=FORMAT_ID,
            server_configuration=LocalhostServerConfiguration,
            team=team_a,
            epsilon=EPSILON_START,
            lr=LR,
            hidden=HIDDEN_DIM,
            log_level=30,
            max_concurrent_battles=MAX_CONCURRENT_BATTLES,
        )
        learner.episode_idx = bi + 1   # <--- add this line


        opponent = RandomPlayer(
            battle_format=FORMAT_ID,
            server_configuration=LocalhostServerConfiguration,
            team=team_b,
            log_level=30,
            max_concurrent_battles=MAX_CONCURRENT_BATTLES,
        )

        # Carry weights/optimizer forward
        if last_learner is not None:
            _carry_model(last_learner, learner)

        # Reset per-episode trackers
        learner.prev_you_hp = 1.0
        learner.prev_opp_hp = 1.0
        learner.prev_you_fainted = 0
        learner.prev_opp_fainted = 0
        learner._last_battle = None   # type: ignore[attr-defined]
        steps = 0

        async def hook(*args, **kwargs):
            nonlocal steps
            steps += 1

            battle = _extract_battle_from_args(*args, **kwargs)
            if battle is None:
                battle = getattr(learner, "current_battle", None)
                if battle is None:
                    return

            # HP-shaped reward
            r = step_reward(learner.prev_you_hp, learner.prev_opp_hp, battle)

            # KO deltas (±10 per faint)
            you_f, opp_f = fainted_counts(battle)
            r += 10.0 * max(0, opp_f - learner.prev_opp_fainted)
            r -= 10.0 * max(0, you_f - learner.prev_you_fainted)
            learner.prev_you_fainted, learner.prev_opp_fainted = you_f, opp_f

            # Log step reward
            learner.learn_step(r)

            # Update prev HP trackers
            you = battle.active_pokemon
            opp = battle.opponent_active_pokemon
            learner.prev_you_hp = float(you.current_hp_fraction if you else 0.0)
            if opp and opp.current_hp is not None:
                learner.prev_opp_hp = float(opp.current_hp_fraction)

            # Remember latest battle for terminal phase
            learner._last_battle = battle  # type: ignore[attr-defined]

        # Register per-turn hook
        learner._post_request_callback = hook

        # --- Run one battle ---
        await learner.battle_against(opponent, n_battles=1)

        # Terminal handling
        b = getattr(learner, "_last_battle", None) or _get_any_battle(learner) or _get_any_battle(opponent)
        if b is None:
            raise RuntimeError("No battle found to compute terminal reward.")

        tr = terminal_reward(b)
        learner.learn_step(tr)

        # One optimize step per episode
        learner.optimize_after_battle(gamma=1.0)

        if b.won:
            wins += 1

        if (bi + 1) % LOG_EVERY == 0:
            ep_ret = getattr(learner, "last_episode_return", None)
            extra = f"  return={ep_ret:.2f}" if ep_ret is not None else ""
            print(f"[Battle {bi+1}/{TOTAL_BATTLES}] eps={learner.epsilon:.3f}  "
                  f"result={'W' if b.won else ('L' if b.lost else 'T')}  wins={wins}{extra}")

        # Keep this learner to carry weights into the next instance
        last_learner = learner

    print(f"Done. Wins: {wins}/{TOTAL_BATTLES}")


if __name__ == "__main__":
    asyncio.run(train_once())
