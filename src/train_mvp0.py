import asyncio
import random
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from .agent2 import LearningPlayerAC
from .config import *  # expects: EVAL_EVERY, EVAL_GAMES, SAVE_CHECKPOINTS, CHECKPOINT_DIR, etc.
from .rewards import step_reward, terminal_reward


# ---------- utils ----------
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
    # carry optional knobs if present
    for attr in ("entropy_beta", "value_coef", "max_grad_norm"):
        if hasattr(from_player, attr):
            setattr(to_player, attr, getattr(from_player, attr))

def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


# ---------- evaluation (periodic) ----------
async def eval_once(base_learner: LearningPlayerAC,
                    team_a: str,
                    team_b: str,
                    games: int) -> Dict[str, float]:
    """
    Run evaluation WITHOUT learning:
      - ε = 0.0 (greedy wrt current policy)
      - no rewards logged into buffers
      - no optimize calls
    Returns: {'win_rate', 'avg_return', 'steps_avg'}
    """
    wins = 0
    term_returns = []
    steps_all = []

    # fresh opponents each eval game; eval agent carries weights from base_learner
    for gi in range(games):
        eval_agent = LearningPlayerAC(
            battle_format=FORMAT_ID,
            server_configuration=LocalhostServerConfiguration,
            team=team_a,
            epsilon=0.0,                    # freeze exploration
            lr=base_learner._lr,            # not used (no learning), but harmless
            hidden=base_learner._hidden,
            log_level=30,
            max_concurrent_battles=MAX_CONCURRENT_BATTLES,
        )
        _carry_model(base_learner, eval_agent)
        eval_agent.episode_idx = -1  # mark as eval

        opponent = RandomPlayer(
            battle_format=FORMAT_ID,
            server_configuration=LocalhostServerConfiguration,
            team=team_b,
            log_level=30,
            max_concurrent_battles=MAX_CONCURRENT_BATTLES,
        )

        steps = 0

        # Eval hook: count steps ONLY (no learn_step calls)
        async def eval_hook(*args, **kwargs):
            nonlocal steps
            steps += 1
            battle = _extract_battle_from_args(*args, **kwargs)
            if battle is None:
                return
            # Do NOT call eval_agent.learn_step(...) here

            # Keep last battle reference for terminal metric extraction
            eval_agent._last_battle = battle  # type: ignore[attr-defined]

        eval_agent._post_request_callback = eval_hook

        await eval_agent.battle_against(opponent, n_battles=1)

        b = getattr(eval_agent, "_last_battle", None) or _get_any_battle(eval_agent) or _get_any_battle(opponent)
        if b is None:
            # If we somehow didn't capture the battle, skip safely
            continue

        # Terminal-only return (consistent and cheap)
        tr = float(terminal_reward(b))
        term_returns.append(tr)
        steps_all.append(steps)
        if b.won:
            wins += 1

        # Clean buffers if any accidental accumulation occurred
        if hasattr(eval_agent, "_clear_buffers"):
            try:
                eval_agent._clear_buffers()
            except Exception:
                pass

    games_played = max(1, len(steps_all))
    metrics = {
        "win_rate": wins / float(games_played),
        "avg_return": float(np.mean(term_returns)) if term_returns else 0.0,
        "steps_avg": float(np.mean(steps_all)) if steps_all else 0.0,
    }
    return metrics


# ---------- training (unchanged behavior + eval hook) ----------
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
    best_eval_wr: float = -1.0

    if SAVE_CHECKPOINTS:
        _ensure_dir(CHECKPOINT_DIR)
        # TensorBoard writer
        run_name = f"run-{datetime.now():%Y%m%d-%H%M%S}"
        log_root = Path(globals().get("TENSORBOARD_LOGDIR", "runs/metagross"))
        log_root.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_root / run_name))

    for bi in range(TOTAL_BATTLES):
        # Fresh instances each battle -> unique auto-generated usernames
        learner = LearningPlayerAC(
            battle_format=FORMAT_ID,
            server_configuration=LocalhostServerConfiguration,
            team=team_a,
            epsilon=eps_for_battle(bi),   # use schedule
            lr=LR,
            hidden=HIDDEN_DIM,
            log_level=30,
            max_concurrent_battles=MAX_CONCURRENT_BATTLES,
        )
        learner.episode_idx = bi + 1

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

        # --- Run one battle (train) ---
        await learner.battle_against(opponent, n_battles=1)

        # Terminal handling
        b = getattr(learner, "_last_battle", None) or _get_any_battle(learner) or _get_any_battle(opponent)
        if b is None:
            raise RuntimeError("No battle found to compute terminal reward.")

        tr = terminal_reward(b)
        learner.learn_step(tr)

        # One optimize step per episode; capture metrics (for TB later)
        losses: Dict[str, float] = learner.optimize_after_battle(gamma=1.0) or {}
        
        # ---- TB: train scalars ----
        step = bi + 1
        ep_ret = getattr(learner, "last_episode_return", None)
        if ep_ret is not None:
            writer.add_scalar("train/episode_return", float(ep_ret), step)

        writer.add_scalar("train/steps_in_battle", float(steps), step)
        writer.add_scalar("train/epsilon", float(learner.epsilon), step)

        for k in ("policy_loss", "value_loss", "entropy", "grad_norm"):
            v = losses.get(k, None)
            if v is not None:
                writer.add_scalar(f"train/{k}", float(v), step)


        if b.won:
            wins += 1

        if (bi + 1) % LOG_EVERY == 0:
            ep_ret = getattr(learner, "last_episode_return", None)
            # optional short loss print
            pl = losses.get("policy_loss", None)
            vl = losses.get("value_loss", None)
            en = losses.get("entropy", None)
            gn = losses.get("grad_norm", None)
            loss_str = ""
            if pl is not None and vl is not None:
                loss_str = f"  pl={pl:.3f} vl={vl:.3f}"
                if en is not None:
                    loss_str += f" ent={en:.3f}"
                if gn is not None:
                    loss_str += f" |g|={gn:.2f}"
            extra = f"  return={ep_ret:.2f}" if ep_ret is not None else ""
            print(f"[Battle {bi+1}/{TOTAL_BATTLES}] eps={learner.epsilon:.3f}  "
                  f"result={'W' if b.won else ('L' if b.lost else 'T')}  wins={wins}{extra}{loss_str}")

        # ---------- periodic evaluation ----------
        if (bi + 1) % EVAL_EVERY == 0:
            base = learner if learner.model is not None else (last_learner or learner)
            eval_metrics = await eval_once(base, team_a=team_a, team_b=team_b, games=EVAL_GAMES)
            wr = eval_metrics["win_rate"]
            ar = eval_metrics["avg_return"]
            st = eval_metrics["steps_avg"]
            print(f"[Eval @ ep {bi+1}] win_rate={wr:.3f}  avg_return={ar:.2f}  steps_avg={st:.1f}")
            # ---- TB: eval scalars ----
            writer.add_scalar("eval/win_rate", float(wr), step)
            writer.add_scalar("eval/avg_return", float(ar), step)
            writer.add_scalar("eval/steps_avg", float(st), step)



            # Save best by eval win rate
            if SAVE_CHECKPOINTS and wr > best_eval_wr and learner.model is not None:
                best_eval_wr = wr
                _ensure_dir(CHECKPOINT_DIR)
                path = Path(CHECKPOINT_DIR) / "best.pt"
                torch.save({"model": learner.model.state_dict()}, str(path))
                print(f"[Checkpoint] New best eval WR {wr:.3f} → saved to {path}")

        # Keep this learner to carry weights into the next instance
        last_learner = learner
    writer.flush()
    writer.close()

    print(f"Done. Wins: {wins}/{TOTAL_BATTLES}")



if __name__ == "__main__":
    asyncio.run(train_once())
