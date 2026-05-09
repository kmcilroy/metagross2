"""Synchronous Actor-Critic trainer.

One battle at a time, with shaped per-turn rewards (HP deltas + KO
bonuses + terminal +/-100). TB scalar names match the async trainer
so runs are directly comparable.

Defaults come from src/config.py; CLI flags override.
"""

import argparse
import asyncio
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from torch.utils.tensorboard import SummaryWriter

from .agent2 import LearningPlayerAC
from .agents.opponents import make_opponent_factory
from .encoder import ENCODER_DIM
from .config import (
    CHECKPOINT_DIR,
    EPSILON_DECAY_BATTLES,
    EPSILON_END,
    EPSILON_START,
    EVAL_EVERY,
    EVAL_GAMES,
    FORMAT_ID,
    HIDDEN_DIM,
    LOG_EVERY,
    LR,
    MAX_CONCURRENT_BATTLES,
    SAVE_CHECKPOINTS,
    SEED,
    TEAM_A_PATH,
    TEAM_B_PATH,
    TENSORBOARD_LOGDIR,
    TOTAL_BATTLES,
)
from .models import list_models
from .rewards import step_reward, terminal_reward


# ---------- helpers ----------
def load_team(path: str) -> str:
    p = (Path(__file__).resolve().parent / path).resolve()
    txt = p.read_text(encoding="utf-8").strip()
    return txt + ("\n" if not txt.endswith("\n") else "")


def fainted_counts(battle):
    you = sum(1 for p in battle.team.values() if p.fainted)
    opp = sum(1 for p in battle.opponent_team.values() if p.fainted)
    return you, opp


def _extract_battle_from_args(*args, **kwargs):
    if args:
        return args[0]
    return kwargs.get("battle", None)


def _get_any_battle(player):
    battles = getattr(player, "battles", None) or getattr(player, "_battles", {})
    if not battles:
        return None
    vals = list(battles.values())
    finished = [b for b in vals if getattr(b, "finished", False)]
    return finished[-1] if finished else vals[-1]


def _carry_model(from_player: LearningPlayerAC, to_player: LearningPlayerAC):
    """Move the model + optimizer + lazy-init knobs to a new player
    instance. Each Showdown battle needs a fresh username so the player
    object is rebuilt; this preserves training state across battles."""
    if getattr(from_player, "model", None) is None:
        return
    to_player.model = from_player.model
    to_player.optimizer = from_player.optimizer
    to_player.device = from_player.device
    to_player._hidden = from_player._hidden
    to_player._lr = from_player._lr
    to_player._arch = from_player._arch
    for attr in ("entropy_beta", "value_coef", "max_grad_norm"):
        if hasattr(from_player, attr):
            setattr(to_player, attr, getattr(from_player, attr))


# ---------- evaluation (periodic) ----------
async def eval_once(base_learner, team_a, team_b, games, fmt, hidden, lr, arch,
                    max_concurrent, opponent_factory):
    """Greedy (eps=0) eval for `games` battles vs the configured opponent. No learning."""
    wins = 0
    term_returns = []
    steps_all = []

    for _ in range(games):
        eval_agent = LearningPlayerAC(
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team_a,
            epsilon=0.0,
            lr=lr,
            hidden=hidden,
            arch=arch,
            log_level=30,
            max_concurrent_battles=max_concurrent,
        )
        _carry_model(base_learner, eval_agent)
        eval_agent.episode_idx = -1

        opponent = opponent_factory.make_player(
            battle_format=fmt,
            team=team_b,
            max_concurrent_battles=max_concurrent,
            state_dim=ENCODER_DIM,
        )

        steps = 0

        async def eval_hook(*args, **kwargs):
            nonlocal steps
            steps += 1
            battle = _extract_battle_from_args(*args, **kwargs)
            if battle is not None:
                eval_agent._last_battle = battle  # type: ignore[attr-defined]

        eval_agent._post_request_callback = eval_hook
        await eval_agent.battle_against(opponent, n_battles=1)

        b = getattr(eval_agent, "_last_battle", None) or _get_any_battle(eval_agent) or _get_any_battle(opponent)
        if b is None:
            continue

        term_returns.append(float(terminal_reward(b)))
        steps_all.append(steps)
        if b.won:
            wins += 1

        if hasattr(eval_agent, "_clear_buffers"):
            try:
                eval_agent._clear_buffers()
            except Exception:
                pass

    n = max(1, len(steps_all))
    return {
        "win_rate": wins / float(n),
        "avg_return": float(np.mean(term_returns)) if term_returns else 0.0,
        "steps_avg": float(np.mean(steps_all)) if steps_all else 0.0,
    }


# ---------- training ----------
async def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    team_a = load_team(args.team_a)
    team_b = load_team(args.team_b)
    opponent_factory = make_opponent_factory(args.opponent)

    def eps_for_battle(bi: int) -> float:
        if bi >= args.epsilon_decay_battles:
            return args.epsilon_end
        t = bi / max(1, args.epsilon_decay_battles)
        return args.epsilon_start * (1 - t) + args.epsilon_end * t

    save_checkpoints = SAVE_CHECKPOINTS and not args.no_checkpoints

    run_name = f"run-{datetime.now():%Y%m%d-%H%M%S}-{args.arch}-h{args.hidden}"
    log_root = Path(args.logdir)
    log_root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_root / run_name))

    if save_checkpoints:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    win_hist = deque(maxlen=100)
    ret_hist = deque(maxlen=100)

    wins = 0
    last_learner: Optional[LearningPlayerAC] = None
    best_eval_wr = -1.0

    for bi in range(args.total_battles):
        learner = LearningPlayerAC(
            battle_format=args.format,
            server_configuration=LocalhostServerConfiguration,
            team=team_a,
            epsilon=eps_for_battle(bi),
            lr=args.lr,
            hidden=args.hidden,
            arch=args.arch,
            log_level=30,
            max_concurrent_battles=args.max_concurrent_battles,
        )
        learner.episode_idx = bi + 1

        opponent = opponent_factory.make_player(
            battle_format=args.format,
            team=team_b,
            max_concurrent_battles=args.max_concurrent_battles,
            state_dim=ENCODER_DIM,
        )

        if last_learner is not None:
            _carry_model(last_learner, learner)

        learner.prev_you_hp = 1.0
        learner.prev_opp_hp = 1.0
        learner.prev_you_fainted = 0
        learner.prev_opp_fainted = 0
        learner._last_battle = None  # type: ignore[attr-defined]

        steps = 0

        async def hook(*hargs, **hkwargs):
            nonlocal steps
            steps += 1
            battle = _extract_battle_from_args(*hargs, **hkwargs) or getattr(learner, "current_battle", None)
            if battle is None:
                return

            r = step_reward(learner.prev_you_hp, learner.prev_opp_hp, battle)

            you_f, opp_f = fainted_counts(battle)
            r += 10.0 * max(0, opp_f - learner.prev_opp_fainted)
            r -= 10.0 * max(0, you_f - learner.prev_you_fainted)
            learner.prev_you_fainted, learner.prev_opp_fainted = you_f, opp_f

            learner.learn_step(r)

            you = battle.active_pokemon
            opp = battle.opponent_active_pokemon
            learner.prev_you_hp = float(you.current_hp_fraction if you else 0.0)
            if opp and opp.current_hp is not None:
                learner.prev_opp_hp = float(opp.current_hp_fraction)

            learner._last_battle = battle  # type: ignore[attr-defined]

        learner._post_request_callback = hook

        await learner.battle_against(opponent, n_battles=1)

        b = getattr(learner, "_last_battle", None) or _get_any_battle(learner) or _get_any_battle(opponent)
        if b is None:
            raise RuntimeError("No battle found to compute terminal reward.")

        learner.learn_step(terminal_reward(b))
        losses: Dict[str, float] = learner.optimize_after_battle(gamma=1.0) or {}

        won = bool(b.won)
        if won:
            wins += 1

        ep_return = float(getattr(learner, "last_episode_return", 0.0))
        win_hist.append(int(won))
        ret_hist.append(ep_return)

        step = bi + 1
        # Per-episode scalars (names match the async trainer)
        writer.add_scalar("train/episode_return", ep_return, step)
        writer.add_scalar("train/win", int(won), step)
        writer.add_scalar("train/steps", float(steps), step)
        writer.add_scalar("train/epsilon", float(learner.epsilon), step)
        for k_metric, k_tb in (
            ("policy_loss", "loss/policy"),
            ("value_loss", "loss/value"),
            ("entropy", "loss/entropy"),
            ("loss_total", "loss/total"),
            ("grad_norm", "loss/grad_norm"),
        ):
            v = losses.get(k_metric)
            if v is not None:
                writer.add_scalar(k_tb, float(v), step)

        # Aggregate scalars (rolling)
        if len(win_hist) > 0:
            writer.add_scalar("agg/winrate_rolling100", float(np.mean(win_hist)), step)
            writer.add_scalar("agg/return_rolling100", float(np.mean(ret_hist)), step)
        writer.add_scalar("agg/winrate_cumulative", wins / float(step), step)

        if (bi + 1) % args.log_every == 0:
            pl = losses.get("policy_loss")
            vl = losses.get("value_loss")
            en = losses.get("entropy")
            gn = losses.get("grad_norm")
            extras = []
            if pl is not None and vl is not None:
                extras.append(f"pl={pl:.3f} vl={vl:.3f}")
                if en is not None:
                    extras.append(f"ent={en:.3f}")
                if gn is not None:
                    extras.append(f"|g|={gn:.2f}")
            extras.append(f"return={ep_return:.2f}")
            tail = "  ".join(extras)
            print(f"[Battle {bi+1}/{args.total_battles}] eps={learner.epsilon:.3f}  "
                  f"result={'W' if won else ('L' if b.lost else 'T')}  wins={wins}  {tail}")

        # Periodic eval
        if (bi + 1) % args.eval_every == 0:
            base = learner if learner.model is not None else (last_learner or learner)
            em = await eval_once(
                base, team_a, team_b, args.eval_games,
                fmt=args.format, hidden=args.hidden, lr=args.lr, arch=args.arch,
                max_concurrent=args.max_concurrent_battles,
                opponent_factory=opponent_factory,
            )
            print(f"[Eval @ ep {bi+1}] win_rate={em['win_rate']:.3f}  "
                  f"avg_return={em['avg_return']:.2f}  steps_avg={em['steps_avg']:.1f}")
            writer.add_scalar("eval/win_rate", em["win_rate"], step)
            writer.add_scalar("eval/avg_return", em["avg_return"], step)
            writer.add_scalar("eval/steps_avg", em["steps_avg"], step)

            if save_checkpoints and em["win_rate"] > best_eval_wr and learner.model is not None:
                best_eval_wr = em["win_rate"]
                Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                ckpt_path = Path(args.checkpoint_dir) / "best.pt"
                torch.save(
                    {"model": learner.model.state_dict(), "arch": args.arch, "hidden": args.hidden},
                    str(ckpt_path),
                )
                print(f"[Checkpoint] New best eval WR {em['win_rate']:.3f} → {ckpt_path}")

        last_learner = learner

    writer.flush()
    writer.close()
    print(f"Done. Wins: {wins}/{args.total_battles}")


def parse_args():
    p = argparse.ArgumentParser(description="Metagross synchronous A2C trainer.")
    p.add_argument("--total-battles", type=int, default=TOTAL_BATTLES)
    p.add_argument("--hidden", type=int, default=HIDDEN_DIM)
    p.add_argument("--arch", type=str, default="mlp_2h", choices=list_models(),
                   help="Model architecture from src.models")
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--epsilon-start", type=float, default=EPSILON_START)
    p.add_argument("--epsilon-end", type=float, default=EPSILON_END)
    p.add_argument("--epsilon-decay-battles", type=int, default=EPSILON_DECAY_BATTLES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--logdir", type=str, default=TENSORBOARD_LOGDIR,
                   help="TensorBoard root; a per-run subdir is created inside")
    p.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    p.add_argument("--no-checkpoints", action="store_true",
                   help="Disable best.pt checkpointing (TB still writes)")
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY,
                   help="Run greedy eval every N battles")
    p.add_argument("--eval-games", type=int, default=EVAL_GAMES)
    p.add_argument("--format", type=str, default=FORMAT_ID)
    p.add_argument("--team-a", type=str, default=TEAM_A_PATH,
                   help="Learner team file (resolved relative to src/)")
    p.add_argument("--team-b", type=str, default=TEAM_B_PATH,
                   help="Opponent team file (resolved relative to src/)")
    p.add_argument("--max-concurrent-battles", type=int, default=MAX_CONCURRENT_BATTLES)
    p.add_argument("--log-every", type=int, default=LOG_EVERY)
    p.add_argument("--opponent", type=str, default="random",
                   help="'random' or 'ac:path/to/checkpoint.pt' for a frozen AC opponent")
    return p.parse_args()


def main():
    asyncio.run(train(parse_args()))


if __name__ == "__main__":
    main()
