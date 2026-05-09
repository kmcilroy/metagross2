"""Metagross A3C (async) — multi-worker Hogwild training.

N concurrent workers train a shared Actor-Critic model asynchronously.
The shared global model lives on CPU shared memory (required by
share_memory()). Each worker keeps a local CPU copy of the same model
for its forward/backward pass and pushes gradients onto the global
model parameters; the optimizer then `step()`s on the global. The agent
itself runs on CUDA.
"""
import argparse
import asyncio
import importlib
import json
import os
import queue
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from src.agents.opponents import OpponentFactory, make_opponent_factory
from src.agents.registry import make_player
from src.encoder import ENCODER_DIM
from src.models import list_models, make_model
from src.psio.accounts import fresh_username
from src.scheduling.tournament import single_match
from src.teams.loader import TeamProvider
from src.training import a2c_loss, compute_gae
from src.utils.poke_helpers import MAX_ACTIONS

SharedAdam = importlib.import_module("src.async.shared_optim").SharedAdam


@dataclass
class WorkerConfig:
    wid: int
    episodes: int
    epsilon_start: float
    epsilon_final: float
    epsilon_decay_episodes: int
    format: str
    max_concurrent_battles: int
    hidden: int
    arch: str
    log_queue: mp.Queue
    verbose: bool
    ent_coef: float
    vf_coef: float
    grad_clip: float
    base_lr: float
    anneal_lr: bool
    team_provider: TeamProvider
    opponent_factory: OpponentFactory


def epsilon_for_episode(ep_idx: int, start: float, final: float, decay_episodes: int) -> float:
    if decay_episodes <= 0:
        return final
    frac = min(1.0, ep_idx / decay_episodes)
    return start + (final - start) * frac


async def _run_one_battle(ac_player, rnd_player, verbose=False, wid=0, ep=0):
    """Run one battle; return (ep_return, traj, won). Reward is +/-1
    on the terminal step (async ignores shaped per-turn rewards)."""
    prev_wins = int(getattr(ac_player, "n_won_battles", 0))
    await ac_player.battle_against(rnd_player, n_battles=1)
    won = (int(getattr(ac_player, "n_won_battles", 0)) - prev_wins) == 1

    traj = []
    if hasattr(ac_player, "pop_rollout"):
        try:
            traj = ac_player.pop_rollout()
        except Exception:
            traj = []

    if traj:
        traj[-1]["reward"] = 1.0 if won else -1.0
        traj[-1]["done"] = True
        ep_return = float(traj[-1]["reward"])
    else:
        ep_return = 0.0

    if hasattr(ac_player, "_clear_buffers"):
        try:
            ac_player._clear_buffers()
        except Exception:
            pass

    if verbose:
        src = "pop_rollout" if traj else "none"
        print(f"[W{wid}] ep={ep} win={int(won)} steps={len(traj)} src={src} return={ep_return:+.2f}")

    return ep_return, traj, won


def worker_entry(global_model, opt, obs_dim: int, cfg: WorkerConfig):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass

    torch.set_num_threads(1)

    local_model = make_model(cfg.arch, state_dim=obs_dim, n_actions=MAX_ACTIONS, hidden=cfg.hidden)
    local_model.load_state_dict(global_model.state_dict())
    local_model.to(torch.device("cpu"))

    match_iter = single_match()

    async def _amain():
        episodes_done = 0
        cfg.team_provider.prepare_once()

        while episodes_done < cfg.episodes:
            if cfg.anneal_lr and cfg.base_lr > 0:
                progress = max(0.0, min(1.0, episodes_done / max(1, cfg.episodes)))
                for g in opt.param_groups:
                    g["lr"] = cfg.base_lr * (1.0 - progress)

            eps = epsilon_for_episode(episodes_done, cfg.epsilon_start, cfg.epsilon_final, cfg.epsilon_decay_episodes)

            spec = next(match_iter)
            team1_str, team1_path, team2_str, team2_path = cfg.team_provider.sample()
            if cfg.verbose and (team1_path or team2_path):
                print(f"[W{cfg.wid}] teams: L={team1_path or '(fixed)'} | O={team2_path or '(fixed)'}")

            ac_player = make_player(
                spec.learner_key,
                username=fresh_username("mga", cfg.wid, episodes_done + 1),
                team=team1_str,
                battle_format=cfg.format,
                max_concurrent_battles=cfg.max_concurrent_battles,
                extra={"epsilon": float(eps), "lr": cfg.base_lr, "hidden": cfg.hidden, "arch": cfg.arch},
            )
            # Opponent comes from --opponent (random | ac:CKPT). Spec ignored
            # for opponent_key now -- factory owns the choice.
            rnd_player = cfg.opponent_factory.make_player(
                battle_format=cfg.format,
                team=team2_str,
                max_concurrent_battles=cfg.max_concurrent_battles,
                state_dim=obs_dim,
            )
            del spec  # silences unused-name warnings; we still call next() to advance the iterator

            # Sync agent's actor-critic from the shared global model so the
            # behavior policy reflects training progress (otherwise the
            # lazy-built model stays at random init).
            ac_player.ensure_model(obs_dim)
            ac_player.model.load_state_dict(global_model.state_dict())

            ep_return, traj, won = await _run_one_battle(
                ac_player, rnd_player, cfg.verbose, cfg.wid, episodes_done + 1
            )
            episodes_done += 1

            # Heartbeat record when the rollout was empty (zero-step battle, etc.)
            if not traj:
                cfg.log_queue.put({
                    "wid": cfg.wid, "episode": episodes_done,
                    "return": float("nan"), "policy_loss": float("nan"),
                    "value_loss": float("nan"), "entropy": float("nan"),
                    "loss_total": float("nan"), "epsilon": eps, "win": int(won),
                    "steps": 0, "grad_norm": float("nan"),
                })
                continue

            # ---- Train on the trajectory with the local CPU model ----
            raw_states = [np.asarray(step["state"], dtype=np.float32).reshape(-1) for step in traj]
            feat_dim = int(raw_states[0].shape[0])
            if feat_dim != local_model.state_dim:
                raise RuntimeError(
                    f"State dim mismatch: rollout={feat_dim} vs model={local_model.state_dim}. "
                    f"Run with --obs-dim {feat_dim} or change the encoder."
                )

            states = torch.from_numpy(np.stack(raw_states, axis=0))
            actions = torch.tensor([step["action"] for step in traj], dtype=torch.long)
            rewards = [step["reward"] for step in traj]
            dones = [step.get("done", False) for step in traj]

            mask_lists = [step.get("mask") for step in traj]
            if mask_lists and mask_lists[0] is not None:
                mask_t = torch.tensor(mask_lists, dtype=torch.bool)
                row_has = mask_t.any(dim=1)
                if not bool(row_has.all()):
                    mask_t = mask_t.clone()
                    mask_t[~row_has] = True  # if a row has no legal actions, treat all as legal
            else:
                mask_t = torch.ones(len(traj), local_model.n_actions, dtype=torch.bool)

            # Bootstrap V from the local model and compute GAE-normalized advantages.
            with torch.no_grad():
                _, values = local_model(states)
                values_list = values.tolist()
            adv, returns = compute_gae(rewards, values_list, dones, gamma=0.99, lam=0.95)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            losses = a2c_loss(
                local_model,
                states, actions, mask_t, returns,
                advantages=adv.detach(),
                value_coef=cfg.vf_coef,
                entropy_coef=cfg.ent_coef,
            )

            # Hogwild: backward through local, copy grads onto global, step on global.
            opt.zero_grad()
            local_model.zero_grad(set_to_none=True)
            losses["total"].backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(local_model.parameters(), cfg.grad_clip))

            for gp, lp in zip(global_model.parameters(), local_model.parameters()):
                if lp.grad is None:
                    continue
                if gp.grad is None:
                    gp.grad = lp.grad.clone()
                else:
                    gp.grad.copy_(lp.grad)
            opt.step()

            local_model.load_state_dict(global_model.state_dict())

            cfg.log_queue.put({
                "wid": cfg.wid,
                "episode": episodes_done,
                "return": float(ep_return),
                "policy_loss": float(losses["policy_loss"].item()),
                "value_loss": float(losses["value_loss"].item()),
                "entropy": float(losses["entropy"].item()),
                "loss_total": float(losses["total"].detach().item()),
                "epsilon": eps,
                "win": int(won),
                "steps": len(traj),
                "grad_norm": grad_norm,
            })

    asyncio.run(_amain())


def _sanitize(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--episodes-per-worker", type=int, default=200)
    parser.add_argument("--epsilon-start", type=float, default=0.20)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=200)
    parser.add_argument("--format", type=str, default="gen1ou")
    parser.add_argument("--max-concurrent-battles", type=int, default=1)
    parser.add_argument("--obs-dim", type=int, default=ENCODER_DIM, help="encoder output dim")
    parser.add_argument("--hidden", type=int, default=256, help="hidden width for actor/critic")
    parser.add_argument("--arch", type=str, default="mlp_2h", choices=list_models(),
                        help="model architecture from src.models")
    parser.add_argument("--logdir", type=str, default="runs/a3c_async")

    # Team source
    parser.add_argument("--team1", type=str, default=None)
    parser.add_argument("--team2", type=str, default=None)
    parser.add_argument("--hf-dir", type=str, default=None)
    parser.add_argument("--hf-dir-learner", type=str, default=None)
    parser.add_argument("--hf-dir-opponent", type=str, default=None)
    parser.add_argument("--hf-refresh", type=str, choices=["episode", "once"], default="episode")

    # Loss / opt
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--opt-eps", type=float, default=1e-8)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--anneal-lr", action="store_true")

    parser.add_argument("--opponent", type=str, default="random",
                        help="'random' or 'ac:path/to/checkpoint.pt' for a frozen AC opponent")

    parser.add_argument("--run-name", type=str, default="auto")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    args.logdir = args.logdir.replace("\\", "/")

    ts = time.strftime("%Y%m%d-%H%M%S")
    if args.run_name.lower() == "auto":
        run_id = f"{ts}_w{args.workers}_arch-{args.arch}_hid{args.hidden}_lr{args.lr:g}"
    else:
        run_id = _sanitize(args.run_name)

    RUN_DIR = os.path.join(args.logdir, run_id).replace("\\", "/")
    CKPT_DIR = os.path.join(RUN_DIR, "checkpoints").replace("\\", "/")
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"[INFO] TensorBoard run dir: {RUN_DIR}")

    hf_dir_learner = args.hf_dir_learner or args.hf_dir
    hf_dir_opponent = args.hf_dir_opponent or args.hf_dir

    fixed_learner = None
    fixed_opponent = None
    if not (hf_dir_learner or hf_dir_opponent):
        if args.team1:
            with open(args.team1, "r", encoding="utf-8") as f:
                fixed_learner = f.read()
        if args.team2:
            with open(args.team2, "r", encoding="utf-8") as f:
                fixed_opponent = f.read()

    team_provider = TeamProvider(
        fmt=args.format,
        dir_learner=Path(hf_dir_learner) if hf_dir_learner else None,
        dir_opponent=Path(hf_dir_opponent) if hf_dir_opponent else None,
        fixed_learner=fixed_learner,
        fixed_opponent=fixed_opponent,
        refresh=args.hf_refresh,
        min_mons=6,
        max_sample_tries=5,
    )

    # Parse --opponent once. FrozenAcOpponent loads the checkpoint here so
    # an invalid path fails before any worker forks.
    opponent_factory = make_opponent_factory(args.opponent)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    global_model = make_model(args.arch, state_dim=args.obs_dim, n_actions=MAX_ACTIONS, hidden=args.hidden)
    global_model.share_memory()

    try:
        opt = SharedAdam(global_model.parameters(), lr=args.lr, eps=args.opt_eps)
    except TypeError:
        opt = SharedAdam(global_model.parameters(), lr=args.lr)

    writer = SummaryWriter(RUN_DIR, flush_secs=10)
    writer.add_text("run/args", json.dumps(vars(args), indent=2))
    writer.add_scalar("meta/started", 1, 0)
    writer.flush()

    log_queue: mp.Queue = mp.Queue(maxsize=1000)

    agg = {
        "global_step": 0,
        "episodes": 0,
        "wins": 0,
        "ret_sum": 0.0,
        "win_hist": deque(maxlen=100),
        "ret_hist": deque(maxlen=100),
        "pl_hist": deque(maxlen=100),
        "vl_hist": deque(maxlen=100),
        "ent_hist": deque(maxlen=100),
    }

    procs = []
    for wid in range(args.workers):
        cfg = WorkerConfig(
            wid=wid,
            episodes=args.episodes_per_worker,
            epsilon_start=args.epsilon_start,
            epsilon_final=args.epsilon_final,
            epsilon_decay_episodes=args.epsilon_decay_episodes,
            format=args.format,
            max_concurrent_battles=args.max_concurrent_battles,
            hidden=args.hidden,
            arch=args.arch,
            log_queue=log_queue,
            verbose=args.verbose,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            grad_clip=args.grad_clip,
            base_lr=args.lr,
            anneal_lr=args.anneal_lr,
            team_provider=team_provider,
            opponent_factory=opponent_factory,
        )
        p = mp.Process(target=worker_entry, args=(global_model, opt, args.obs_dim, cfg), daemon=False)
        p.start()
        procs.append(p)

    try:
        while any(p.is_alive() for p in procs) or not log_queue.empty():
            try:
                rec = log_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WARN] log_queue error: {e}")
                continue

            ret_val = rec.get("return", float("nan"))
            pol_val = rec.get("policy_loss", float("nan"))
            val_val = rec.get("value_loss", float("nan"))
            ent_val = rec.get("entropy", float("nan"))
            tot_val = rec.get("loss_total", float("nan"))

            if not np.isnan(ret_val):
                agg["episodes"] += 1
                agg["wins"] += int(rec.get("win", 0))
                agg["ret_sum"] += float(ret_val)
                agg["win_hist"].append(int(rec.get("win", 0)))
                agg["ret_hist"].append(float(ret_val))
                if not np.isnan(pol_val):
                    agg["pl_hist"].append(float(pol_val))
                if not np.isnan(val_val):
                    agg["vl_hist"].append(float(val_val))
                if not np.isnan(ent_val):
                    agg["ent_hist"].append(float(ent_val))

            agg["global_step"] += 1
            gs = agg["global_step"]

            # Per-episode scalars (standardized: same names sync trainer writes)
            writer.add_scalar("train/episode_return", ret_val, gs)
            writer.add_scalar("train/win", rec.get("win", 0), gs)
            writer.add_scalar("train/steps", rec.get("steps", 0), gs)
            writer.add_scalar("train/epsilon", rec.get("epsilon", float("nan")), gs)
            if not np.isnan(pol_val):
                writer.add_scalar("loss/policy", pol_val, gs)
                writer.add_scalar("loss/value", val_val, gs)
                writer.add_scalar("loss/entropy", ent_val, gs)
                writer.add_scalar("loss/total", tot_val, gs)
                writer.add_scalar("loss/grad_norm", rec.get("grad_norm", float("nan")), gs)

            # Aggregate scalars
            if agg["episodes"] > 0:
                writer.add_scalar("agg/winrate_cumulative", agg["wins"] / agg["episodes"], gs)
                writer.add_scalar("agg/return_avg_cumulative", agg["ret_sum"] / agg["episodes"], gs)
            if len(agg["win_hist"]) > 0:
                writer.add_scalar("agg/winrate_rolling100", float(np.mean(agg["win_hist"])), gs)
                writer.add_scalar("agg/return_rolling100", float(np.mean(agg["ret_hist"])), gs)
            if len(agg["pl_hist"]) > 0:
                writer.add_scalar("agg/policy_loss_ma100", float(np.mean(agg["pl_hist"])), gs)
                writer.add_scalar("agg/value_loss_ma100", float(np.mean(agg["vl_hist"])), gs)
                writer.add_scalar("agg/entropy_ma100", float(np.mean(agg["ent_hist"])), gs)

            if agg["episodes"] > 0 and agg["episodes"] % 50 == 0:
                ckpt_path = os.path.join(CKPT_DIR, f"a3c_global_ep{agg['episodes']}.pt").replace("\\", "/")
                torch.save({"model": global_model.state_dict(), "arch": args.arch, "hidden": args.hidden}, ckpt_path)
    finally:
        for i, p in enumerate(procs):
            if p.exitcode not in (0, None):
                print(f"[WARN] Worker {i} exited with code {p.exitcode}")
        for p in procs:
            if p.is_alive():
                p.join(timeout=1)
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
