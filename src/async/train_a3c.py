# Metagross A3C (async) — multi-worker training (modular wiring)
# ---------------------------------------------------------------
# N concurrent workers train a shared Actor–Critic model asynchronously
# (Hogwild-style). The shared model stays on CPU for shared_memory().
# Your agent can still run CUDA internally via --device cuda.

import os
import time
import argparse
import asyncio
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque
import numpy as np
import json
import queue

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# ------------------------- Project imports -------------------------
from src.encoder import ENCODER_DIM
from src.utils.poke_helpers import MAX_ACTIONS
from src.agent2 import LearningPlayerAC  # for typing only

# new modular bits
from src.teams.loader import TeamProvider            # 1) team sampling + per-gen normalization
from src.agents.registry import make_player          # 2) agent factory via registry
from src.scheduling.tournament import single_match   # 3) simple scheduler; swap for round-robin later
from src.psio.accounts import fresh_username         # 4) username jitter helper

from poke_env.player.baselines import RandomPlayer

import importlib
SharedAdam = importlib.import_module("src.async.shared_optim").SharedAdam


# ------------------------------- Global AC network -------------------------------
class GlobalAC(nn.Module):
    """Actor-Critic aligned with src.agent2.ActorCritic.

    forward(states) -> (logits_all, values)
      logits_all: [B, n_actions], values: [B, 1]
    """
    def __init__(self, state_dim: int, hidden: int = 256, n_actions: int = MAX_ACTIONS):
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)

        self.actor = nn.Sequential(
            nn.Linear(state_dim + n_actions, hidden),
            nn.ReLU(),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: torch.Tensor):
        # states: [B, state_dim]
        B = states.shape[0]
        A = self.n_actions
        s_exp = states.unsqueeze(1).expand(B, A, self.state_dim)
        onehot = torch.eye(A, device=states.device, dtype=states.dtype).unsqueeze(0).expand(B, A, A)
        sa = torch.cat([s_exp, onehot], dim=-1).reshape(B * A, self.state_dim + A)
        logits = self.actor(sa).reshape(B, A)
        values = self.critic(states)
        return logits, values


# ----------------------------- Advantage utilities -----------------------------
def compute_gae(rewards: List[float], values: List[float], dones: List[bool],
                gamma=0.99, lam=0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
    adv = [0.0] * T
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        done = dones[t]
        delta = rewards[t] + (0 if done else gamma * next_value) - values[t]
        gae = delta + (0 if done else gamma * lam) * gae
        adv[t] = gae
        next_value = values[t]
    returns = [a + v for a, v in zip(adv, values)]
    return torch.tensor(adv, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


# ----------------------------- Worker process logic -----------------------------
@dataclass
class WorkerConfig:
    wid: int
    episodes: int
    t_max: int
    epsilon_start: float
    epsilon_final: float
    epsilon_decay_episodes: int
    format: str
    max_concurrent_battles: int
    device: str
    hidden: int
    log_queue: mp.Queue
    verbose: bool
    ent_coef: float
    vf_coef: float
    grad_clip: float
    base_lr: float
    anneal_lr: bool
    # NEW: inject modular providers instead of raw strings/dirs
    team_provider: TeamProvider


def epsilon_for_episode(ep_idx: int, start: float, final: float, decay_episodes: int) -> float:
    if decay_episodes <= 0:
        return final
    frac = min(1.0, ep_idx / decay_episodes)
    return start + (final - start) * frac


async def _run_one_battle(ac_player, rnd_player, n_actions: int, verbose=False, wid=0, ep=0):
    """Run 1 battle and reconstruct a rollout from the learner (AC) buffers."""
    prev_wins = int(getattr(ac_player, "n_won_battles", 0))
    await ac_player.battle_against(rnd_player, n_battles=1)
    post_wins = int(getattr(ac_player, "n_won_battles", 0))
    won = (post_wins - prev_wins) == 1

    # best-effort rollout extraction
    if hasattr(ac_player, "pop_rollout"):
        try:
            traj = ac_player.pop_rollout()
        except Exception:
            traj = []
    else:
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
        print(f"[W{wid}] ep={ep} win={int(won)} steps={len(traj)} src={'pop_rollout' if traj else 'none'} return={ep_return:+.2f}")

    return ep_return, traj, won


def worker_entry(global_model: GlobalAC, opt: torch.optim.Optimizer,
                 obs_dim: int, n_actions: int, cfg: WorkerConfig):
    # Windows event loop policy (no-op elsewhere)
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

    torch.set_num_threads(1)

    local_model = GlobalAC(state_dim=obs_dim, hidden=cfg.hidden, n_actions=n_actions)
    local_model.load_state_dict(global_model.state_dict())
    local_model.to(torch.device('cpu'))

    # simple perpetual schedule: AC vs Random each episode (swappable later)
    match_iter = single_match()

    async def _amain():
        episodes_done = 0

        # If provider wants to preselect teams once, do it now
        cfg.team_provider.prepare_once()

        while episodes_done < cfg.episodes:
            # Anneal LR across this worker's budget
            if cfg.anneal_lr and cfg.base_lr > 0:
                progress = max(0.0, min(1.0, episodes_done / max(1, cfg.episodes)))
                new_lr = cfg.base_lr * (1.0 - progress)
                for g in opt.param_groups:
                    g['lr'] = new_lr

            eps = epsilon_for_episode(episodes_done, cfg.epsilon_start, cfg.epsilon_final, cfg.epsilon_decay_episodes)

            # Schedule + teams
            spec = next(match_iter)  # has .learner_key / .opponent_key
            team1_str, team1_path, team2_str, team2_path = cfg.team_provider.sample()
            if cfg.verbose and (team1_path or team2_path):
                print(f"[W{cfg.wid}] teams: L={team1_path or '(fixed)'} | O={team2_path or '(fixed)'}")

            # Build players via registry (handles account/server/format/etc.)
            ac_player = make_player(
                spec.learner_key,
                username=fresh_username("mga", cfg.wid, episodes_done + 1),
                team=team1_str,
                battle_format=cfg.format,
                max_concurrent_battles=cfg.max_concurrent_battles,
                extra={"epsilon": float(eps), "lr": 3e-4, "hidden": cfg.hidden},
            )
            rnd_player = make_player(
                spec.opponent_key,
                username=fresh_username("mgr", cfg.wid, episodes_done + 1),
                team=team2_str,
                battle_format=cfg.format,
                max_concurrent_battles=cfg.max_concurrent_battles,
            )

            # Optional CUDA push for the learner's internals
            if cfg.device == 'cuda' and torch.cuda.is_available():
                try:
                    if hasattr(ac_player, 'to_device'):
                        ac_player.to_device('cuda')
                    else:
                        for attr in ('model', 'policy', 'value'):
                            m = getattr(ac_player, attr, None)
                            if isinstance(m, nn.Module):
                                m.to('cuda')
                except Exception:
                    pass

            # ---- One battle ----
            ep_return, traj, won = await _run_one_battle(
                ac_player, rnd_player, n_actions, cfg.verbose, cfg.wid, episodes_done + 1
            )
            episodes_done += 1

            # ---- If no buffers, send NaNs to keep TB heartbeat ----
            if not traj:
                cfg.log_queue.put({
                    'wid': cfg.wid,
                    'episode': episodes_done,
                    'return': float('nan'),
                    'policy_loss': float('nan'),
                    'value_loss': float('nan'),
                    'entropy': float('nan'),
                    'epsilon': eps,
                    'win': int(won),
                    'steps': 0,
                    'grad_norm': float('nan'),
                })
                continue

            # ---- Train on trajectory with local model ----
            raw_states = [np.asarray(step['state'], dtype=np.float32).reshape(-1) for step in traj]
            feat_dim = int(raw_states[0].shape[0])
            if feat_dim != local_model.state_dim:
                raise RuntimeError(
                    f"State dim mismatch: rollout={feat_dim} vs model={local_model.state_dim}. "
                    f"Run with --obs-dim {feat_dim} or modify your agent to store encoded states."
                )
            states = torch.from_numpy(np.stack(raw_states, axis=0))
            actions = torch.tensor([step['action'] for step in traj], dtype=torch.long)
            rewards = [step['reward'] for step in traj]
            dones = [step.get('done', False) for step in traj]
            masks = [step.get('mask') for step in traj]

            with torch.no_grad():
                _, values = local_model(states)
                values = values.squeeze(-1).tolist()

            # GAE + advantage normalization (variance reduction)
            adv, rets = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Forward for loss
            logits, value_pred = local_model(states)

            # Mask BEFORE softmax: set illegal logits to -inf, and guard empty rows
            if masks and masks[0] is not None:
                mask_t = torch.tensor(masks, dtype=torch.bool)
                row_has = mask_t.any(dim=1)
                if not bool(row_has.all()):
                    safe = mask_t.clone()
                    safe[~row_has] = True  # if a row has no legal actions, treat all as legal
                    mask_t = safe
                logits = logits.masked_fill(~mask_t, float('-inf'))

            logp_all = F.log_softmax(logits, dim=-1)
            logp_taken = logp_all.gather(1, actions.view(-1, 1)).squeeze(1)
            policy_loss = -(logp_taken * adv.detach()).mean()

            value_loss = ((value_pred.squeeze(1) - rets)**2).mean()

            with torch.no_grad():
                p_all = torch.softmax(logits, dim=-1)
                entropy = -(p_all * torch.log(p_all.clamp_min(1e-8))).sum(dim=-1).mean()

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            # Hogwild: push local grads into shared global model
            opt.zero_grad()
            if hasattr(local_model, 'zero_grad'):
                local_model.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(local_model.parameters(), cfg.grad_clip))

            for gp, lp in zip(global_model.parameters(), local_model.parameters()):
                if lp.grad is None:
                    continue
                if gp.grad is None:
                    gp.grad = lp.grad.clone()
                else:
                    gp.grad.copy_(lp.grad)
            opt.step()

            # Sync local with updated global
            local_model.load_state_dict(global_model.state_dict())

            cfg.log_queue.put({
                'wid': cfg.wid,
                'episode': episodes_done,
                'return': float(ep_return),
                'policy_loss': float(policy_loss.item()),
                'value_loss': float(value_loss.item()),
                'entropy': float(entropy.item()),
                'epsilon': eps,
                'win': int(won),
                'steps': len(traj),
                'grad_norm': grad_norm,
            })

    # Run the async loop in this worker
    asyncio.run(_amain())


# ----------------------------------- Launcher -----------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--episodes-per-worker', type=int, default=200)
    parser.add_argument('--t-max', type=int, default=128, help='n-step batch size (future use)')
    parser.add_argument('--epsilon-start', type=float, default=0.20)
    parser.add_argument('--epsilon-final', type=float, default=0.05)
    parser.add_argument('--epsilon-decay-episodes', type=int, default=200)
    parser.add_argument('--format', type=str, default='gen1ou')
    parser.add_argument('--max-concurrent-battles', type=int, default=1)
    parser.add_argument('--obs-dim', type=int, default=ENCODER_DIM, help='encoder output dim')
    parser.add_argument('--n-actions', type=int, default=MAX_ACTIONS, help='fixed action space size')
    parser.add_argument('--hidden', type=int, default=256, help='hidden width for actor/critic (global/local/agent)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--logdir', type=str, default='runs/a3c_async')

    # Team source (TeamProvider will use these)
    parser.add_argument('--team1', type=str, default=None, help='Path to fixed learner team file (Showdown export)')
    parser.add_argument('--team2', type=str, default=None, help='Path to fixed opponent team file (Showdown export)')
    parser.add_argument('--hf-dir', type=str, default=None, help='Directory to sample teams for BOTH players')
    parser.add_argument('--hf-dir-learner', type=str, default=None, help='Directory to sample teams for the learner')
    parser.add_argument('--hf-dir-opponent', type=str, default=None, help='Directory to sample teams for the opponent')
    parser.add_argument('--hf-refresh', type=str, choices=['episode', 'once'], default='episode')

    # Loss / opt
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--opt-eps', type=float, default=1e-8)
    parser.add_argument('--ent-coef', type=float, default=0.005)
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--grad-clip', type=float, default=0.5)
    parser.add_argument('--anneal-lr', action='store_true')

    parser.add_argument('--run-name', type=str, default='auto')
    parser.add_argument('--verbose', action='store_true', help='Print a per-episode line from each worker')


    args = parser.parse_args()
    args.logdir = args.logdir.replace("\\", "/")

    # --- unique run directory so TB shows a new line per run ---
    def _sanitize(s: str) -> str:
        return ''.join(ch for ch in s if ch.isalnum() or ch in ('-', '_'))

    ts = time.strftime("%Y%m%d-%H%M%S")
    if getattr(args, "run_name", "auto").lower() == 'auto':
        run_id = f"{ts}_w{args.workers}_obs{args.obs_dim}_na{args.n_actions}_hid{args.hidden}_lr{args.lr:g}"
    else:
        run_id = _sanitize(args.run_name)

    RUN_DIR = os.path.join(args.logdir, run_id).replace("\\", "/")
    CKPT_DIR = os.path.join(RUN_DIR, "checkpoints").replace("\\", "/")
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"[INFO] TensorBoard run dir: {RUN_DIR}")

    # ---- Build TeamProvider from args (HF dirs take precedence) ----
    from pathlib import Path
    hf_dir_learner = args.hf_dir_learner or args.hf_dir
    hf_dir_opponent = args.hf_dir_opponent or args.hf_dir

    fixed_learner = None
    fixed_opponent = None
    if not (hf_dir_learner or hf_dir_opponent):
        # fixed file mode
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

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    global_model = GlobalAC(state_dim=args.obs_dim, hidden=args.hidden, n_actions=args.n_actions)
    global_model.share_memory()

    # SharedAdam may or may not accept eps; try both
    try:
        opt = SharedAdam(global_model.parameters(), lr=args.lr, eps=args.opt_eps)
    except TypeError:
        opt = SharedAdam(global_model.parameters(), lr=args.lr)

    # TensorBoard writer + aggregator state
    writer = SummaryWriter(RUN_DIR, flush_secs=10)

    # Bootstrap TB so an event file exists even if workers crash early
    writer.add_text('run/args', json.dumps(vars(args), indent=2))
    writer.add_scalar('meta/started', 1, 0)
    writer.flush()

    log_queue: mp.Queue = mp.Queue(maxsize=1000)

    agg = {
        "global_step": 0,           # monotonic step for TB
        "episodes": 0,              # total across workers (valid ones)
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
            t_max=args.t_max,
            epsilon_start=args.epsilon_start,
            epsilon_final=args.epsilon_final,
            epsilon_decay_episodes=args.epsilon_decay_episodes,
            format=args.format,
            max_concurrent_battles=args.max_concurrent_battles,
            device=args.device,
            hidden=args.hidden,
            log_queue=log_queue,
            verbose=args.verbose,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            grad_clip=args.grad_clip,
            base_lr=args.lr,
            anneal_lr=args.anneal_lr,
            team_provider=team_provider,
        )
        p = mp.Process(
            target=worker_entry,
            args=(global_model, opt, args.obs_dim, args.n_actions, cfg),
            daemon=False
        )
        p.start()
        procs.append(p)

    try:
        # Keep draining logs until all workers are done AND the queue is empty
        while any(p.is_alive() for p in procs) or not log_queue.empty():
            try:
                rec = log_queue.get(timeout=1.0)
            except queue.Empty:
                time.sleep(0.1)
                continue
            except Exception as e:
                print(f"[WARN] log_queue error: {e}")
                time.sleep(0.1)
                continue

            # ----------------- process one training record -----------------
            ret_val = rec.get('return', float('nan'))
            pol_val = rec.get('policy_loss', float('nan'))
            val_val = rec.get('value_loss', float('nan'))
            ent_val = rec.get('entropy', float('nan'))

            # Update aggregate stats (skip NaN returns)
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

            # Monotonic global step
            agg["global_step"] += 1
            gs = agg["global_step"]

            # Per-episode scalars
            writer.add_scalar('train/episode_return', ret_val, gs)
            if not np.isnan(pol_val):
                writer.add_scalar('loss/policy', pol_val, gs)
                writer.add_scalar('loss/value', val_val, gs)
                writer.add_scalar('stats/entropy', ent_val, gs)
                writer.add_scalar('stats/grad_norm', rec.get('grad_norm', float('nan')), gs)
            writer.add_scalar('stats/epsilon', rec.get('epsilon', float('nan')), gs)
            writer.add_scalar('train/win', rec.get('win', 0), gs)
            writer.add_scalar('train/steps', rec.get('steps', 0), gs)

            # Aggregate scalars
            if agg["episodes"] > 0:
                writer.add_scalar('agg/winrate_cumulative', agg["wins"] / agg["episodes"], gs)
                writer.add_scalar('agg/return_avg_cumulative', agg["ret_sum"] / agg["episodes"], gs)
            if len(agg["win_hist"]) > 0:
                writer.add_scalar('agg/winrate_rolling100', np.mean(agg["win_hist"]), gs)
                writer.add_scalar('agg/return_rolling100', np.mean(agg["ret_hist"]), gs)
            if len(agg["pl_hist"]) > 0:
                writer.add_scalar('agg/policy_loss_ma100', np.mean(agg["pl_hist"]), gs)
                writer.add_scalar('agg/value_loss_ma100', np.mean(agg["vl_hist"]), gs)
                writer.add_scalar('agg/entropy_ma100', np.mean(agg["ent_hist"]), gs)

            # Periodic checkpoint every 50 *valid* episodes
            if agg["episodes"] > 0 and agg["episodes"] % 50 == 0:
                ckpt_path = os.path.join(CKPT_DIR, f"a3c_global_ep{agg['episodes']}.pt").replace("\\", "/")
                torch.save(global_model.state_dict(), ckpt_path)

            time.sleep(0.05)
    finally:
        # Report worker exit codes to aid debugging
        for i, p in enumerate(procs):
            if p.exitcode not in (0, None):
                print(f"[WARN] Worker {i} exited with code {p.exitcode}")
        for p in procs:
            if p.is_alive():
                p.join(timeout=1)
        writer.flush()
        writer.close()


if __name__ == '__main__':
    main()
