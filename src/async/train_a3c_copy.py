# Metagross A3C (async) — multi-worker training
# ------------------------------------------------
# N concurrent workers train a shared Actor–Critic model asynchronously
# (Hogwild-style). The shared model stays on CPU for shared_memory().
# Your agent can still run CUDA internally via --device cuda.

import os
import time
import argparse
import asyncio
import random
import string
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

# ------------------------- Project imports -------------------------
from src.agent2 import LearningPlayerAC, ActorCritic  # noqa: F401
from src.encoder import ENCODER_DIM
from src.utils.poke_helpers import MAX_ACTIONS

from poke_env.player.baselines import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

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
    team1: str
    team2: str
    verbose: bool


def _uname(prefix: str, wid: int) -> str:
    token = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
    raw = f"{prefix}{wid}{token}"
    name = ''.join(ch for ch in raw.lower() if ch.isalnum())
    return name[:18]


def _extract_rollout_from_agent(ac_player: LearningPlayerAC):
    """Try several ways to get a trajectory. Returns (traj_list, used_method_str)."""
    if hasattr(ac_player, "pop_rollout"):
        try:
            r = ac_player.pop_rollout()
            if isinstance(r, list) and (len(r) == 0 or isinstance(r[0], dict)):
                return r, "pop_rollout"
        except Exception:
            pass

    candidates = [
        ("_states", "_actions", "_legal_sets"),
        ("states", "actions", "legal_sets"),
        ("episode_states", "episode_actions", "episode_legal_sets"),
    ]
    for s, a, l in candidates:
        if hasattr(ac_player, s) and hasattr(ac_player, a) and hasattr(ac_player, l):
            states_raw = list(getattr(ac_player, s))
            actions = list(getattr(ac_player, a))
            legal_sets = list(getattr(ac_player, l))
            T = min(len(states_raw), len(actions), len(legal_sets))
            traj = []
            for t in range(T):
                legal = legal_sets[t] or []
                mask = [False] * MAX_ACTIONS
                for j in legal:
                    jj = int(j)
                    if 0 <= jj < MAX_ACTIONS:
                        mask[jj] = True
                traj.append({
                    "state": states_raw[t],
                    "action": int(actions[t]) if t < len(actions) else 0,
                    "reward": 0.0,
                    "done": False,
                    "mask": mask,
                })
            return traj, f"buffers:{s},{a},{l}"

    return [], "none"


async def run_episode_vs_random(ac_player: LearningPlayerAC, rnd: RandomPlayer, verbose=False, wid=0, ep=0):
    """Run one battle and reconstruct a rollout. Returns (ep_return, traj, source, won)."""
    prev_wins = int(getattr(ac_player, "n_won_battles", 0))
    await ac_player.battle_against(rnd, n_battles=1)
    post_wins = int(getattr(ac_player, "n_won_battles", 0))
    won = (post_wins - prev_wins) == 1

    traj, source = _extract_rollout_from_agent(ac_player)
    if traj:
        traj[-1]["reward"] = 1.0 if won else -1.0
        traj[-1]["done"] = True
        ep_return = float(traj[-1]["reward"])
    else:
        ep_return = 0.0  # heartbeat if buffers were empty

    if hasattr(ac_player, "_clear_buffers"):
        try:
            ac_player._clear_buffers()
        except Exception:
            pass

    if verbose:
        print(f"[W{wid}] ep={ep} win={int(won)} steps={len(traj)} src={source} return={ep_return:+.2f}")

    return ep_return, traj, source, won


def epsilon_for_episode(ep_idx: int, start: float, final: float, decay_episodes: int) -> float:
    if decay_episodes <= 0:
        return final
    frac = min(1.0, ep_idx / decay_episodes)
    return start + (final - start) * frac


def worker_entry(global_model: GlobalAC, opt: torch.optim.Optimizer, obs_dim: int, n_actions: int, cfg: WorkerConfig):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

    torch.set_num_threads(1)

    local_model = GlobalAC(state_dim=obs_dim, hidden=cfg.hidden, n_actions=n_actions)
    local_model.load_state_dict(global_model.state_dict())
    local_model.to(torch.device('cpu'))

    server = LocalhostServerConfiguration

    ac_name = _uname("mga", cfg.wid)
    rnd_name = _uname("mgr", cfg.wid)

    ac_player = LearningPlayerAC(
        epsilon=float(cfg.epsilon_start),
        lr=3e-4,
        hidden=cfg.hidden,
        account_configuration=AccountConfiguration(ac_name, None),
        server_configuration=server,
        battle_format=cfg.format,
        max_concurrent_battles=cfg.max_concurrent_battles,
        team=cfg.team1,
    )

    if cfg.device == 'cuda' and torch.cuda.is_available():
        try:
            if hasattr(ac_player, 'to_device'):
                ac_player.to_device('cuda')
            else:
                if hasattr(ac_player, 'model') and isinstance(ac_player.model, nn.Module):
                    ac_player.model.to('cuda')
                if hasattr(ac_player, 'policy') and isinstance(ac_player.policy, nn.Module):
                    ac_player.policy.to('cuda')
                if hasattr(ac_player, 'value') and isinstance(ac_player.value, nn.Module):
                    ac_player.value.to('cuda')
        except Exception:
            pass

    rnd_player = RandomPlayer(
        account_configuration=AccountConfiguration(rnd_name, None),
        server_configuration=server,
        battle_format=cfg.format,
        max_concurrent_battles=cfg.max_concurrent_battles,
        team=cfg.team2,
    )

    ent_coef = 0.01
    vf_coef = 0.5
    grad_clip = 1.0

    async def _amain():
        episodes_done = 0
        while episodes_done < cfg.episodes:
            eps = epsilon_for_episode(episodes_done, cfg.epsilon_start, cfg.epsilon_final, cfg.epsilon_decay_episodes)
            try:
                ac_player.epsilon = float(eps)
            except Exception:
                pass

            ep_return, traj, src, won = await run_episode_vs_random(ac_player, rnd_player, cfg.verbose, cfg.wid, episodes_done + 1)
            episodes_done += 1

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
                logits, values = local_model(states)
                values = values.squeeze(-1).tolist()

            adv, rets = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)

            logits, value_pred = local_model(states)
            logp_all = torch.log_softmax(logits, dim=-1)

            mask_t = None
            if masks and masks[0] is not None:
                mask_t = torch.tensor(masks, dtype=torch.bool)
                logp_all = torch.where(mask_t, logp_all, torch.full_like(logp_all, -1e9))

            logp_taken = logp_all.gather(1, actions.view(-1, 1)).squeeze(1)
            policy_loss = -(logp_taken * adv.detach()).mean()
            value_loss = ((value_pred.squeeze(1) - rets)**2).mean()

            with torch.no_grad():
                p_all = torch.softmax(logits, dim=-1)
                if mask_t is not None:
                    p_all = p_all * mask_t.float()
                    p_all = p_all / (p_all.sum(dim=-1, keepdim=True) + 1e-8)
                logp = torch.log(p_all.clamp_min(1e-8))
                entropy = -(p_all * logp).sum(dim=-1).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            opt.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(local_model.parameters(), grad_clip))

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
    parser.add_argument('--team1', type=str, default=None, help='Path to Showdown export team for the learner')
    parser.add_argument('--team2', type=str, default=None, help='Path to Showdown export team for the opponent')
    parser.add_argument('--verbose', action='store_true', help='Print a per-episode line from each worker')
    args = parser.parse_args()

    # Normalize Windows path so TB finds it consistently
    args.logdir = args.logdir.replace("\\", "/")

    def _read_team_or_default(path_hint, fallback_rel):
        if path_hint is not None:
            with open(path_hint, 'r', encoding='utf-8') as f:
                return f.read()
        try:
            with open(fallback_rel, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                "No team provided. Pass --team1 and --team2 (paths to Showdown-exported teams) "
                "or create defaults at ./teams/gen1ou/learner.txt and ./teams/gen1ou/opponent.txt"
            )

    team1_str = _read_team_or_default(args.team1, './teams/gen1ou/learner.txt')
    team2_str = _read_team_or_default(args.team2, './teams/gen1ou/opponent.txt')

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    global_model = GlobalAC(state_dim=args.obs_dim, hidden=args.hidden, n_actions=args.n_actions)
    global_model.share_memory()
    opt = SharedAdam(global_model.parameters(), lr=3e-4)

    # TensorBoard writer + aggregator state
    writer = SummaryWriter(args.logdir, flush_secs=10)
    log_queue: mp.Queue = mp.Queue(maxsize=1000)

    os.makedirs("checkpoints", exist_ok=True)

    agg = {
        "global_step": 0,           # monotonic step for TB
        "episodes": 0,              # total across workers
        "wins": 0,
        "ret_sum": 0.0,
        "win_hist": deque(maxlen=100),
        "ret_hist": deque(maxlen=100),
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
            team1=team1_str,
            team2=team2_str,
            verbose=args.verbose,
        )
        p = mp.Process(
            target=worker_entry,
            args=(global_model, opt, args.obs_dim, args.n_actions, cfg),
            daemon=False
        )
        p.start()
        procs.append(p)

    try:
        while any(p.is_alive() for p in procs):
            try:
                rec = log_queue.get(timeout=1.0)
                # Update aggregate stats (skip NaN records)
                if not np.isnan(rec['return']):
                    agg["episodes"] += 1
                    agg["wins"] += int(rec.get("win", 0))
                    agg["ret_sum"] += float(rec['return'])
                    agg["win_hist"].append(int(rec.get("win", 0)))
                    agg["ret_hist"].append(float(rec['return']))

                # Monotonic global step
                agg["global_step"] += 1
                gs = agg["global_step"]

                # Per-episode scalars
                writer.add_scalar('train/episode_return', rec['return'], gs)
                if not np.isnan(rec['policy_loss']):
                    writer.add_scalar('loss/policy', rec['policy_loss'], gs)
                    writer.add_scalar('loss/value', rec['value_loss'], gs)
                    writer.add_scalar('stats/entropy', rec['entropy'], gs)
                    writer.add_scalar('stats/grad_norm', rec.get('grad_norm', float('nan')), gs)
                writer.add_scalar('stats/epsilon', rec['epsilon'], gs)
                writer.add_scalar('train/win', rec.get('win', 0), gs)
                writer.add_scalar('train/steps', rec.get('steps', 0), gs)

                # Aggregate scalars
                if agg["episodes"] > 0:
                    writer.add_scalar('agg/winrate_cumulative', agg["wins"] / agg["episodes"], gs)
                    writer.add_scalar('agg/return_avg_cumulative', agg["ret_sum"] / agg["episodes"], gs)
                if len(agg["win_hist"]) > 0:
                    writer.add_scalar('agg/winrate_rolling100', np.mean(agg["win_hist"]), gs)
                    writer.add_scalar('agg/return_rolling100', np.mean(agg["ret_hist"]), gs)

                # Optional console echo
                if args.verbose:
                    print(f"[TB] wid={rec['wid']} ep={rec['episode']} ret={rec['return']} "
                          f"win={rec.get('win',0)} pol={rec['policy_loss']} val={rec['value_loss']} "
                          f"ent={rec['entropy']} eps={rec['epsilon']} gnorm={rec.get('grad_norm','-')}")

                # Periodic checkpoint every 50 *valid* episodes
                if agg["episodes"] % 50 == 0 and agg["episodes"] > 0:
                    ckpt_path = f"checkpoints/a3c_global_ep{agg['episodes']}.pt"
                    torch.save(global_model.state_dict(), ckpt_path)
            except Exception:
                pass
            time.sleep(0.1)
    finally:
        for p in procs:
            if p.is_alive():
                p.join(timeout=1)
        writer.flush()
        writer.close()


if __name__ == '__main__':
    main()
