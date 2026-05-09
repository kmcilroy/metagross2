# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Metagross trains a Pokémon Showdown battle agent against a locally-hosted Showdown server using `poke-env`. Two training entry points exist:

- **MVP0 (synchronous)**: `src/train_mvp0.py` — one battle at a time, Actor–Critic, TensorBoard + checkpoint + periodic eval. Reads its knobs from `src/config.py`.
- **A3C (async, multi-process)**: `src/async/train_a3c.py` — Hogwild-style; shared `GlobalAC` lives on CPU shared memory, N worker processes run battles concurrently and push gradients into the shared model via `SharedAdam`. Reads its knobs from `argparse` only — `src/config.py` is ignored here.

## Prerequisites

A local Pokémon Showdown server must be running before any training or smoke test — `poke-env` connects via `LocalhostServerConfiguration`. The setup scripts assume **Windows + PowerShell**; on Linux/macOS replicate the steps manually (create venv; install PyTorch from the appropriate CUDA wheel index; `pip install poke-env==0.5.2 numpy pandas tqdm`; `git clone https://github.com/smogon/pokemon-showdown.git showdown && (cd showdown && npm install)`).

`requirements.txt` is empty in source control — `scripts/setup.ps1` regenerates it on first run with `poke-env==0.5.2 numpy pandas tqdm`. PyTorch is installed separately. **`LearningPlayerAC` raises `RuntimeError` on init if CUDA is unavailable**, so a CUDA-capable PyTorch build is mandatory for both trainers (the smoke test does not require it).

## Common commands

All Python entry points are run as modules from the repo root so relative imports (`from .config import ...`) resolve.

```powershell
# First-time setup (creates .venv, installs torch+deps, clones Showdown, npm install)
scripts/setup.ps1

# Start the local Showdown server (required before training/smoke test)
scripts/start_showdown.ps1

# Smoke test: two RandomPlayers play 3 battles. Verifies Showdown + poke-env wiring. No CUDA needed.
python -m src.smoke_test

# Synchronous training (TOTAL_BATTLES, eps schedule, eval, checkpoints from src/config.py)
python -m src.train_mvp0

# Async A3C training (CLI args override defaults; see argparse in train_a3c.py)
python -m src.async.train_a3c --workers 4 --episodes-per-worker 200 --hidden 256 \
    --team1 teams/team_a.txt --team2 teams/team_b.txt --logdir runs/a3c_async

# Async training with team sampling (Hugging Face teams under teams/hf/<group>/<tier>/)
python -m src.async.train_a3c --workers 4 --hf-dir teams/hf/competitive/gen1ou --hf-refresh episode

# Plot per-turn reward distribution from CSVs written when LOG_TURN_BY_TURN=True
python scripts/analyze_logs.py --logdir logs

# View training curves
tensorboard --logdir runs/metagross           # sync run
tensorboard --logdir runs/a3c_async           # async run
```

There is no test suite, lint config, or formatter wired up. `src/smoke_test.py` is the closest thing to an integration check — running it after any change to encoding, action wiring, or poke-env upgrades is a good smell test.

## Architecture

### State, action, reward triple
- **State** (`src/encoder.py`): a fixed 13-dim float vector (`ENCODER_DIM`) — HP fractions for both actives, opp-HP-unknown flag, fainted counts, paralysis/sleep flags, 4-slot move-availability mask, switch count. Designed to be gen-agnostic for MVP0.
- **Action space** (`src/utils/poke_helpers.py`): a **fixed 10-slot space** (`MAX_ACTIONS`). Slots 0–3 = moves indexed into `battle.available_moves`; slots 4–9 = switches indexed into `battle.available_switches`. Per turn, `enumerate_legal_indices(battle)` returns the legal subset; `order_from_index(player, battle, idx)` maps an index back to a `BattleOrder` via `player.create_order(...)`. Always train with this masking — illegal logits must be `-inf` before softmax (async trainer) or filtered through the legal list (sync agent).
- **Rewards** (`src/rewards.py`): shaped per-turn — `+3.0 * dmg_dealt − 3.0 * dmg_taken` on HP fractions, plus heal terms. The sync trainer (`train_mvp0.py`) separately adds `±10` per fainted Pokémon delta. Terminal reward in `terminal_reward(battle)`: `+100/-100/0`. **Async trainer ignores shaped rewards** — it uses `±1` for win/loss only, set on the last step of the trajectory.

### Player / model lifecycle (`LearningPlayerAC` in `src/agent2.py`)
- Extends poke-env's `Player`. The `ActorCritic` model is **lazily constructed** on the first call to `choose_move` once `state_dim` is observed. Use `ensure_model(state_dim)` to build it eagerly when you know the dim — needed by the async worker to load global weights before the first move.
- Per-turn flow: `choose_move` encodes state, builds `[state ⊕ one-hot(action)]` rows for legal actions only, samples via masked softmax (or ε-random), and appends to in-memory rollout buffers (`_states`, `_actions`, `_legal_sets`, `_step_logs`).
- Two consumption paths for those buffers:
  - **Sync (`train_mvp0.py`)**: `learn_step(reward)` is called from the per-turn hook; `optimize_after_battle(gamma)` computes returns + A2C losses, optimizer-steps, dumps `episode_XXXX.csv`, and clears buffers.
  - **Async (`train_a3c.py`)**: no `learn_step` calls; `pop_rollout()` drains the buffers into `[{state, action, reward, done, mask}, ...]` for the worker to compute the loss against the local model. The worker explicitly calls `_clear_buffers()` afterwards.
- Architecture: actor and critic are real two-hidden-layer MLPs (`Linear → ReLU → Linear → ReLU → Linear`). The actor takes `[state ⊕ one-hot(MAX_ACTIONS)]`, the critic takes `state`.

### Per-turn hook injection (sync trainer)
`src/train_mvp0.py` does not subclass anything: it sets `learner._post_request_callback = hook` where `hook` is a closure that (a) extracts the current `battle`, (b) computes `step_reward(...)` and KO deltas, (c) calls `learner.learn_step(r)`, and (d) updates `prev_you_hp/prev_opp_hp/prev_*_fainted` trackers on the learner. This is the integration point for any reward-shaping changes.

### "Fresh learner per battle" pattern
Showdown rejects duplicate usernames within a session, so **each battle creates a brand-new `LearningPlayerAC` instance** with an auto-generated username. To keep training stateful:
- **Sync trainer**: `_carry_model(from_player, to_player)` copies `model`, `optimizer`, `device`, and the lazy-init knobs (`_hidden`, `_lr`, `entropy_beta`, `value_coef`, `max_grad_norm`) from the previous learner.
- **Async trainer**: `ac_player.ensure_model(obs_dim)` then `ac_player.model.load_state_dict(global_model.state_dict())` at the top of each episode pulls the latest shared weights into the new learner. Without this sync the agent acts on random init and the rollout becomes meaningless.

### Async A3C topology
`GlobalAC` (in `train_a3c.py`) is the **shared model** — same shape as `LearningPlayerAC.ActorCritic` so state_dicts are interchangeable. It lives on CPU (required by `share_memory()`) and is shared across workers. `SharedAdam` (`src/async/shared_optim.py`) keeps optimizer moments in shared memory too. Each worker:
1. Builds a **local CPU copy** of `GlobalAC` and syncs it from the shared model.
2. Per episode: samples teams from `TeamProvider`, builds an `ac_player` via the registry, eagerly initializes its model and **loads global weights into it**, then runs one battle vs Random.
3. Drains the trajectory via `ac_player.pop_rollout()`. If empty (e.g., zero-step battle), emits a NaN log record and continues.
4. Forwards the trajectory through `local_model` (CPU tensors) for log-probs and value predictions; computes GAE, masked policy loss (illegal slots → `-inf` before log_softmax), value MSE, entropy bonus.
5. Backwards through `local_model`, **Hogwild-copies** local grads onto `global_model.parameters()`, `opt.step()`, then reloads `local_model` from `global_model` and repeats.

The launcher process drains a `mp.Queue` of per-episode log records and writes scalars/aggregates to TensorBoard. `mp.set_start_method('spawn')` is forced for cross-platform safety. Workers explicitly call `torch.set_num_threads(1)` to avoid intra-op thread contention.

### Modular swap-in points (async only)
- `src/teams/loader.py::TeamProvider` — sample teams from a directory or use fixed strings; `normalize_for_format` injects `Ability: No Ability` and strips `Nature/EVs/IVs` for `gen1*` formats. `min_mons=6` retry guard rejects malformed teams. `refresh="episode"` resamples per battle; `"once"` locks the choice via `prepare_once()`.
- `src/agents/registry.py::make_player(key, ...)` — factory keyed by `"ac"` (LearningPlayerAC) or `"random"` (RandomPlayer). Add new agents by calling `register(AgentSpec(...))` at import time.
- `src/scheduling/tournament.py` — `single_match()` yields `MatchSpec(learner_key="ac", opponent_key="random")` forever; `round_robin(keys)` enumerates all-pairs. Swap iterators here to change the matchup distribution.
- `src/psio/accounts.py::fresh_username(prefix, wid, ep)` — short unique username with a `uuid4` suffix, capped at 18 chars.

### Config & paths
`src/config.py` is the single knob panel for the sync trainer: format ID, team paths, total battles, ε schedule, hidden dim, LR, entropy/value coefficients, gradient clipping, eval cadence/games, TB logdir, checkpoint dir, per-turn CSV logging. Paths use forward slashes and `pathlib`-friendly strings — works on Linux/macOS and Windows. `TEAM_*_PATH` is resolved relative to `src/` by `train_mvp0.load_team()`; `TENSORBOARD_LOGDIR`, `CHECKPOINT_DIR`, `LOG_DIR` are relative to the launch directory.

The async trainer ignores `src/config.py` and takes everything from `argparse`.

## Conventions

- **Always run as modules**: `python -m src.train_mvp0`, never `python src/train_mvp0.py` (relative imports break).
- **Lazy model init**: in the sync path the model is built on first `choose_move` (encoder dim auto-detected). In the async path call `ensure_model(state_dim)` before the first move so weights can be loaded.
- **Carry / sync weights between battle-scoped player instances**. Sync uses `_carry_model`; async uses `ensure_model` + `load_state_dict(global_model.state_dict())`. Forgetting either resets the behavior policy every episode.
- **Action masking is mandatory**. The async trainer masks logits with `-inf` before `log_softmax`; the sync agent indexes into the legal subset directly. Unmasked softmax over all 10 slots will sample illegal actions.
- **CSV logs** land in `LOG_DIR` (default `logs/`) as `episode_XXXX.csv` when `LOG_TURN_BY_TURN=True`. They contain per-step `value`, `legal`, `action_idx`, `action_label`, `probs`, `entropy`, `reward`, `return`, `advantage` — useful for offline analysis via `scripts/analyze_logs.py`.
- **`teams/hf/`** is gitignored except for its README and `.gitignore`; downloaded HF teams populate it but are not committed.
- **Build outputs** (`runs/`, `logs/`, `checkpoints/`, `__pycache__/`, `*.pt`, `.venv/`, `showdown/`) are all in `.gitignore`. Don't commit them.
