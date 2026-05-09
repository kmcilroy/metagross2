# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Metagross trains a Pokémon Showdown battle agent against a locally-hosted Showdown server using `poke-env`. There are two trainers; both share the model registry, A2C loss, and TensorBoard scalar layout, so swapping architectures or comparing runs across trainers Just Works.

- **MVP0 (synchronous)**: `src/train_mvp0.py` — one battle at a time, shaped per-turn rewards, periodic eval, best-checkpoint by eval win-rate. Defaults from `src/config.py`; everything overridable from CLI.
- **A3C (async, multi-process)**: `src/async/train_a3c.py` — Hogwild-style; shared `make_model(...)` instance lives on CPU shared memory, N workers run battles concurrently and push gradients via `SharedAdam`. Configured purely via CLI.

## Prerequisites

A local Pokémon Showdown server must be running before any training or smoke test — `poke-env` connects via `LocalhostServerConfiguration`. The setup scripts assume **Windows + PowerShell**; on Linux/macOS replicate the steps manually (create venv; install PyTorch from the appropriate CUDA wheel index; `pip install poke-env==0.5.2 numpy pandas tqdm`; `git clone https://github.com/smogon/pokemon-showdown.git showdown && (cd showdown && npm install)`).

`requirements.txt` is empty in source control — `scripts/setup.ps1` regenerates it on first run with `poke-env==0.5.2 numpy pandas tqdm`. PyTorch is installed separately. **`LearningPlayerAC` raises `RuntimeError` on init if CUDA is unavailable**, so a CUDA-capable PyTorch build is mandatory for both trainers (the smoke test does not require it).

## Common commands

All Python entry points are run as modules from the repo root.

```powershell
# First-time setup (creates .venv, installs torch+deps, clones Showdown, npm install)
scripts/setup.ps1

# Start the local Showdown server (required before training/smoke test)
scripts/start_showdown.ps1

# Smoke test: two RandomPlayers play 3 battles. Verifies Showdown + poke-env wiring. No CUDA needed.
python -m src.smoke_test

# Synchronous training, defaults from config.py
python -m src.train_mvp0

# Synchronous training, swap architecture and tweak a few knobs
python -m src.train_mvp0 --arch mlp_1h --hidden 64 --total-battles 300 --logdir runs/exp_baseline
python -m src.train_mvp0 --arch mlp_2h --hidden 128 --total-battles 300 --logdir runs/exp_baseline
python -m src.train_mvp0 --arch mlp_direct --hidden 256 --total-battles 300 --logdir runs/exp_baseline

# Synchronous training against a frozen prior checkpoint
python -m src.train_mvp0 --opponent ac:checkpoints/best.pt --logdir runs/self_play_step1

# Async A3C training
python -m src.async.train_a3c --workers 4 --episodes-per-worker 200 --hidden 256 \
    --team1 teams/team_a.txt --team2 teams/team_b.txt --logdir runs/a3c_async

# Async with team sampling (Hugging Face teams under teams/hf/<group>/<tier>/)
python -m src.async.train_a3c --workers 4 --hf-dir teams/hf/competitive/gen1ou --hf-refresh episode

# Plot per-turn reward distribution from CSVs (sync only; async doesn't write per-turn CSVs)
python scripts/analyze_logs.py --logdir logs

# View training curves; same TB scalar names mean sync/async runs overlay cleanly
tensorboard --logdir runs
```

There is no test suite, lint config, or formatter. `src/smoke_test.py` is the closest thing to an integration check — running it after any change to encoding, action wiring, or poke-env upgrades is a good smell test.

## Architecture

### State, action, reward triple
- **State** (`src/encoder.py`): a fixed 13-dim float vector (`ENCODER_DIM`) — HP fractions for both actives, opp-HP-unknown flag, fainted counts, paralysis/sleep flags, 4-slot move-availability mask, switch count.
- **Action space** (`src/utils/poke_helpers.py`): a **fixed 10-slot space** (`MAX_ACTIONS`). Slots 0–3 = moves indexed into `battle.available_moves`; slots 4–9 = switches indexed into `battle.available_switches`. `enumerate_legal_indices(battle)` returns the per-turn legal subset; `order_from_index(player, battle, idx)` maps a slot back to a `BattleOrder`. **Action masking is mandatory** — illegal logits must be `-inf` before softmax. The unified loss does this automatically.
- **Rewards** (`src/rewards.py`): shaped per-turn — `+3.0 * dmg_dealt − 3.0 * dmg_taken` on HP fractions plus heal terms. The sync trainer adds `±10` per fainted-Pokémon delta, terminal `±100`. **Async trainer ignores shaped rewards** — it uses `±1` for win/loss only, set on the last step of the trajectory.

### Model registry (`src/models.py`)

Single source of truth for architectures. All models implement:

```python
forward(states: Tensor[B, state_dim]) -> (logits: Tensor[B, n_actions], values: Tensor[B])
```

Built-ins: `mlp_2h` (default, two-hidden-layer state⊕one-hot actor + state→V critic), `mlp_1h` (single-hidden-layer ablation), `mlp_direct` (state→[A] logits actor, no per-action expansion — faster, but state_dict is **not** compatible with the other two).

To add a new arch:

1. Subclass `nn.Module` with the forward contract above.
2. Add an entry to `_REGISTRY` at the bottom of `src/models.py`.
3. Pass `--arch <name>` to either trainer.

### Shared loss (`src/training/loss.py`)

Both trainers drive their inner step through `a2c_loss(model, states, actions, masks, returns, advantages=None, value_coef, entropy_coef)`. It returns `{total, policy_loss, value_loss, entropy, advantages, values, probs}` — the gradient-attached `total` for backprop, detached scalars for TB, and detached per-step tensors for the sync agent's CSV row enrichment. `compute_gae`, `discounted_returns`, and `legal_mask_from_lists` live alongside it.

A change to the A2C loss formula lands in one place and propagates to both trainers.

### Player / model lifecycle (`LearningPlayerAC` in `src/agent2.py`)
- Extends poke-env's `Player`. The model is **lazily constructed** on the first call to `choose_move` once `state_dim` is observed. Use `ensure_model(state_dim)` to build it eagerly when you know the dim — needed by the async worker to load global weights before the first move, and by `FrozenAcOpponent` to load checkpoint weights.
- Per-turn flow: `choose_move` encodes state, runs **one forward pass** through the model (returns logits over all 10 slots and the value estimate), masks illegal slots to `-inf`, samples (or ε-randoms), appends to rollout buffers (`_states`, `_actions`, `_legal_sets`, `_step_logs`).
- Two consumption paths for those buffers:
  - **Sync (`train_mvp0.py`)**: `learn_step(reward)` is called from the per-turn hook; `optimize_after_battle(gamma)` builds tensors, calls `a2c_loss`, optimizer-steps, dumps `episode_XXXX.csv`, clears buffers.
  - **Async (`train_a3c.py`)**: no `learn_step` calls; `pop_rollout()` drains the buffers into `[{state, action, reward, done, mask}, ...]` for the worker to compute the loss against the local model. The worker explicitly calls `_clear_buffers()` afterwards.

### Per-turn hook injection (sync trainer)
`src/train_mvp0.py` does not subclass anything: it sets `learner._post_request_callback = hook` where `hook` is a closure that (a) extracts the current `battle`, (b) computes `step_reward(...)` and KO deltas, (c) calls `learner.learn_step(r)`, (d) updates `prev_*` trackers. This is the integration point for any reward-shaping changes.

### "Fresh learner per battle" pattern
Showdown rejects duplicate usernames within a session, so each battle creates a brand-new `LearningPlayerAC` instance with an auto-generated username. To keep training stateful:
- **Sync**: `_carry_model(from_player, to_player)` copies the model, optimizer, device, and lazy-init knobs (including `_arch`).
- **Async**: `ac_player.ensure_model(obs_dim)` then `ac_player.model.load_state_dict(global_model.state_dict())` at the top of each episode pulls the latest shared weights into the new learner. Without this sync the agent acts on random init.

### Async A3C topology
The shared global model (built via `make_model(arch, ...)`) lives on CPU (required by `share_memory()`). `SharedAdam` (`src/async/shared_optim.py`) keeps optimizer moments in shared memory too. Each worker:
1. Builds a **local CPU copy** of the same model and syncs it from the global.
2. Per episode: samples teams from `TeamProvider`, builds an `ac_player` via the registry, eagerly initializes its model and **loads global weights into it**, builds the opponent via `cfg.opponent_factory`, runs one battle.
3. Drains the trajectory via `ac_player.pop_rollout()`. Empty rollout → emits a NaN log record and continues.
4. Forwards through `local_model` for log-probs and value predictions; computes GAE, `a2c_loss`.
5. Backwards through `local_model`, **Hogwild-copies** local grads onto the global model, `opt.step()`, then reloads `local_model` from the global.

Workers explicitly call `torch.set_num_threads(1)` to avoid intra-op contention. `mp.set_start_method('spawn')` is forced.

### Opponent selection (`src/agents/opponents.py`)
Both trainers parse `--opponent` once at startup via `make_opponent_factory(spec)`:
- `random` → `RandomOpponent`
- `ac:path/to/checkpoint.pt` → `FrozenAcOpponent` — reads the checkpoint, caches `{arch, hidden, state_dict}`, mints a fresh `LearningPlayerAC` (ε=0, eval mode) per battle and copies the cached weights in. Bad paths fail at parse time, before any worker forks.

Checkpoints saved by either trainer are dicts of the form `{"model": state_dict, "arch": str, "hidden": int}`. `FrozenAcOpponent` will also accept a bare state_dict (defaults to `mlp_2h` / `hidden=128`).

### Other modular swap-in points (async)
- `src/teams/loader.py::TeamProvider` — sample teams from a directory or use fixed strings; `normalize_for_format` injects `Ability: No Ability` and strips `Nature/EVs/IVs` for `gen1*` formats. `min_mons=6` retry guard rejects malformed teams. `refresh="episode"` resamples per battle; `"once"` locks the choice via `prepare_once()`.
- `src/agents/registry.py::make_player(key, ...)` — factory for the **learner** (the registry's `opponent_key` is no longer consulted; the opponent factory owns that decision).
- `src/scheduling/tournament.py` — `single_match()` yields a `MatchSpec` perpetually; `round_robin(keys)` enumerates all-pairs. Currently only `learner_key` is used.
- `src/psio/accounts.py::fresh_username(prefix, wid, ep)` — short unique username, capped at 18 chars.

### TensorBoard scalar layout (both trainers write the same names)

| Group | Scalars |
|---|---|
| `train/` | `episode_return`, `win`, `steps`, `epsilon` |
| `loss/`  | `policy`, `value`, `entropy`, `total`, `grad_norm` |
| `agg/`   | `winrate_rolling100`, `return_rolling100`, `winrate_cumulative`, `return_avg_cumulative` (async also: `policy_loss_ma100`, `value_loss_ma100`, `entropy_ma100`) |
| `eval/`  | `win_rate`, `avg_return`, `steps_avg` (sync only — async has no eval phase) |

Run dirs are timestamped and include the architecture in the name, so `tensorboard --logdir runs/` overlays sweeps as separate lines.

### Config & paths
`src/config.py` is the **defaults panel for the sync trainer**. Every constant there has a matching `--flag` on `train_mvp0`, so config edits are optional — useful for nailing down the "default experiment", but never required. Paths use forward slashes and `pathlib`-friendly strings.

The async trainer ignores `src/config.py` and takes everything from `argparse`.

## Conventions

- **Always run as modules**: `python -m src.train_mvp0`, never `python src/train_mvp0.py` (relative imports break).
- **Eager model build before loading weights**: any time you need to load a state_dict (async sync, frozen opponent, resume), call `ensure_model(state_dim)` first.
- **Carry / sync weights between battle-scoped player instances**. Sync uses `_carry_model`; async uses `ensure_model` + `load_state_dict(global_model.state_dict())`. Forgetting either resets the behavior policy every episode.
- **Action masking is mandatory** — `a2c_loss` does it for you when given a `[T, A]` boolean mask. `legal_mask_from_lists` builds the mask from per-step legal-index lists.
- **CSV logs** land in `LOG_DIR` (default `logs/`) as `episode_XXXX.csv` when `LOG_TURN_BY_TURN=True` (sync only). Per-step columns: `value`, `legal`, `action_idx`, `action_label`, `probs`, `entropy`, `reward`, `return`, `advantage`, `probs_train`. Async does not write per-turn CSVs.
- **`teams/hf/`** is gitignored except for its README and `.gitignore`; downloaded HF teams populate it but are not committed.
- **Build outputs** (`runs/`, `logs/`, `checkpoints/`, `__pycache__/`, `*.pt`, `.venv/`, `showdown/`) are all in `.gitignore`. Don't commit them.

## Adding a new training regime

A new regime (PPO, off-policy, etc.) is a new trainer file (e.g. `src/train_ppo.py`) that:

1. Constructs a model via `make_model(arch, ...)`.
2. Drives `LearningPlayerAC` through battles, draining rollouts via `pop_rollout()`.
3. Replaces `a2c_loss` with whatever loss the regime needs.
4. Writes the same `train/`, `loss/`, `agg/` TB scalars so curves overlay with existing runs.

The `models.py` factory and `agents/opponents.py` carry over unchanged.
