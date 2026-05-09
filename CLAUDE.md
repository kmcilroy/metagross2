# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Metagross trains a Pokémon Showdown battle agent against a locally-hosted Showdown server using `poke-env`. Two training entry points exist:

- **MVP0 (synchronous)**: `src/train_mvp0.py` — one battle at a time, Actor–Critic, TensorBoard + checkpoint + periodic eval.
- **A3C (async, multi-process)**: `src/async/train_a3c.py` — Hogwild-style; shared `GlobalAC` lives on CPU shared memory, N worker processes run battles concurrently and push gradients into the shared model via `SharedAdam`.

`src/agent.py` (REINFORCE `LearningPlayer`) and `src/train_mvp0_orig.py` are earlier versions kept for reference; new work should target `agent2.py` / `train_mvp0.py` / `async/train_a3c.py`.

## Prerequisites

A local Pokémon Showdown server must be running before any training or smoke test — `poke-env` connects via `LocalhostServerConfiguration`. The setup scripts assume **Windows + PowerShell**; on Linux/macOS you must replicate the steps manually (create venv, `pip install torch ...` with the appropriate index, `pip install poke-env==0.5.2 numpy pandas tqdm`, clone `https://github.com/smogon/pokemon-showdown.git` to `./showdown/`, `npm install`).

`requirements.txt` in the repo is intentionally empty — `scripts/setup.ps1` regenerates it with the canonical pin (`poke-env==0.5.2`, `numpy`, `pandas`, `tqdm`). PyTorch is installed separately from the CUDA 12.4 wheel index, not from `requirements.txt`. `LearningPlayerAC` raises on init if CUDA is unavailable, so a CUDA-capable PyTorch build is mandatory for training (the smoke test does not require it).

## Common commands

All Python entry points are run as modules from the repo root so relative imports (`from .config import ...`) resolve.

```powershell
# First-time setup (creates .venv, installs torch+deps, clones Showdown, npm install)
scripts/setup.ps1

# Start the local Showdown server (required before training/smoke test)
scripts/start_showdown.ps1

# Smoke test: two RandomPlayers play 3 battles. Verifies Showdown + poke-env wiring.
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

There is no test suite, lint config, or formatter wired up. `src/smoke_test.py` is the closest thing to an integration check.

## Architecture

### State, action, reward triple
- **State** (`src/encoder.py`): a fixed 13-dim float vector (`ENCODER_DIM`) — HP fractions for both actives, opp-HP-unknown flag, fainted counts, paralysis/sleep flags, 4-slot move-availability mask, switch count. Designed to be gen-agnostic for MVP0.
- **Action space** (`src/utils/poke_helpers.py`): a **fixed 10-slot space** (`MAX_ACTIONS`). Slots 0–3 = moves indexed into `battle.available_moves`; slots 4–9 = switches indexed into `battle.available_switches`. Per turn, `enumerate_legal_indices(battle)` returns the legal subset; `order_from_index(player, battle, idx)` maps an index back to a `BattleOrder` via `player.create_order(...)`. Always train with this masking — illegal logits must be `-inf` before softmax (see `train_a3c.py` mask handling) or filtered through the legal list (`agent2.py`).
- **Rewards** (`src/rewards.py`): shaped per-turn — `+3.0 * dmg_dealt − 3.0 * dmg_taken` on HP fractions, plus heal terms. The training loop in `train_mvp0.py` separately adds `±10` per fainted Pokémon delta. Terminal reward in `terminal_reward(battle)`: `+100/-100/0`.

### Player / model lifecycle
- `LearningPlayerAC` (`src/agent2.py`) extends poke-env's `Player`. The `ActorCritic` model is **lazily constructed** on the first call to `choose_move` once `state_dim` is observed — do not instantiate the model up-front.
- Per-turn flow: `choose_move` encodes state, builds `[state ⊕ one-hot(action)]` rows for legal actions only, samples via masked softmax (or ε-random), and appends to in-memory rollout buffers (`_states`, `_actions`, `_legal_sets`, `_step_logs`).
- `learn_step(reward)` is called by the training loop's per-turn hook to push the shaped reward; `optimize_after_battle(gamma)` computes discounted returns, A2C-style policy + value + entropy losses, gradient-clips, optimizer-steps, dumps `episode_XXXX.csv` (when `LOG_TURN_BY_TURN`), and clears buffers.

### Per-turn hook injection (sync trainer)
`src/train_mvp0.py` does not subclass anything: it sets `learner._post_request_callback = hook` where `hook` is a closure that (a) extracts the current `battle`, (b) computes `step_reward(...)` and KO deltas, (c) calls `learner.learn_step(r)`, and (d) updates `prev_you_hp/prev_opp_hp/prev_*_fainted` trackers on the learner. This is the integration point for shaping changes.

### Username & "fresh learner per battle" pattern
Showdown rejects duplicate usernames within a session, so **each battle creates a brand-new `LearningPlayerAC` instance** with an auto-generated username. To keep training stateful, `_carry_model(from_player, to_player)` copies `model`, `optimizer`, `device`, and lazy-init knobs (`_hidden`, `_lr`, `entropy_beta`, `value_coef`, `max_grad_norm`) from the previous learner. The async trainer instead uses `src/psio/accounts.fresh_username(prefix, wid, ep)` to mint short unique names with a `uuid` suffix, retrying on `|nametaken|` errors.

### Async A3C topology
`GlobalAC` (in `train_a3c.py`) is the **shared model** — it lives on CPU and is passed to each worker via `share_memory()`. `SharedAdam` (`src/async/shared_optim.py`) keeps optimizer moments in shared memory too. Each worker:
1. Builds a **local CPU copy** of `GlobalAC`, runs one battle on the AC vs. Random matchup,
2. Reconstructs the rollout via `ac_player.pop_rollout()` (best-effort; if the agent doesn't expose it, the worker emits NaN log records to keep TB heartbeating),
3. Computes GAE + advantage normalization, masked policy loss, value loss, entropy bonus,
4. **Hogwild-copies** local grads onto `global_model.parameters()` and calls `opt.step()`,
5. Reloads local weights from the global model, repeats.

The launcher process drains a `mp.Queue` of per-episode log records and writes scalars/aggregates to TensorBoard. `mp.set_start_method('spawn')` is forced for cross-platform safety. Workers explicitly `torch.set_num_threads(1)` to avoid intra-op thread contention. CUDA is opt-in per worker via `--device cuda` and only pushes the agent's *internal* nets to GPU; the shared model stays on CPU because `share_memory()` requires it.

### Modular swap-in points
The async trainer wires four small abstractions, all under `src/`:
- `teams/loader.py::TeamProvider` — sample teams from a directory or use fixed strings; `normalize_for_format` injects `Ability: No Ability` and strips `Nature/EVs/IVs` for `gen1*` formats. `min_mons=6` retry guard rejects malformed teams. `refresh="episode"` resamples per battle; `"once"` locks the choice via `prepare_once()`.
- `agents/registry.py::make_player(key, ...)` — factory keyed by `"ac"` (LearningPlayerAC) or `"random"` (RandomPlayer). Add new agents by calling `register(AgentSpec(...))` at import time.
- `scheduling/tournament.py` — `single_match()` yields `MatchSpec(learner_key="ac", opponent_key="random")` forever; `round_robin(keys)` enumerates all-pairs. Swap iterators here to change the matchup distribution.
- `psio/accounts.py::fresh_username` — username minting with retry-on-collision.

### Config & paths
`src/config.py` is the single knob panel for the sync trainer: format ID, team paths, total battles, ε schedule, hidden dim, LR, entropy/value coefficients, gradient clipping, eval cadence/games, TB logdir, checkpoint dir, per-turn CSV logging. Note: paths use Windows raw-string syntax (`r"..\teams\team_a.txt"` is resolved relative to `src/`; `r"runs\metagross"` and `r"checkpoints"` are relative to wherever the trainer is launched). On non-Windows hosts, forward slashes work too — but be careful not to mix.

The async trainer ignores `src/config.py` and takes everything from `argparse`.

## Conventions

- **Always run as modules**: `python -m src.train_mvp0`, never `python src/train_mvp0.py` (relative imports break).
- **Lazy model init**: never construct `ActorCritic`/`ActionPolicy` before observing a state — `state_dim` is only known once `encode_battle` has run on a real battle. The encoder dim is currently 13 but `agent2.py` reads it from `len(s_np)` rather than from `ENCODER_DIM`.
- **Carry weights between battle-scoped player instances** with `_carry_model`. Forgetting this resets training every episode.
- **Action masking is mandatory**. The async trainer masks logits with `-inf`; the sync agent indexes into the legal subset directly. Either approach is fine, but unmasked softmax over all 10 slots will sample illegal actions.
- **CSV logs** land in `LOG_DIR` (default `logs/`) as `episode_XXXX.csv` when `LOG_TURN_BY_TURN=True`. They contain per-step `value`, `legal`, `action_idx`, `action_label`, `probs`, `entropy`, `reward`, `return`, `advantage` — useful for offline analysis via `scripts/analyze_logs.py`.
- **`teams/hf/`** is gitignored except for its README and `.gitignore`; downloaded HF teams populate it but are not committed.
