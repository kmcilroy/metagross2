# Minimal, readable knobs for MVP0

FORMAT_ID = "gen1ou"

TEAM_A_PATH = r"..\teams\team_a.txt"
TEAM_B_PATH = r"..\teams\team_b.txt"

# Training
TOTAL_BATTLES = 600             # keep small for MVP0
MAX_CONCURRENT_BATTLES = 1     # parallel later
STEPS_PER_BATTLE_CAP = 1000    # safety cap

# Policy / optimizer
HIDDEN_DIM = 128
LR = 1e-3
EPSILON_START = 0.20
EPSILON_END = 0.05
EPSILON_DECAY_BATTLES = 100     # linear decay over first N battles
SEED = 42
# Policy sampling / regularization
SOFTMAX_TEMPERATURE = 1.0   # lower (0.7–0.9) = peakier; higher (1.2) = more exploratory
ENTROPY_BETA = 0.01         # 0.0 disables; 0.005–0.02 is a good start

# Actor–Critic extras
VALUE_COEF = 0.5          # scale for critic MSE loss
MAX_GRAD_NORM = 1.0       # gradient clipping; set 0 or None to disable


# Turn-by-turn CSV logging
LOG_TURN_BY_TURN = True
LOG_DIR = r"..\logs"     # relative to src/; CSVs will appear in metagross\logs\

# Logging
LOG_EVERY = 1


# === TensorBoard / Eval / Checkpoint ===
TENSORBOARD_LOGDIR = r"runs\metagross"   # relative to repo root (Windows-safe)
EVAL_EVERY = 25                           # episodes
EVAL_GAMES = 10
WINRATE_ROLL = 100                        # rolling window size
WINRATE_EMA_ALPHA = 0.10
SAVE_CHECKPOINTS = True
CHECKPOINT_DIR = r"checkpoints"


LOG_TURN_BY_TURN = True   # set False to disable CSVs
LOG_DIR = r"logs"         # folder relative to repo root

