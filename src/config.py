# Minimal, readable knobs for MVP0

FORMAT_ID = "gen1ou"

# Paths use forward slashes (cross-platform via pathlib).
# TEAM_*_PATH is resolved relative to src/ by train_mvp0.load_team().
TEAM_A_PATH = "../teams/team_a.txt"
TEAM_B_PATH = "../teams/team_b.txt"

# Training
TOTAL_BATTLES = 600             # keep small for MVP0
MAX_CONCURRENT_BATTLES = 1     # parallel later

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

# Console logging
LOG_EVERY = 1

# Per-turn CSV logging (one file per episode under LOG_DIR)
LOG_TURN_BY_TURN = True
LOG_DIR = "logs"

# TensorBoard / Eval / Checkpoint (paths relative to launch directory)
TENSORBOARD_LOGDIR = "runs/metagross"
EVAL_EVERY = 25                           # episodes
EVAL_GAMES = 10
SAVE_CHECKPOINTS = True
CHECKPOINT_DIR = "checkpoints"
