"""
Central configuration: game parameters, bucket counts, CFR hyperparameters.
"""

# Game parameters (from PROJECT_FORMULATION)
NUM_PLAYERS = 3
STARTING_STACK_BB = 20
SMALL_BLIND_BB = 0.5
BIG_BLIND_BB = 1.0
INITIAL_POT_BB = SMALL_BLIND_BB + BIG_BLIND_BB  # 1.5

# Preflop acting order: BTN(0), SB(1), BB(2)
# Postflop acting order: SB(1), BB(2), BTN(0)
PREFLOP_ORDER = [0, 1, 2]
POSTFLOP_ORDER = [1, 2, 0]

# Action abstraction: 10 actions per round
NUM_ACTIONS = 10

# Information abstraction: bucket counts per round
PREFLOP_BUCKETS = 15
FLOP_BUCKETS = 50
TURN_BUCKETS = 50
RIVER_BUCKETS = 50

# CFR training
T_MAX_DEFAULT = 100_000
LOG_INTERVAL = 1_000
USE_LINEAR_CFR = True
LINEAR_CFR_CUTOFF = 100_000
PRUNE_THRESHOLD = -10_000_000
PRUNE_WARM_UP_ITERATIONS = 10_000
PRUNE_SKIP_PROBABILITY = 0.95

# Team objective hyperparameters
SMOOTH_LAMBDA: float = 1.0
RISK_OFFSET: float = 21.0   # must exceed max loss (STARTING_STACK_BB = 20)

# Bucket table paths (relative to project root or data/)
DEFAULT_BUCKET_DIR = "data"
PREFLOP_BUCKETS_FILE = "preflop_buckets.pkl"
FLOP_BUCKETS_FILE = "flop_buckets.pkl"
TURN_BUCKETS_FILE = "turn_buckets.pkl"
RIVER_BUCKETS_FILE = "river_buckets.pkl"

# Online search (Phase 2)
SUBGAME_DEPTH_LIMIT = 3
BIAS_FACTOR = 4.0
SUBGAME_LEAF_ROLLOUTS = 10
SUBGAME_CFR_ITERATIONS = 200

# Parallel training defaults
PARALLEL_WORKERS: int = 4
PARALLEL_BATCH_SIZE: int = 24   # traversals per logical iteration; must be multiple of NUM_PLAYERS (3)

# Evaluation
EVAL_HANDS_DEFAULT = 1
# EVAL_HANDS_DEFAULT = 50_000
EVAL_BLOCK_SIZE = 500
