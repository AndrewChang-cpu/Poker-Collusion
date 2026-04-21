"""
Central configuration: game parameters, bucket counts, CFR hyperparameters.
Modified for 3-player Leduc Hold'em validation.
"""

# Game parameters
NUM_PLAYERS = 3
STARTING_STACK_BB = 20
SMALL_BLIND_BB = 0.5
BIG_BLIND_BB = 1.0
INITIAL_POT_BB = SMALL_BLIND_BB + BIG_BLIND_BB  # 1.5

# Preflop acting order: BTN(0), SB(1), BB(2)
# Postflop acting order: SB(1), BB(2), BTN(0)
PREFLOP_ORDER = [0, 1, 2]
POSTFLOP_ORDER = [1, 2, 0]

# Action abstraction: 10 actions per round (Unchanged per request)
NUM_ACTIONS = 10

# Information abstraction: Leduc-specific bucket counts
# Preflop: 4 ranks (J, Q, K, A)
# Flop: 4 hole ranks * 4 board ranks = 16 combinations
PREFLOP_BUCKETS = 4
FLOP_BUCKETS = 16
TURN_BUCKETS = 0  # Not used in Leduc
RIVER_BUCKETS = 0 # Not used in Leduc

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
RISK_OFFSET: float = 21.0

# Bucket table paths (Bypassed in Leduc mode)
DEFAULT_BUCKET_DIR = "data"
PREFLOP_BUCKETS_FILE = "preflop_buckets.pkl"
FLOP_BUCKETS_FILE = "flop_buckets.pkl"

# Online search (Phase 2)
SUBGAME_DEPTH_LIMIT = 2
BIAS_FACTOR = 4.0
SUBGAME_LEAF_ROLLOUTS = 10
SUBGAME_CFR_ITERATIONS = 200

# Parallel training defaults
PARALLEL_WORKERS: int = 4
PARALLEL_BATCH_SIZE: int = 24

# Evaluation
EVAL_HANDS_DEFAULT = 10_000
EVAL_BLOCK_SIZE = 500