"""
Central configuration: game parameters, bucket counts, CFR hyperparameters.
"""

# Game parameters (from PROJECT_FORMULATION)
NUM_PLAYERS = 3
STARTING_STACK_BB = 20
SMALL_BLIND_BB = 0.5
BIG_BLIND_BB = 1.0
INITIAL_POT_BB = SMALL_BLIND_BB + BIG_BLIND_BB  # 1.5

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
PRUNE_THRESHOLD = -300
PRUNE_WARM_UP_ITERATIONS = 100
PRUNE_SKIP_PROBABILITY = 0.95

# Bucket table paths (relative to project root or data/)
DEFAULT_BUCKET_DIR = "data"
PREFLOP_BUCKETS_FILE = "preflop_buckets.pkl"
FLOP_BUCKETS_FILE = "flop_buckets.pkl"
TURN_BUCKETS_FILE = "turn_buckets.pkl"
RIVER_BUCKETS_FILE = "river_buckets.pkl"

# Evaluation
EVAL_HANDS_DEFAULT = 50_000
EVAL_BLOCK_SIZE = 500
