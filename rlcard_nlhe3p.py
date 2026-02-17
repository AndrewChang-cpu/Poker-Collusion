"""
RLCard-based 3-Player No-Limit Texas Hold'em game module.

Wraps RLCard's no-limit-holdem environment to provide the same interface
that cfr.py expects:
  - deal_new_hand() -> state
  - get_current_player(state) -> int
  - get_legal_actions(state) -> list
  - get_info_key(state, player) -> str
  - is_terminal(state) -> bool
  - get_payoffs(state) -> list[float]
  - apply_action(state, action) -> state
  - is_chance_node(state) -> bool
  - sample_chance(state) -> state

Requirements:
    pip install rlcard
"""

import rlcard
import numpy as np


# ============================================================
# Shared environment with step_back enabled for tree traversal.
#
# IMPORTANT DESIGN NOTE:
# RLCard's env is stateful (one internal game state). MCCFR needs
# to branch the game tree: try action A, recurse, undo, try action B.
# We handle this with step_back(). The "state" object we pass around
# is just a lightweight snapshot for info extraction — the real game
# state lives inside the env.
# ============================================================

NUM_PLAYERS = 3

# Module-level env — initialized once via init_env()
_env = None


def init_env(seed=None):
    """Initialize the RLCard environment. Call once before training."""
    global _env
    config = {
        'allow_step_back': True,
        'game_num_players': NUM_PLAYERS,
    }
    if seed is not None:
        config['seed'] = seed
    _env = rlcard.make('no-limit-holdem', config=config)
    return _env


def _get_env():
    global _env
    if _env is None:
        init_env()
    return _env


# ============================================================
# State snapshot — lightweight object for passing around
# ============================================================

class RLCardState:
    """
    Lightweight snapshot of the RLCard env state at a point in time.
    The actual game state is managed by RLCard internally.
    """
    def __init__(self, player_id, raw_obs, raw_legal_actions, is_over, action_history):
        self.player_id = player_id
        self.raw_obs = raw_obs
        self.raw_legal_actions = raw_legal_actions
        self.is_over = is_over
        self.action_history = list(action_history)  # copy


def _snapshot():
    """Take a snapshot of the current env state."""
    env = _get_env()
    is_over = env.is_over()

    if is_over:
        return RLCardState(
            player_id=-1,
            raw_obs={},
            raw_legal_actions=[],
            is_over=True,
            action_history=_action_history,
        )

    player_id = env.get_player_id()
    state = env.get_state(player_id)

    return RLCardState(
        player_id=player_id,
        raw_obs=state['raw_obs'],
        raw_legal_actions=state['raw_legal_actions'],
        is_over=False,
        action_history=_action_history,
    )


# Track action history at module level (reset each hand)
_action_history = []


# ============================================================
# Game interface functions (matching what cfr.py expects)
# ============================================================

def deal_new_hand():
    """Start a new hand. Returns initial state snapshot."""
    global _action_history
    env = _get_env()
    env.reset()
    _action_history = []
    return _snapshot()


def get_current_player(state):
    """Return current player ID (0, 1, or 2)."""
    return state.player_id


def get_legal_actions(state):
    """Return list of legal action strings."""
    return [str(a) for a in state.raw_legal_actions]


def is_terminal(state):
    """Check if the hand is over."""
    return state.is_over


def get_payoffs(state):
    """Return payoffs for each player at terminal state."""
    env = _get_env()
    payoffs = env.get_payoffs()
    return list(payoffs)


def apply_action(state, action):
    """
    Apply action to the environment and return new state snapshot.

    IMPORTANT: This modifies the shared env. For MCCFR tree branching,
    cfr.py must call undo_action() to step back after recursing.
    """
    global _action_history
    env = _get_env()
    env.step(action, raw_action=True)
    _action_history.append(str(action))
    return _snapshot()


def undo_action():
    """
    Step back one action in the game tree.
    Used by MCCFR to explore multiple branches from the same state.
    """
    global _action_history
    env = _get_env()
    env.step_back()
    if _action_history:
        _action_history.pop()


def is_chance_node(state):
    """
    RLCard handles chance events (dealing) internally.
    From our perspective, there are no explicit chance nodes.
    """
    return False


def sample_chance(state):
    """No-op since RLCard handles chance internally."""
    return state


# ============================================================
# Information set key construction
# ============================================================

def _hand_bucket(hand_cards, public_cards):
    """
    Simple hand strength bucketing.

    Preflop: Based on card ranks + suitedness (10 buckets)
    Postflop: Based on hand category from RLCard's raw_obs (8 buckets)
    """
    if not public_cards:
        # Preflop bucketing by rank
        return _preflop_bucket(hand_cards)
    else:
        # Postflop: use simple rank-based heuristic
        return _postflop_bucket(hand_cards, public_cards)


def _parse_card(card_str):
    """Parse RLCard card string like 'HQ', 'S2', 'DT' -> (rank_idx, suit)."""
    suit = card_str[0]  # H, S, D, C
    rank = card_str[1:]  # 2-9, T, J, Q, K, A
    rank_order = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
                  '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    return rank_order.get(rank, 0), suit


def _preflop_bucket(hand_cards, num_buckets=10):
    """Bucket preflop hands by rank + suitedness."""
    if not hand_cards or len(hand_cards) < 2:
        return 0

    r1, s1 = _parse_card(hand_cards[0])
    r2, s2 = _parse_card(hand_cards[1])

    high = max(r1, r2)
    low = min(r1, r2)
    suited = (s1 == s2)

    score = high * 13 + low
    if high == low:
        score += 100  # pair bonus
    if suited:
        score += 20

    max_score = 12 * 13 + 12 + 100 + 20  # AA suited
    bucket = int((score / (max_score + 1)) * num_buckets)
    return min(bucket, num_buckets - 1)


def _postflop_bucket(hand_cards, public_cards, num_buckets=8):
    """
    Simple postflop bucketing by estimated hand category.
    Uses a rough heuristic based on pair/high card detection.
    For better performance, replace with a proper hand evaluator.
    """
    all_ranks = []
    for c in (hand_cards or []):
        r, _ = _parse_card(c)
        all_ranks.append(r)
    for c in (public_cards or []):
        r, _ = _parse_card(c)
        all_ranks.append(r)

    if not all_ranks:
        return 0

    from collections import Counter
    counts = Counter(all_ranks)
    max_count = max(counts.values())

    # Rough hand category scoring
    if max_count >= 4:
        category = 7  # quads
    elif max_count == 3:
        if len([v for v in counts.values() if v >= 2]) >= 2:
            category = 6  # full house
        else:
            category = 3  # trips
    elif max_count == 2:
        pairs = len([v for v in counts.values() if v == 2])
        if pairs >= 2:
            category = 2  # two pair
        else:
            category = 1  # one pair
    else:
        category = 0  # high card / possible straight/flush (simplified)

    bucket = int((category / 8.0) * num_buckets)
    return min(bucket, num_buckets - 1)


def get_info_key(state, player):
    """
    Build information set key from what the player knows:
    - Their hand (bucketed)
    - Public cards (bucketed with hand)
    - Action history
    """
    raw_obs = state.raw_obs
    if not raw_obs:
        return "terminal"

    # RLCard uses different key names across versions/games
    hand = raw_obs.get('hand', raw_obs.get('hand_cards', []))
    public = raw_obs.get('public_cards', raw_obs.get('public_card', []))

    # Normalize to list of strings
    if isinstance(hand, str):
        hand = [hand]
    if hand is None:
        hand = []
    if isinstance(public, str):
        public = [public] if public else []
    if public is None:
        public = []

    # Determine round from public card count
    num_public = len(public) if public else 0
    if num_public == 0:
        round_idx = 0  # preflop
    elif num_public <= 3:
        round_idx = 1  # flop
    elif num_public == 4:
        round_idx = 2  # turn
    else:
        round_idx = 3  # river

    bucket = _hand_bucket(hand, public)

    # Action history string (ensure all elements are strings)
    hist_str = ",".join(str(a) for a in state.action_history) if state.action_history else ""

    return f"{round_idx}|{bucket}|{hist_str}"