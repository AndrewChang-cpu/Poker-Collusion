"""
Simplified 3-Player No-Limit Texas Hold'em engine.

Features:
- Full deck, proper dealing, 4 betting rounds (preflop/flop/turn/river)
- Fixed action abstraction: fold, check/call, bet/raise half-pot, bet/raise pot, all-in
- Simple hand strength bucketing for information abstraction
- Self-contained hand evaluator (no external deps)

Stack size: 100 big blinds (BB). Small blind = 0.5 BB, Big blind = 1 BB.
"""

import numpy as np
from copy import deepcopy
from itertools import combinations
from collections import Counter


# ============================================================
# Card representation: 0-51
# suit = card // 13, rank = card % 13 (0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A)
# ============================================================

RANK_NAMES = "23456789TJQKA"
SUIT_NAMES = "shdc"

def card_str(card):
    return RANK_NAMES[card % 13] + SUIT_NAMES[card // 13]

def card_rank(card):
    return card % 13

def card_suit(card):
    return card // 13


# ============================================================
# Hand evaluator - returns a comparable hand rank tuple
# Higher tuple = better hand
# ============================================================

def evaluate_hand(cards):
    """
    Evaluate best 5-card hand from a list of cards (5-7 cards).
    Returns a tuple that can be compared: higher = better.
    """
    best = None
    card_list = list(cards)
    for combo in combinations(card_list, 5):
        score = _score_5(combo)
        if best is None or score > best:
            best = score
    return best


def _score_5(cards):
    """Score a 5-card hand. Returns comparable tuple."""
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]

    is_flush = len(set(suits)) == 1

    # Check straight
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        # Ace-low straight (A-2-3-4-5)
        if unique_ranks == [12, 3, 2, 1, 0]:
            is_straight = True
            straight_high = 3  # 5-high straight

    # Count rank frequencies
    counts = Counter(ranks)
    freq = sorted(counts.values(), reverse=True)
    # Ranks sorted by frequency then rank value
    ranks_by_freq = sorted(counts.keys(), key=lambda r: (counts[r], r), reverse=True)

    if is_straight and is_flush:
        return (8, straight_high)       # Straight flush
    if freq == [4, 1]:
        return (7, *ranks_by_freq)       # Four of a kind
    if freq == [3, 2]:
        return (6, *ranks_by_freq)       # Full house
    if is_flush:
        return (5, *ranks)               # Flush
    if is_straight:
        return (4, straight_high)        # Straight
    if freq == [3, 1, 1]:
        return (3, *ranks_by_freq)       # Three of a kind
    if freq == [2, 2, 1]:
        return (2, *ranks_by_freq)       # Two pair
    if freq == [2, 1, 1, 1]:
        return (1, *ranks_by_freq)       # One pair
    return (0, *ranks)                   # High card


# ============================================================
# Hand strength bucketing (simple information abstraction)
# ============================================================

def hand_strength_bucket(hole_cards, board_cards, num_buckets=10):
    """
    Estimate hand strength via Monte Carlo rollout and bucket it.

    For preflop: use a precomputed canonical hand ranking.
    For postflop: quick Monte Carlo estimate of equity.
    """
    if len(board_cards) == 0:
        # Preflop: bucket by canonical hand strength
        return _preflop_bucket(hole_cards, num_buckets)
    else:
        # Postflop: Monte Carlo equity estimation
        return _postflop_bucket(hole_cards, board_cards, num_buckets)


def _preflop_bucket(hole_cards, num_buckets):
    """Simple preflop bucketing by card ranks and suitedness."""
    r1, r2 = sorted([card_rank(hole_cards[0]), card_rank(hole_cards[1])], reverse=True)
    suited = card_suit(hole_cards[0]) == card_suit(hole_cards[1])

    # Simple strength score: high card combo + suited bonus + pair bonus
    score = r1 * 13 + r2
    if r1 == r2:
        score += 100  # pair bonus
    if suited:
        score += 20   # suited bonus

    # Map to bucket (0 to num_buckets-1)
    max_score = 12 * 13 + 12 + 100 + 20  # AA suited
    bucket = int((score / (max_score + 1)) * num_buckets)
    return min(bucket, num_buckets - 1)


def _postflop_bucket(hole_cards, board_cards, num_buckets, num_rollouts=10):
    """Fast hand strength estimation using current hand rank."""
    # Instead of Monte Carlo, just evaluate current hand strength directly
    # This is much faster and still provides useful bucketing
    all_cards = list(hole_cards) + list(board_cards)
    if len(all_cards) >= 5:
        hand_score = evaluate_hand(all_cards)
        # Map hand category (0-8) to bucket
        category = hand_score[0]  # 0=high card through 8=straight flush
        # Spread across buckets
        bucket = int((category / 9.0) * num_buckets)
        return min(bucket, num_buckets - 1)
    else:
        # Not enough cards yet, use simple rank-based bucket
        ranks = sorted([card_rank(c) for c in hole_cards], reverse=True)
        score = ranks[0] * 13 + ranks[1]
        bucket = int((score / (12 * 13 + 12 + 1)) * num_buckets)
        return min(bucket, num_buckets - 1)


# ============================================================
# Game State
# ============================================================

ROUNDS = ["preflop", "flop", "turn", "river"]

# Actions
FOLD = "fold"
CHECK = "check"
CALL = "call"
BET_HALF = "bet_half"   # bet/raise half pot
BET_POT = "bet_pot"     # bet/raise full pot
ALL_IN = "all_in"

ALL_ACTIONS = [FOLD, CHECK, CALL, BET_HALF, BET_POT, ALL_IN]

NUM_PLAYERS = 3
STARTING_STACK = 100  # in big blinds
SMALL_BLIND = 0.5
BIG_BLIND = 1.0


class NLHEState:
    def __init__(self):
        self.deck = list(range(52))
        self.hole_cards = [[], [], []]  # 2 cards per player
        self.board = []                  # community cards
        self.round_idx = 0               # 0=preflop, 1=flop, 2=turn, 3=river
        self.stacks = [STARTING_STACK] * NUM_PLAYERS
        self.pot = 0.0
        self.bets = [0.0] * NUM_PLAYERS  # current round bets per player
        self.active = [True] * NUM_PLAYERS  # hasn't folded
        self.all_in = [False] * NUM_PLAYERS
        self.current_player = 0
        self.actions_this_round = []     # list of (player, action) for current round
        self.history = []                # full history: list of (round, player, action)
        self.num_actions_this_round = 0
        self.last_raiser = -1
        self.done = False

    def copy(self):
        return deepcopy(self)


# ============================================================
# Game logic
# ============================================================

def deal_new_hand():
    """Deal a fresh 3-player NLHE hand."""
    state = NLHEState()
    np.random.shuffle(state.deck)

    # Deal hole cards
    idx = 0
    for p in range(NUM_PLAYERS):
        state.hole_cards[p] = [state.deck[idx], state.deck[idx + 1]]
        idx += 2
    state.deck_idx = idx  # track where we are in the deck

    # Post blinds: player 0 = dealer/button, player 1 = SB, player 2 = BB
    state.stacks[1] -= SMALL_BLIND
    state.bets[1] = SMALL_BLIND
    state.stacks[2] -= BIG_BLIND
    state.bets[2] = BIG_BLIND
    state.pot = SMALL_BLIND + BIG_BLIND

    # Preflop action starts with player 0 (UTG/button in 3-player)
    state.current_player = 0
    state.last_raiser = 2  # BB is the "raiser" for preflop purposes

    return state


def get_current_player(state):
    return state.current_player


def is_terminal(state):
    return state.done


def is_chance_node(state):
    """Check if we need to deal community cards."""
    # This is handled internally in _advance_round
    return False


def sample_chance(state):
    return state


def get_legal_actions(state):
    """Return list of legal abstract actions for current player."""
    p = state.current_player
    actions = []

    if not state.active[p] or state.all_in[p]:
        return []

    max_bet = max(state.bets)
    my_bet = state.bets[p]
    to_call = max_bet - my_bet

    if to_call > 0:
        actions.append(FOLD)
        if state.stacks[p] >= to_call:
            actions.append(CALL)
        else:
            # Can only go all-in for less
            actions.append(ALL_IN)
            return actions
    else:
        actions.append(CHECK)

    # Raise/bet options (only if we have chips beyond calling)
    chips_after_call = state.stacks[p] - to_call
    if chips_after_call > 0:
        current_pot = state.pot + to_call  # pot after we call

        half_pot = current_pot * 0.5
        full_pot = current_pot * 1.0

        if half_pot > 0 and chips_after_call >= half_pot:
            actions.append(BET_HALF)
        if full_pot > 0 and chips_after_call >= full_pot:
            actions.append(BET_POT)
        if ALL_IN not in actions:
            actions.append(ALL_IN)

    return actions


def apply_action(state, action):
    """Apply action and return new state. Handles round transitions."""
    new = state.copy()
    p = new.current_player

    max_bet = max(new.bets)
    to_call = max_bet - new.bets[p]

    if action == FOLD:
        new.active[p] = False
    elif action == CHECK:
        pass  # no money movement
    elif action == CALL:
        amount = min(to_call, new.stacks[p])
        new.stacks[p] -= amount
        new.bets[p] += amount
        new.pot += amount
    elif action == BET_HALF:
        call_amount = min(to_call, new.stacks[p])
        new.stacks[p] -= call_amount
        new.bets[p] += call_amount
        new.pot += call_amount
        # Then raise half pot
        raise_amount = min(new.pot * 0.5, new.stacks[p])
        new.stacks[p] -= raise_amount
        new.bets[p] += raise_amount
        new.pot += raise_amount
        new.last_raiser = p
    elif action == BET_POT:
        call_amount = min(to_call, new.stacks[p])
        new.stacks[p] -= call_amount
        new.bets[p] += call_amount
        new.pot += call_amount
        # Then raise full pot
        raise_amount = min(new.pot * 1.0, new.stacks[p])
        new.stacks[p] -= raise_amount
        new.bets[p] += raise_amount
        new.pot += raise_amount
        new.last_raiser = p
    elif action == ALL_IN:
        amount = new.stacks[p]
        new.stacks[p] = 0
        new.bets[p] += amount
        new.pot += amount
        new.all_in[p] = True
        if amount > to_call:
            new.last_raiser = p

    new.history.append((new.round_idx, p, action))
    new.actions_this_round.append((p, action))
    new.num_actions_this_round += 1

    # Check if hand is over (only 1 active player)
    active_count = sum(new.active)
    if active_count == 1:
        _resolve_hand(new)
        return new

    # Find next player and check if round is over
    _advance_to_next_player(new)

    return new


def _advance_to_next_player(state):
    """Find next active player or advance to next round."""
    # Count players who can still act (active and not all-in)
    can_act = [i for i in range(NUM_PLAYERS) if state.active[i] and not state.all_in[i]]

    if len(can_act) <= 1:
        # No more action possible, run out remaining board and resolve
        _run_out_board(state)
        _resolve_hand(state)
        return

    # Check if betting round is complete
    round_complete = _is_round_complete(state)

    if round_complete:
        _advance_round(state)
        return

    # Find next active player who can act
    next_p = (state.current_player + 1) % NUM_PLAYERS
    while not state.active[next_p] or state.all_in[next_p]:
        next_p = (next_p + 1) % NUM_PLAYERS
    state.current_player = next_p


def _is_round_complete(state):
    """Check if all active non-all-in players have acted and bets are equal."""
    can_act = [i for i in range(NUM_PLAYERS) if state.active[i] and not state.all_in[i]]

    if len(can_act) == 0:
        return True

    # All active players must have acted at least once
    acted_players = set(p for p, a in state.actions_this_round)
    for p in can_act:
        if p not in acted_players:
            return False

    # All bets must be equal among active players (excluding all-in players who are short)
    bets = [state.bets[p] for p in can_act]
    if len(set(bets)) > 1:
        return False

    # If someone raised, everyone after must have had a chance to respond
    if state.last_raiser >= 0:
        raiser_action_idx = None
        for idx, (p, a) in enumerate(state.actions_this_round):
            if p == state.last_raiser and a in [BET_HALF, BET_POT, ALL_IN]:
                raiser_action_idx = idx

        if raiser_action_idx is not None:
            # All other active players must have acted AFTER the raise
            for p in can_act:
                if p == state.last_raiser:
                    continue
                acted_after = False
                for idx, (ap, aa) in enumerate(state.actions_this_round):
                    if idx > raiser_action_idx and ap == p:
                        acted_after = True
                        break
                if not acted_after:
                    return False

    return True


def _advance_round(state):
    """Move to next betting round (deal community cards)."""
    state.round_idx += 1
    state.actions_this_round = []
    state.num_actions_this_round = 0
    state.last_raiser = -1

    # Reset per-round bets
    state.bets = [0.0] * NUM_PLAYERS

    if state.round_idx > 3:
        # Showdown after river
        _resolve_hand(state)
        return

    # Deal community cards
    if state.round_idx == 1:  # flop
        for _ in range(3):
            state.board.append(state.deck[state.deck_idx])
            state.deck_idx += 1
    elif state.round_idx in [2, 3]:  # turn, river
        state.board.append(state.deck[state.deck_idx])
        state.deck_idx += 1

    # Postflop: action starts with first active player after button (player 1, then 2, then 0)
    for offset in range(1, NUM_PLAYERS + 1):
        p = offset % NUM_PLAYERS
        if state.active[p] and not state.all_in[p]:
            state.current_player = p
            return

    # Everyone is all-in, run out board
    _run_out_board(state)
    _resolve_hand(state)


def _run_out_board(state):
    """Deal remaining community cards when no more betting is possible."""
    while len(state.board) < 5:
        if state.round_idx == 0:
            # Deal flop
            for _ in range(3):
                state.board.append(state.deck[state.deck_idx])
                state.deck_idx += 1
            state.round_idx = 1
        else:
            state.board.append(state.deck[state.deck_idx])
            state.deck_idx += 1
            state.round_idx += 1


def _resolve_hand(state):
    """Determine winner and distribute pot."""
    state.done = True
    active_players = [p for p in range(NUM_PLAYERS) if state.active[p]]

    if len(active_players) == 1:
        # Everyone else folded — winner gets the pot
        winner = active_players[0]
        state.stacks[winner] += state.pot
        return

    # Showdown: best hand wins
    # (simplified: no side pots for now)
    best_hand = None
    winner = -1
    for p in active_players:
        hand = evaluate_hand(state.hole_cards[p] + state.board)
        if best_hand is None or hand > best_hand:
            best_hand = hand
            winner = p

    # Give pot to winner
    state.stacks[winner] += state.pot


def get_payoffs(state):
    """
    Return net profit/loss for each player in big blinds.
    Positive = won, negative = lost.
    """
    payoffs = []
    for p in range(NUM_PLAYERS):
        payoffs.append(state.stacks[p] - STARTING_STACK)
    return payoffs


def get_info_key(state, player):
    """
    Information set key: what the player knows.
    = hand bucket + board bucket + abstracted betting history

    We abstract the history to just the sequence of action TYPES per round
    to keep the number of info sets manageable.
    """
    hole = tuple(state.hole_cards[player])
    board = tuple(state.board)
    round_idx = state.round_idx

    # Bucket the hand
    num_buckets = 10 if round_idx == 0 else 8
    bucket = hand_strength_bucket(hole, board, num_buckets)

    # Abstract betting history: just action names in sequence
    hist_str = ",".join(a for _, _, a in state.history)

    return f"{round_idx}|{bucket}|{hist_str}"
