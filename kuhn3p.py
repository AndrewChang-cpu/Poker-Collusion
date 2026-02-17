"""
3-Player Kuhn Poker game engine.

Rules:
- Deck has 4 cards: {0, 1, 2, 3} (think J, Q, K, A)
- Each player antes 1 chip, gets 1 card
- Single betting round: players can Pass(check/fold) or Bet(1 chip)
- After betting round, highest card among active players wins the pot
- This is small enough to verify CFR convergence.
"""

import numpy as np
from copy import deepcopy


NUM_PLAYERS = 3
DECK = [0, 1, 2, 3]
ACTIONS = ["pass", "bet"]


class KuhnState:
    def __init__(self):
        self.cards = None          # list of 3 cards, one per player
        self.history = []          # list of actions taken
        self.active = [True] * 3   # who hasn't folded
        self.pot = [1, 1, 1]       # ante of 1 each
        self.current_player = 0
        self.bets_this_round = 0   # how many bets have been made

    def copy(self):
        return deepcopy(self)


def deal_new_hand():
    """Deal a new 3-player Kuhn hand."""
    state = KuhnState()
    cards = list(np.random.choice(DECK, size=3, replace=False))
    state.cards = cards
    return state


def get_current_player(state):
    return state.current_player


def get_legal_actions(state):
    return ACTIONS  # always pass or bet


def get_info_key(state, player):
    """
    Info set key: player's card + the action history they've observed.
    e.g., "2:pass-bet" means player holds card 2, saw pass then bet.
    """
    card = state.cards[player]
    history_str = "-".join(state.history) if state.history else ""
    return f"{card}:{history_str}"


def is_terminal(state):
    """
    Terminal conditions for 3-player Kuhn:
    - All 3 players have acted AND no bet was made (all passed) -> showdown
    - A bet was made and all remaining players have responded -> showdown/fold resolution
    """
    h = state.history
    n = len(h)

    if n == 0:
        return False

    # All 3 passed -> showdown
    if h == ["pass", "pass", "pass"]:
        return True

    # Someone bet, and the remaining players after the bettor have all responded
    if "bet" in h:
        bet_idx = h.index("bet")
        actions_after_bet = n - bet_idx - 1
        players_who_need_to_respond = 2  # the other 2 players after bettor
        if actions_after_bet >= players_who_need_to_respond:
            return True

    return False


def get_payoffs(state):
    """Return payoffs for each player at a terminal state."""
    h = state.history

    # All passed -> highest card wins the pot (3 chips total, winner gains 2)
    if h == ["pass", "pass", "pass"]:
        winner = max(range(3), key=lambda p: state.cards[p])
        payoffs = [-1.0, -1.0, -1.0]  # everyone anted 1
        payoffs[winner] = 2.0          # winner gets pot of 3, net +2
        return payoffs

    # Someone bet
    bet_idx = h.index("bet")
    bettor = bet_idx  # player index who bet (0-indexed by action order)

    # Determine who called vs folded after the bet
    callers = [bettor]
    folders = []
    for i, action in enumerate(h[bet_idx + 1:]):
        player = (bettor + 1 + i) % 3
        if action == "bet":  # "bet" after a bet = call
            callers.append(player)
        else:  # "pass" after a bet = fold
            folders.append(player)

    # Pot contributions: ante 1 for all, +1 for callers/bettor
    pot_contributions = [1.0] * 3
    for p in callers:
        pot_contributions[p] = 2.0

    total_pot = sum(pot_contributions)

    # Highest card among callers wins
    winner = max(callers, key=lambda p: state.cards[p])

    payoffs = [-pot_contributions[p] for p in range(3)]
    payoffs[winner] += total_pot

    return payoffs


def apply_action(state, action):
    """Apply action and return new state."""
    new_state = state.copy()
    new_state.history.append(action)

    # Advance current player
    new_state.current_player = len(new_state.history) % 3

    # But if we're past the first 3 actions, current_player depends on bet position
    h = new_state.history
    if "bet" in h:
        bet_idx = h.index("bet")
        bettor = bet_idx
        if len(h) > bet_idx + 1:
            # Next player to act after bet
            actions_after_bet = len(h) - bet_idx - 1
            new_state.current_player = (bettor + actions_after_bet) % 3

    # If not yet at a bet, it's just sequential
    if "bet" not in h:
        new_state.current_player = len(h) % 3

    return new_state


def is_chance_node(state):
    """Kuhn has no mid-game chance nodes (all cards dealt at start)."""
    return False


def sample_chance(state):
    return state
