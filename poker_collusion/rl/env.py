"""
Turn-based RL wrapper around the 3-player Leduc engine.

Learner seats act through `step`; every other seat is played by a fixed
opponent policy inside the wrapper. Actor observations contain only what the
acting seat may legally see. `global_state` (all hole cards) is for a
centralized critic during training only (CTDE, MAPPO-style).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from poker_collusion.config import NUM_ACTIONS, NUM_PLAYERS, SMOOTH_LAMBDA, STARTING_STACK_BB
from poker_collusion.env import game_logic as g
from poker_collusion.env.game_state import DEAL, deal_new_hand, get_payoffs

NUM_RANKS = 4
HISTORY_LEN = 12  # Leduc lines with 3 players rarely exceed this
_ACT_FEAT = NUM_PLAYERS + NUM_ACTIONS + 1  # actor, action, street
OBS_DIM = NUM_RANKS + (NUM_RANKS + 1) + 2 + NUM_PLAYERS + 3 * NUM_PLAYERS + 2 + HISTORY_LEN * _ACT_FEAT
GLOBAL_DIM = OBS_DIM + NUM_PLAYERS * NUM_RANKS

OpponentFn = Callable[[object, int, List[int]], np.ndarray]


def _one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if 0 <= i < n:
        v[i] = 1.0
    return v


def observe(state, seat: int) -> np.ndarray:
    """Actor observation for `seat`: own card, public board, public betting."""
    s = STARTING_STACK_BB
    board = int(state.board[0]) if state.board else -1
    hist = np.zeros((HISTORY_LEN, _ACT_FEAT), dtype=np.float32)
    street, k = 0, 0
    entries = []
    for a in state.action_history:
        if a == DEAL:
            street = 1
            continue
        entries.append((state.actor_history[k], int(a), street))
        k += 1
    for row, (actor, a, st) in enumerate(entries[-HISTORY_LEN:]):
        hist[row, actor] = 1.0
        hist[row, NUM_PLAYERS + a] = 1.0
        hist[row, -1] = st
    to_call = max(state.bets) - state.bets[seat]
    return np.concatenate([
        _one_hot(int(state.hole_cards[seat][0]), NUM_RANKS),
        _one_hot(board, NUM_RANKS + 1) if board >= 0 else _one_hot(NUM_RANKS, NUM_RANKS + 1),
        np.array([state.round_idx, float(len(entries) > HISTORY_LEN)], dtype=np.float32),
        _one_hot(seat, NUM_PLAYERS),
        np.array(state.stacks, dtype=np.float32) / s,
        np.array(state.bets, dtype=np.float32) / s,
        np.array(state.active, dtype=np.float32),
        np.array([state.pot / (NUM_PLAYERS * s), to_call / s], dtype=np.float32),
        hist.ravel(),
    ])


def global_state(state, seat: int) -> np.ndarray:
    """Critic input: actor observation plus every seat's hole card."""
    cards = [_one_hot(int(state.hole_cards[p][0]), NUM_RANKS) for p in range(NUM_PLAYERS)]
    return np.concatenate([observe(state, seat)] + cards)


def legal_mask(state) -> np.ndarray:
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    m[g.get_legal_actions(state)] = True
    return m


class LeducTeamEnv:
    """
    Episode = one hand. `reset()` returns the first learner decision;
    `step(action)` applies it, auto-plays opponents and chance, and returns
    the next learner decision or a terminal result.

    Reward: at terminal, each learner seat gets `team_reward(payoffs)`
    (utilitarian team sum by default; `smooth` adds SMOOTH_LAMBDA * min),
    0 before. Payoffs are in BB.
    """

    def __init__(self, learner_seats: Sequence[int], opponent: OpponentFn,
                 team_objective: str = "utilitarian"):
        self.learner_seats = tuple(sorted(learner_seats))
        self.opponent = opponent
        if team_objective not in ("utilitarian", "smooth", "individual"):
            raise ValueError(f"unsupported team_objective {team_objective!r}")
        self.team_objective = team_objective
        self.state = None

    def team_reward(self, payoffs: Sequence[float], seat: int) -> float:
        if self.team_objective == "individual":
            return float(payoffs[seat])
        team = [payoffs[s] for s in self.learner_seats]
        if self.team_objective == "smooth":  # penalizes sacrificing one teammate (chip dumping)
            return float(sum(team) + SMOOTH_LAMBDA * min(team))
        return float(sum(team))

    def reset(self, state=None) -> Dict:
        self.state = state if state is not None else deal_new_hand()
        return self._advance()

    def step(self, action: int) -> Dict:
        legal = g.get_legal_actions(self.state)
        if action not in legal:
            raise ValueError(f"illegal action {action}; legal={legal}")
        self.state = g.apply_action(self.state, int(action))
        return self._advance()

    def _advance(self) -> Dict:
        s = self.state
        while not s.done:
            if s.chance_pending:
                s = g.sample_chance(s)
                continue
            p = s.current_player
            if p in self.learner_seats:
                self.state = s
                return {"done": False, "seat": p, "obs": observe(s, p),
                        "global": global_state(s, p), "mask": legal_mask(s)}
            legal = g.get_legal_actions(s)
            probs = self.opponent(s, p, legal)
            s = g.apply_action(s, legal[int(np.random.choice(len(legal), p=probs))])
        self.state = s
        payoffs = get_payoffs(s)
        return {"done": True, "payoffs": payoffs,
                "rewards": {q: self.team_reward(payoffs, q) for q in self.learner_seats}}
