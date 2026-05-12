"""
Shared fixtures and stub game for CFR correctness tests.
Fixed: Added **kwargs to get_info_key for compatibility.
"""

import numpy as np
import pytest

class FakeState:
    def __init__(self, kind='terminal', traverser=0, payoffs=None,
                 legal_actions=None, value_table=None, actor=0):
        self.kind = kind
        self.traverser = traverser
        self.actor = actor
        self.payoffs = payoffs or [0.0, 0.0, 0.0]
        self.legal_actions = legal_actions or [0, 1]
        self.value_table = value_table or {}
        self.done = (kind == 'terminal')
        self.chance_pending = False
        self.action_history = ()

class StubGame:
    def __init__(self, root_state: FakeState):
        self._root = root_state

    def deal_new_hand(self):
        return self._root

    def is_terminal(self, state):
        return state.kind == 'terminal'

    def is_chance_node(self, state):
        return state.kind == 'chance'

    def sample_chance(self, state):
        raise NotImplementedError

    def get_current_player(self, state):
        if state.done or state.chance_pending:
            return -1
        return state.actor

    def get_legal_actions(self, state):
        if state.done or state.chance_pending:
            return []
        return list(state.legal_actions)

    def get_info_key(self, state, player, **kwargs):
        """Accept and ignore additional arguments like team_seats."""
        return (player, tuple(state.legal_actions))

    def get_payoffs(self, state):
        return list(state.payoffs)

    def apply_action(self, state, action_index):
        traverser = state.traverser
        payoff = state.value_table.get(action_index, 0.0)
        payoffs = [0.0, 0.0, 0.0]
        payoffs[traverser] = payoff
        terminal = FakeState(
            kind='terminal',
            traverser=traverser,
            payoffs=payoffs,
        )
        terminal.action_history = state.action_history + (action_index,)
        return terminal

PLAN_V_TABLE = {0: 0.0, 1: 0.5, 2: 1.5, 3: 1.5, 4: 1.5, 5: 1.5, 6: 1.5, 7: 1.5, 8: 1.5, 9: 1.5}
PLAN_LEGAL = list(range(10))

@pytest.fixture
def plan_root():
    return FakeState(kind='traverser_root', traverser=0, actor=0,
                     legal_actions=PLAN_LEGAL, value_table=PLAN_V_TABLE)

@pytest.fixture
def plan_game(plan_root):
    return StubGame(plan_root)