"""
Shared fixtures and stub game for CFR correctness tests.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal state objects
# ---------------------------------------------------------------------------

class FakeState:
    """
    Minimal state object accepted by CFRTrainer.  The game module (StubGame)
    interprets the 'kind' field to decide what to return at each node.

    kind options:
      'traverser_root'  – the one node where the traverser acts; legal actions
                          come from self.legal_actions; values come from
                          self.value_table (dict mapping action_index -> float).
      'terminal'        – terminal node; payoff for all players in self.payoffs.
      'opponent_action' – opponent decision; legal = self.legal_actions,
                          always samples index 0 (via seeded numpy).
    """

    def __init__(self, kind='terminal', traverser=0, payoffs=None,
                 legal_actions=None, value_table=None, actor=0):
        self.kind = kind
        self.traverser = traverser          # which player is the traverser
        self.actor = actor                  # who acts at this node
        self.payoffs = payoffs or [0.0, 0.0, 0.0]
        self.legal_actions = legal_actions or [0, 1]
        self.value_table = value_table or {}  # action -> terminal payoff for traverser
        self.done = (kind == 'terminal')
        self.chance_pending = False
        self.action_history = ()


class StubGame:
    """
    A fully deterministic stub game module implementing the CFRGame protocol.

    The 'root state' is a traverser decision node with a fixed value table.
    After the traverser picks an action, the next state is a terminal node
    whose payoff is looked up from value_table[action].

    This makes the entire recursion depth=1 from the traverser's perspective:
        traverser acts → terminal leaf.

    The traverser's payoff at that leaf equals value_table[action].
    """

    def __init__(self, root_state: FakeState):
        self._root = root_state

    def deal_new_hand(self):
        return self._root

    def is_terminal(self, state):
        return state.kind == 'terminal'

    def is_chance_node(self, state):
        return state.kind == 'chance'

    def sample_chance(self, state):
        raise NotImplementedError("chance not used in these tests")

    def get_current_player(self, state):
        if state.done or state.chance_pending:
            return -1
        return state.actor

    def get_legal_actions(self, state):
        if state.done or state.chance_pending:
            return []
        return list(state.legal_actions)

    def get_info_key(self, state, player):
        # Unique key per (actor, legal_actions tuple) — deterministic.
        return (player, tuple(state.legal_actions))

    def get_payoffs(self, state):
        return list(state.payoffs)

    def apply_action(self, state, action_index):
        """
        Applying an action from the root always produces a terminal.
        The traverser's payoff at that terminal is value_table[action_index].
        """
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


# ---------------------------------------------------------------------------
# Standard root state for P0 first-move examples from PLAN.md
# ---------------------------------------------------------------------------
# Legal actions: 0..9  (all 10 abstract actions available)
# v[0]=0.0, v[1]=0.5, v[2..9]=1.5
PLAN_V_TABLE = {
    0: 0.0,
    1: 0.5,
    2: 1.5,
    3: 1.5,
    4: 1.5,
    5: 1.5,
    6: 1.5,
    7: 1.5,
    8: 1.5,
    9: 1.5,
}
PLAN_LEGAL = list(range(10))


@pytest.fixture
def plan_root():
    """Root state matching the PLAN.md scenario: P0 traverser, 10 legal actions."""
    return FakeState(
        kind='traverser_root',
        traverser=0,
        actor=0,
        legal_actions=PLAN_LEGAL,
        value_table=PLAN_V_TABLE,
    )


@pytest.fixture
def plan_game(plan_root):
    return StubGame(plan_root)
