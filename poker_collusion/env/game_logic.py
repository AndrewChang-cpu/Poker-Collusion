"""
Game logic: legal actions, apply_action, sample_chance, round advancement.

apply_action and sample_chance are copy-on-write: they return a new NLHEState
and never mutate their input. The caller owns the returned state.
"""

from poker_collusion.config import NUM_PLAYERS, STARTING_STACK_BB, POSTFLOP_ORDER
from poker_collusion.env.game_state import NLHEState, DEAL
from poker_collusion.env.hand_eval import evaluate_hand
from poker_collusion.abstraction.actions import get_legal_action_indices, action_index_to_chips


def get_current_player(state):
    if state.done or state.chance_pending:
        return -1
    return state.current_player


def get_legal_actions(state):
    if state.done or state.chance_pending:
        return []
    return get_legal_action_indices(state)


def is_terminal(state):
    return state.done


def is_chance_node(state):
    return state.chance_pending and not state.done


def sample_chance(state):
    """
    Deal next street (flop/turn/river). Returns a new state with board cards added,
    round_idx advanced, bets reset, and current_player set to first postflop actor.
    """
    assert state.chance_pending, "sample_chance called but chance_pending is False"
    assert not state.done, "sample_chance called on terminal state"

    s = state.copy()
    n = 3 if s.round_idx == 0 else 1
    for _ in range(n):
        s.board.append(s.deck[s.deck_idx])
        s.deck_idx += 1
    s.action_history = s.action_history + (DEAL,)
    s.round_idx += 1
    s.chance_pending = False
    s.bets = (0.0,) * NUM_PLAYERS
    s.last_raiser = -1
    s.last_raise_amount = 0.0

    # Past river or board full -> resolve immediately
    if s.round_idx > 3 or len(s.board) >= 5:
        _resolve_hand(s)
        return s

    # Set first postflop actor: first active, non-all-in player in POSTFLOP_ORDER
    for p in POSTFLOP_ORDER:
        if s.active[p] and not s.all_in[p]:
            s.current_player = p
            break
    else:
        _run_out_board_and_resolve(s)

    return s


def apply_action(state, action_index):
    """
    Apply action (index 0..9) for state.current_player. Returns a new state.
    Never mutates the input state.
    """
    assert not state.done, "apply_action called on terminal state"
    assert not state.chance_pending, "apply_action called while chance_pending"
    assert 0 <= action_index <= 9, f"invalid action_index {action_index}"
    p = state.current_player
    assert state.active[p], f"apply_action: player {p} is not active"
    assert not state.all_in[p], f"apply_action: player {p} is already all-in"

    s = state.copy()
    is_fold, total_bet = action_index_to_chips(s, action_index)

    s.action_history = s.action_history + (action_index,)
    s.actor_history = s.actor_history + (p,)

    if is_fold:
        s.active = s.active[:p] + (False,) + s.active[p + 1:]
    else:
        add = total_bet - s.bets[p]
        assert add >= 0, f"apply_action: negative chip addition {add} for player {p}"
        new_stack = s.stacks[p] - add
        s.stacks = s.stacks[:p] + (new_stack,) + s.stacks[p + 1:]
        s.bets = s.bets[:p] + (total_bet,) + s.bets[p + 1:]
        s.pot += add
        prev_max = max(s.bets[q] for q in range(NUM_PLAYERS) if q != p)
        if add > 0 and total_bet > prev_max:
            s.last_raiser = p
            s.last_raise_amount = total_bet - prev_max
        if new_stack <= 0:
            s.all_in = s.all_in[:p] + (True,) + s.all_in[p + 1:]

    # Single active player -> they win uncontested
    if sum(s.active) == 1:
        _resolve_hand(s)
        return s

    _advance_to_next_player(s)
    return s


def _advance_to_next_player(state):
    """Mutates state (internal use only — state is already a fresh copy)."""
    can_act = [p for p in range(NUM_PLAYERS) if state.active[p] and not state.all_in[p]]
    if len(can_act) == 0:
        _run_out_board_and_resolve(state)
        return
    if _is_round_complete(state):
        if state.round_idx >= 3:
            _resolve_hand(state)
        else:
            state.chance_pending = True
            state.current_player = -1
        return
    # Advance to next active, non-all-in player
    next_p = (state.current_player + 1) % NUM_PLAYERS
    while not state.active[next_p] or state.all_in[next_p]:
        next_p = (next_p + 1) % NUM_PLAYERS
    state.current_player = next_p


def _street_start_index(state):
    """Return the index in action_history where the current street's actions begin."""
    hist = state.action_history
    for i in range(len(hist) - 1, -1, -1):
        if hist[i] == DEAL:
            return i + 1
    return 0


def _street_actors(state):
    """
    Return the slice of actor_history corresponding to the current street.
    actor_history has one entry per non-DEAL action, in order. We find how
    many non-DEAL actions occurred before the current street and slice from there.
    """
    street_start = _street_start_index(state)
    actor_offset = sum(1 for a in state.action_history[:street_start] if a != DEAL)
    return state.actor_history[actor_offset:]


def _is_round_complete(state):
    """
    The round is complete when:
    1. All active non-all-in players have equal bets, AND
    2. Every active non-all-in player has acted at least once this street, AND
    3. Every active non-all-in player has acted after the last raise (if any).
    """
    can_act = [p for p in range(NUM_PLAYERS) if state.active[p] and not state.all_in[p]]
    if not can_act:
        return True

    # Condition 1: equal bets
    if len(set(state.bets[p] for p in can_act)) > 1:
        return False

    street_acts = _street_actors(state)  # list of player indices who acted this street, in order

    # Condition 2: everyone has acted at least once
    for p in can_act:
        if p not in street_acts:
            return False

    # Condition 3: if there was a raise, everyone has acted after the last raise
    if state.last_raiser >= 0:
        street_start = _street_start_index(state)
        street_action_history = [a for a in state.action_history[street_start:] if a != DEAL]
        assert len(street_action_history) == len(street_acts), (
            f"street_action_history length {len(street_action_history)} "
            f"!= street_acts length {len(street_acts)}"
        )

        last_raise_pos = -1
        for i, (actor, action) in enumerate(zip(street_acts, street_action_history)):
            if actor == state.last_raiser and action not in (0, 1):
                last_raise_pos = i

        if last_raise_pos >= 0:
            actors_after_raise = street_acts[last_raise_pos + 1:]
            for p in can_act:
                if p == state.last_raiser:
                    continue
                if p not in actors_after_raise:
                    return False

    return True


def _run_out_board_and_resolve(state):
    """Mutates state (internal use only)."""
    while len(state.board) < 5:
        n = 3 if state.round_idx == 0 else 1
        for _ in range(n):
            state.board.append(state.deck[state.deck_idx])
            state.deck_idx += 1
        state.action_history = state.action_history + (DEAL,)
        state.round_idx += 1
    _resolve_hand(state)


def _resolve_hand(state):
    """Mutates state (internal use only). Sets done=True and distributes pot."""
    state.done = True
    active = [p for p in range(NUM_PLAYERS) if state.active[p]]
    assert len(active) >= 1, "resolve_hand: no active players"
    if len(active) == 1:
        w = active[0]
        state.stacks = state.stacks[:w] + (state.stacks[w] + state.pot,) + state.stacks[w + 1:]
        return
    contributions = [STARTING_STACK_BB - state.stacks[p] for p in range(NUM_PLAYERS)]
    _resolve_side_pots(state, active, contributions)


def _resolve_side_pots(state, active_players, contributions):
    """Distribute pot among active players using side pot rules. Mutates state."""
    levels = sorted(set(contributions[p] for p in range(NUM_PLAYERS) if contributions[p] > 0))
    prev = 0.0
    for level in levels:
        eligible_count = [p for p in range(NUM_PLAYERS) if contributions[p] >= level]
        slice_size = (level - prev) * len(eligible_count)
        if slice_size <= 0:
            prev = level
            continue
        eligible_win = [p for p in eligible_count if state.active[p]]
        if not eligible_win:
            prev = level
            continue
        best_hand = None
        winners = []
        for p in eligible_win:
            h = evaluate_hand(state.hole_cards[p] + state.board)
            if best_hand is None or h > best_hand:
                best_hand = h
                winners = [p]
            elif h == best_hand:
                winners.append(p)
        assert len(winners) > 0
        share = slice_size / len(winners)
        stacks = list(state.stacks)
        for w in winners:
            stacks[w] += share
        state.stacks = tuple(stacks)
        prev = level
