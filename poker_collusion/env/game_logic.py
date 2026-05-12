"""
Game logic for Leduc Hold'em: legal actions, apply_action, sample_chance, round advancement.
Modified to remove circular imports and resolve precision issues via rounding.
Step 2: Added remainder distribution to prevent chip bleeding in split pots.
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
    Deal next street for Leduc (one community card). Returns a new state with 
    board cards added, round_idx advanced, bets reset, and current_player updated.
    """
    assert state.chance_pending, "sample_chance called but chance_pending is False"
    assert not state.done, "sample_chance called on terminal state"

    s = state.copy()
    # Leduc deals exactly 1 card for the community board
    s.board.append(s.deck[s.deck_idx])
    s.deck_idx += 1
    s.action_history = s.action_history + (DEAL,)
    s.round_idx += 1
    s.chance_pending = False
    s.bets = (0.0,) * NUM_PLAYERS
    s.last_raiser = -1
    s.last_raise_amount = 0.0

    # Leduc game ends after round 1 (Flop) betting. 
    if s.round_idx > 1:
        _resolve_hand(s)
        return s

    # Set first postflop actor: first active, non-all-in player in POSTFLOP_ORDER
    for p in POSTFLOP_ORDER:
        if s.active[p] and not s.all_in[p]:
            s.current_player = p
            break
    else:
        # All players are all-in or folded; resolve immediately
        _resolve_hand(s)

    return s


def apply_action(state, action_index):
    """
    Apply action (index 0..9) for state.current_player. Returns a new state.
    Uses round(val, 2) to ensure precision safety for comparison logic.
    """
    assert not state.done, "apply_action called on terminal state"
    assert not state.chance_pending, "apply_action called while chance_pending"
    p = state.current_player

    s = state.copy()
    is_fold, total_bet = action_index_to_chips(s, action_index)
    total_bet = round(total_bet, 2)

    s.action_history = s.action_history + (action_index,)
    s.actor_history = s.actor_history + (p,)

    if is_fold:
        s.active = s.active[:p] + (False,) + s.active[p + 1:]
    else:
        add = round(total_bet - s.bets[p], 2)
        new_stack = round(s.stacks[p] - add, 2)
        s.stacks = s.stacks[:p] + (new_stack,) + s.stacks[p + 1:]
        s.bets = s.bets[:p] + (total_bet,) + s.bets[p + 1:]
        s.pot = round(s.pot + add, 2)
        
        prev_max = max(s.bets[q] for q in range(NUM_PLAYERS) if q != p)
        if add > 0 and total_bet > prev_max:
            s.last_raiser = p
            s.last_raise_amount = round(total_bet - prev_max, 2)
        if new_stack <= 0:
            s.all_in = s.all_in[:p] + (True,) + s.all_in[p + 1:]

    # Single active player -> they win uncontested
    if sum(s.active) == 1:
        _resolve_hand(s)
        return s

    _advance_to_next_player(s)
    return s


def _advance_to_next_player(state):
    """Advance current_player or transition to chance/terminal state."""
    can_act = [p for p in range(NUM_PLAYERS) if state.active[p] and not state.all_in[p]]
    if len(can_act) == 0:
        _run_out_board_and_resolve(state)
        return
    if _is_round_complete(state):
        # Leduc concludes after the Flop (round 1)
        if state.round_idx >= 1:
            _resolve_hand(state)
        else:
            state.chance_pending = True
            state.current_player = -1
        return
        
    next_p = (state.current_player + 1) % NUM_PLAYERS
    while not state.active[next_p] or state.all_in[next_p]:
        next_p = (next_p + 1) % NUM_PLAYERS
    state.current_player = next_p


def _is_round_complete(state):
    """
    Round is complete if bets are equal and everyone has had a chance to act.
    """
    can_act = [p for p in range(NUM_PLAYERS) if state.active[p] and not state.all_in[p]]
    if not can_act:
        return True

    # Condition 1: equal bets
    if len(set(state.bets[p] for p in can_act)) > 1:
        return False

    # Condition 2: everyone has acted at least once this street
    hist = state.action_history
    start = 0
    for i in range(len(hist) - 1, -1, -1):
        if hist[i] == DEAL:
            start = i + 1
            break
    
    actor_offset = sum(1 for x in hist[:start] if x != DEAL)
    street_acts = state.actor_history[actor_offset:]
    
    for p in can_act:
        if p not in street_acts:
            return False

    # Condition 3: everyone has acted after the last raise
    if state.last_raiser >= 0:
        last_raise_pos = -1
        street_action_history = [a for a in state.action_history[start:] if a != DEAL]
        for i, (actor, action) in enumerate(zip(street_acts, street_action_history)):
            if actor == state.last_raiser and action not in (0, 1):
                last_raise_pos = i

        if last_raise_pos >= 0:
            actors_after_raise = street_acts[last_raise_pos + 1:]
            for p in can_act:
                if p != state.last_raiser and p not in actors_after_raise:
                    return False

    return True


def _run_out_board_and_resolve(state):
    """Deal remaining board cards for showdown evaluation."""
    while len(state.board) < 1:
        state.board.append(state.deck[state.deck_idx])
        state.deck_idx += 1
        state.round_idx += 1
        state.action_history = state.action_history + (DEAL,)
    _resolve_hand(state)


def _resolve_hand(state):
    """Distribute pot. Removed redundant internal import of _resolve_side_pots."""
    state.done = True
    active = [p for p in range(NUM_PLAYERS) if state.active[p]]
    if len(active) == 1:
        w = active[0]
        state.stacks = state.stacks[:w] + (round(state.stacks[w] + state.pot, 2),) + state.stacks[w + 1:]
        return
    
    contributions = [round(STARTING_STACK_BB - state.stacks[p], 2) for p in range(NUM_PLAYERS)]
    _resolve_side_pots(state, active, contributions)


def _resolve_side_pots(state, active_players, contributions):
    """
    Distribute pot among active players based on contributions and hand rank.
    Step 2: awarding the rounding remainder to the first winner to maintain chip total.
    """
    levels = sorted(set(c for c in contributions if c > 0))
    prev = 0.0
    for level in levels:
        eligible_count = [p for p in range(NUM_PLAYERS) if contributions[p] >= level]
        slice_size = round((level - prev) * len(eligible_count), 2)
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
            # Leduc hand evaluation (hole card + community card)
            h = evaluate_hand(state.hole_cards[p] + state.board)
            if best_hand is None or h > best_hand:
                best_hand, winners = h, [p]
            elif h == best_hand:
                winners.append(p)
        
        # Calculate individual share and remaining 'cents' due to rounding
        share = round(slice_size / len(winners), 2)
        total_distributed = round(share * len(winners), 2)
        remainder = round(slice_size - total_distributed, 2)
        
        stacks = list(state.stacks)
        for i, w in enumerate(winners):
            add = share
            if i == 0:
                # Award the rounding remainder to the first winner (OOP player)
                add = round(add + remainder, 2)
            stacks[w] = round(stacks[w] + add, 2)
        state.stacks = tuple(stacks)
        prev = level