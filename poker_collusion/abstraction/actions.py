"""
Action sets by round (10 each) and legality filtering.
"""

from poker_collusion.config import NUM_PLAYERS, STARTING_STACK_BB, NUM_ACTIONS

PREFLOP_RAISE_BB = [2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 12.0]
POSTFLOP_POT_MULT = [0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.5]


def _to_call(state):
    p = state.current_player
    return max(state.bets) - state.bets[p]


def _max_bet(state):
    return max(state.bets)


def _pot_for_acting(state):
    return state.pot + _to_call(state)


def _min_raise_total(state):
    return _max_bet(state) + state.last_raise_amount


def get_legal_action_indices(state):
    """Return legal indices [0..9] with redundancy fix."""
    p = state.current_player
    if not state.active[p] or state.all_in[p]:
        return []
    to_call = _to_call(state)
    stack = state.stacks[p]
    is_preflop = state.round_idx == 0
    legal = []

    if to_call > 0:
        legal.append(0)  # Fold
        if stack >= to_call:
            legal.append(1)  # Call
    else:
        legal.append(1)  # Check

    chips_after_call = stack - to_call
    if chips_after_call <= 0:
        if stack > 0 and 9 not in legal:
            legal.append(9)
        return sorted(legal)

    if is_preflop:
        min_raise_total = _min_raise_total(state)
        seen_totals = set()
        for i, total_bb in enumerate(PREFLOP_RAISE_BB):
            if total_bb < min_raise_total:
                continue
            if total_bb > stack:
                break
            if total_bb in seen_totals:
                continue
            seen_totals.add(total_bb)
            legal.append(2 + i)
        if stack > 0 and (stack >= min_raise_total or to_call > 0):
            if stack not in seen_totals:
                legal.append(9)
    else:
        pot = _pot_for_acting(state)
        max_bet = _max_bet(state)
        min_raise_total = _min_raise_total(state) if max_bet > 0 else 0
        seen_totals = set()
        for i, mult in enumerate(POSTFLOP_POT_MULT):
            bet_amount = pot * mult
            total_bet = to_call + bet_amount
            if max_bet > 0 and total_bet < min_raise_total:
                continue
            if total_bet > stack:
                continue
            if total_bet in seen_totals:
                continue
            seen_totals.add(total_bet)
            legal.append(2 + i)
        if stack > 0:
            all_in_total = stack
            if all_in_total >= min_raise_total or to_call > 0:
                if all_in_total not in seen_totals:
                    legal.append(9)

    return sorted(legal)


def action_index_to_chips(state, action_index):
    p = state.current_player
    to_call = _to_call(state)
    stack = state.stacks[p]
    is_preflop = state.round_idx == 0

    if action_index == 0:
        return True, state.bets[p]
    if action_index == 1:
        if to_call > 0:
            return False, state.bets[p] + min(to_call, stack)
        return False, state.bets[p]

    if action_index == 9:
        return False, state.bets[p] + stack

    if is_preflop:
        total_bb = PREFLOP_RAISE_BB[action_index - 2]
        return False, total_bb
    else:
        pot = _pot_for_acting(state)
        mult = POSTFLOP_POT_MULT[action_index - 2]
        bet_amount = pot * mult
        total_bet = to_call + bet_amount
        return False, min(total_bet, stack)