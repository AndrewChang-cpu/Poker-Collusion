"""
Game logic: handles actions and round transitions.
"""

from poker_collusion.config import NUM_PLAYERS, STARTING_STACK_BB
from poker_collusion.env.game_state import NLHEState, DEAL
from poker_collusion.env.hand_eval import evaluate_hand
from poker_collusion.abstraction.actions import get_legal_action_indices, action_index_to_chips

def get_current_player(state):
    return -1 if (state.done or state.chance_pending) else state.current_player

def get_legal_actions(state):
    return [] if (state.done or state.chance_pending) else get_legal_action_indices(state)

def is_terminal(state): return state.done
def is_chance_node(state): return state.chance_pending and not state.done

def sample_chance(state):
    if not state.chance_pending or state.done: return state
    n = 3 if state.round_idx == 0 else 1
    for _ in range(n):
        state.board.append(state.deck[state.deck_idx])
        state.deck_idx += 1
    state.undo_stack.append(("DEAL", n, list(state.bets), state.last_raiser, state.last_raise_amount))
    state.action_history.append(DEAL)
    state.round_idx += 1
    state.chance_pending = False
    state.bets, state.last_raiser, state.last_raise_amount = [0.0]*NUM_PLAYERS, -1, 0.0
    if state.round_idx > 3 or len(state.board) >= 5:
        _resolve_hand(state)
    else:
        for offset in range(1, NUM_PLAYERS + 1):
            p = offset % NUM_PLAYERS
            if state.active[p] and not state.all_in[p]:
                state.current_player = p
                break
        else: _run_out_board_and_resolve(state)
    return state

def apply_action(state, action_index):
    p = state.current_player
    is_fold, total_bet = action_index_to_chips(state, action_index)
    state.undo_stack.append({
        "stacks": list(state.stacks), "pot": state.pot, "bets": list(state.bets),
        "active": list(state.active), "all_in": list(state.all_in),
        "last_raiser": state.last_raiser, "last_raise_amount": state.last_raise_amount,
        "current_player": state.current_player,
    })
    state.action_history.append(action_index)
    if is_fold: state.active[p] = False
    else:
        add = total_bet - state.bets[p]
        state.stacks[p] -= add
        state.bets[p], state.pot = total_bet, state.pot + add
        if add > 0 and total_bet > max([state.bets[q] for q in range(NUM_PLAYERS) if q != p] + [0]):
            state.last_raiser, state.last_raise_amount = p, add
        if state.stacks[p] <= 0: state.all_in[p] = True
    if sum(state.active) == 1: _resolve_hand(state)
    else: _advance_to_next_player(state)
    return state

def _advance_to_next_player(state):
    can_act = [i for i in range(NUM_PLAYERS) if state.active[i] and not state.all_in[i]]
    # Only run out when nobody can bet anymore (everyone still in is all-in).
    # If len(can_act)==1, that player may still need to call/fold to an all-in — do not skip them.
    if len(can_act) == 0:
        _run_out_board_and_resolve(state)
        return
    if _is_round_complete(state):
        if state.round_idx >= 3:
            _resolve_hand(state)
        else:
            state.chance_pending, state.current_player = True, -1
        return
    next_p = (state.current_player + 1) % NUM_PLAYERS
    while not state.active[next_p] or state.all_in[next_p]:
        next_p = (next_p + 1) % NUM_PLAYERS
    state.current_player = next_p

def _is_round_complete(state):
    can_act = [i for i in range(NUM_PLAYERS) if state.active[i] and not state.all_in[i]]
    if not can_act: return True
    actions_since_deal = []
    for i in range(len(state.undo_stack) - 1, -1, -1):
        entry = state.undo_stack[i]
        if isinstance(entry, tuple) and entry[0] == "DEAL": break
        if isinstance(entry, dict):
            # Fix: Ensure we don't compare "DEAL" with an integer
            hist_item = state.action_history[i]
            is_raise = (hist_item != DEAL and hist_item >= 2)
            actions_since_deal.insert(0, (entry["current_player"], i, is_raise))
    if not actions_since_deal: return False
    if not all(p in {a[0] for a in actions_since_deal} for p in can_act): return False
    if len(set(state.bets[p] for p in can_act)) > 1: return False
    last_raise_pos = max([i for i, a in enumerate(actions_since_deal) if a[2]] + [-1])
    if last_raise_pos != -1:
        raiser = actions_since_deal[last_raise_pos][0]
        for p in can_act:
            if p != raiser and not any(a[0] == p for a in actions_since_deal[last_raise_pos+1:]):
                return False
    return True

def _run_out_board_and_resolve(state):
    while len(state.board) < 5:
        n = 3 if state.round_idx == 0 else 1
        for _ in range(n):
            state.board.append(state.deck[state.deck_idx]); state.deck_idx += 1
        state.round_idx += 1
    _resolve_hand(state)

def _resolve_hand(state):
    state.done = True
    active = [p for p in range(NUM_PLAYERS) if state.active[p]]
    if len(active) == 1:
        state.stacks[active[0]] += state.pot
    else:
        contributions = [STARTING_STACK_BB - state.stacks[p] for p in range(NUM_PLAYERS)]
        levels = sorted(set(c for c in contributions if c > 0))
        prev = 0
        for level in levels:
            eligible = [p for p in range(NUM_PLAYERS) if contributions[p] >= level]
            side_pot = (level - prev) * len(eligible)
            winners = [p for p in eligible if state.active[p]]
            if winners:
                best_h = max(evaluate_hand(state.hole_cards[p] + state.board) for p in winners)
                pot_winners = [p for p in winners if evaluate_hand(state.hole_cards[p] + state.board) == best_h]
                for w in pot_winners: state.stacks[w] += side_pot / len(pot_winners)
            prev = level