"""
Game logic for Leduc: 2 streets, 1 board card.
"""

from poker_collusion.config import NUM_PLAYERS, POSTFLOP_ORDER, STARTING_STACK_BB
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
    assert state.chance_pending and not state.done
    s = state.copy()
    s.board.append(s.deck[s.deck_idx])
    s.deck_idx += 1
    s.action_history = s.action_history + (DEAL,)
    s.round_idx += 1
    s.chance_pending = False
    s.bets = (0.0,) * NUM_PLAYERS
    s.last_raiser = -1
    s.last_raise_amount = 0.0

    if s.round_idx > 1:
        _resolve_hand(s)
        return s

    for p in POSTFLOP_ORDER:
        if s.active[p] and not s.all_in[p]:
            s.current_player = p
            break
    else:
        _resolve_hand(s)
    return s

def apply_action(state, action_index):
    s = state.copy()
    p = s.current_player
    is_fold, total_bet = action_index_to_chips(s, action_index)
    total_bet = round(total_bet, 1)
    s.action_history = s.action_history + (action_index,)
    s.actor_history = s.actor_history + (p,)

    if is_fold:
        s.active = s.active[:p] + (False,) + s.active[p + 1:]
    else:
        add = total_bet - s.bets[p]
        s.stacks = s.stacks[:p] + (round(s.stacks[p] - add, 1),) + s.stacks[p + 1:]
        s.bets = s.bets[:p] + (total_bet,) + s.bets[p + 1:]
        s.pot += add
        prev_max = max(s.bets[q] for q in range(NUM_PLAYERS) if q != p)
        if add > 0 and total_bet > prev_max:
            s.last_raiser = p
            s.last_raise_amount = total_bet - prev_max
        if s.stacks[p] <= 0:
            s.all_in = s.all_in[:p] + (True,) + s.all_in[p + 1:]

    if sum(s.active) == 1:
        _resolve_hand(s)
        return s
    _advance_to_next_player(s)
    return s

def _advance_to_next_player(state):
    can_act = [p for p in range(NUM_PLAYERS) if state.active[p] and not state.all_in[p]]
    if not can_act:
        if state.round_idx == 0:
            state.chance_pending = True
            state.current_player = -1
        else:
            _resolve_hand(state)
        return
    if _is_round_complete(state):
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
    can_act = [p for p in range(NUM_PLAYERS) if state.active[p] and not state.all_in[p]]
    if not can_act: return True
    if len(set(state.bets[p] for p in can_act)) > 1: return False
    
    hist = state.action_history
    start = 0
    for i in range(len(hist)-1, -1, -1):
        if hist[i] == DEAL:
            start = i + 1
            break
    street_acts = state.actor_history[sum(1 for x in hist[:start] if x != DEAL):]
    for p in can_act:
        if p not in street_acts: return False
    return True

def _resolve_hand(state):
    state.done = True
    active = [p for p in range(NUM_PLAYERS) if state.active[p]]
    if len(active) == 1:
        w = active[0]
        state.stacks = state.stacks[:w] + (state.stacks[w] + state.pot,) + state.stacks[w + 1:]
        return
    contributions = [STARTING_STACK_BB - state.stacks[p] for p in range(NUM_PLAYERS)]
    _resolve_side_pots(state, active, contributions)

def _resolve_side_pots(state, active_players, contributions):
    """Side pot resolution (Modified from NLHE logic to be Leduc-safe)."""
    levels = sorted(set(c for c in contributions if c > 0))
    prev = 0.0
    for level in levels:
        eligible_count = [p for p in range(NUM_PLAYERS) if contributions[p] >= level]
        slice_size = (level - prev) * len(eligible_count)
        eligible_win = [p for p in eligible_count if state.active[p]]
        if not eligible_win:
            prev = level
            continue
        best_hand = None
        winners = []
        for p in eligible_win:
            h = evaluate_hand(state.hole_cards[p] + state.board)
            if best_hand is None or h > best_hand:
                best_hand, winners = h, [p]
            elif h == best_hand:
                winners.append(p)
        share = slice_size / len(winners)
        stacks = list(state.stacks)
        for w in winners:
            stacks[w] += share
        state.stacks = tuple(stacks)
        prev = level