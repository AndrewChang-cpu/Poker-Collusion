"""
Game logic: legal actions, apply_action, undo_action, chance nodes, round advancement.
"""

# Set by apply_action and sample_chance so undo_action() can undo without explicit state.
_current_state = None

from poker_collusion.config import NUM_PLAYERS, STARTING_STACK_BB
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
    """Deal next street (flop/turn/river), append DEAL to history, advance round."""
    global _current_state
    _current_state = state
    
    if not state.chance_pending or state.done:
        return state
    n = 3 if state.round_idx == 0 else 1
    for _ in range(n):
        state.board.append(state.deck[state.deck_idx])
        state.deck_idx += 1
    state.action_history.append(DEAL)
    state.round_idx += 1
    state.chance_pending = False
    
    # Snapshot for DEAL with stacks and pot for perfect reversion
    state.undo_stack.append(("DEAL", n, list(state.bets), state.last_raiser, 
                             state.last_raise_amount, list(state.stacks), state.pot))
    
    state.bets = [0.0] * NUM_PLAYERS
    state.last_raiser = -1
    state.last_raise_amount = 0.0
    if state.round_idx > 3 or len(state.board) >= 5:
        _resolve_hand(state)
        return state
    for offset in range(1, NUM_PLAYERS + 1):
        p = (offset) % NUM_PLAYERS
        if state.active[p] and not state.all_in[p]:
            state.current_player = p
            break
    else:
        _run_out_board_and_resolve(state)
    return state


def apply_action(state, action_index):
    """Apply action (index 0..9), update state, possibly set chance_pending. Mutates state."""
    global _current_state
    _current_state = state
    p = state.current_player
    is_fold, total_bet = action_index_to_chips(state, action_index)
    # Snapshot for undo
    state.undo_stack.append({
        "stacks": list(state.stacks),
        "pot": state.pot,
        "bets": list(state.bets),
        "active": list(state.active),
        "all_in": list(state.all_in),
        "last_raiser": state.last_raiser,
        "last_raise_amount": state.last_raise_amount,
        "current_player": state.current_player,
    })
    state.action_history.append(action_index)

    if is_fold:
        state.active[p] = False
    else:
        add = total_bet - state.bets[p]
        state.stacks[p] -= add
        state.bets[p] = total_bet
        state.pot += add
        if add > 0 and total_bet > max(state.bets[q] for q in range(NUM_PLAYERS) if q != p):
            state.last_raiser = p
            state.last_raise_amount = add
        if state.stacks[p] <= 0:
            state.all_in[p] = True

    active_count = sum(state.active)
    if active_count == 1:
        _resolve_hand(state)
        return state

    _advance_to_next_player(state)
    return state


def undo_action(state=None):
    """Undo last apply_action or sample_chance with run-out support."""
    global _current_state
    if state is None:
        state = _current_state
    if state is None or not state.undo_stack:
        return
        
    top = state.undo_stack.pop()
    
    if isinstance(top, tuple) and top[0] == "RUNOUT":
        _, added_cards, added_rounds = top
        state.deck_idx -= added_cards
        for _ in range(added_cards):
            state.board.pop()
        state.round_idx -= added_rounds
        state.done = False
        return undo_action(state)
        
    if isinstance(top, tuple) and top[0] == "DEAL":
        _, n, prev_bets, prev_raiser, prev_raise_amt, prev_stacks, prev_pot = top
        state.action_history.pop()
        state.round_idx -= 1
        state.deck_idx -= n
        for _ in range(n):
            state.board.pop()
        state.chance_pending = True
        state.done = False
        state.current_player = -1
        state.bets = list(prev_bets)
        state.last_raiser = prev_raiser
        state.last_raise_amount = prev_raise_amt
        state.stacks = list(prev_stacks)
        state.pot = prev_pot
        return
        
    state.action_history.pop()
    state.stacks = top["stacks"]
    state.pot = top["pot"]
    state.bets = top["bets"]
    state.active = top["active"]
    state.all_in = top["all_in"]
    state.last_raiser = top["last_raiser"]
    state.last_raise_amount = top["last_raise_amount"]
    state.current_player = top["current_player"]
    state.done = False


def _advance_to_next_player(state):
    can_act = [i for i in range(NUM_PLAYERS) if state.active[i] and not state.all_in[i]]
    if len(can_act) <= 1:
        _run_out_board_and_resolve(state)
        return
    if _is_round_complete(state):
        if state.round_idx >= 3:
            _resolve_hand(state)
            return
        state.chance_pending = True
        state.current_player = -1
        return
    next_p = (state.current_player + 1) % NUM_PLAYERS
    while not state.active[next_p] or state.all_in[next_p]:
        next_p = (next_p + 1) % NUM_PLAYERS
    state.current_player = next_p


def _who_acted_this_round(state):
    """Correctly identify actors by checking snapshots in history."""
    hist = state.action_history
    start = 0
    for i in range(len(hist) - 1, -1, -1):
        if hist[i] == DEAL:
            start = i + 1
            break
    round_actions = hist[start:]
    num_round_actions = len(round_actions)
    if num_round_actions == 0:
        return set()
    acted = set()
    snapshots = state.undo_stack[-num_round_actions:]
    for s in snapshots:
        if isinstance(s, dict):
            acted.add(s["current_player"])
    return acted


def _is_round_complete(state):
    can_act = [i for i in range(NUM_PLAYERS) if state.active[i] and not state.all_in[i]]
    if not can_act:
        return True
    acted_this_round = _who_acted_this_round(state)
    for p in can_act:
        if p not in acted_this_round:
            return False
    bets_active = [state.bets[p] for p in can_act]
    if len(set(bets_active)) > 1:
        return False
    if state.last_raiser >= 0 and state.last_raiser in can_act:
        hist = state.action_history
        start = 0
        for i in range(len(hist) - 1, -1, -1):
            if hist[i] == DEAL:
                start = i + 1
                break
        round_actions = hist[start:]
        is_preflop = state.round_idx == 0
        order = [0, 1, 2] if is_preflop else [1, 2, 0]
        raise_idx = -1
        cur = 0
        for i, a in enumerate(round_actions):
            if a == DEAL:
                break
            player = order[cur % len(order)]
            if a not in (0, 1) and player == state.last_raiser:
                raise_idx = i
            cur += 1
        if raise_idx >= 0:
            for p in can_act:
                if p == state.last_raiser:
                    continue
                found_after = False
                cur = 0
                for i, a in enumerate(round_actions):
                    if a == DEAL:
                        break
                    player = order[cur % len(order)]
                    if i > raise_idx and player == p:
                        found_after = True
                        break
                    cur += 1
                if not found_after:
                    return False
    return True


def _run_out_board_and_resolve(state):
    """Run out community cards and resolve with undo tracking."""
    added_cards = 0
    added_rounds = 0
    while len(state.board) < 5:
        n = 3 if state.round_idx == 0 else 1
        if state.deck_idx + n > len(state.deck):
            break
        for _ in range(n):
            state.board.append(state.deck[state.deck_idx])
            state.deck_idx += 1
            added_cards += 1
        state.round_idx += 1
        added_rounds += 1
    if added_cards > 0 or added_rounds > 0:
        state.undo_stack.append(("RUNOUT", added_cards, added_rounds))
    _resolve_hand(state)


def _resolve_hand(state):
    state.done = True
    active = [p for p in range(NUM_PLAYERS) if state.active[p]]
    if len(active) == 1:
        state.stacks[active[0]] += state.pot
        return
    contributions = [STARTING_STACK_BB - state.stacks[p] for p in range(NUM_PLAYERS)]
    _resolve_side_pots(state, active, contributions)


def _resolve_side_pots(state, active_players, contributions):
    """Distribute pot supporting split pots."""
    levels = sorted(set(contributions[p] for p in range(NUM_PLAYERS) if contributions[p] > 0))
    prev = 0
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
        share = slice_size / len(winners)
        for w in winners:
            state.stacks[w] += share
        prev = level