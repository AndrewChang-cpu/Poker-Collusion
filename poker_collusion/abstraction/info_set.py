"""
Leduc Info Set: (round_idx, rank_bucket, history).
Supports "Shared Information" (psychic) keys for colluding teams.
"""

_DEAL = "DEAL"

def get_info_key(state, player, team_seats=None):
    """
    Generate hashable keys for Leduc Hold'em.
    
    If team_seats is provided and 'player' is in that team, the key is 
    augmented with the hole cards of all teammates.
    """
    my_hole = state.hole_cards[player][0]
    round_idx = state.round_idx

    # 1. Determine the "card bucket" component
    if team_seats and player in team_seats:
        # Psychic variant: Include teammate hole cards in the key
        teammates = [state.hole_cards[s][0] for s in team_seats if s != player]
        if round_idx == 0:
            # Bucket is a tuple of (my_card, teammate_cards...)
            bucket = (my_hole, tuple(teammates))
        else:
            board_rank = state.board[0]
            # Bucket is (my_card, teammate_cards..., board)
            bucket = (my_hole, tuple(teammates), board_rank)
    else:
        # Standard variant: Only my cards and board
        if round_idx == 0:
            bucket = my_hole
        else:
            board_rank = state.board[0]
            bucket = (my_hole * 4) + board_rank

    # 2. Generate the history component
    history = []
    actor_idx = 0
    for a in state.action_history:
        if a == _DEAL:
            history.append(_DEAL)
        else:
            history.append((state.actor_history[actor_idx], a))
            actor_idx += 1
            
    return (round_idx, bucket, tuple(history))