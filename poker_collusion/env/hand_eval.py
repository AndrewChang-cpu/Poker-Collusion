"""
Leduc Hand Evaluator: Pair > High Card.
"""

def evaluate_hand(cards):
    """
    Evaluate a 2-card Leduc hand (1 hole, 1 board).
    Input: list of 2 card ranks [hole, board].
    Output: integer score where higher is better.
    """
    if len(cards) < 2:
        # Pre-showdown or folded scenarios
        return cards[0] if cards else -1
        
    hole = cards[0]
    board = cards[1]
    
    if hole == board:
        # Pair: score starts at 100 to ensure any pair beats any high card
        return 100 + hole
    else:
        # High Card
        return hole