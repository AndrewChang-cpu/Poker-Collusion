from poker_collusion.env.hand_eval import evaluate_hand

def test_evaluate_hand_pair():
    # Pair of Aces (Rank 3) vs Pair of Jacks (Rank 0)
    assert evaluate_hand([3, 3]) == 103
    assert evaluate_hand([0, 0]) == 100
    assert evaluate_hand([3, 3]) > evaluate_hand([0, 0])

def test_evaluate_hand_high_card():
    # Ace high (Rank 3) vs King high (Rank 2)
    assert evaluate_hand([3, 1]) == 3
    assert evaluate_hand([2, 3]) == 2
    assert evaluate_hand([3, 1]) > evaluate_hand([2, 3])

def test_pair_beats_high_card():
    # Pair of Jacks (100) vs Ace high (3)
    assert evaluate_hand([0, 0]) > evaluate_hand([3, 2])

def test_pre_showdown_eval():
    # Test evaluation with only hole card (used in some logic/debug)
    assert evaluate_hand([3]) == 3
    assert evaluate_hand([]) == -1