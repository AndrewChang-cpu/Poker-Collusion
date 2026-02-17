"""
Main entry point for 3-player Pluribus-style CFR training.

Usage:
    python main.py kuhn    # Validate on 3-player Kuhn poker (fast)
    python main.py nlhe    # Train on 3-player No-Limit Hold'em
"""

import sys
import time


def run_kuhn():
    """Train and evaluate on 3-player Kuhn poker (validation)."""
    import kuhn3p as game
    from cfr import CFRTrainer
    from evaluate import evaluate

    print("=" * 60)
    print("3-Player Kuhn Poker - CFR Validation")
    print("=" * 60)

    trainer = CFRTrainer(game, num_players=3, use_linear_cfr=True)

    start = time.time()
    trainer.train(num_iterations=50000, log_interval=10000)
    elapsed = time.time() - start
    print(f"\nTraining time: {elapsed:.1f}s")

    # Print some learned strategies
    print("\n--- Sample Strategies ---")
    strategies = trainer.get_all_strategies()
    # Show a few interesting info sets
    sample_keys = sorted(strategies.keys())[:20]
    for key in sample_keys:
        actions, probs = strategies[key]
        prob_str = ", ".join(f"{a}:{p:.3f}" for a, p in zip(actions, probs))
        print(f"  {key:<25} -> {prob_str}")

    # Evaluate by playing many hands
    print("\n--- Evaluation ---")
    evaluate(game, trainer, num_hands=50000)

    trainer.save("kuhn3p_strategy.pkl")


def run_nlhe():
    """Train and evaluate on 3-player NLHE."""
    import nlhe3p as game
    from cfr import CFRTrainer
    from evaluate import evaluate, evaluate_with_variance

    print("=" * 60)
    print("3-Player No-Limit Hold'em - MCCFR Training")
    print("=" * 60)
    print(f"Stack: {game.STARTING_STACK} BB")
    print(f"Blinds: {game.SMALL_BLIND}/{game.BIG_BLIND}")
    print(f"Actions: {game.ALL_ACTIONS}")
    print()

    trainer = CFRTrainer(
        game,
        num_players=3,
        use_linear_cfr=True,
        prune_threshold=-300,
    )

    # Train - adjust iterations based on available time
    # 10k iterations is a starting point; more = better strategy
    start = time.time()
    trainer.train(num_iterations=10000, log_interval=1000)
    elapsed = time.time() - start
    print(f"\nTraining time: {elapsed:.1f}s")
    print(f"Info sets discovered: {len(trainer.regret_sum)}")

    # Evaluate
    print("\n--- Evaluation ---")
    evaluate_with_variance(game, trainer, num_hands=5000)

    trainer.save("nlhe3p_strategy.pkl")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "kuhn"

    if mode == "kuhn":
        run_kuhn()
    elif mode == "nlhe":
        run_nlhe()
    else:
        print(f"Unknown mode: {mode}. Use 'kuhn' or 'nlhe'.")
