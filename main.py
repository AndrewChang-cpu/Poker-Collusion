"""
Main entry point for 3-player Pluribus-style CFR training.

Usage:
    python main.py kuhn         # Validate on 3-player Kuhn poker (fast, ~20s)
    python main.py nlhe         # Train on built-in 3-player NLHE engine
    python main.py rlcard       # Train on RLCard 3-player NLHE (requires: pip install rlcard)
"""

import sys
import time


def run_kuhn():
    """Train and evaluate on 3-player Kuhn poker (validation)."""
    import kuhn3p as game
    from cfr import CFRTrainer
    from evaluate import evaluate

    print("=" * 60)
    print("3-Player Kuhn Poker — CFR Validation")
    print("=" * 60)

    trainer = CFRTrainer(game, num_players=3, use_linear_cfr=True)
    start = time.time()
    trainer.train(num_iterations=50_000, log_interval=10_000)
    print(f"Time: {time.time() - start:.1f}s")

    # Print sample strategies
    print("\n--- Sample Strategies ---")
    for key, (actions, probs) in sorted(trainer.get_all_strategies().items())[:20]:
        prob_str = ", ".join(f"{a}:{p:.3f}" for a, p in zip(actions, probs))
        print(f"  {key:<25} -> {prob_str}")

    print("\n--- Evaluation ---")
    evaluate(game, trainer, num_hands=50_000)
    trainer.save("kuhn3p_strategy.pkl")


def run_nlhe():
    """Train and evaluate on built-in 3-player NLHE engine."""
    import nlhe3p as game
    from cfr import CFRTrainer
    from evaluate import evaluate_with_variance

    print("=" * 60)
    print("3-Player NLHE — MCCFR Training (built-in engine)")
    print("=" * 60)

    trainer = CFRTrainer(game, num_players=3, use_linear_cfr=True, prune_threshold=None)
    start = time.time()
    trainer.train(num_iterations=10_000, log_interval=1_000)
    print(f"Time: {time.time() - start:.1f}s")

    print("\n--- Evaluation ---")
    evaluate_with_variance(game, trainer, num_hands=5_000)
    trainer.save("nlhe3p_strategy.pkl")


def run_rlcard():
    """Train and evaluate on RLCard 3-player NLHE."""
    try:
        import rlcard_nlhe3p as game
    except ImportError:
        print("ERROR: Could not import rlcard_nlhe3p.")
        print("Make sure RLCard is installed: pip install rlcard")
        sys.exit(1)

    from cfr import CFRTrainer
    from evaluate import evaluate_with_variance

    print("=" * 60)
    print("3-Player NLHE — MCCFR Training (RLCard)")
    print("=" * 60)

    # Initialize the RLCard environment
    game.init_env(seed=42)

    trainer = CFRTrainer(game, num_players=3, use_linear_cfr=True, prune_threshold=None)
    start = time.time()
    trainer.train(num_iterations=10_000, log_interval=1_000)
    print(f"Time: {time.time() - start:.1f}s")

    print("\n--- Evaluation ---")
    evaluate_with_variance(game, trainer, num_hands=5_000)
    trainer.save("rlcard_nlhe3p_strategy.pkl")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "kuhn"

    modes = {
        "kuhn": run_kuhn,
        # "nlhe": run_nlhe,
        "nlhe": run_rlcard,
    }

    if mode in modes:
        modes[mode]()
    else:
        print(f"Unknown mode: {mode}")
        print(f"Available: {', '.join(modes.keys())}")