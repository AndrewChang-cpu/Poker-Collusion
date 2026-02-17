# Pluribus-Style 3-Player NLHE Self-Play

Milestone 1 implementation: MCCFR self-play training for 3-player No-Limit Texas Hold'em.

## Project Structure

```
pluribus_3p/
├── main.py        # Entry point (train Kuhn or NLHE)
├── cfr.py         # MCCFR trainer (game-agnostic)
├── kuhn3p.py      # 3-player Kuhn poker (validation game)
├── nlhe3p.py      # 3-player NLHE engine + hand evaluator + abstraction
└── evaluate.py    # Evaluation: mbb/g with confidence intervals
```

## Usage

```bash
# Step 1: Validate CFR on Kuhn poker (fast, ~20s)
python main.py kuhn

# Step 2: Train on 3-player NLHE
python main.py nlhe
```

## How It Works

1. **Game engine** (`nlhe3p.py`): Self-contained 3-player NLHE with:
   - Full deck, 4 betting rounds (preflop/flop/turn/river)
   - Action abstraction: fold, check, call, half-pot, pot, all-in
   - Hand strength bucketing (preflop: rank-based, postflop: hand category)

2. **MCCFR trainer** (`cfr.py`): External sampling Monte Carlo CFR with:
   - Linear CFR weighting (iteration t weighted by t)
   - Regret matching for strategy computation
   - Optional pruning of very negative regret actions
   - Works with ANY game module that implements the game interface

3. **Evaluation** (`evaluate.py`): Plays trained agents against each other, reports mbb/g with standard errors.

## Game Interface

Both `kuhn3p.py` and `nlhe3p.py` implement the same interface:

```python
deal_new_hand() -> state
get_current_player(state) -> int
get_legal_actions(state) -> list
get_info_key(state, player) -> str
is_terminal(state) -> bool
get_payoffs(state) -> list[float]
apply_action(state, action) -> state
is_chance_node(state) -> bool
sample_chance(state) -> state
```

To swap games, just change the import in `main.py`.

## Training Parameters

Key knobs to adjust in `main.py`:
- `num_iterations`: More = better strategy (start with 10k, aim for 50k+)
- `use_linear_cfr`: True recommended (faster convergence)
- `prune_threshold`: Set to -300 or None to disable

## For Milestone 2 (Cooperation)

The CFR trainer is game-agnostic. To implement cooperative agents:
1. Create a modified trainer that optimizes a **shared objective** for 2 of the 3 players
2. The shared objective: maximize `payoff[p1] + payoff[p2]` (minimizing opponent's return)
3. Train the cooperative pair using the same MCCFR framework but with the joint utility function
4. Evaluate cooperative pair vs. the baseline single-agent strategy

## Dependencies

- Python 3.8+
- NumPy
- (Optional) RLCard — can replace `nlhe3p.py` with RLCard's environment for richer game logic

## Notes

- The hand evaluator and game engine are fully self-contained (no external poker libraries)
- Info set count grows with training iterations; for serious training, consider coarser abstraction
- The blueprint strategy alone (no real-time search) is the baseline — this matches your project scope
