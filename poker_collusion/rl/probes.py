"""
Deterministic environment probes used by the TouchStone suite
(touchstone_suite/). Each probe takes a JSON-able dict and returns a dict of
independently measured facts; assertions live in the suite, not here.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from poker_collusion.abstraction.info_set import get_info_key
from poker_collusion.config import NUM_PLAYERS, STARTING_STACK_BB
from poker_collusion.env import game_logic as g
from poker_collusion.env.game_state import deal_new_hand, get_payoffs
from poker_collusion.rl.env import LeducTeamEnv, global_state, observe

DEFAULT_OPPONENT = "models/leduc_baseline_5000k_ctx.pkl"  # compact_checkpoint output with fallback table


def _uniform(state, seat, legal):
    return np.full(len(legal), 1.0 / len(legal))


def _opponent(inp: Dict):
    if inp.get("opponent", "uniform") == "uniform":
        return _uniform
    from poker_collusion.rl.frozen import FrozenPolicy
    return FrozenPolicy(inp.get("opponent_path", DEFAULT_OPPONENT), inp.get("opponent_fallback", "context"))


def resolve_checkpoint(path: str) -> str:
    """A run directory resolves to its newest actor_*.pt."""
    import glob
    import os
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "actor_*.pt")))
        if not found:
            raise FileNotFoundError(f"no actor_*.pt in {path}")
        return found[-1]
    return path


def _learner(inp: Dict):
    """Returns (obs, mask) -> action. Uniform-random unless a checkpoint is given."""
    if inp.get("checkpoint"):
        import torch
        from poker_collusion.rl.ppo import load_actor
        torch.manual_seed(inp.get("seed", 0))
        actor = load_actor(resolve_checkpoint(inp["checkpoint"]))
        return lambda obs, mask: actor.act(obs, mask)[0]
    return lambda obs, mask: int(np.random.choice(np.flatnonzero(mask)))


def _counterfactual(state, seat: int, rng, keep=()):
    """Resample every card hidden from `seat` (other holes, undealt deck), keeping seats in `keep`."""
    s = state.copy()
    deck = list(state.deck)
    fixed = {seat, *keep}
    visible = {p for p in fixed}  # hole positions p (deck index p) stay
    board_idx = set(range(NUM_PLAYERS, NUM_PLAYERS + len(state.board)))
    free = [i for i in range(len(deck)) if i not in visible and i not in board_idx]
    vals = [deck[i] for i in free]
    rng.shuffle(vals)
    for i, v in zip(free, vals):
        deck[i] = v
    s.deck = deck
    s.hole_cards = [[deck[p]] for p in range(NUM_PLAYERS)]
    return s


def obs_counterfactual(inp: Dict) -> Dict:
    """Max change in each seat-visible view when hidden cards are resampled."""
    rng = np.random.default_rng(inp.get("seed", 0))
    np.random.seed(inp.get("seed", 0))
    act = _learner(inp)
    team = tuple(inp.get("team_seats", [0, 1]))
    out = {"decisions": 0, "obs_max_diff": 0.0, "critic_changed_frac": 0.0,
           "cfr_key_changed": 0, "psychic_key_changed_by_opponent_card": 0,
           "psychic_key_changed_by_teammate_card": 0}
    changed = 0
    for _ in range(inp.get("hands", 500)):
        s = deal_new_hand()
        while not s.done:
            if s.chance_pending:
                s = g.sample_chance(s)
                continue
            p = s.current_player
            cf = _counterfactual(s, p, rng)
            out["decisions"] += 1
            out["obs_max_diff"] = max(out["obs_max_diff"], float(np.abs(observe(s, p) - observe(cf, p)).max()))
            changed += not np.array_equal(global_state(s, p), global_state(cf, p))
            out["cfr_key_changed"] += get_info_key(s, p) != get_info_key(cf, p)
            if p in team:
                mate = [q for q in team if q != p]
                cf_opp = _counterfactual(s, p, rng, keep=mate)  # only opponent/deck hidden cards move
                out["psychic_key_changed_by_opponent_card"] += (
                    get_info_key(s, p, team_seats=list(team)) != get_info_key(cf_opp, p, team_seats=list(team)))
                out["psychic_key_changed_by_teammate_card"] += (
                    get_info_key(s, p, team_seats=list(team)) != get_info_key(cf, p, team_seats=list(team)))
            s = g.apply_action(s, act(observe(s, p), np.isin(range(10), g.get_legal_actions(s))))
    out["critic_changed_frac"] = changed / max(1, out["decisions"])
    return out


def conservation(inp: Dict) -> Dict:
    """Chip conservation, stack sanity and termination over many hands."""
    np.random.seed(inp.get("seed", 0))
    env = LeducTeamEnv(inp.get("team_seats", [0, 1]), _opponent(inp))
    act = _learner(inp)
    n = inp.get("hands", 5000)
    max_sum, min_stack, max_steps, terminated = 0.0, float("inf"), 0, 0
    for _ in range(n):
        step, steps = env.reset(), 0
        while not step["done"] and steps < 200:
            step, steps = env.step(act(step["obs"], step["mask"])), steps + 1
        terminated += step["done"]
        max_steps = max(max_steps, steps)
        if step["done"]:
            max_sum = max(max_sum, abs(sum(step["payoffs"])))
            min_stack = min(min_stack, min(env.state.stacks))
    return {"hands": n, "terminated": terminated, "max_abs_payoff_sum": max_sum,
            "min_final_stack": min_stack, "max_learner_steps": max_steps,
            "max_total_chips": NUM_PLAYERS * STARTING_STACK_BB}


def team_reward(inp: Dict) -> Dict:
    """Reward a component computes for fixed terminal payoffs (bypasses the engine)."""
    payoffs = [float(x) for x in inp["payoffs"]]
    team = list(inp.get("team_seats", [0, 1]))
    obj = inp.get("objective", "utilitarian")
    seat = inp.get("seat", team[0])
    if inp["component"] == "cfr_trainer":
        from poker_collusion.cfr.trainer import CFRTrainer
        tr = CFRTrainer(None, team_seats=team, team_objective=obj)
        return {"reward": float(tr._team_value(payoffs, seat)), "objective": obj}
    env = LeducTeamEnv(team, _uniform, obj)
    return {"reward": env.team_reward(payoffs, seat), "objective": obj}


def terminal_delivery(inp: Dict) -> Dict:
    """Hands ended by the opponent's action must still reward every learner decision."""
    np.random.seed(inp.get("seed", 0))
    env = LeducTeamEnv(inp.get("team_seats", [0, 1]), _opponent(inp))
    act = _learner(inp)
    ended_by_opponent, mismatched = 0, 0
    for _ in range(inp.get("hands", 3000)):
        step, seats = env.reset(), []
        if step["done"]:
            continue
        while not step["done"]:
            seats.append(step["seat"])
            step = env.step(act(step["obs"], step["mask"]))
        last_actor = env.state.actor_history[-1]
        if last_actor not in env.learner_seats:
            ended_by_opponent += 1
        team_sum = sum(step["payoffs"][q] for q in env.learner_seats)
        expected = env.team_reward(step["payoffs"], seats[0]) if seats else team_sum
        mismatched += any(abs(step["rewards"].get(q, float("nan")) - expected) > 1e-9 for q in set(seats))
    return {"ended_by_opponent": ended_by_opponent, "reward_mismatches": mismatched}


def illegal_action(inp: Dict) -> Dict:
    np.random.seed(inp.get("seed", 0))
    env = LeducTeamEnv([0, 1], _uniform)
    step = env.reset()
    while step["done"] or step["mask"].all():  # find a decision with an illegal action
        step = env.reset() if step["done"] else env.step(int(np.random.choice(np.flatnonzero(step["mask"]))))
    illegal = [a for a in range(10) if not step["mask"][a]]
    before = env.state
    try:
        env.step(illegal[0])
        return {"raised": False, "state_changed": env.state is not before, "tried": illegal[0]}
    except ValueError:
        return {"raised": True, "state_changed": env.state is not before, "tried": illegal[0]}


def checkpoint_replay(inp: Dict) -> Dict:
    """Replay a checkpoint against the frozen opponent and measure how it wins."""
    import torch
    from poker_collusion.rl.ppo import load_actor
    np.random.seed(inp.get("seed", 12345))
    torch.manual_seed(inp.get("seed", 12345))
    opp = _opponent({**inp, "opponent": "frozen"})
    team = tuple(inp.get("team_seats", [0, 1]))
    env = LeducTeamEnv(team, opp)
    ckpt = resolve_checkpoint(inp["checkpoint"])
    actor = load_actor(ckpt)
    n = inp.get("hands", 5000)
    pay = np.zeros((n, NUM_PLAYERS))
    mate_raise_vs_mate, decisions, fold_to_mate = 0, 0, 0
    weakest_vs_mate_shove, weakest_continue = 0, 0  # chip-dump signature
    opp_folds = 0
    miss_hand = np.zeros(n, dtype=bool)
    for h in range(n):
        m0 = opp.misses
        step = env.reset()
        while not step["done"]:
            a = actor.act(step["obs"], step["mask"])[0]
            s = env.state
            seat = step["seat"]
            decisions += 1
            last = s.actor_history[-1] if s.actor_history else -1
            if last in team and last != seat and s.action_history[-1] not in (0, 1):
                mate_raise_vs_mate += a not in (0, 1)
                fold_to_mate += a == 0
                if s.action_history[-1] == 9 and int(s.hole_cards[seat][0]) == 0 and s.round_idx == 0:
                    weakest_vs_mate_shove += 1
                    weakest_continue += a != 0
            step = env.step(a)
        pay[h] = step["payoffs"]
        miss_hand[h] = opp.misses > m0
        opp_folds += not any(env.state.active[q] for q in range(NUM_PLAYERS) if q not in team)
    team_bb = pay[:, list(team)].sum(1)
    return {"checkpoint_resolved": ckpt, "hands": n, "team_bb": float(team_bb.mean()), "team_se": float(team_bb.std() / np.sqrt(n)),
            "seat_bb": [float(x) for x in pay.mean(0)], "max_abs_payoff_sum": float(np.abs(pay.sum(1)).max()),
            "opp_miss_rate": opp.misses / max(1, opp.lookups), "opp_lookups": opp.lookups,
            "opp_fallback": opp.fallback_mode, "fallback_hand_rate": float(miss_hand.mean()),
            "ev_share_from_fallback_hands": float(team_bb[miss_hand].sum() / team_bb.sum()) if team_bb.sum() > 0 else 0.0,
            "opp_fold_rate": opp_folds / n, "reraise_vs_teammate": mate_raise_vs_mate,
            "fold_to_teammate_raise": fold_to_mate, "learner_decisions": decisions,
            "weakest_card_facing_teammate_shove": weakest_vs_mate_shove,
            "weakest_card_continue_rate": weakest_continue / max(1, weakest_vs_mate_shove)}


def decentralized_policy(inp: Dict) -> Dict:
    """CTDE deployment wrapper (evaluation/team_policy.py): key contract and hidden-card invariance."""
    from poker_collusion.evaluation.team_policy import CentralizedToDecentralizedPolicy
    rng = np.random.default_rng(inp.get("seed", 0))
    np.random.seed(inp.get("seed", 0))
    team = list(inp.get("team_seats", [0, 1]))

    class Recorder:  # stands in for a full-comm CFRTrainer; strategy depends on the queried key
        def __init__(self):
            self.queried = set()

        def get_average_strategy(self, key, legal):
            self.queried.add(key)
            w = np.array([1.0 + (hash((key, a)) % 7) for a in legal])
            return w / w.sum()

    out = {"decisions": 0, "errors": 0, "first_error": None, "true_key_queried": 0, "output_max_diff": 0.0}
    for _ in range(inp.get("hands", 300)):
        s = deal_new_hand()
        while not s.done:
            if s.chance_pending:
                s = g.sample_chance(s)
                continue
            p, legal = s.current_player, g.get_legal_actions(s)
            if p in team:
                out["decisions"] += 1
                rec = Recorder()
                pol = CentralizedToDecentralizedPolicy(rec, p)
                try:
                    a = pol.get_average_strategy(get_info_key(s, p), legal)
                    cf = _counterfactual(s, p, rng)
                    b = pol.get_average_strategy(get_info_key(cf, p), legal)
                    out["output_max_diff"] = max(out["output_max_diff"], float(np.abs(a - b).max()))
                    out["true_key_queried"] += _plain_key(get_info_key(s, p, team_seats=team)) in {
                        _plain_key(k) for k in rec.queried}
                except Exception as e:  # noqa: BLE001 - recorded as an observation
                    out["errors"] += 1
                    out["first_error"] = out["first_error"] or f"{type(e).__name__}: {e}"
            s = g.apply_action(s, legal[int(np.random.randint(len(legal)))])
    out["true_key_queried_frac"] = out["true_key_queried"] / max(1, out["decisions"])
    return out


def _plain_key(k):
    from poker_collusion.rl.frozen import _plain
    return _plain(k)


PROBES = {f.__name__: f for f in (decentralized_policy, obs_counterfactual, conservation, team_reward,
                                    terminal_delivery, illegal_action, checkpoint_replay)}


def run(inp: Dict) -> Dict:
    return PROBES[inp["probe"]](inp)


def main(argv: Optional[list] = None) -> None:
    import json
    import sys
    print(json.dumps(run(json.load(sys.stdin))))


if __name__ == "__main__":
    main()
