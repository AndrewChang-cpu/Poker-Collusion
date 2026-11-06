import numpy as np
import pytest

from poker_collusion.rl import probes
from poker_collusion.rl.env import OBS_DIM, LeducTeamEnv


def _uniform(state, seat, legal):
    return np.full(len(legal), 1.0 / len(legal))


def test_episode_rewards_team_sum_and_obs_shape():
    np.random.seed(0)
    env = LeducTeamEnv([0, 1], _uniform)
    for _ in range(200):
        step = env.reset()
        while not step["done"]:
            assert step["obs"].shape == (OBS_DIM,)
            assert step["seat"] in (0, 1)
            step = env.step(int(np.flatnonzero(step["mask"])[0]))
        p = step["payoffs"]
        assert abs(sum(p)) < 1e-9
        assert step["rewards"][0] == step["rewards"][1] == pytest.approx(p[0] + p[1])


def test_actor_obs_ignores_hidden_cards():
    out = probes.obs_counterfactual({"hands": 200, "seed": 1})
    assert out["obs_max_diff"] == 0
    assert out["critic_changed_frac"] > 0.5


def test_illegal_action_raises():
    assert probes.illegal_action({"seed": 2}) == {"raised": True, "state_changed": False, "tried": 2}


def test_decentralized_policy_uses_current_key_format():
    out = probes.decentralized_policy({"hands": 100, "seed": 3})
    assert out["errors"] == 0, out["first_error"]
    assert out["true_key_queried_frac"] == 1.0
    assert out["output_max_diff"] == 0.0
