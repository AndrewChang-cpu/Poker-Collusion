"""
MAPPO for the Leduc team seats: one shared decentralized actor (own
observation + seat id) and a centralized critic that sees all hole cards
during training only. See docs/RL_ALGORITHM.md for the rationale.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from poker_collusion.config import NUM_ACTIONS
from poker_collusion.rl.env import GLOBAL_DIM, OBS_DIM, LeducTeamEnv


@dataclass
class PPOConfig:
    hands_per_update: int = 4096
    updates: int = 500
    epochs: int = 4
    minibatch: int = 2048
    lr: float = 3e-4
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    reward_scale: float = 0.1  # BB -> roughly unit-scale returns
    checkpoint_every: int = 25
    eval_hands: int = 20000
    seed: int = 0


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = _mlp(OBS_DIM, hidden, NUM_ACTIONS)

    def dist(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.net(obs).masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False):
        d = self.dist(torch.as_tensor(obs)[None], torch.as_tensor(mask)[None])
        a = d.probs.argmax(-1) if greedy else d.sample()
        return int(a), float(d.log_prob(a))

    @torch.no_grad()
    def probs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.dist(torch.as_tensor(obs)[None], torch.as_tensor(mask)[None]).probs[0].numpy()


def save_actor(actor: Actor, path: str, meta: Dict) -> None:
    torch.save({"state_dict": actor.state_dict(), "meta": meta}, path)


def load_actor(path: str) -> Actor:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    actor = Actor(ck["meta"].get("hidden", 128))
    actor.load_state_dict(ck["state_dict"])
    actor.eval()
    return actor


def play_hands(env: LeducTeamEnv, actor: Actor, n: int, record: bool = False):
    """Roll out n hands. Returns per-hand payoffs and, if record, decision data."""
    payoffs = np.zeros((n, 3))
    data: List[Dict] = []
    for h in range(n):
        step = env.reset()
        hand: List[Dict] = []
        while not step["done"]:
            a, logp = actor.act(step["obs"], step["mask"])
            if record:
                hand.append({"seat": step["seat"], "obs": step["obs"], "global": step["global"],
                             "mask": step["mask"], "action": a, "logp": logp})
            step = env.step(a)
        payoffs[h] = step["payoffs"]
        for d in hand:
            d["reward"] = step["rewards"][d["seat"]]
        data.extend(hand)
    return payoffs, data


def train(env: LeducTeamEnv, cfg: PPOConfig, out_dir: str,
          eval_env: Optional[LeducTeamEnv] = None) -> Actor:
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(2)
    actor, critic = Actor(cfg.hidden), _mlp(GLOBAL_DIM, cfg.hidden, 1)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr, eps=1e-5)
    log_path = os.path.join(out_dir, "train_log.jsonl")
    meta = {**asdict(cfg), "learner_seats": list(env.learner_seats), "team_objective": env.team_objective}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    t0 = time.time()
    for u in range(1, cfg.updates + 1):
        payoffs, data = play_hands(env, actor, cfg.hands_per_update, record=True)
        obs = torch.as_tensor(np.stack([d["obs"] for d in data]))
        glob = torch.as_tensor(np.stack([d["global"] for d in data]))
        mask = torch.as_tensor(np.stack([d["mask"] for d in data]))
        act = torch.as_tensor([d["action"] for d in data])
        old_logp = torch.as_tensor([d["logp"] for d in data])
        # Single terminal reward, gamma=1: the return of every decision is the hand's team payoff.
        ret = torch.as_tensor([d["reward"] for d in data], dtype=torch.float32) * cfg.reward_scale
        with torch.no_grad():
            adv = ret - critic(glob).squeeze(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(data)
        for _ in range(cfg.epochs):
            for idx in torch.randperm(n).split(cfg.minibatch):
                d = actor.dist(obs[idx], mask[idx])
                ratio = torch.exp(d.log_prob(act[idx]) - old_logp[idx])
                pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[idx]).mean()
                vloss = (critic(glob[idx]).squeeze(-1) - ret[idx]).pow(2).mean()
                ent = d.entropy().mean()
                loss = pg + cfg.vf_coef * vloss - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), cfg.max_grad_norm)
                opt.step()

        team = payoffs[:, list(env.learner_seats)].sum(1)
        rec = {"update": u, "hands": u * cfg.hands_per_update, "decisions": n,
               "train_team_bb": float(team.mean()), "entropy": float(ent.detach()),
               "vloss": float(vloss.detach()), "elapsed_s": round(time.time() - t0, 1)}
        if u % cfg.checkpoint_every == 0 or u == cfg.updates:
            path = os.path.join(out_dir, f"actor_{u:04d}.pt")
            save_actor(actor, path, {**meta, "update": u})
            if eval_env is not None:
                rec.update(evaluate(eval_env, actor, cfg.eval_hands, seed=12345))
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    return actor


def evaluate(env: LeducTeamEnv, actor: Actor, n: int, seed: int = 12345) -> Dict:
    state = np.random.get_state()
    np.random.seed(seed)
    opp = env.opponent
    l0, m0 = getattr(opp, "lookups", 0), getattr(opp, "misses", 0)
    payoffs, _ = play_hands(env, actor, n)
    np.random.set_state(state)
    team = payoffs[:, list(env.learner_seats)].sum(1)
    out = {"eval_team_bb": float(team.mean()), "eval_team_se": float(team.std() / np.sqrt(n)),
           "eval_seat_bb": [float(x) for x in payoffs.mean(0)],
           "eval_sum_abs_max": float(np.abs(payoffs.sum(1)).max())}
    if hasattr(opp, "lookups"):
        out["opp_miss_rate"] = (opp.misses - m0) / max(1, opp.lookups - l0)
    return out
