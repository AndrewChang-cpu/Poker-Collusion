# System Design: Poker Bot Web Frontend

## Architecture

```
┌─────────────────────────────┐        ┌──────────────────────────────────────┐
│   React Frontend (Vercel)   │        │   FastAPI Backend (Railway)           │
│                             │        │                                       │
│  useGame hook               │──────► │  POST /api/session/new               │
│  ├─ session UUID (memory)   │◄──────  │  POST /api/session/{id}/action       │
│  ├─ game state              │        │  POST /api/session/{id}/next_hand     │
│  └─ stagger queue           │        │  POST /api/session/{id}/switch_mode   │
│                             │        │  GET  /api/health                     │
│  Components                 │        │                                       │
│  ├─ ModeSelect              │        │  Session store (in-memory dict)       │
│  ├─ CollusionBanner         │        │  ├─ game_mode                         │
│  ├─ PokerTable              │        │  ├─ runner (Leduc or NLHE engine)     │
│  │   ├─ PlayerArea ×3       │        │  ├─ state (NLHEState)                 │
│  │   ├─ CommunityCards      │        │  └─ cumulative_bb                     │
│  │   └─ PotDisplay          │        │                                       │
│  ├─ ActionPanel             │        │  Models (loaded at startup):          │
│  ├─ ActionLog               │        │  ├─ leduc_baseline (Strategy)         │
│  └─ LoadingScreen           │        │  ├─ leduc_collusion (Strategy)        │
└─────────────────────────────┘        │  └─ nlhe_baseline (Strategy)         │
                                       │                                       │
                                       │  Game engines:                        │
                                       │  ├─ poker_collusion/ (Leduc)          │
                                       │  └─ poker_collusion_nlhe/ (NLHE)      │
                                       └──────────────────────────────────────┘
```

**Key flow:** Frontend generates a UUID on page load (memory only, no storage). Every request includes `X-Session-ID`. Backend maintains session state in a Python dict. All bot turns and chance nodes are resolved server-side in a single request; the frontend receives an ordered `bot_actions_this_turn` list and plays it back with 600ms stagger.

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Frontend framework | React 18 + Vite | Fast build, straightforward state model |
| Styling | Tailwind CSS v3 | Utility-first, no design system conflicts |
| Card assets | `react-playing-cards@0.6.3` (LGPL-3.0) | React components, verified on npm |
| Backend | FastAPI + Uvicorn | Native Python, async, typed |
| Frontend deploy | Vercel (free tier) | Auto-deploy from GitHub, CDN-backed |
| Backend deploy | Railway (free, $5/mo credit) | No spin-down, push-to-deploy from GitHub |
| No database, no WebSockets, no auth, no Redis | | |

## Repository Structure

```
poker-collusion/
├── poker_collusion/             # Leduc engine — DO NOT MODIFY
├── poker_collusion_nlhe/        # NLHE engine — restored from git HEAD~8
│   ├── config.py                # NLHE config (15 preflop, 50 postflop buckets, 52-card deck)
│   ├── env/                     # game_state.py, game_logic.py, hand_eval.py (NLHE)
│   ├── abstraction/             # actions.py, bucketing.py, info_set.py (NLHE)
│   └── cfr/strategy.py          # same interface as poker_collusion version
├── backend/
│   ├── main.py                  # FastAPI app, CORS, lifespan model loading
│   ├── session.py               # Session dataclass + in-memory store with 30-min expiry
│   ├── runner.py                # GameRunner base + LeducRunner + NLHERunner
│   ├── serializer.py            # NLHEState → GameStateResponse (Pydantic)
│   └── requirements.txt         # fastapi uvicorn pydantic numpy tqdm scikit-learn
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts               # Typed fetch wrappers, session ID management
│   │   ├── hooks/useGame.ts     # All game state, stagger queue, error handling
│   │   └── components/
│   │       ├── LoadingScreen.tsx
│   │       ├── ModeSelect.tsx
│   │       ├── CollusionBanner.tsx
│   │       ├── PokerTable.tsx
│   │       ├── PlayerArea.tsx   # Reusable: user and each bot seat
│   │       ├── CardFace.tsx     # SVG card renderer
│   │       ├── CommunityCards.tsx
│   │       ├── ActionPanel.tsx
│   │       └── ActionLog.tsx
│   ├── package.json
│   └── vite.config.ts           # VITE_API_URL env var for backend URL
├── models/
│   ├── leduc_baseline_5000k.pkl
│   ├── leduc_obs_signal_1m.pkl
│   └── blueprint_v4_21600.pkl
└── data/                        # Precomputed bucket tables (NLHE)
```

## Data Model

### Session (in-memory, backend)
```python
@dataclass
class Session:
    game_mode: Literal["leduc_baseline", "leduc_collusion", "nlhe"]
    runner: GameRunner       # holds loaded strategies + engine refs
    state: NLHEState         # mutable game state
    cumulative_bb: float     # running total across hands this session
    last_active: float       # time.time() for 30-min expiry check
    hand_number: int         # increments each hand, shown in log
```

### Game modes → models → seats
| Mode | P0 (BTN) strategy | P1 (SB) strategy | P2 (BB) |
|---|---|---|---|
| `leduc_baseline` | `leduc_baseline_5000k.pkl` | `leduc_baseline_5000k.pkl` | Human |
| `leduc_collusion` | `leduc_obs_signal_1m.pkl` | `leduc_obs_signal_1m.pkl` | Human |
| `nlhe` | `blueprint_v4_21600.pkl` | `blueprint_v4_21600.pkl` | Human |

### GameStateResponse (JSON to frontend)
```json
{
  "game_mode": "leduc_baseline",
  "street": "preflop",
  "hand_number": 4,
  "hole_cards_p2": [{"rank": "Q", "suit": "clubs"}],
  "community_cards": [],
  "stacks": [19.5, 19.0, 20.0],
  "bets": [0.5, 1.0, 0.0],
  "pot": 1.5,
  "is_terminal": false,
  "current_player": 2,
  "legal_actions": [
    {"index": 0, "label": "Fold"},
    {"index": 1, "label": "Call 1.0 BB"},
    {"index": 2, "label": "Raise to 2 BB"},
    {"index": 9, "label": "All-in 20 BB"}
  ],
  "bot_actions_this_turn": [
    {"type": "action", "player": 0, "player_name": "BTN", "label": "BTN: Raise to 2 BB"},
    {"type": "action", "player": 1, "player_name": "SB", "label": "SB: Call"}
  ],
  "all_in_seats": [],
  "hand_result": null,
  "cumulative_bb": -2.5
}
```

`hand_result` (non-null when `is_terminal: true`):
```json
{
  "payoffs_bb": [3.5, -2.0, -1.5],
  "showdown_cards": {
    "0": [{"rank": "A", "suit": "spades"}],
    "1": [{"rank": "K", "suit": "hearts"}],
    "2": [{"rank": "Q", "suit": "clubs"}]
  },
  "hand_description": "BTN wins (pair of Aces)"
}
```

## API Contracts

```
GET /api/health
  Response: { "models_loaded": { "leduc_baseline": true, "leduc_collusion": true, "nlhe": true } }

POST /api/session/new
  Headers: X-Session-ID: <uuid>
  Body:    { "game_mode": "leduc_baseline" | "leduc_collusion" | "nlhe" }
  Response: GameStateResponse
  Side-effect: deals new hand, auto-advances through BTN/SB preflop actions,
               returns state with bot_actions_this_turn populated

POST /api/session/{session_id}/action
  Body:    { "action_index": 0-9 }
  Response: GameStateResponse
  Side-effect: applies user action, runs bot turns + chance nodes until
               is_terminal or current_player == 2, populates bot_actions_this_turn

POST /api/session/{session_id}/next_hand
  Response: GameStateResponse
  Side-effect: deals new hand, increments hand_number, preserves cumulative_bb,
               auto-advances through BTN/SB preflop actions

POST /api/session/{session_id}/switch_mode
  Body:    { "game_mode": "leduc_baseline" | "leduc_collusion" | "nlhe" }
  Response: GameStateResponse
  Side-effect: creates new session state, resets cumulative_bb to 0,
               deals new hand, auto-advances preflop bots
```

### Backend action resolution loop

> **Critical engine difference:** The Leduc engine (`poker_collusion/`) returns a **new state** from `apply_action` — the original is not mutated. The NLHE engine (`poker_collusion_nlhe/`) mutates in-place. Both patterns must be handled. The simplest approach: always reassign (`state = apply_action(state, action)`) and have the NLHE runner return `state` unchanged when its `apply_action` returns `None`.

```python
def resolve_until_user_turn(state, runner, bot_actions):
    while not is_terminal(state) and get_current_player(state) != 2:
        if is_chance_node(state):
            card = sample_chance(state)
            result = apply_action(state, card)
            state = result if result is not None else state  # Leduc returns new; NLHE mutates
            bot_actions.append({"type": "deal", "cards": serialize_cards([card])})
        else:
            player = get_current_player(state)
            info_key = get_info_key(state)
            legal = get_legal_actions(state)
            action = runner.strategy_for(player).sample_action(info_key, legal)
            label = f"{player_name(player)}: {get_action_description(state, action)}"
            result = apply_action(state, action)
            state = result if result is not None else state
            bot_actions.append({"type": "action", "player": player,
                                 "player_name": player_name(player), "label": label})
    return state, bot_actions  # return updated state too
```

> **Edge case:** If both bots fold before `get_current_player(state) == 2` is ever reached, the loop exits with `is_terminal(state) == True`. The caller must handle this: `bot_actions_this_turn` will be populated (the fold events), and `hand_result` will be non-null immediately. The frontend shows the stagger then goes straight to hand-result — the user action panel never appears.

## Auth & Authorization
N/A — public demo, no auth.

## Brownfield Context

### Must not change
- `poker_collusion/` package — Leduc engine, all training and evaluation scripts
- `models/*.pkl` files — read-only at runtime
- `scripts/`, `tests/`, `data/` — untouched

### NLHE engine restoration procedure
```bash
# For each file that changed between NLHE and Leduc refactor:
git show HEAD~8:poker_collusion/<file>.py > poker_collusion_nlhe/<file>.py
# Update all internal imports: s/poker_collusion/poker_collusion_nlhe/g
# Verify:
python -c "from poker_collusion_nlhe.env import deal_new_hand; s=deal_new_hand(); print(len(s.hole_cards[0]))"
# Expected: 2
```

Files to restore (24 total): `config.py`, `env/game_state.py`, `env/game_logic.py`, `env/hand_eval.py`, `abstraction/actions.py`, `abstraction/bucketing.py`, `abstraction/info_set.py`, `cfr/strategy.py`, `cfr/trainer.py`, `cfr/debug.py`, `evaluation/*`, `search/*`, `terminal_display.py`, `typing_defs.py`, plus `__init__.py` files.

> **Post-restore addition required:** The NLHE `abstraction/actions.py` at HEAD~8 does **not** contain `get_action_description`. This function was added in the Leduc refactor. After restoring the NLHE engine, manually add `get_action_description(state, action_index) -> str` to `poker_collusion_nlhe/abstraction/actions.py`. The function signature and logic can be copied directly from `poker_collusion/abstraction/actions.py` (Leduc version), substituting `NLHE_PREFLOP_RAISE_BB` and `NLHE_POSTFLOP_POT_MULT` constants.

### Read-only imports used in backend
```python
# Leduc
from poker_collusion.env import (deal_new_hand, get_current_player, get_legal_actions,
    get_info_key, is_terminal, get_payoffs, apply_action, is_chance_node, sample_chance)
from poker_collusion.cfr.strategy import Strategy
from poker_collusion.abstraction.actions import get_action_description

# NLHE — identical interface, different namespace
from poker_collusion_nlhe.env import (...)
from poker_collusion_nlhe.cfr.strategy import Strategy
from poker_collusion_nlhe.abstraction.actions import get_action_description
```
