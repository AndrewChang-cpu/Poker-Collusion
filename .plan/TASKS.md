# Tasks: Poker Bot Web Frontend
> Generated: 2026-05-14
> Source: .plan/
> Total: 20 tasks | Starting points: 3 (T-01, T-02, T-07)

## Dependency Graph

```
T-01 · Restore NLHE engine
└── T-04* · GameRunner (runner.py)
    └── T-06* · FastAPI app + endpoints (main.py)

T-02 · Backend project setup
├── T-03 · Session store (session.py)
│   └── T-06* · FastAPI app + endpoints (main.py)
├── T-04* · GameRunner (runner.py)
└── T-05 · Serializer + Pydantic models (serializer.py)
    └── T-06* · FastAPI app + endpoints (main.py)

T-07 · Frontend scaffold
├── T-08 · API client (api.ts)
│   └── T-09 · Game state hook (useGame.ts)
│       └── T-20* · App.tsx + Header.tsx (full wiring)
└── T-10 · Dark Precision CSS + Tailwind
    ├── T-11 · CardFace.tsx
    │   ├── T-12 · PlayerArea.tsx
    │   │   └── T-19* · PokerTable.tsx
    │   └── T-13 · CommunityCards.tsx
    │       └── T-19* · PokerTable.tsx
    ├── T-14 · ActionPanel.tsx
    │   └── T-19* · PokerTable.tsx
    ├── T-15 · ActionLog.tsx
    │   └── T-19* · PokerTable.tsx
    ├── T-16 · LoadingScreen.tsx
    │   └── T-20* · App.tsx + Header.tsx
    ├── T-17 · ModeSelect.tsx
    │   └── T-20* · App.tsx + Header.tsx
    └── T-18 · CollusionBanner.tsx
        └── T-20* · App.tsx + Header.tsx

* T-04 also depends on T-01
* T-06 also depends on T-03, T-05
* T-19 also depends on T-13, T-14, T-15
* T-20 also depends on T-09, T-16, T-17, T-18
```

---

## Tasks

### T-01 · Restore NLHE engine from git history
**Status:** `done`
**Depends on:** none
**Files:** `[new] poker_collusion_nlhe/` (24 files — full package mirror of `poker_collusion/` at HEAD~8)
**What:** Extract the pre-Leduc NLHE engine from git history into a new `poker_collusion_nlhe/` package.
1. For each of the 24 files listed in SYSTEM-DESIGN.md Brownfield section, run: `git show HEAD~8:poker_collusion/<file>.py > poker_collusion_nlhe/<file>.py`
2. Replace all `from poker_collusion.` and `import poker_collusion.` strings with `poker_collusion_nlhe` throughout the new package (all 24 files).
3. Add `get_action_description(state, action_index) -> str` to `poker_collusion_nlhe/abstraction/actions.py`. Copy the implementation verbatim from `poker_collusion/abstraction/actions.py` — it uses `PREFLOP_RAISE_BB` and `POSTFLOP_POT_MULT` constants already defined in the NLHE actions file.
4. Create `poker_collusion_nlhe/__init__.py` if not already extracted.
**Done when:**
- `python -c "from poker_collusion_nlhe.env import deal_new_hand; s=deal_new_hand(); assert len(s.hole_cards[0])==2"` exits 0
- `python -c "from poker_collusion_nlhe.abstraction.actions import get_action_description; print('ok')"` exits 0

---

### T-02 · Backend project setup
**Status:** `done`
**Depends on:** none
**Files:** `[new] backend/__init__.py`, `[new] backend/requirements.txt`
**What:** Create the `backend/` directory with an empty `__init__.py` and a `requirements.txt` containing:
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
numpy>=1.19
tqdm
scikit-learn>=0.24
```
**Done when:** `pip install -r backend/requirements.txt` exits 0.

---

### T-03 · Session store
**Status:** `done`
**Depends on:** T-02
**Files:** `[new] backend/session.py`
**What:** Define:
- `Session` dataclass with fields: `game_mode: Literal["leduc_baseline","leduc_collusion","nlhe"]`, `runner: Any`, `state: Any`, `cumulative_bb: float`, `last_active: float`, `hand_number: int`.
- `SessionStore` class with:
  - `get(session_id: str) -> Session | None` — returns session or None if not found or expired (>1800s since `last_active`).
  - `set(session_id: str, session: Session) -> None` — stores session, updates `last_active = time.time()`.
  - `touch(session_id: str) -> None` — updates `last_active` on each request.
  - `prune_expired() -> None` — removes all entries where `time.time() - session.last_active > 1800`.
- Module-level singleton: `store = SessionStore()`.
**Done when:** `python -c "from backend.session import store, Session; print('ok')"` exits 0.

---

### T-04 · GameRunner abstraction + Leduc/NLHE runners
**Status:** `done`
**Depends on:** T-01, T-02
**Files:** `[new] backend/runner.py`
**What:** Implement three classes and one helper function:

`GameRunner` (abstract base): defines interface methods — `deal_new_hand()`, `apply_action(state, action)`, `get_current_player(state)`, `get_legal_actions(state)`, `get_info_key(state)`, `is_terminal(state)`, `is_chance_node(state)`, `sample_chance(state)`, `get_payoffs(state)`, `get_action_description(state, action)`, `strategy_for(player: int) -> Strategy`.

`LeducRunner(GameRunner)`: wraps `poker_collusion` functions. Constructor takes `strategy_p0: Strategy, strategy_p1: Strategy`. `apply_action` captures and returns the new state (Leduc returns a new state object). `strategy_for(player)` returns `strategy_p0` if player==0 else `strategy_p1`.

`NLHERunner(GameRunner)`: wraps `poker_collusion_nlhe` functions. Same constructor. `apply_action` calls the mutating NLHE version and returns the original state object (which was mutated in-place).

`resolve_until_user_turn(runner: GameRunner, state) -> tuple[state, list[dict]]`: runs the loop from SYSTEM-DESIGN.md — while not terminal and current_player != 2, either handle chance node or sample+apply bot action. Returns `(final_state, bot_actions_list)` where each bot_action is `{"type": "action"|"deal", "player": int|None, "player_name": str|None, "label": str}`.

`make_runner(game_mode: str, strategies: dict) -> GameRunner`: factory returning the right runner for the mode, passing the correct strategies.
**Done when:** `python -c "from backend.runner import LeducRunner, NLHERunner, resolve_until_user_turn, make_runner; print('ok')"` exits 0.

---

### T-05 · Serializer and Pydantic models
**Status:** `done`
**Depends on:** T-02
**Files:** `[new] backend/serializer.py`
**What:** Define Pydantic v2 models:
- `CardModel(rank: str, suit: str)`
- `LegalAction(index: int, label: str)`
- `BotEvent(type: str, player: int | None = None, player_name: str | None = None, label: str | None = None, cards: list[CardModel] | None = None)`
- `HandResult(payoffs_bb: list[float], showdown_cards: dict[str, list[CardModel]], hand_description: str)`
- `GameStateResponse` with all fields from SYSTEM-DESIGN.md schema: `game_mode`, `street`, `hand_number`, `hole_cards_p2`, `community_cards`, `stacks`, `bets`, `pot`, `is_terminal`, `current_player`, `legal_actions`, `bot_actions_this_turn`, `all_in_seats`, `hand_result`, `cumulative_bb`.

Implement `serialize_state(state, runner: GameRunner, session: Session, bot_actions: list[dict]) -> GameStateResponse`:
- Maps `state.round_idx` to street name (`0="preflop"`, `1="flop"` for Leduc; `0="preflop"`, `1="flop"`, `2="turn"`, `3="river"` for NLHE).
- Converts `state.hole_cards[2]` to `list[CardModel]` using a rank int→str map (`{0:"A",1:"K",2:"Q",3:"J"}` for Leduc, full 52-card map for NLHE).
- Converts `state.board` to `list[CardModel]`.
- Builds `legal_actions` by calling `runner.get_legal_actions(state)` and `runner.get_action_description(state, idx)` for each.
- Builds `hand_result` when `runner.is_terminal(state)` — calls `runner.get_payoffs(state)` and includes all hole cards.
**Done when:** `python -c "from backend.serializer import GameStateResponse, serialize_state; print('ok')"` exits 0.

---

### T-06 · FastAPI app + all endpoints
**Status:** `done`
**Depends on:** T-03, T-04, T-05
**Files:** `[new] backend/main.py`
**What:** FastAPI application with:

**Startup (lifespan):** Load all three models using `Strategy.load(path)` from `poker_collusion.cfr.strategy` and `poker_collusion_nlhe.cfr.strategy`. Store in module-level `STRATEGIES: dict[str, Strategy | None]` with keys `"leduc_baseline"`, `"leduc_collusion"`, `"nlhe"`. Missing files → set value to `None`.

**CORS:** `CORSMiddleware` with origins from `os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")`.

**`GET /api/health`:** Returns `{"models_loaded": {"leduc_baseline": bool, "leduc_collusion": bool, "nlhe": bool}}`.

**`POST /api/session/new`** (header: `X-Session-ID: str`, body: `{"game_mode": str}`): Creates a `Session` via `make_runner`, calls `runner.deal_new_hand()`, calls `resolve_until_user_turn(runner, state)`, stores session, returns `serialize_state(...)`.

**`POST /api/session/{session_id}/action`** (body: `{"action_index": int}`): Looks up session via `store.get(session_id)` — returns 404 if missing. Validates `action_index` is in `runner.get_legal_actions(state)` — returns 400 if invalid. Applies user action via `runner.apply_action(state, action_index)`. Calls `resolve_until_user_turn`. If terminal, updates `session.cumulative_bb += payoffs[2]`. Returns `serialize_state(...)`.

**`POST /api/session/{session_id}/next_hand`:** Increments `session.hand_number`, calls `runner.deal_new_hand()`, calls `resolve_until_user_turn`, returns `serialize_state(...)`.

**`POST /api/session/{session_id}/switch_mode`** (body: `{"game_mode": str}`): Resets `cumulative_bb=0`, `hand_number=1`, creates new runner for mode, deals new hand, resolves preflop bots, stores updated session, returns `serialize_state(...)`.
**Done when:**
- `uvicorn backend.main:app --port 8000` starts without error output
- `curl -s http://localhost:8000/api/health | python3 -m json.tool` prints JSON with `models_loaded` containing three boolean keys

---

### T-07 · Frontend project scaffold
**Status:** `done`
**Depends on:** none
**Files:** `[new] frontend/package.json`, `[new] frontend/vite.config.ts`, `[new] frontend/tsconfig.json`, `[new] frontend/index.html`, `[new] frontend/src/main.tsx`, `[new] frontend/.env.local`, `[new] frontend/tailwind.config.js`, `[new] frontend/postcss.config.js`
**What:**
1. From repo root: `npm create vite@latest frontend -- --template react-ts` (accept all prompts).
2. `cd frontend && npm install`.
3. `npm install react-playing-cards tailwindcss postcss autoprefixer`.
4. `npx tailwindcss init -p` (generates `tailwind.config.js` and `postcss.config.js`).
5. Set `tailwind.config.js` content array to `["./index.html", "./src/**/*.{ts,tsx}"]`.
6. In `vite.config.ts`, ensure the default config is present (no changes needed beyond scaffold).
7. Create `frontend/.env.local` with content: `VITE_API_URL=http://localhost:8000`.
8. In `frontend/index.html`, add Google Fonts `<link>` in `<head>`: `https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;500&display=swap`.
**Done when:** `cd frontend && npm run build` exits 0 with no TypeScript errors.

---

### T-08 · API client (api.ts)
**Status:** `done`
**Depends on:** T-07
**Files:** `[new] frontend/src/api.ts`
**What:** Export the following TypeScript interfaces matching SYSTEM-DESIGN.md schemas exactly: `CardModel`, `LegalAction`, `BotEvent`, `HandResult`, `GameStateResponse`, `HealthResponse`. Export typed async functions:
- `generateSessionId(): string` — returns `crypto.randomUUID()`.
- `fetchHealth(): Promise<HealthResponse>` — `GET ${VITE_API_URL}/api/health`.
- `newSession(sessionId: string, gameMode: GameMode): Promise<GameStateResponse>` — `POST /api/session/new` with `X-Session-ID` header and JSON body.
- `submitAction(sessionId: string, actionIndex: number): Promise<GameStateResponse>` — `POST /api/session/{sessionId}/action`.
- `nextHand(sessionId: string): Promise<GameStateResponse>` — `POST /api/session/{sessionId}/next_hand`.
- `switchMode(sessionId: string, gameMode: GameMode): Promise<GameStateResponse>` — `POST /api/session/{sessionId}/switch_mode`.

All functions throw if response is not 2xx (preserve the response status for 404 detection). Export `type GameMode = "leduc_baseline" | "leduc_collusion" | "nlhe"`.
**Done when:** `cd frontend && npx tsc --noEmit` reports 0 errors.

---

### T-09 · Game state hook (useGame.ts)
**Status:** `done`
**Depends on:** T-08
**Files:** `[new] frontend/src/hooks/useGame.ts`
**What:** Export `useGame()` returning:
```ts
{
  phase: Phase,            // 'loading'|'mode-select'|'collusion-banner'|'preflop-bot-acting'|'user-turn'|'bot-acting'|'hand-result'|'error'
  gameState: GameStateResponse | null,
  logEntries: LogEntry[],  // accumulated action log entries
  errorMessage: string | null,
  availableModes: Record<GameMode, boolean>,
  startGame: (mode: GameMode) => void,
  submitAction: (actionIndex: number) => void,
  nextHand: () => void,
  switchMode: (mode: GameMode) => void,
  retryLastAction: () => void,
  dismissBanner: () => void,
}
```
Internal implementation:
- `sessionId`: generated once with `generateSessionId()` on mount, stored in a `useRef`.
- On mount: poll `fetchHealth()` every 2s until at least one model is loaded → set `availableModes`, transition phase to `'mode-select'`.
- `startGame(mode)`: calls `newSession`, on success sets `gameState`, appends bot preflop events to stagger queue, transitions to `'preflop-bot-acting'`, begins stagger.
- Stagger: `useEffect` watching `staggerQueue` — pops one entry every 600ms using `setTimeout` chain, appends to `logEntries`, when queue empty transitions to `'user-turn'` (or `'hand-result'` if `gameState.is_terminal`).
- `submitAction(actionIndex)`: disables panel (set phase `'bot-acting'`), calls API, on success appends bot_actions_this_turn to stagger queue, begins stagger.
- On any API error: stores the failed call as `lastAction`, sets `phase='error'`, sets `errorMessage`.
- `retryLastAction()`: re-executes `lastAction`. If response is 404, sets `errorMessage='Session expired — refresh to restart'`.
- `switchMode(mode)`: resets `logEntries=[]`, calls `switchMode` API, if mode is `'leduc_collusion'` sets phase to `'collusion-banner'` (banner dismissed via `dismissBanner()` which then deals the hand).
- `LogEntry` type: `{ handNumber: number, type: 'action'|'deal'|'result'|'separator', text: string }`.
**Done when:** `cd frontend && npx tsc --noEmit` reports 0 errors for `useGame.ts`.

---

### T-10 · Dark Precision CSS + Tailwind config
**Status:** `done`
**Depends on:** T-07
**Files:** `frontend/src/index.css`
**What:** Replace scaffolded `index.css` content with:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --bg: #0D0F14;
  --surface: #161A23;
  --surface2: #1C2130;
  --border: #252A35;
  --gold: #E8C547;
  --blue: #4A9EFF;
  --red: #E84747;
  --green: #47E87A;
  --text: #E0E4EF;
  --text-dim: #5A6070;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'IBM Plex Mono', monospace;
  min-height: 100vh;
}

.font-display {
  font-family: 'Bebas Neue', sans-serif;
  letter-spacing: 0.1em;
}
```
**Done when:** `frontend/src/index.css` contains all 10 CSS custom property definitions and the three Tailwind directives.

---

### T-11 · CardFace component
**Status:** `done`
**Depends on:** T-10
**Files:** `[new] frontend/src/components/CardFace.tsx`
**What:** Export:
```ts
interface CardFaceProps {
  rank: string;       // e.g. "A", "K", "Q", "J", "2"..."10"
  suit: string;       // "spades"|"hearts"|"diamonds"|"clubs"
  faceDown?: boolean; // default false
  size?: 'sm' | 'md'; // 'sm'=38×54px (player areas), 'md'=44×62px (community)
  revealed?: boolean; // triggers rotateY reveal animation when changed from false→true
}
```
- `faceDown=true`: render a `<div>` with background `#1E2535`, border `1px solid #2E3545`, dimensions from `size`.
- `faceDown=false`: render the appropriate `react-playing-cards` component for the rank+suit.
- Apply CSS transition: `transition: transform 0.3s ease` with `transform: rotateY(180deg)` when `revealed` changes from false to true.
**Done when:** `cd frontend && npx tsc --noEmit` reports 0 errors.

---

### T-12 · PlayerArea component
**Status:** `done`
**Depends on:** T-11
**Files:** `[new] frontend/src/components/PlayerArea.tsx`
**What:** Export:
```ts
interface PlayerAreaProps {
  seat: 0 | 1 | 2;
  name: string;          // "BTN · P0", "SB · P1", "BB · YOU · P2"
  stack: number;
  bet: number;
  cards: CardModel[];    // empty array = no cards yet
  faceDown: boolean;     // true for bots during play, false for user
  isActiveTurn: boolean; // blue border on bot areas during stagger
  isAllIn: boolean;
  isCollusion: boolean;  // red border + red name for BTN/SB in collusion mode
  revealCards?: boolean; // triggers reveal animation on showdown
}
```
Renders: bordered panel (`border: 1px solid var(--border)`), override to `var(--blue)` if `isActiveTurn`, `var(--red)` if `isCollusion`, `var(--gold)` if `seat===2`. Player name in Bebas Neue (red if `isCollusion`, blue if `isActiveTurn`, gold if `seat===2`, else dim). Stack + bet in IBM Plex Mono. ALL-IN badge (`background: #2A1515; color: var(--red); border: 1px solid var(--red); font-family: Bebas Neue; font-size: 11px`) when `isAllIn`. Row of `<CardFace>` components sized 'sm'.
**Done when:** `cd frontend && npx tsc --noEmit` reports 0 errors.

---

### T-13 · CommunityCards component
**Status:** `done`
**Depends on:** T-11
**Files:** `[new] frontend/src/components/CommunityCards.tsx`
**What:** Export:
```ts
interface CommunityCardsProps {
  cards: CardModel[];
  gameMode: GameMode;
  pot: number;
}
```
- `gameMode === 'nlhe'` → render 5 card slots. Otherwise → render 1 card slot.
- For each slot: if a card exists at that index, render `<CardFace rank suit size='md'>`. If empty, render a dashed placeholder `<div>` (border: `1px dashed var(--border)`, background: `var(--surface2)`, dimensions 44×62px).
- Above the cards, render pot display: "Pot: X.X BB" in Bebas Neue font-size 26px.
- Below the cards, render street label derived from `cards.length`: 0→"PREFLOP", 1→"FLOP" (Leduc) or 3→"FLOP", 4→"TURN", 5→"RIVER" (NLHE). Font-size 10px, uppercase, letter-spacing 2px, color `var(--text-dim)`.
**Done when:** `cd frontend && npx tsc --noEmit` reports 0 errors.

---

### T-14 · ActionPanel component
**Status:** `done`
**Depends on:** T-10
**Files:** `[new] frontend/src/components/ActionPanel.tsx`
**What:** Export:
```ts
interface ActionPanelProps {
  phase: Phase;
  legalActions: LegalAction[];
  handResult: HandResult | null;
  onAction: (actionIndex: number) => void;
  onNextHand: () => void;
}
```
Renders one of three views:
- **`user-turn`**: render a button for each `legalAction`. Buttons: `border: 1px solid var(--border); background: transparent; font-family: IBM Plex Mono`. Hover: `border-color: var(--gold); color: var(--gold)`. Fold button hover: `border-color: var(--red); color: var(--red)`. All enabled.
- **`preflop-bot-acting` | `bot-acting`**: render "Waiting for opponents..." with three animated dots (same CSS keyframe as in mockup-bot-acting.html).
- **`hand-result`**: render winner label from `handResult.hand_description` (Bebas Neue 18px), per-seat payoffs for all 3 seats (green if positive, red if negative), "DEAL NEXT HAND" button (`background: var(--gold); color: var(--bg); font-family: Bebas Neue`). Button calls `onNextHand`.
All buttons `disabled` in any phase other than `user-turn`.
**Done when:** `cd frontend && npx tsc --noEmit` reports 0 errors.

---

### T-15 · ActionLog component
**Status:** `done`
**Depends on:** T-10
**Files:** `[new] frontend/src/components/ActionLog.tsx`
**What:** Export:
```ts
interface LogEntry { handNumber: number; type: 'action'|'deal'|'result'|'separator'; text: string; }
interface ActionLogProps { entries: LogEntry[]; }
```
Renders a scrollable feed:
- Container: `max-height: 180px; overflow-y: auto; border: 1px solid var(--border); background: var(--surface); padding: 12px 16px`.
- Each entry is one `<div>` line:
  - `action`: `color: var(--text)`
  - `deal` | `result`: `color: var(--gold)`
  - `separator`: `color: var(--border); font-size: 10px`
- New entries animate in: CSS class with `@keyframes logAppear { from { opacity: 0; transform: translateX(-5px) } to { opacity: 1; transform: none } }` applied on mount, duration 200ms.
- `useEffect` scrolls the container to `scrollTop = scrollHeight` whenever `entries` changes.
**Done when:** `cd frontend && npx tsc --noEmit` reports 0 errors.

---

### T-16 · LoadingScreen component
**Status:** `done`
**Depends on:** T-10
**Files:** `[new] frontend/src/components/LoadingScreen.tsx`
**What:** Export `LoadingScreen()` — a full-viewport centered layout (flex, items-center, justify-center) with:
- Outer box: `border: 1px solid var(--border); background: var(--surface); padding: 48px 64px; min-width: 380px`.
- "POKER BOT" heading: Bebas Neue 28px, letter-spacing 4px.
- "CONNECTING TO SERVER" subtitle: IBM Plex Mono 12px, `color: var(--text-dim)`, uppercase.
- Progress bar: `background: var(--border); height: 2px; width: 100%`. Inner fill: `background: var(--gold); height: 100%` with CSS animation alternating width 30%→80%.
- "LOADING MODELS..." text: `color: var(--gold)` with blink animation.
- Three animated dots: each `6×6px; border-radius: 50%` pulsing gold sequentially.
**Done when:** `LoadingScreen.tsx` exists and `cd frontend && npx tsc --noEmit` reports 0 errors.

---

### T-17 · ModeSelect component
**Status:** `done`
**Depends on:** T-10
**Files:** `[new] frontend/src/components/ModeSelect.tsx`
**What:** Export:
```ts
interface ModeSelectProps {
  availableModes: Record<GameMode, boolean>;
  onStart: (mode: GameMode) => void;
}
```
Renders three radio-style mode cards in a column. Each card: `border: 1px solid var(--border); background: var(--surface); padding: 20px 24px; cursor: pointer`. Selected card: `border-color: var(--gold)`. On click: updates selected mode in local state.

Mode display names and metadata:
- `leduc_baseline` → "LEDUC (BASELINE)" / "Self-play equilibrium · 1 hole card, 1 community card" / tag: "SELF-PLAY · leduc_baseline_5000k" (blue tag)
- `leduc_collusion` → "LEDUC (COLLUSION)" / "BTN+SB are a team against you (BB). Observable signaling." / tag: "OBSERVABLE SIGNALING · leduc_obs_signal_1m" (red tag)
- `nlhe` → "NLHE (BASELINE)" / "Full NLHE, 20 BB stacks, 2 hole cards, up to 5 community cards" / tag: "SELF-PLAY · blueprint_v4_21600" (blue tag)

If `availableModes[mode] === false`: card is opacity 0.5, `cursor: not-allowed`, radio button does not respond to clicks, shows tooltip "Model unavailable".

"START GAME" button below all cards: `background: var(--gold); color: var(--bg); font-family: Bebas Neue; font-size: 20px; width: 100%; padding: 16px`. Calls `onStart(selectedMode)`.
**Done when:** `cd frontend && npx tsc --noEmit` reports 0 errors.

---

### T-18 · CollusionBanner component
**Status:** `done`
**Depends on:** T-10
**Files:** `[new] frontend/src/components/CollusionBanner.tsx`
**What:** Export `CollusionBanner({ onDismiss }: { onDismiss: () => void })`. Renders a fixed-position overlay (`position: fixed; inset: 0; background: rgba(13,15,20,0.75); z-index: 50; display: flex; align-items: center; justify-content: center`) containing a centered panel (`width: 560px; border: 1px solid var(--red); background: var(--bg)`):
- Red header bar: `background: var(--red); padding: 14px 24px`; "⚠ COLLUSION MODE" in Bebas Neue / `color: var(--bg)`.
- Body padding 28px.
- "BTN AND SB ARE A COORDINATED TEAM" title in Bebas Neue.
- Three-column seat grid (BTN, SB, BB/YOU): BTN+SB boxes have `border-color: var(--red); background: #2A1515`; BB box has `border-color: var(--gold)`. Name in appropriate color.
- Explanatory text: "The BTN and SB bots were trained together using an observable signaling strategy — they optimize for their combined payoff, effectively colluding against you (BB). Can you detect patterns in their play?"
- "I UNDERSTAND — START GAME" button: `background: var(--red); color: white; font-family: Bebas Neue; width: 100%; padding: 14px`. Calls `onDismiss`.
**Done when:** `cd frontend && npx tsc --noEmit` reports 0 errors.

---

### T-19 · PokerTable component
**Status:** `done`
**Depends on:** T-12, T-13, T-14, T-15
**Files:** `[new] frontend/src/components/PokerTable.tsx`
**What:** Export:
```ts
interface PokerTableProps {
  gameState: GameStateResponse;
  phase: Phase;
  logEntries: LogEntry[];
  activeStaggerSeat: number | null; // which bot seat is "currently acting" during stagger
  isCollusion: boolean;
  onAction: (actionIndex: number) => void;
  onNextHand: () => void;
}
```
Layout: `max-width: 800px; margin: 0 auto; padding: 20px 32px; display: grid; grid-template-rows: auto 1fr auto auto auto` with areas: bots / center / user / actions / log.

- Bots row: two `<PlayerArea>` side by side — BTN (seat=0) and SB (seat=1), both `faceDown={phase !== 'hand-result'}`, `isCollusion={isCollusion}`, `isActiveTurn={activeStaggerSeat === seat}`, `revealCards={phase === 'hand-result'}`.
- Center: `<CommunityCards cards={gameState.community_cards} gameMode={gameState.game_mode} pot={gameState.pot}>`.
- User: `<PlayerArea seat={2} faceDown={false} cards={gameState.hole_cards_p2} isCollusion={false} isActiveTurn={phase==='user-turn'}>`.
- Actions: `<ActionPanel phase={phase} legalActions={gameState.legal_actions} handResult={gameState.hand_result} onAction={onAction} onNextHand={onNextHand}>`.
- Log: `<ActionLog entries={logEntries}>`.
**Done when:** `cd frontend && npx tsc --noEmit` reports 0 errors.

---

### T-20 · App.tsx + Header.tsx (full wiring)
**Status:** `done`
**Depends on:** T-09, T-16, T-17, T-18, T-19
**Files:** `frontend/src/App.tsx` (replace scaffold), `[new] frontend/src/components/Header.tsx`
**What:**

`Header.tsx`: Export `Header({ bankroll, gameMode, availableModes, onSwitchMode }: HeaderProps)`. Renders: `<header>` with `border-bottom: 1px solid var(--border); padding: 12px 32px; display: flex; justify-content: space-between`. Left: "POKER BOT" logo (Bebas Neue 22px, `color: var(--gold)`). Right: bankroll display (`+X.X BB` green / `-X.X BB` red) + `<select>` with three mode options (disabled if not available). Changing the select calls `onSwitchMode(newMode)`.

`App.tsx`: Call `useGame()`. Wire up all state and handlers:
- Always render `<Header bankroll={gameState?.cumulative_bb ?? 0} ...>`.
- Render `<LoadingScreen>` when `phase === 'loading'`.
- Render `<ModeSelect availableModes onStart={startGame}>` when `phase === 'mode-select'`.
- Render `<CollusionBanner onDismiss={dismissBanner}>` overlay when `phase === 'collusion-banner'` (on top of PokerTable or ModeSelect).
- Render `<PokerTable ...>` when phase is any of: `preflop-bot-acting`, `user-turn`, `bot-acting`, `hand-result`. Pass `activeStaggerSeat` (derived from which bot_action is currently being staggered), `isCollusion={gameState?.game_mode === 'leduc_collusion'}`.
- Render error banner (per mockup-error.html) when `phase === 'error'`: `position: fixed; top: header-height; left: 0; right: 0; background: var(--red-bg); border-bottom: 1px solid var(--red)`. Contains error message + "RETRY" button calling `retryLastAction()`.

**Done when:**
- `cd frontend && npm run build` exits 0 with no TypeScript errors
- `cd frontend && npm run dev` starts on `http://localhost:5173` without console errors
- Opening `http://localhost:5173` while backend is offline shows "POKER BOT" header and the LoadingScreen (verifiable by visual inspection)

---

## Open Questions
None — all ambiguities resolved or inferred.
