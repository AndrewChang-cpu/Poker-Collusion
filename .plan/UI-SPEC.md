# UI Spec: Poker Bot Web Frontend

## Aesthetic Direction

**Dark Precision** — trading terminal meets card room. The product is a research demo of AI poker agents, so it reads technical-serious rather than casino-flashy.

| Token | Value | Role |
|---|---|---|
| `--bg` | `#0D0F14` | Page background |
| `--surface` | `#161A23` | Player areas, panels |
| `--surface2` | `#1C2130` | Action panel, secondary surfaces |
| `--border` | `#252A35` | All borders |
| `--gold` | `#E8C547` | Active states, YOUR player area, CTA buttons, winning outcomes |
| `--blue` | `#4A9EFF` | Active bot indicator, action log player names |
| `--red` | `#E84747` | Negative bankroll, collusion border tint, error states |
| `--green` | `#47E87A` | Positive bankroll |
| `--text` | `#E0E4EF` | Primary text |
| `--text-dim` | `#5A6070` | Labels, secondary info |

**Typography:**
- Display / headings: `Bebas Neue` (Google Fonts) — condensed, bold, uppercase. Used for: logo, player names, button text, hand result title.
- Body / numbers: `IBM Plex Mono` — all numeric values (stack sizes, BB amounts, pot). Used for: everything else.

**Layout:**  Single-column centered layout, max-width 800px. Player areas are sharp-cornered bordered rectangles, not rounded cards. No drop shadows. No background textures.

---

## UI States

| State | File | Description |
|---|---|---|
| `loading` | [mockup-loading.html](mockup-loading.html) | Cold start: polling `/api/health`, progress indicator |
| `mode-select` | [mockup-mode-select.html](mockup-mode-select.html) | Mode picker: 3 options, Start Game button |
| `collusion-banner` | [mockup-collusion-banner.html](mockup-collusion-banner.html) | Info overlay when Leduc (Collusion) is selected |
| `preflop-bot-acting` | [mockup-preflop-bot-acting.html](mockup-preflop-bot-acting.html) | Hand just dealt, BTN+SB acting before user's first turn |
| `user-turn` | [mockup-user-turn.html](mockup-user-turn.html) | User (BB) to act — action panel with legal buttons |
| `bot-acting` | [mockup-bot-acting.html](mockup-bot-acting.html) | User acted, bots responding mid-hand, panel disabled |
| `hand-result` | [mockup-hand-result.html](mockup-hand-result.html) | Hand over — cards revealed, payoffs, Deal Next Hand |
| `error` | [mockup-error.html](mockup-error.html) | Network error banner with Retry, game dimmed behind |

---

## UX Flows

### New hand start (user is BB)
```
Deal called
  → backend: deal_new_hand() → resolve BTN/SB preflop actions → return state
  → frontend: show preflop-bot-acting state
  → stagger bot_actions_this_turn in action log (600ms each)
  → after stagger complete: transition to user-turn state
```

### User acts mid-hand
```
User clicks action button
  → buttons disabled immediately
  → POST /action
  → backend resolves all bot turns + chance nodes → returns state + bot_actions_this_turn
  → frontend: if bot_actions_this_turn is empty and terminal → hand-result state
  → else: show bot-acting state (panel disabled)
  → stagger bot_actions_this_turn in log (600ms each)
  → after stagger: if terminal → hand-result state, else → user-turn state
```

### User folds
```
User clicks Fold
  → POST /action
  → backend: apply fold, then runs remaining BTN vs SB resolution to terminal
  → returns full bot_actions_this_turn (SB/BTN actions + community deals + final result)
  → frontend staggers full sequence (may be 3-5 events) → then hand-result state
```

### Mode switch
```
User selects new mode from dropdown
  → POST /switch_mode
  → bankroll resets to 0
  → if switching TO "Leduc (Collusion)": show collusion-banner overlay
  → after banner dismissed (or if not collusion): new hand dealt → preflop-bot-acting state
```

---

## Component Structure

```
App
├── LoadingScreen              — polls /api/health, blocks until ready
├── Header
│   ├── Logo                   — "POKER BOT" in Bebas Neue / gold
│   ├── BankrollDisplay        — "+X.X BB" / "-X.X BB", colored green/red
│   └── ModeDropdown           — 3 options; switching calls switch_mode
│
├── ModeSelect                 — shown before first hand; disappears after Start Game
│
├── CollusionBanner            — overlay shown when collusion mode selected
│
└── PokerTable                 — main game view, shown once a hand is in progress
    ├── BotsRow
    │   ├── PlayerArea (BTN)   — face-down card, stack, bet, ALL-IN badge if applicable
    │   │                        red border tint in collusion mode
    │   └── PlayerArea (SB)    — same; active-turn blue border during bot stagger
    ├── CenterArea
    │   ├── PotDisplay         — "X.X BB" in Bebas Neue
    │   └── CommunityCards     — 1 slot (Leduc) or 5 slots (NLHE); empty slots dashed border
    ├── UserArea (BB)          — gold border; face-up hole cards; YOUR TURN badge when active
    ├── ActionPanel            — shows legal action buttons OR "Waiting..." OR hand-result content
    │   ├── ActionButtons      — disabled at all times except user's turn
    │   ├── WaitingIndicator   — shown during stagger + preflop bot-acting phase
    │   └── HandResult         — winner label, per-seat payoffs, Deal Next Hand button
    └── ActionLog              — last 10 events, newest at bottom, hand separators
```

---

## Collusion Mode Visual Distinction

When game_mode is `leduc_collusion`:
- BTN and SB `PlayerArea` borders: `1px solid var(--red)` instead of `var(--border)`
- BTN and SB player name labels rendered in `var(--red)` instead of `var(--text-dim)`
- No "ALLY" label — the collusion-banner explains the dynamic once at session start

---

## Card Rendering

**Library:** `react-playing-cards@0.6.3` (LGPL-3.0, verified on npm) — React components for all 52 standard playing cards.

**Face-up:** SVG card rendered at 38×54px (bot areas) or 44×62px (center community card).

**Face-down:** Solid dark rectangle `#1E2535` with `1px solid #2E3545` border. Same dimensions as face-up. No back-of-card pattern needed.

**Leduc deck:** Only A, K, Q, J used across 3 suits (clubs, diamonds, hearts). Cards dealt normally from the shuffled deck — no suit pre-assignment.

**NLHE deck:** All 52 standard cards.

**Reveal animation:** On showdown, face-down cards in opponent areas flip to face-up with a brief CSS `rotateY` transition (0.3s, triggered by state change to `is_terminal: true`).

---

## ALL-IN Badge

When a seat's stack is 0 BB, render a small badge next to their player name:
```
[ALL-IN]  — font: Bebas Neue 11px, background: #2A1515, color: var(--red), border: 1px solid var(--red)
```
Stack display shows `0.0 BB`.

---

## Action Log Format

Each entry is one line. Format:
```
H{hand_number}  {actor}: {action text}            ← player action
H{hand_number}  Community: {rank}{suit} dealt      ← chance event (gold color)
H{hand_number}  {player} wins {X} BB              ← result (gold color)
── Hand #{N} ──────────────────────────────       ← hand separator (dim border color)
```

**No entry cap.** The log shows every event for the current hand in progress, plus one summary line for the previous hand (e.g., `H3 | BTN wins 6.0 BB`). When a new hand begins, all prior hand events are replaced by that single summary line. Newest entries animate in with a 200ms fade+slide-left. The log area has a max-height with `overflow-y: auto` to prevent layout overflow on very long NLHE hands.
