# Plan: Poker Bot Web Frontend
> Generated: 2026-05-14
> Type: brownfield
> Documents: [PRD.md](PRD.md) · [SYSTEM-DESIGN.md](SYSTEM-DESIGN.md) · [UI-SPEC.md](UI-SPEC.md)

## Overview
**What:** A web application where a user plays 3-player poker as BB (P2) against two CFR-trained bot opponents. Three game modes are available via a single dropdown: Leduc Baseline (both bots play self-play equilibrium), Leduc Collusion (BTN+SB bots use an observable-signaling team strategy against the user), and NLHE Baseline (full 52-card NLHE with self-play bots). Stacks reset to 20 BB each hand; cumulative gain/loss is tracked across the session. Bot actions play out staggered (600ms each), including preflop bot actions before the user's first turn each hand.

**Why:** The trained bots are accessible only via CLI scripts. A web UI makes them demonstrable to non-technical audiences without a terminal, and lets viewers experience the collusion dynamic directly — as the BB being colluded against.

**Who:** Single user per session — the developer, collaborators, or a demo/presentation audience.

## Definition of Done
- [ ] `GET /api/health` returns `{"models_loaded": {"leduc_baseline": true, "leduc_collusion": true, "nlhe": true}}` with all three models confirmed loaded
- [ ] NLHE engine restored: `from poker_collusion_nlhe.env import deal_new_hand; s = deal_new_hand(); assert len(s.hole_cards[0]) == 2` passes
- [ ] Mode-select screen shows three labeled options; clicking "Start Game" deals a hand and triggers the preflop bot-acting stagger
- [ ] In Leduc Collusion mode, the collusion info banner appears before the game starts
- [ ] User's hole card(s) render as SVG playing cards (not placeholders) in all three modes
- [ ] BTN and SB act preflop with visible 600ms stagger in the action log before the user's action panel appears
- [ ] User's action panel shows only legal actions; buttons are disabled at all times except when it is the user's (BB's) turn
- [ ] After user acts, bot responses appear staggered in the action log with 600ms delay per event
- [ ] After user folds, remaining bot-vs-bot resolution plays out staggered in the action log before the result is shown
- [ ] If both bots fold preflop before the user ever acts, the hand-result view is shown directly after the preflop stagger completes — the user's action panel never appears for that hand
- [ ] At hand end, opponent hole cards are revealed and payoffs are shown for all three seats
- [ ] Bankroll updates correctly across 5 consecutive hands (displayed total equals sum of hand payoffs)
- [ ] "Deal Next Hand" starts a new hand with bankroll preserved and action log appended
- [ ] Switching game mode via dropdown resets bankroll to 0 and deals a fresh hand
- [ ] Page refresh resets bankroll to 0 and creates a new session
- [ ] In collusion mode, BTN and SB player areas are visually distinguished (red border tint) from baseline mode
- [ ] Error banner with Retry button appears on API failure; session-expired path shows "Session expired — refresh to restart"
- [ ] Cold start: "Connecting to server..." state blocks play until `/api/health` succeeds
- [ ] `vite build` succeeds with `VITE_API_URL` controlling the backend URL (no hardcoded localhost)
- [ ] Backend starts with `uvicorn backend.main:app --reload` from repo root

## Out of Scope
- PluribusBot subgame search — strategy lookup only, no `search/` module used
- Additional collusion variants (CTDE, free-comm) — only observable signaling (`leduc_obs_signal_1m.pkl`) for now
- Multi-user concurrent sessions
- Game history persistence across page refreshes
- Mobile layout optimization
- Sound effects or animated card-deal transitions
- Seat rotation (user is always BB)
- Any modification to `poker_collusion/`, training scripts, or evaluation scripts

## Assumptions
- NLHE bucket tables confirmed present: `data/` contains `preflop_buckets.pkl`, `flop_buckets.pkl`, `turn_buckets.pkl`, `river_buckets.pkl`
- Railway's standard plan (512MB RAM) can hold all three models (~400MB total) simultaneously
- `react-playing-cards@0.6.3` renders all 52 standard cards needed for NLHE and the A/K/Q/J ranks needed for Leduc

## Open Questions
None — all factual uncertainties resolved via codebase investigation.
