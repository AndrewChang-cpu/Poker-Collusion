# PRD: Poker Bot Web Frontend

## Functional Requirements

- FR-01: A dropdown in the header offers three game modes: "Leduc (Baseline)", "Leduc (Collusion)", "NLHE (Baseline)". Selecting a different mode resets the session — bankroll returns to 0 and a new hand is dealt.
- FR-02: When "Leduc (Collusion)" is selected, an info banner appears before the first hand explaining that BTN and SB are a coordinated team optimizing against the user (BB). The user dismisses the banner to begin play.
- FR-03: The game starts with a mode-select screen. The user picks a mode and clicks "Start Game". Cards are not dealt until this action.
- FR-04: The user always sits in the BB seat (P2). BTN is P0, SB is P1. This is fixed across all modes and all hands.
- FR-05: Each hand begins with a 20 BB stack for all players. On deal, BTN and SB post blinds (0.5 BB and 1 BB respectively), BB posts nothing. These deductions are reflected immediately in stack displays.
- FR-06: When a hand is dealt, the backend runs BTN and SB's preflop actions before returning control. Their actions appear staggered in the action log (600ms per action). The user's action panel does not appear until the stagger is complete.
- FR-07: The user's hole card(s) are shown face-up as SVG playing cards. Opponent cards are face-down at all times except showdown.
- FR-08: Community cards are revealed face-up when the chance node fires (Leduc: 1 card after preflop; NLHE: 3 cards on flop, 1 on turn, 1 on river). Community card reveal is part of the stagger sequence.
- FR-09: The user's action panel displays only the legal actions for the current state as labeled buttons (e.g., "Fold", "Check", "Call 1.0 BB", "Raise to 3 BB", "All-in 20 BB"). All buttons are disabled except during the user's turn.
- FR-10: After the user acts, the action panel is disabled. Backend resolves all subsequent bot actions and chance events and returns them as an ordered list. The frontend displays each item staggered at 600ms intervals. Once the stagger completes, the panel re-enables (if it is the user's turn again) or the hand-result view appears (if terminal).
- FR-11: When the user folds, the remaining bot-vs-bot resolution plays out in full (staggered) before the hand-result view appears.
- FR-12: At hand end, all players' hole cards are revealed. The result area shows each player's payoff in BB (positive or negative). Split pots are communicated solely through the payoff numbers (0.0 BB profit each).
- FR-13: A player who is all-in displays an "ALL-IN" badge on their player area with a stack of 0 BB.
- FR-14: A bankroll counter in the header shows cumulative BB gain/loss for the session. It updates after each hand.
- FR-15: The action log (bottom of the table area) shows all events for the current hand in progress, plus one summary line for the previous hand (e.g., "Hand #3 | BTN wins 6.0 BB"). There is no entry cap — the log grows to fit the current hand, then resets when the next hand begins. Each new hand is separated by a visible divider. Newest events appear at the bottom.
- FR-16: In Leduc (Collusion) mode, BTN and SB player areas display a visual indicator (red border tint) distinguishing them as coordinated opponents.
- FR-17: If the API call fails, an error banner appears with a "Retry" button that resends the last action. If the retry response indicates session-not-found (expired after 30min idle), the banner message updates to "Session expired — refresh to restart".
- FR-18: On page load, if the backend is cold-starting, a "Connecting to server..." screen blocks the mode-select UI and polls `GET /api/health` until the backend is ready.
- FR-19: Page refresh generates a new session — bankroll resets to 0, no hand in progress.

## Critical User Journeys

### CUJ-1: Standard Leduc Baseline hand
1. User lands on page → "Connecting to server..." polling (if cold start), then mode-select screen appears
2. User selects "Leduc (Baseline)" from dropdown, clicks "Start Game"
3. Hand is dealt. BTN posts 0.5 BB, SB posts 1.0 BB. BTN and SB preflop actions stagger in the log over ~1.2s. User's hole card appears face-up (SVG).
4. User's action panel appears with legal actions (e.g., Fold / Call 1.0 BB / Raise 2 BB / All-in)
5. User clicks "Raise 2 BB". Panel disables. SB response staggers in (600ms), BB check/call staggers in (1200ms).
6. Community card flips face-up in center (next stagger event at 1800ms).
7. SB acts postflop (stagger), then user's panel re-enables.
8. User checks. BTN responds (stagger). Hand is terminal.
9. All cards revealed. Result area shows payoffs. Bankroll updates. "Deal Next Hand" button appears.
10. User clicks "Deal Next Hand". New hand starts from step 3. Action log appends a divider and new hand events.

### CUJ-2: Leduc Collusion mode — user folds
1. User selects "Leduc (Collusion)" → collusion info banner appears ("BTN and SB are a coordinated team...")
2. User dismisses banner, clicks "Start Game"
3. Hand dealt. BTN/SB preflop bot actions stagger. Player area borders are red-tinted.
4. User sees their hole card and legal actions. User clicks "Fold".
5. Remaining BTN vs SB resolution staggers in the log (their postflop actions, community card, result).
6. Result shown. User's payoff is displayed (e.g., -1.0 BB for the blind posted). Bankroll updates.

### CUJ-3: Network error mid-hand
1. User clicks an action. API call fails (network down).
2. Error banner appears: "Connection failed — Retry".
3. User clicks Retry. API call succeeds → game continues normally.
4. If Retry returns 404 (session expired): banner updates to "Session expired — refresh to restart".

### CUJ-4: Switching game modes
1. Mid-session, user opens the dropdown and selects "NLHE (Baseline)".
2. Bankroll resets to 0. New session starts. NLHE hand is dealt (user sees 2 hole cards, 5-slot board area).
3. BTN and SB preflop actions stagger. User acts as BB.

## Non-Functional Requirements

- NFR-perf: `/api/session/{id}/action` response time < 800ms (strategy lookup only, no subgame search). The 600ms stagger is client-side only — server returns all bot actions in one payload.
- NFR-sec: CORS restricted to the Vercel deployment origin (not wildcard `*`). No auth required. No sensitive user data stored.
- NFR-rely: All three models loaded at backend startup. If any model file is missing, that mode's option is grayed out in the dropdown with a tooltip "Model unavailable". Other modes remain functional.
- NFR-a11y: Not specified.
- NFR-i18n: Not specified.
- NFR-audit: Not specified.
