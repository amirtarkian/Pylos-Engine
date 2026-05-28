# Take-back + Unity-style Camera Controls — Design

Date: 2026-05-27
Status: Approved by user, ready for implementation plan

## Summary

Two independent features for the Pylos web frontend:

1. **Undo Last Turn** in Human-vs-Human mode — a button that rewinds the most recent (or in-progress) turn, including any associated removal sub-actions.
2. **Unity-style camera controls** — Three.js scene gets a Unity Scene-view-style camera (Left=Orbit, Middle=Pan, Right=Fly with WASD/QE/Shift) as the default, plus a settings panel that lets the user rebind every mouse button and fly key.

The features touch disjoint code paths and ship together because they share the same release.

## Feature 1 — Undo Last Turn (HvH)

### Scope

- **Only in Human vs Human mode.** The Undo button is hidden in HvAI and AIvAI.
- **Whole-turn granularity.** Undo always rewinds to the state that existed *before* the player whose turn it currently is (or was last) made their placement/raise — i.e. the placement and any removal sub-actions of that turn are all undone in one click.
- Disabled when: no moves yet, game over, or viewing history.

### Server changes (`engine/server.py`)

Each game session keeps an **action stack** alongside its `PylosGame`:

- Every `move` / `remove` / `skip_removal` that successfully mutates the game pushes the resolved action index (or a `"skip_removal"` sentinel) onto the stack.
- New WebSocket message:

  ```json
  {"type": "take_back"}
  ```

- Handler:
  1. Reject if `mode != "human_vs_human"` or stack is empty → send `error`.
  2. Pop actions belonging to the in-progress or most recently completed turn — that's the trailing run of removal/skip entries plus the preceding placement/raise.
  3. Reconstruct the game by `PylosGame()` + replaying the remaining stack.
  4. Broadcast a `state` message with an added `take_back: true` flag so the client knows to retract its move-history row.

Why replay instead of deep-copying state per move? Pylos games are short (≤ ~30 actions), replay is microseconds, and the invariant `state == replay(init, action_stack)` is easy to reason about. No `copy.deepcopy` of NN-adjacent objects required.

### Client changes (`web/src/main.js`, `web/index.html`)

- New **"Undo Move"** button in the controls strip, next to "New Game".
- Visible only when `selModeEl.value === "human_vs_human"`. Disabled when `boardSnapshots.length <= 1`, `gameOver`, or `isViewingHistory()`.
- Click handler sends `{"type": "take_back"}`.
- On the returning `state` message with `take_back: true`:
  - Pop one entry from `boardSnapshots`.
  - Remove the matching cell(s) from the move-history panel (last cell, and if it was a white cell, remove the empty paired row).
  - Clear `pendingHumanMove` and `pendingAiRemovals` / `pendingAiFormation` (defensive — these should already be null in HvH).
  - Reset `viewIndex = null`, refresh nav buttons and move position indicator.

### Edge cases

- **Mid-removal-phase undo.** If the current player has placed a piece and is mid-removal, Undo rewinds the whole turn including the placement (matches "whole last turn" intent — they retake the placement).
- **Right after a turn boundary.** Undo rewinds the just-completed turn; the same player whose turn it now is becomes the next mover (not the player who just moved).
- **Concurrent click during AI thought.** Not applicable; HvH only.

## Feature 2 — Unity-style Camera Controls

### Defaults

The application's first-load defaults are the **Unity preset**:

| Input | Action |
|---|---|
| Left mouse drag | Orbit around target |
| Middle mouse drag | Pan |
| Right mouse drag | Fly mode (FPS free-look) |
| Wheel | Zoom in Orbit mode; adjust fly speed in Fly mode |
| W / A / S / D | Forward / Left / Back / Right (Fly mode) |
| Q / E | Down / Up (Fly mode) |
| Shift | Speed boost (Fly mode) |

The current (pre-existing) OrbitControls scheme is not retained as a preset.

### Controls module — new file `web/src/camera-controls.js`

A small facade over Three.js `OrbitControls`, exposing two operating modes that share one camera.

**Orbit mode (default).** Delegates to OrbitControls. The user's bindings configure `controls.mouseButtons` — each of `LEFT/MIDDLE/RIGHT` maps to one of `THREE.MOUSE.ROTATE`, `THREE.MOUSE.PAN`, `THREE.MOUSE.DOLLY`, or `null`. The Fly action is special-cased outside OrbitControls.

**Fly mode (modal, while the "Fly" mouse button is held).**

- `pointerdown` on the Fly button:
  - Disable OrbitControls.
  - Call `renderer.domElement.requestPointerLock()`.
  - Read current camera yaw/pitch from its quaternion into local state.
- `pointermove` while pointer-locked:
  - Update yaw/pitch from `event.movementX / movementY` (with a sensitivity constant).
  - Apply the new orientation to the camera.
- `requestAnimationFrame` loop while in Fly mode:
  - Translate the camera along its local axes from currently-held WASD/QE keys.
  - Shift multiplies the speed (boost).
- `wheel` while in Fly mode:
  - Adjust an internal `flySpeed` (instead of OrbitControls dolly).
- `pointerup` on the Fly button:
  - Exit pointer lock.
  - Set OrbitControls' `target` to `camera.position + forward * 5` so subsequent orbits feel natural.
  - Re-enable OrbitControls.

### Settings panel — additions to `web/index.html` + `web/src/main.js`

A new collapsible **"Camera Controls"** card in the existing right-sidebar glass UI.

**Mouse bindings.** Three rows — Left / Middle / Right — each a `<select>` of `{ Orbit, Pan, Fly, None }`. Validation rules:

- At most one mouse button may have `Fly` assigned (UI shows an inline warning otherwise).
- At least one button must be `Orbit` or `Fly` (otherwise the user has no way to rotate the camera; UI shows warning).
- Multiple buttons mapped to the same orbit-mode action (e.g. two = Pan) are allowed.

**Fly keys.** Seven rebindable keys: Forward, Back, Left, Right, Up, Down, Speed-boost. Each is a button labelled with the current key (e.g. "W"). Clicking puts it into capture mode; the next `keydown` becomes the new binding. Escape cancels. Duplicate key assignments highlighted with a warning but allowed (last write wins).

**Reset button.** One button: **"Reset to Unity defaults"**. Restores the table above.

**Persistence.** A single localStorage key, `pylos.camera.bindings`, holding a JSON object:

```json
{
  "version": 1,
  "mouse": { "left": "orbit", "middle": "pan", "right": "fly" },
  "fly": {
    "forward": "KeyW", "back": "KeyS", "left": "KeyA", "right": "KeyD",
    "up": "KeyE", "down": "KeyQ", "boost": "ShiftLeft"
  }
}
```

On load, parse and validate. If parse fails, schema mismatches, or `version` differs, fall back to defaults and overwrite. Keys use DOM `KeyboardEvent.code` strings.

### Click-vs-drag isolation

The piece-placement handler in `main.js` currently uses pointer events; need to ensure camera drags don't trigger placement. Approach: in the existing pointerdown/pointerup logic for piece clicks, record the pointerdown screen position and suppress the click if pointerup occurred more than 4 px away. (Will verify existing code first — partial drag-suppression may already exist.)

### Pointer lock UX

When Fly mode is entered the cursor disappears (browser pointer lock). On `pointerup` and on Esc we exit pointer lock. The OS-level Esc-to-release behavior is standard and acceptable. If pointer lock is denied by the browser (e.g. user denied prompt), the controls module falls back to free-mouse fly: rotates based on canvas-relative `movementX/Y`.

## Out of scope

- Touch / mobile camera controls (existing code already deals with this; no changes).
- A keyboard-only orbit mode (arrow-key panning, etc.). The skill answer was specifically *fly cam*, not arrow-pan.
- Multi-step "Redo".
- Server-side persistence of camera bindings (per-browser localStorage only).
- Take-back vs the AI (deferred; this spec is HvH only).

## Testing

**Manual (user-driven, after dev server is restarted):**

- HvH: make moves on both sides, including a placement that triggers a removal phase. Verify Undo rewinds the entire turn, the move-history row is removed, and the player who just moved is back on the clock.
- HvH: try Undo with no moves made → button disabled.
- HvAI: confirm the Undo button is hidden.
- Camera: on a fresh browser profile (cleared localStorage), confirm Unity bindings apply. Hold Right, drag → free-look; WASD moves; Shift boosts; wheel changes fly speed.
- Rebinding: remap Right → Orbit, Left → Fly via the settings UI. Reload. Confirm bindings persist.
- Click-vs-drag: rotate the camera over a board cell, release. Confirm no sphere is placed.

**Automated (light):**

- A pure-function unit test for the camera-bindings JSON validator: round-trips defaults through localStorage shape; rejects malformed/missing fields; falls back to defaults on version mismatch.
- A server-side unit test for the take-back action stack: replays an action sequence with mid-turn removals, calls take-back, asserts the game state and turn pointer match the pre-turn snapshot.

## File touch list

New files:

- `web/src/camera-controls.js` — Unity-mode + Orbit-mode controls facade.
- `engine/tests/test_take_back.py` — unit test for the server action stack.

Modified files:

- `engine/server.py` — action stack, `take_back` message handler.
- `web/src/scene.js` — initialize via the new camera-controls module instead of raw `OrbitControls`.
- `web/src/main.js` — Undo button wiring, history-row retraction, Camera Controls settings panel state.
- `web/index.html` — Undo button markup, Camera Controls collapsible card markup.

No changes to: `engine/game.py`, `engine/mcts.py`, `engine/models.py`, `web/src/network.js`, `web/src/game.js`.
