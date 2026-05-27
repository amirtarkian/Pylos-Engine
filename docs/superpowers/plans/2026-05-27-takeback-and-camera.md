# Take-back + Unity Camera Controls — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "Undo Move" button for Human-vs-Human games and replace the camera controls with a Unity-style scheme (Left=Orbit, Middle=Pan, Right=Fly with WASD/QE/Shift) with a rebindable settings panel.

**Architecture:**
- Server keeps an action stack per game session; take-back pops one turn's worth of actions and replays the rest from a fresh `PylosGame()`.
- Client gets a new `camera-controls.js` module wrapping `OrbitControls` with a Fly mode that activates via the configured mouse button + pointer lock + WASD/QE.
- Bindings live in `localStorage["pylos.camera.bindings"]`; a new collapsible card in the side panel edits them.

**Tech Stack:** Python 3.13 + FastAPI WebSocket (server), Three.js 0.160 (frontend), pytest + `fastapi.testclient.TestClient` (tests). No new packages.

**Spec:** [`docs/superpowers/specs/2026-05-27-takeback-and-camera-design.md`](../specs/2026-05-27-takeback-and-camera-design.md)

**Branch:** `feat/takeback-and-camera` (already created with the spec commit).

---

## File Structure

**Create:**
- `web/src/camera-bindings.js` — pure helpers (defaults, validate, load/save).
- `web/src/camera-controls.js` — Orbit + Fly controls facade over OrbitControls.
- `engine/tests/test_take_back.py` — server-side action-stack + take-back tests.
- `web/tests/camera-bindings.test.js` — Node test for the bindings serializer (runnable via Bun).

**Modify:**
- `engine/server.py` — per-session action stack, `take_back` message handler.
- `web/src/scene.js` — initialize via `camera-controls.js`; rewire pointer-click to fire on `pointerup` with a 4-px drag threshold so camera drags don't place pieces.
- `web/src/main.js` — Undo button wiring, history-row retraction, Camera Controls settings panel state.
- `web/index.html` — Undo button markup; Camera Controls collapsible card markup; minimal styles.

**Not touched:** `engine/game.py`, `engine/mcts.py`, `engine/models.py`, `web/src/network.js`, `web/src/game.js`.

---

## Task 1 — Server: action stack + `take_back` message

Goal: server tracks every successful action; new `take_back` message rewinds one whole turn.

**Files:**
- Modify: `engine/server.py` (sections at lines ~534 game-loop locals, ~582 after `new_game`, ~663 after `move`, ~702 after `remove`, ~737 after `skip_removal`, ~776 unknown-message fallthrough).
- Create: `engine/tests/test_take_back.py`.

- [ ] **Step 1: Write the failing test**

Create `engine/tests/test_take_back.py`:

```python
"""Tests for the server's take_back action (HvH only)."""

from fastapi.testclient import TestClient
from server import app


def _make_hvh_session():
    client = TestClient(app)
    ws = client.websocket_connect("/game").__enter__()
    ws.send_json({"type": "new_game", "mode": "human_vs_human"})
    initial = ws.receive_json()
    assert initial["type"] == "state"
    assert initial["turn"] == "white"
    return client, ws, initial


def _legal_place_action(state_msg, level, row, col):
    """Find the action index for placing at (level,row,col) in legal_moves."""
    for m in state_msg["legal_moves"]:
        if (
            m["type"] == "place"
            and m["level"] == level
            and m["row"] == row
            and m["col"] == col
        ):
            return m["action"]
    raise AssertionError(f"No legal place at L{level}({row},{col})")


def test_take_back_rejected_without_game():
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "take_back"})
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_take_back_rejected_in_human_vs_ai():
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_ai", "human_color": "white"})
        _ = ws.receive_json()  # initial state
        ws.send_json({"type": "take_back"})
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_take_back_rejected_with_empty_stack():
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        _ = ws.receive_json()
        ws.send_json({"type": "take_back"})
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_take_back_one_placement_returns_to_initial_state():
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        initial = ws.receive_json()
        action = _legal_place_action(initial, 0, 0, 0)

        ws.send_json({"type": "move", "action": {"action": action}})
        after_move = ws.receive_json()
        assert after_move["type"] == "state"
        assert after_move["turn"] == "black"
        # Reserves: white spent one
        assert after_move["reserves"]["white"] == initial["reserves"]["white"] - 1

        ws.send_json({"type": "take_back"})
        rolled_back = ws.receive_json()
        assert rolled_back["type"] == "state"
        assert rolled_back.get("take_back") is True
        assert rolled_back["turn"] == initial["turn"]
        assert rolled_back["reserves"] == initial["reserves"]
        assert rolled_back["board"] == initial["board"]


def test_take_back_rewinds_only_one_turn():
    """White places, black places, take_back -> black should be back on the clock."""
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        s0 = ws.receive_json()
        a1 = _legal_place_action(s0, 0, 0, 0)
        ws.send_json({"type": "move", "action": {"action": a1}})
        s1 = ws.receive_json()
        assert s1["turn"] == "black"

        a2 = _legal_place_action(s1, 0, 1, 1)
        ws.send_json({"type": "move", "action": {"action": a2}})
        s2 = ws.receive_json()
        assert s2["turn"] == "white"

        ws.send_json({"type": "take_back"})
        rolled = ws.receive_json()
        assert rolled["type"] == "state"
        assert rolled.get("take_back") is True
        assert rolled["turn"] == "black"
        # White piece from move 1 is still on the board
        assert rolled["reserves"]["white"] == s0["reserves"]["white"] - 1
        assert rolled["reserves"]["black"] == s0["reserves"]["black"]
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd /Users/amir/Documents/GitHub/Pylos-Engine && uv run --with httpx pytest engine/tests/test_take_back.py -v`

Expected: FAIL — server replies "Unknown message type: take_back" for the take_back send paths.

- [ ] **Step 3: Add action-stack locals to the game-loop in `engine/server.py`**

After line 532 (`shared_delay = [1500]`), add an action stack local:

```python
    # Per-session action history for take-back. Each entry is either an int
    # (action index applied via PylosGame.step) or one of the sentinel strings
    # "remove:L,R,C" / "skip_removal".
    action_stack: list = []
```

- [ ] **Step 4: Reset the stack on `new_game`**

Inside the `if msg_type == "new_game":` block in `engine/server.py`, immediately after `game = PylosGame()` (around line 548), add:

```python
                action_stack = []
```

- [ ] **Step 5: Record actions after each successful mutation**

After `game.step(action, auto_remove=False)` in the `move` handler (around line 663), add:

```python
                action_stack.append(int(action))
```

After `game.step_removal(level, row, col)` succeeds in the `remove` handler (around line 702), inside the `if not game.step_removal(...)` block-exit success path (right after the `if`), add:

```python
                action_stack.append(f"remove:{level},{row},{col}")
```

After `game.skip_removal()` in the `skip_removal` handler (around line 737), add:

```python
                action_stack.append("skip_removal")
```

- [ ] **Step 6: Add the `take_back` handler before the unknown-type `else`**

In `engine/server.py`, immediately before the `else:` that emits "Unknown message type" (around line 776), insert:

```python
            elif msg_type == "take_back":
                if game is None:
                    await ws.send_json({"type": "error", "message": "No active game"})
                    continue
                if mode != "human_vs_human":
                    await ws.send_json({"type": "error", "message": "Take-back is only available in Human vs Human"})
                    continue
                if not action_stack:
                    await ws.send_json({"type": "error", "message": "Nothing to take back"})
                    continue

                # Pop trailing remove/skip_removal entries, then the placement/raise that started the turn.
                while action_stack and isinstance(action_stack[-1], str):
                    action_stack.pop()
                if action_stack:
                    action_stack.pop()  # the placement/raise action int

                # Replay remaining actions from a fresh game.
                game = PylosGame()
                for entry in action_stack:
                    if isinstance(entry, int):
                        game.step(entry, auto_remove=False)
                    elif isinstance(entry, str) and entry.startswith("remove:"):
                        _, coords = entry.split(":", 1)
                        l, r, c = (int(x) for x in coords.split(","))
                        game.step_removal(l, r, c)
                    elif entry == "skip_removal":
                        game.skip_removal()

                state = make_state_msg(game)
                state["take_back"] = True
                await ws.send_json(state)
```

- [ ] **Step 7: Run tests, verify they pass**

Run: `cd /Users/amir/Documents/GitHub/Pylos-Engine && uv run --with httpx pytest engine/tests/test_take_back.py -v`

Expected: PASS for all 5 tests.

- [ ] **Step 8: Run the full server test suite to make sure nothing regressed**

Run: `cd /Users/amir/Documents/GitHub/Pylos-Engine && uv run --with httpx pytest engine/tests/test_server.py engine/tests/test_take_back.py -v`

Expected: PASS — no existing tests broken.

- [ ] **Step 9: Commit**

```bash
git add engine/server.py engine/tests/test_take_back.py
git commit -m "feat(server): add take_back message for HvH undo

Server keeps an action stack per session and replays from a fresh
PylosGame to roll back exactly one turn (placement/raise + any
removal/skip sub-actions). Rejected in HvAI and AIvAI."
```

---

## Task 2 — Client: Undo button + history retraction

Goal: render an "Undo Move" button in HvH; on click, send `take_back` and clean up the move-history UI when the server returns the rolled-back state.

**Files:**
- Modify: `web/index.html` (add button next to `#btn-new-game`).
- Modify: `web/src/main.js` (button wiring, retraction helper, state-message handler hook).

- [ ] **Step 1: Add the Undo button markup**

In `web/index.html`, change the existing block at line 840–842:

```html
    <div class="control-group">
      <button id="btn-new-game">New Game</button>
    </div>
```

…to:

```html
    <div class="control-group">
      <button id="btn-new-game">New Game</button>
      <button id="btn-undo-move" class="hidden" style="margin-top:8px;">Undo Move</button>
    </div>
```

- [ ] **Step 2: Grab a DOM ref for the button**

In `web/src/main.js`, add this line right after `const btnNewGameEl = document.getElementById("btn-new-game");` (around line 47):

```javascript
const btnUndoMoveEl = document.getElementById("btn-undo-move");
```

- [ ] **Step 3: Add a `updateUndoButtonVisibility()` helper**

In `web/src/main.js`, just above the `// ── UI state updates` section header (around line 273), add:

```javascript
function updateUndoButtonVisibility() {
  if (!btnUndoMoveEl) return;
  const inHvH = selModeEl.value === "human_vs_human";
  const hasMoves = boardSnapshots.length > 1;
  const allowed = inHvH && hasMoves && !gameOver && !isViewingHistory();
  btnUndoMoveEl.classList.toggle("hidden", !inHvH);
  btnUndoMoveEl.disabled = !allowed;
}
```

- [ ] **Step 4: Wire the button click**

In `web/src/main.js`, in the event-binding block (find `btnNewGameEl.addEventListener("click", startNewGame);` around line 1158) add the next line:

```javascript
  btnUndoMoveEl.addEventListener("click", () => {
    if (btnUndoMoveEl.disabled) return;
    send({ type: "take_back" });
    setStatus("Taking back...", "info", 1500);
  });
```

- [ ] **Step 5: Add a `retractLastMoveFromHistory()` helper**

In `web/src/main.js`, immediately after `function clearMoveHistory()` (after the closing brace around line 165), add:

```javascript
function retractLastMoveFromHistory() {
  // Drop the last filled move cell. If the row only had a white cell, remove the row.
  if (!currentMoveRow) return;
  const blackCell = currentMoveRow.querySelector(".move-cell.black-move");
  if (blackCell) {
    blackCell.className = "move-cell placeholder";
    blackCell.textContent = "…";
    blackCell.removeAttribute("data-idx");
    blackCell.replaceWith(blackCell.cloneNode(true)); // strip listeners
    return;
  }
  // No black cell yet -> drop the whole row, decrement turn number.
  currentMoveRow.remove();
  turnNumber = Math.max(0, turnNumber - 1);
  currentMoveRow = moveHistoryEl.lastElementChild || null;
  if (moveCount > 0) moveCount--;
}
```

- [ ] **Step 6: Handle the server's `take_back` flag in the state-message branch**

In `web/src/main.js`, find the `state` handler (search for `msg.type === "state"` — around the area near line 525-570 where snapshots are saved). At the very start of the state branch, before snapshot save logic, insert:

```javascript
      if (msg.take_back) {
        boardSnapshots.pop();
        retractLastMoveFromHistory();
        pendingHumanMove = null;
        viewIndex = null;
        gameOver = false;
        updateNavButtons();
        updateMovePosition();
        updateUndoButtonVisibility();
        // Fall through: the rest of the state branch updates board/turn/reserves
        // from msg as usual, which is exactly what we want.
      }
```

(Find the precise location: the block that begins with `} else if (msg.type === "state") {`. Verify with `grep -n '"state"' web/src/main.js`.)

- [ ] **Step 7: Refresh the button on every relevant state change**

In `web/src/main.js`:

(a) In `updateNavButtons()` (around line 207), add as the final line before the closing brace:

```javascript
  updateUndoButtonVisibility();
```

(b) In the mode-`change` listener for `selModeEl` (search `selModeEl.addEventListener("change"` — around line 1251 area, "Initial mode visibility"), add at the end:

```javascript
  updateUndoButtonVisibility();
```

(c) In `clearMoveHistory()` (after the existing body), add at the end:

```javascript
  updateUndoButtonVisibility();
```

(d) In `startNewGame()` (search `function startNewGame` — around line 703), at the very end before the closing brace add:

```javascript
  updateUndoButtonVisibility();
```

- [ ] **Step 8: Manual test (launch the dev server, drive the UI)**

Run (in another shell):

```bash
cd /Users/amir/Documents/GitHub/Pylos-Engine && bash run.sh
```

Then in a browser at `http://localhost:8000/`:
1. Select **Human vs Human**, click **New Game**. Confirm the "Undo Move" button appears and is disabled.
2. Place a white piece on level 0. Confirm the button enables.
3. Click **Undo Move**. Confirm the piece disappears, white reserves go back to 15, the move-history row is removed, and turn indicator says **WHITE'S TURN**.
4. Make two placements (white, black), then click **Undo Move**. Confirm only black's move is undone; black is on the clock again.
5. Switch to **Human vs AI**. Confirm the Undo button disappears.

Expected: All 5 manual checks pass.

- [ ] **Step 9: Commit**

```bash
git add web/index.html web/src/main.js
git commit -m "feat(web): add Undo Move button for HvH

Button is visible only in Human vs Human, sends take_back to the
server, and retracts the last move-history row when the server
returns the rolled-back state."
```

---

## Task 3 — Camera bindings: pure module + persistence

Goal: factor camera-binding defaults, validation, and localStorage I/O into a tiny pure module so it's testable and reusable from the settings panel and the controls module.

**Files:**
- Create: `web/src/camera-bindings.js`.
- Create: `web/tests/camera-bindings.test.js`.

- [ ] **Step 1: Write the failing test**

Create `web/tests/camera-bindings.test.js`:

```javascript
// Run with: bun test web/tests/camera-bindings.test.js
import { test, expect, beforeEach } from "bun:test";
import {
  DEFAULT_BINDINGS,
  validateBindings,
  loadBindings,
  saveBindings,
  STORAGE_KEY,
} from "../src/camera-bindings.js";

// Minimal localStorage mock for Node/Bun.
class MemStore {
  constructor() { this.m = new Map(); }
  getItem(k) { return this.m.has(k) ? this.m.get(k) : null; }
  setItem(k, v) { this.m.set(k, String(v)); }
  removeItem(k) { this.m.delete(k); }
  clear() { this.m.clear(); }
}

beforeEach(() => {
  globalThis.localStorage = new MemStore();
});

test("DEFAULT_BINDINGS is the Unity preset", () => {
  expect(DEFAULT_BINDINGS.mouse).toEqual({ left: "orbit", middle: "pan", right: "fly" });
  expect(DEFAULT_BINDINGS.fly.forward).toBe("KeyW");
  expect(DEFAULT_BINDINGS.fly.back).toBe("KeyS");
  expect(DEFAULT_BINDINGS.fly.left).toBe("KeyA");
  expect(DEFAULT_BINDINGS.fly.right).toBe("KeyD");
  expect(DEFAULT_BINDINGS.fly.up).toBe("KeyE");
  expect(DEFAULT_BINDINGS.fly.down).toBe("KeyQ");
  expect(DEFAULT_BINDINGS.fly.boost).toBe("ShiftLeft");
  expect(DEFAULT_BINDINGS.version).toBe(1);
});

test("validateBindings accepts the defaults", () => {
  expect(validateBindings(DEFAULT_BINDINGS)).toBe(true);
});

test("validateBindings rejects missing fields", () => {
  expect(validateBindings(null)).toBe(false);
  expect(validateBindings({})).toBe(false);
  expect(validateBindings({ mouse: {}, fly: {} })).toBe(false);
  expect(validateBindings({ version: 1, mouse: { left: "orbit" }, fly: DEFAULT_BINDINGS.fly })).toBe(false);
});

test("validateBindings rejects unknown mouse actions", () => {
  const bad = JSON.parse(JSON.stringify(DEFAULT_BINDINGS));
  bad.mouse.left = "teleport";
  expect(validateBindings(bad)).toBe(false);
});

test("validateBindings rejects wrong version", () => {
  const bad = JSON.parse(JSON.stringify(DEFAULT_BINDINGS));
  bad.version = 999;
  expect(validateBindings(bad)).toBe(false);
});

test("loadBindings returns defaults when storage is empty", () => {
  expect(loadBindings()).toEqual(DEFAULT_BINDINGS);
});

test("loadBindings returns defaults when JSON is malformed", () => {
  localStorage.setItem(STORAGE_KEY, "{not json");
  expect(loadBindings()).toEqual(DEFAULT_BINDINGS);
});

test("saveBindings + loadBindings round-trip", () => {
  const custom = JSON.parse(JSON.stringify(DEFAULT_BINDINGS));
  custom.mouse.left = "fly";
  custom.mouse.right = "orbit";
  custom.fly.boost = "ControlLeft";
  saveBindings(custom);
  expect(loadBindings()).toEqual(custom);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/amir/Documents/GitHub/Pylos-Engine && bun test web/tests/camera-bindings.test.js`

Expected: FAIL with "Cannot find module ../src/camera-bindings.js".

(If `bun` isn't installed: `brew install oven-sh/bun/bun` per the user's "Bun over npm/pnpm" preference.)

- [ ] **Step 3: Write the module**

Create `web/src/camera-bindings.js`:

```javascript
/**
 * camera-bindings.js -- pure helpers for camera control bindings.
 *
 * Bindings shape:
 *   {
 *     version: 1,
 *     mouse: { left: "orbit"|"pan"|"fly"|"none", middle: ..., right: ... },
 *     fly:   { forward, back, left, right, up, down, boost }   // KeyboardEvent.code strings
 *   }
 */

export const STORAGE_KEY = "pylos.camera.bindings";
export const CURRENT_VERSION = 1;
export const MOUSE_ACTIONS = ["orbit", "pan", "fly", "none"];
export const FLY_KEYS = ["forward", "back", "left", "right", "up", "down", "boost"];

export const DEFAULT_BINDINGS = Object.freeze({
  version: CURRENT_VERSION,
  mouse: { left: "orbit", middle: "pan", right: "fly" },
  fly: {
    forward: "KeyW",
    back: "KeyS",
    left: "KeyA",
    right: "KeyD",
    up: "KeyE",
    down: "KeyQ",
    boost: "ShiftLeft",
  },
});

export function validateBindings(b) {
  if (!b || typeof b !== "object") return false;
  if (b.version !== CURRENT_VERSION) return false;
  if (!b.mouse || typeof b.mouse !== "object") return false;
  for (const btn of ["left", "middle", "right"]) {
    if (!MOUSE_ACTIONS.includes(b.mouse[btn])) return false;
  }
  if (!b.fly || typeof b.fly !== "object") return false;
  for (const k of FLY_KEYS) {
    if (typeof b.fly[k] !== "string" || b.fly[k].length === 0) return false;
  }
  return true;
}

export function loadBindings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return cloneDefaults();
    const parsed = JSON.parse(raw);
    if (!validateBindings(parsed)) return cloneDefaults();
    return parsed;
  } catch (_e) {
    return cloneDefaults();
  }
}

export function saveBindings(b) {
  if (!validateBindings(b)) throw new Error("invalid bindings");
  localStorage.setItem(STORAGE_KEY, JSON.stringify(b));
}

function cloneDefaults() {
  return JSON.parse(JSON.stringify(DEFAULT_BINDINGS));
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/amir/Documents/GitHub/Pylos-Engine && bun test web/tests/camera-bindings.test.js`

Expected: PASS — all 8 tests green.

- [ ] **Step 5: Commit**

```bash
git add web/src/camera-bindings.js web/tests/camera-bindings.test.js
git commit -m "feat(web): camera-bindings module (defaults, validate, persist)

Pure helpers for the camera control bindings: Unity preset as the
default, validation against a fixed schema, and localStorage round-
trip with safe fallback on parse/version errors."
```

---

## Task 4 — Camera controls module (Orbit + Fly)

Goal: produce a controls facade that wraps `OrbitControls` for Orbit mode and implements Fly mode (pointer lock + WASD/QE + Shift) for the configured Fly mouse button.

**Files:**
- Create: `web/src/camera-controls.js`.

- [ ] **Step 1: Create the module skeleton**

Create `web/src/camera-controls.js`:

```javascript
/**
 * camera-controls.js -- Orbit + Fly controls facade.
 *
 * Usage:
 *   const cc = createCameraControls(camera, domElement, { bindings });
 *   cc.update();         // call once per render frame
 *   cc.setBindings(b);   // swap bindings at runtime
 *   cc.dispose();
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { DEFAULT_BINDINGS } from "./camera-bindings.js";

const MOUSE_BUTTON_INDEX = { left: 0, middle: 1, right: 2 };

const ORBIT_ACTION_MAP = {
  orbit: THREE.MOUSE.ROTATE,
  pan: THREE.MOUSE.PAN,
  // "fly" and "none" -> not handled by OrbitControls; set to null below.
};

const FLY_SPEED_BASE = 4.0;       // world units per second
const FLY_SPEED_BOOST = 3.0;      // multiplier when boost held
const FLY_LOOK_SENS = 0.0025;     // radians per pixel of pointer movement
const FLY_WHEEL_SPEED_STEP = 0.2; // multiplicative
const FLY_EXIT_DISTANCE = 5;      // target placed this far in front on exit

export function createCameraControls(camera, dom, opts = {}) {
  let bindings = opts.bindings ? structuredClone(opts.bindings) : structuredClone(DEFAULT_BINDINGS);

  const orbit = new OrbitControls(camera, dom);
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.08;
  orbit.minDistance = 4;
  orbit.maxDistance = 15;
  orbit.maxPolarAngle = Math.PI * 0.44;

  // Fly state
  let flying = false;
  let flySpeed = 1.0; // multiplier on FLY_SPEED_BASE, tuned by wheel
  const heldKeys = new Set();
  let yaw = 0, pitch = 0;
  let lastTime = performance.now();

  function applyBindingsToOrbit() {
    const m = bindings.mouse;
    orbit.mouseButtons = {
      LEFT:   ORBIT_ACTION_MAP[m.left]   ?? null,
      MIDDLE: ORBIT_ACTION_MAP[m.middle] ?? null,
      RIGHT:  ORBIT_ACTION_MAP[m.right]  ?? null,
    };
  }
  applyBindingsToOrbit();

  function flyButtonIndex() {
    const m = bindings.mouse;
    if (m.left === "fly") return 0;
    if (m.middle === "fly") return 1;
    if (m.right === "fly") return 2;
    return -1;
  }

  function syncYawPitchFromCamera() {
    const e = new THREE.Euler().setFromQuaternion(camera.quaternion, "YXZ");
    yaw = e.y;
    pitch = e.x;
  }

  function applyYawPitchToCamera() {
    const maxPitch = Math.PI / 2 - 0.001;
    pitch = Math.max(-maxPitch, Math.min(maxPitch, pitch));
    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, 0, "YXZ"));
    camera.quaternion.copy(q);
  }

  function enterFly() {
    if (flying) return;
    flying = true;
    orbit.enabled = false;
    syncYawPitchFromCamera();
    if (dom.requestPointerLock) {
      try { dom.requestPointerLock(); } catch (_) { /* fall back to free mouse */ }
    }
  }

  function exitFly() {
    if (!flying) return;
    flying = false;
    if (document.pointerLockElement === dom && document.exitPointerLock) {
      document.exitPointerLock();
    }
    // Place orbit target a fixed distance in front of the camera.
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    orbit.target.copy(camera.position).addScaledVector(forward, FLY_EXIT_DISTANCE);
    orbit.enabled = true;
    orbit.update();
    heldKeys.clear();
  }

  function onPointerDown(ev) {
    if (ev.button === flyButtonIndex()) {
      ev.preventDefault();
      enterFly();
    }
  }

  function onPointerUp(ev) {
    if (ev.button === flyButtonIndex() && flying) {
      ev.preventDefault();
      exitFly();
    }
  }

  function onPointerMove(ev) {
    if (!flying) return;
    const dx = ev.movementX || 0;
    const dy = ev.movementY || 0;
    yaw   -= dx * FLY_LOOK_SENS;
    pitch -= dy * FLY_LOOK_SENS;
    applyYawPitchToCamera();
  }

  function onKeyDown(ev) {
    if (!flying) return;
    heldKeys.add(ev.code);
  }

  function onKeyUp(ev) {
    heldKeys.delete(ev.code);
  }

  function onWheel(ev) {
    if (!flying) return;
    ev.preventDefault();
    const dir = Math.sign(ev.deltaY);
    if (dir > 0) flySpeed = Math.max(0.1, flySpeed - FLY_WHEEL_SPEED_STEP);
    else if (dir < 0) flySpeed = Math.min(10, flySpeed + FLY_WHEEL_SPEED_STEP);
  }

  function onContextMenu(ev) {
    // If right-mouse is bound to anything (fly/orbit/pan/none), suppress the browser menu.
    if (bindings.mouse.right !== "none") ev.preventDefault();
  }

  function onPointerLockChange() {
    // If the user pressed Esc to release the lock while still holding the mouse, exit fly.
    if (flying && document.pointerLockElement !== dom) {
      exitFly();
    }
  }

  dom.addEventListener("pointerdown", onPointerDown);
  dom.addEventListener("pointerup", onPointerUp);
  dom.addEventListener("pointermove", onPointerMove);
  dom.addEventListener("wheel", onWheel, { passive: false });
  dom.addEventListener("contextmenu", onContextMenu);
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  document.addEventListener("pointerlockchange", onPointerLockChange);

  function update() {
    const now = performance.now();
    const dt = Math.min(0.05, (now - lastTime) / 1000);
    lastTime = now;

    if (flying) {
      const k = bindings.fly;
      const boost = heldKeys.has(k.boost) ? FLY_SPEED_BOOST : 1.0;
      const speed = FLY_SPEED_BASE * flySpeed * boost * dt;

      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
      const worldUp = new THREE.Vector3(0, 1, 0);

      if (heldKeys.has(k.forward)) camera.position.addScaledVector(forward, speed);
      if (heldKeys.has(k.back))    camera.position.addScaledVector(forward, -speed);
      if (heldKeys.has(k.left))    camera.position.addScaledVector(right, -speed);
      if (heldKeys.has(k.right))   camera.position.addScaledVector(right, speed);
      if (heldKeys.has(k.up))      camera.position.addScaledVector(worldUp, speed);
      if (heldKeys.has(k.down))    camera.position.addScaledVector(worldUp, -speed);
    } else {
      orbit.update();
    }
  }

  function setBindings(nextBindings) {
    if (flying) exitFly();
    bindings = structuredClone(nextBindings);
    applyBindingsToOrbit();
  }

  function dispose() {
    if (flying) exitFly();
    orbit.dispose();
    dom.removeEventListener("pointerdown", onPointerDown);
    dom.removeEventListener("pointerup", onPointerUp);
    dom.removeEventListener("pointermove", onPointerMove);
    dom.removeEventListener("wheel", onWheel);
    dom.removeEventListener("contextmenu", onContextMenu);
    window.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("keyup", onKeyUp);
    document.removeEventListener("pointerlockchange", onPointerLockChange);
  }

  return {
    update,
    setBindings,
    dispose,
    get orbit() { return orbit; },
    get isFlying() { return flying; },
  };
}
```

- [ ] **Step 2: Commit (manual integration test happens in Task 5)**

```bash
git add web/src/camera-controls.js
git commit -m "feat(web): Unity-style camera controls (Orbit + Fly modes)

Wraps OrbitControls for orbit/pan/zoom and adds a Fly mode that
activates on the configured mouse button: pointer lock for FPS-style
look, WASD/QE for movement, Shift for boost, wheel adjusts fly speed."
```

---

## Task 5 — Wire `camera-controls` into `scene.js`; fix click-vs-drag

Goal: replace the raw `OrbitControls` in `scene.js` with the new module, and change the piece-click handler to fire on pointer*up* with a drag threshold so camera drags don't place pieces.

**Files:**
- Modify: `web/src/scene.js`.

- [ ] **Step 1: Swap the import and constructor**

In `web/src/scene.js`, replace line 15:

```javascript
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
```

with:

```javascript
import { createCameraControls } from "./camera-controls.js";
import { loadBindings } from "./camera-bindings.js";
```

- [ ] **Step 2: Use the new constructor in `init`**

In `web/src/scene.js`, replace lines 87–95 (the `// Controls` block):

```javascript
  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 1.5, 0);
  controls.minDistance = 4;
  controls.maxDistance = 15;
  controls.maxPolarAngle = Math.PI * 0.44; // ~80 degrees
  controls.update();
```

with:

```javascript
  // Controls (Unity-style: Left=Orbit, Middle=Pan, Right=Fly)
  controls = createCameraControls(camera, renderer.domElement, {
    bindings: loadBindings(),
  });
  controls.orbit.target.set(0, 1.5, 0);
  controls.orbit.update();
```

- [ ] **Step 3: Confirm the render loop still calls `controls.update()`**

The facade exposes `.update()` with the same signature OrbitControls did, so `_animate`'s existing `controls.update()` call needs no changes. Verify:

Run: `grep -n "controls.update\|controls.orbit" /Users/amir/Documents/GitHub/Pylos-Engine/web/src/scene.js`

Expected after edits: exactly one `controls.update()` call (in `_animate`) and one `controls.orbit.update()` call (in `init`, added in Step 2).

- [ ] **Step 4: Export a setter so `main.js` can push new bindings at runtime**

In `web/src/scene.js`, add at the end of the file (after the existing exports):

```javascript
export function setCameraBindings(bindings) {
  if (controls && controls.setBindings) controls.setBindings(bindings);
}
```

- [ ] **Step 5: Rewire piece picking to fire on pointer-up with a 4-px drag threshold**

In `web/src/scene.js`, replace the entire `onPointerDown` function (lines 502–517) with two functions:

```javascript
// Track drag distance so camera drags don't trigger piece placement.
let _pointerDownState = null; // { button, x, y }
const DRAG_PX_THRESHOLD = 4;

function onPointerDown(event) {
  if (event.button !== 0) {
    _pointerDownState = null;
    return;
  }
  _pointerDownState = { button: 0, x: event.clientX, y: event.clientY };
}

function onPointerUp(event) {
  if (event.button !== 0 || _pointerDownState === null) return;
  const dx = event.clientX - _pointerDownState.x;
  const dy = event.clientY - _pointerDownState.y;
  _pointerDownState = null;
  if (dx * dx + dy * dy > DRAG_PX_THRESHOLD * DRAG_PX_THRESHOLD) return;

  const intersects = _getIntersects(event);
  if (intersects.length === 0) return;

  const obj = intersects[0].object;
  const ud = obj.userData;

  if (ud.type === "sphere" && onSphereClick) {
    onSphereClick(ud.level, ud.row, ud.col, ud.player);
  } else if (ud.type === "position" && onPositionClick) {
    onPositionClick(ud.level, ud.row, ud.col);
  }
}
```

- [ ] **Step 6: Register the pointer-up listener**

In `web/src/scene.js`, at line ~195 where `addEventListener("pointerdown", onPointerDown, false)` lives, add right after it:

```javascript
  renderer.domElement.addEventListener("pointerup", onPointerUp, false);
```

- [ ] **Step 7: Manual smoke test**

Start the server (`bash run.sh`) and reload `http://localhost:8000/`. Verify:

1. Hold **Left mouse** and drag — camera orbits.
2. Hold **Middle mouse** and drag — camera pans.
3. Hold **Right mouse** — cursor disappears (pointer lock); moving the mouse looks around; WASD moves; Q/E descends/ascends; Shift accelerates. Release Right — cursor returns and OrbitControls resumes.
4. Click a position (no drag) — sphere drops as expected.
5. Drag the camera over a position and release — no spurious sphere placed.
6. Mouse wheel in Orbit mode — zoom. Mouse wheel while holding Right — fly speed changes (visible by observing how fast WASD moves the camera).

Expected: all 6 checks pass.

- [ ] **Step 8: Commit**

```bash
git add web/src/scene.js
git commit -m "feat(web): switch scene.js to Unity-style camera + drag-safe clicks

Initialize camera through the new camera-controls module loaded from
saved bindings. Piece picking now fires on pointer-up with a 4-px
drag threshold so left-drag orbiting no longer places pieces."
```

---

## Task 6 — Camera Controls settings panel UI

Goal: collapsible "Camera Controls" card with three mouse-button selects, seven key-capture buttons, and a "Reset to Unity defaults" button. Changes apply live via `setCameraBindings`.

**Files:**
- Modify: `web/index.html` (markup + styles for the card).
- Modify: `web/src/main.js` (wiring + key-capture state).

- [ ] **Step 1: Add the card markup**

In `web/index.html`, just before `</div>` that closes `#side-panel` (find `<div id="move-history-wrapper">` — the wrapper closes a few lines below; insert immediately after that wrapper closes and before side-panel closes). The card:

```html
    <div id="camera-controls-card" class="control-group" style="margin-top:12px;">
      <button id="cam-controls-toggle" class="glass" style="
        width:100%;padding:8px;display:flex;justify-content:space-between;
        align-items:center;background:transparent;color:#d4d8e8;
        border:1px solid rgba(255,255,255,0.1);border-radius:6px;
        font-size:11px;text-transform:uppercase;letter-spacing:1px;cursor:pointer;">
        <span>Camera Controls</span><span id="cam-controls-caret">&#9656;</span>
      </button>
      <div id="cam-controls-body" class="hidden" style="margin-top:8px;font-size:11px;color:#d4d8e8;">
        <div style="margin-bottom:6px;color:#7a82a0;">Mouse buttons</div>
        <label style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
          Left
          <select id="cam-mouse-left" style="flex:1;">
            <option value="orbit">Orbit</option><option value="pan">Pan</option>
            <option value="fly">Fly</option><option value="none">None</option>
          </select>
        </label>
        <label style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
          Middle
          <select id="cam-mouse-middle" style="flex:1;">
            <option value="orbit">Orbit</option><option value="pan">Pan</option>
            <option value="fly">Fly</option><option value="none">None</option>
          </select>
        </label>
        <label style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
          Right
          <select id="cam-mouse-right" style="flex:1;">
            <option value="orbit">Orbit</option><option value="pan">Pan</option>
            <option value="fly">Fly</option><option value="none">None</option>
          </select>
        </label>
        <div id="cam-bindings-warning" style="color:#ffb86b;font-size:10px;min-height:14px;margin-bottom:6px;"></div>
        <div style="margin-bottom:6px;color:#7a82a0;">Fly keys</div>
        <div id="cam-fly-keys-grid" style="display:grid;grid-template-columns:auto 1fr;gap:4px 8px;align-items:center;">
          <span>Forward</span><button class="cam-key-btn" data-key="forward"></button>
          <span>Back</span><button class="cam-key-btn" data-key="back"></button>
          <span>Left</span><button class="cam-key-btn" data-key="left"></button>
          <span>Right</span><button class="cam-key-btn" data-key="right"></button>
          <span>Up</span><button class="cam-key-btn" data-key="up"></button>
          <span>Down</span><button class="cam-key-btn" data-key="down"></button>
          <span>Boost</span><button class="cam-key-btn" data-key="boost"></button>
        </div>
        <button id="cam-reset-defaults" style="margin-top:8px;width:100%;padding:6px;">
          Reset to Unity defaults
        </button>
      </div>
    </div>
```

Add to the existing `<style>` block near the top of `index.html` (search for `.hidden` — there's an existing style block):

```html
    .cam-key-btn {
      padding: 4px 8px;
      font-family: 'SF Mono', monospace;
      font-size: 11px;
      background: rgba(255,255,255,0.05);
      color: #d4d8e8;
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 4px;
      cursor: pointer;
    }
    .cam-key-btn.capturing {
      background: rgba(74,108,247,0.25);
      border-color: #4a6cf7;
    }
```

- [ ] **Step 2: Wire selectors and apply-on-change in `main.js`**

In `web/src/main.js`:

(a) Near the top with the other imports, add:

```javascript
import { DEFAULT_BINDINGS, loadBindings, saveBindings, validateBindings } from "./camera-bindings.js";
import { setCameraBindings } from "./scene.js";
```

(b) Just below the existing DOM-ref declarations (after `btnUndoMoveEl`), add:

```javascript
const camToggleEl = document.getElementById("cam-controls-toggle");
const camCaretEl = document.getElementById("cam-controls-caret");
const camBodyEl = document.getElementById("cam-controls-body");
const camMouseEls = {
  left: document.getElementById("cam-mouse-left"),
  middle: document.getElementById("cam-mouse-middle"),
  right: document.getElementById("cam-mouse-right"),
};
const camWarningEl = document.getElementById("cam-bindings-warning");
const camKeyBtnEls = Array.from(document.querySelectorAll(".cam-key-btn"));
const camResetEl = document.getElementById("cam-reset-defaults");
```

(c) Add the binding-state code as a new section near the bottom of `main.js`, just above the final initialization block (search for `// Initial mode visibility`):

```javascript
// ── Camera bindings UI ──────────────────────────────────────────
let camBindings = loadBindings();
let camCapturingKey = null; // "forward"|...|null

function paintCamUI() {
  for (const btn of ["left", "middle", "right"]) {
    camMouseEls[btn].value = camBindings.mouse[btn];
  }
  for (const el of camKeyBtnEls) {
    const which = el.dataset.key;
    el.textContent = camBindings.fly[which];
    el.classList.toggle("capturing", camCapturingKey === which);
  }
  // Warnings
  const flyCount = Object.values(camBindings.mouse).filter((v) => v === "fly").length;
  const hasRotate = Object.values(camBindings.mouse).some((v) => v === "orbit" || v === "fly");
  const warnings = [];
  if (flyCount > 1) warnings.push("Multiple buttons set to Fly; only one will activate fly mode.");
  if (!hasRotate) warnings.push("No button can rotate the camera (assign Orbit or Fly).");
  camWarningEl.textContent = warnings.join(" ");
}

function persistAndApply() {
  if (!validateBindings(camBindings)) {
    camBindings = JSON.parse(JSON.stringify(DEFAULT_BINDINGS));
  }
  saveBindings(camBindings);
  setCameraBindings(camBindings);
  paintCamUI();
}

camToggleEl.addEventListener("click", () => {
  const open = !camBodyEl.classList.contains("hidden");
  camBodyEl.classList.toggle("hidden", open);
  camCaretEl.innerHTML = open ? "▸" : "▾";
});

for (const btn of ["left", "middle", "right"]) {
  camMouseEls[btn].addEventListener("change", (e) => {
    camBindings.mouse[btn] = e.target.value;
    persistAndApply();
  });
}

for (const el of camKeyBtnEls) {
  el.addEventListener("click", () => {
    camCapturingKey = el.dataset.key;
    paintCamUI();
  });
}

window.addEventListener("keydown", (e) => {
  if (!camCapturingKey) return;
  if (e.code === "Escape") {
    camCapturingKey = null;
    paintCamUI();
    return;
  }
  e.preventDefault();
  e.stopImmediatePropagation(); // don't trigger nav shortcuts (← → Space Home End)
  camBindings.fly[camCapturingKey] = e.code;
  camCapturingKey = null;
  persistAndApply();
}, true); // capture phase so the key isn't consumed by fly-mode or nav handlers

camResetEl.addEventListener("click", () => {
  camBindings = JSON.parse(JSON.stringify(DEFAULT_BINDINGS));
  persistAndApply();
});

paintCamUI();
```

- [ ] **Step 3: Manual UI test**

Restart the server and reload the page.

1. Click **Camera Controls** in the side panel. Confirm it expands.
2. Change **Left** to "Pan". Try to orbit with left mouse — should pan now.
3. Change **Right** to "Orbit", **Left** back to "Orbit". The warning "Multiple buttons can rotate" should NOT appear (mixing orbit and fly is OK; only "no rotation" or "multiple fly" warn).
4. Set **Right** to "Fly" and **Left** to "Fly". Confirm the warning about multiple Fly buttons appears; verify only one of them actually triggers fly mode.
5. Click the **Forward** key button. It turns blue. Press **ArrowUp**. The button label updates to "ArrowUp" and Forward in fly mode now uses ArrowUp.
6. Click **Reset to Unity defaults**. All settings revert; Forward = "KeyW" again.
7. Reload the page. Confirm your custom settings (before reset) would have persisted (so reload before resetting in step 6 to actually test this). Re-test by setting Left=Pan, reload, confirm Left=Pan after reload.

Expected: all 7 checks pass.

- [ ] **Step 4: Commit**

```bash
git add web/index.html web/src/main.js
git commit -m "feat(web): camera controls settings panel

Collapsible side-panel card to remap mouse buttons (Orbit/Pan/Fly/None)
and rebind fly-mode keys via click-then-press capture. Changes apply
live through setCameraBindings, persist to localStorage, and validate
against the bindings schema."
```

---

## Task 7 — End-to-end smoke + final commit

Goal: regression-check the full feature flow.

- [ ] **Step 1: Run all backend tests**

Run: `cd /Users/amir/Documents/GitHub/Pylos-Engine && uv run --with httpx pytest engine/tests/ -v`

Expected: PASS for all tests (no new failures vs baseline).

- [ ] **Step 2: Run the camera-bindings unit test**

Run: `cd /Users/amir/Documents/GitHub/Pylos-Engine && bun test web/tests/camera-bindings.test.js`

Expected: PASS.

- [ ] **Step 3: End-to-end manual walkthrough**

Start the server (`bash run.sh`) and in a fresh browser profile (or after clearing `pylos.camera.bindings` from localStorage):

1. Default load → Camera Controls panel shows Left=Orbit, Middle=Pan, Right=Fly; fly keys are WASD/QE/Shift.
2. HvH new game: place a piece, click Undo, piece is gone. ✓
3. Play a sequence ending in a "square" formation that triggers removal phase. Remove one piece; click Undo. The placement and the removal both unwind in one click. ✓
4. Hold Right-mouse: pointer locks, mouse-look works, WASD/QE move, Shift boosts, wheel changes fly speed. Release Right: orbit resumes naturally pointing at the new target. ✓
5. Drag with Left-mouse over a board cell, release — no piece placed. ✓
6. Open Camera Controls, swap Right and Left actions, reload, settings persist. ✓
7. Switch mode to Human vs AI — Undo button disappears. ✓

Expected: all 7 manual checks pass.

- [ ] **Step 4: Final no-op commit (optional)**

If anything was tweaked during E2E, commit. Otherwise skip.

- [ ] **Step 5: Push the branch (optional)**

```bash
git push -u origin feat/takeback-and-camera
```

(Skip if not pushing yet — the user will decide whether to open a PR.)

---

## Notes for the implementer

- **`structuredClone` availability**: required for the controls module's `bindings` deep-copy. Available in all supported Three.js targets (Chrome 98+, Safari 15.4+, Firefox 94+).
- **Pointer lock prompt**: the first time Right-drag is used in a new browser profile, the user sees a brief OS-level prompt. This is expected.
- **Existing tests reference `httpx`**: the `requirements.txt` already lists `httpx>=0.25.0`, so `uv run --with httpx` is redundant but harmless if the venv didn't pick it up. Drop `--with httpx` if it's already in the venv.
- **Click-vs-drag threshold**: 4 px is a forgiving value; can be tuned down if accidental drags-while-placing become an issue.
