# Pylos Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Pylos board game with AlphaZero AI training (forked from ImprovedTinyZero), checkpoint-based difficulty selection, and an interactive 3D Three.js visualization.

**Architecture:** Python backend (game engine + MCTS training + FastAPI WebSocket server) with a vanilla Three.js frontend. The game engine expands the action space to 303 actions (30 placements + 273 raises). Training saves periodic checkpoints that the web UI exposes as difficulty levels.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Numba, FastAPI, Three.js (ES modules via CDN)

---

## Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `engine/__init__.py`
- Create: `engine/tests/__init__.py`
- Create: `web/index.html` (placeholder)

**Step 1: Create requirements.txt**

```
torch>=2.1.0
numpy>=1.26.0
numba>=0.58.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0
pyyaml>=6.0
tqdm>=4.66.0
wandb>=0.16.0
pytest>=7.4.0
```

**Step 2: Create directory structure**

```bash
mkdir -p engine/tests engine/checkpoints web/src
touch engine/__init__.py engine/tests/__init__.py
```

**Step 3: Create minimal web/index.html placeholder**

```html
<!DOCTYPE html>
<html><head><title>Pylos</title></head><body><h1>Pylos</h1></body></html>
```

**Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 5: Commit**

```bash
git add requirements.txt engine/ web/
git commit -m "scaffold: project structure and dependencies"
```

---

## Task 2: Core Game Engine — Board Representation & Placement

**Files:**
- Create: `engine/game.py`
- Create: `engine/tests/test_game.py`

**Step 1: Write failing tests for board init and placement**

```python
# engine/tests/test_game.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from game import PylosGame

class TestBoardInit:
    def test_board_levels(self):
        g = PylosGame()
        assert len(g.board) == 4
        assert g.board[0].shape == (4, 4)
        assert g.board[1].shape == (3, 3)
        assert g.board[2].shape == (2, 2)
        assert g.board[3].shape == (1, 1)

    def test_initial_reserves(self):
        g = PylosGame()
        assert g.reserves[1] == 15
        assert g.reserves[-1] == 15

    def test_initial_turn(self):
        g = PylosGame()
        assert g.turn == 1

    def test_action_space_size(self):
        g = PylosGame()
        assert g.action_space == 303  # 30 placements + 273 raises

    def test_observation_shape(self):
        g = PylosGame()
        assert g.observation_shape == (32,)  # 30 positions + 2 reserves

class TestPlacement:
    def test_place_on_empty_level0(self):
        g = PylosGame()
        assert g.place(0, 0, 0) == True
        assert g.board[0][0, 0] == 1
        assert g.reserves[1] == 14

    def test_place_on_occupied(self):
        g = PylosGame()
        g.place(0, 0, 0)
        g.turn = -1
        assert g.place(0, 0, 0) == False

    def test_place_unsupported_level1(self):
        g = PylosGame()
        assert g.place(1, 0, 0) == False  # no support

    def test_place_supported_level1(self):
        g = PylosGame()
        # fill 2x2 on level 0 to support (1,0,0)
        for r, c in [(0,0), (0,1), (1,0), (1,1)]:
            g.board[0][r, c] = 1
            g.reserves[1] -= 1
        assert g.place(1, 0, 0) == True

    def test_place_no_reserves(self):
        g = PylosGame()
        g.reserves[1] = 0
        assert g.place(0, 0, 0) == False

class TestSupport:
    def test_level0_always_supported(self):
        g = PylosGame()
        assert g.is_supported(0, 0, 0) == True
        assert g.is_supported(0, 3, 3) == True

    def test_level1_needs_4_below(self):
        g = PylosGame()
        assert g.is_supported(1, 0, 0) == False
        g.board[0][0, 0] = 1
        g.board[0][0, 1] = 1
        g.board[0][1, 0] = 1
        assert g.is_supported(1, 0, 0) == False  # missing (1,1)
        g.board[0][1, 1] = -1
        assert g.is_supported(1, 0, 0) == True  # any color supports
```

**Step 2: Run tests — verify they fail**

```bash
cd /Users/atarkian2/Documents/GitHub/Pylos-Engine && python -m pytest engine/tests/test_game.py -v
```

Expected: FAIL — `ModuleNotFoundError` or `ImportError`

**Step 3: Implement PylosGame — init, reset, placement, support**

Fork from ImprovedTinyZero's `pylos/game.py` and extend:

```python
# engine/game.py
import numpy as np


class PylosGame:
    """Pylos board game with full action space (placements + raises)."""

    LEVEL_SIZES = [4, 3, 2, 1]

    def __init__(self):
        # Position index mapping: index -> (level, row, col)
        self.index_to_coords = []
        for lvl, size in enumerate(self.LEVEL_SIZES):
            for r in range(size):
                for c in range(size):
                    self.index_to_coords.append((lvl, r, c))

        # Reverse mapping: (level, row, col) -> index
        self.coords_to_index = {}
        for idx, coords in enumerate(self.index_to_coords):
            self.coords_to_index[coords] = idx

        # Pre-compute all raise pairs: (src_index, dst_index) where dst is higher
        self.raise_pairs = []
        for src_idx, (sl, sr, sc) in enumerate(self.index_to_coords):
            for dst_idx, (dl, dr, dc) in enumerate(self.index_to_coords):
                if dl > sl:
                    self.raise_pairs.append((src_idx, dst_idx))

        # Action space: 30 placements + 273 raises = 303
        self.num_placement_actions = len(self.index_to_coords)  # 30
        self.num_raise_actions = len(self.raise_pairs)  # 273
        self.action_space = self.num_placement_actions + self.num_raise_actions  # 303

        # Raise action index -> (src_idx, dst_idx)
        self.raise_action_to_pair = {
            self.num_placement_actions + i: pair
            for i, pair in enumerate(self.raise_pairs)
        }

        self.observation_shape = (self.num_placement_actions + 2,)  # 30 board + 2 reserves = 32

        self.reset()

    def reset(self):
        self.board = [np.zeros((s, s), dtype=int) for s in self.LEVEL_SIZES]
        self.turn = 1  # 1=White, -1=Black
        self.reserves = {1: 15, -1: 15}
        self.last_move = None
        self.actions_stack = []
        self._removal_pending = False

    # ---- Display ----
    def __str__(self):
        lines = []
        for lvl, layer in enumerate(self.board):
            lines.append(f"Level {lvl}:")
            for r in range(layer.shape[0]):
                row = []
                for c in range(layer.shape[1]):
                    v = layer[r, c]
                    row.append("W" if v == 1 else ("B" if v == -1 else "."))
                lines.append("  " + " ".join(row))
        lines.append(f"Reserves: W={self.reserves[1]} B={self.reserves[-1]}")
        return "\n".join(lines)

    # ---- Board queries ----
    def piece_has_top(self, level, r, c):
        """Check if a piece at (level, r, c) supports anything above it."""
        if level >= len(self.board) - 1:
            return False
        for dr in (0, -1):
            for dc in (0, -1):
                nr, nc = r + dr, c + dc
                upper = self.board[level + 1]
                if 0 <= nr < upper.shape[0] and 0 <= nc < upper.shape[1]:
                    if upper[nr, nc] != 0:
                        return True
        return False

    def is_supported(self, level, r, c):
        """Check if position (level, r, c) is supported from below."""
        if level == 0:
            return True
        b = self.board[level - 1]
        return (
            b[r, c] != 0
            and b[r + 1, c] != 0
            and b[r, c + 1] != 0
            and b[r + 1, c + 1] != 0
        )

    # ---- Moves ----
    def place(self, level, r, c):
        """Place a reserve sphere. Returns True on success."""
        if self.reserves[self.turn] <= 0:
            return False
        if self.board[level][r, c] != 0:
            return False
        if not self.is_supported(level, r, c):
            return False
        self.board[level][r, c] = self.turn
        self.reserves[self.turn] -= 1
        self.last_move = (level, r, c)
        return True

    def raise_piece(self, sl, sr, sc, dl, dr, dc):
        """Raise own piece from (sl,sr,sc) to (dl,dr,dc). Returns True on success."""
        if self.board[sl][sr, sc] != self.turn:
            return False
        if self.piece_has_top(sl, sr, sc):
            return False
        if dl <= sl:
            return False
        if self.board[dl][dr, dc] != 0:
            return False
        # Temporarily remove src to check support at dst
        self.board[sl][sr, sc] = 0
        supported = self.is_supported(dl, dr, dc)
        if not supported:
            self.board[sl][sr, sc] = self.turn  # restore
            return False
        self.board[dl][dr, dc] = self.turn
        self.last_move = (dl, dr, dc)
        return True

    # ---- Formation detection ----
    def check_square(self, level, r, c):
        """Check if placing at (level,r,c) completed a 2x2 square."""
        layer = self.board[level]
        player = layer[r, c]
        if player == 0:
            return False
        for dr in (0, -1):
            for dc in (0, -1):
                rr, cc = r + dr, c + dc
                if rr < 0 or cc < 0 or rr + 1 >= layer.shape[0] + 1 or cc + 1 >= layer.shape[1] + 1:
                    continue
                if rr + 1 > layer.shape[0] - 1 or cc + 1 > layer.shape[1] - 1:
                    continue
                sq = layer[rr:rr + 2, cc:cc + 2]
                if sq.shape == (2, 2) and np.all(sq == player):
                    return True
        return False

    def check_line(self, level, r, c):
        """Check if placing at (level,r,c) completed a line (advanced rules)."""
        player = self.board[level][r, c]
        if player == 0:
            return False
        if level == 0:  # 4-in-a-row
            if np.all(self.board[0][r, :] == player):
                return True
            if np.all(self.board[0][:, c] == player):
                return True
        elif level == 1:  # 3-in-a-row
            if np.all(self.board[1][r, :] == player):
                return True
            if np.all(self.board[1][:, c] == player):
                return True
        return False

    def check_for_removal(self):
        """Check if last move triggers removal phase."""
        if not self.last_move:
            return False
        lvl, r, c = self.last_move
        return self.check_square(lvl, r, c) or self.check_line(lvl, r, c)

    def get_removable_pieces(self):
        """Get list of (level, row, col) for current player's removable pieces."""
        removable = []
        for lvl, layer in enumerate(self.board):
            for r in range(layer.shape[0]):
                for c in range(layer.shape[1]):
                    if layer[r, c] == self.turn and not self.piece_has_top(lvl, r, c):
                        removable.append((lvl, r, c))
        return removable

    def remove(self, level, r, c):
        """Remove own unsupported piece. Returns True on success."""
        if self.board[level][r, c] != self.turn:
            return False
        if self.piece_has_top(level, r, c):
            return False
        self.board[level][r, c] = 0
        self.reserves[self.turn] += 1
        return True

    # ---- Win conditions ----
    def top_filled(self):
        return self.board[-1][0, 0] != 0

    def has_move(self):
        """Check if current player has any legal action."""
        return len(self.get_legal_actions()) > 0

    def get_result(self):
        """Return 1 (white wins), -1 (black wins), or None (game ongoing)."""
        if self.top_filled():
            return self.board[-1][0, 0]
        if not self.has_move():
            return -self.turn  # current player loses
        return None

    def get_first_person_result(self):
        result = self.get_result()
        if result is not None:
            return result * self.turn
        return None

    @staticmethod
    def swap_result(result):
        return -result

    # ---- AlphaZero interface ----
    def get_legal_actions(self):
        """Return list of legal action indices."""
        actions = []
        # Placement actions (0-29)
        if self.reserves[self.turn] > 0:
            for idx, (lvl, r, c) in enumerate(self.index_to_coords):
                if self.board[lvl][r, c] == 0 and self.is_supported(lvl, r, c):
                    actions.append(idx)
        # Raise actions (30-302)
        for action_idx, (src_idx, dst_idx) in self.raise_action_to_pair.items():
            sl, sr, sc = self.index_to_coords[src_idx]
            dl, dr, dc = self.index_to_coords[dst_idx]
            if (self.board[sl][sr, sc] == self.turn
                    and not self.piece_has_top(sl, sr, sc)
                    and self.board[dl][dr, dc] == 0
                    and self.is_supported(dl, dr, dc)):
                # Temporarily remove src to verify dst support still holds
                self.board[sl][sr, sc] = 0
                still_supported = self.is_supported(dl, dr, dc)
                self.board[sl][sr, sc] = self.turn
                if still_supported:
                    actions.append(action_idx)
        return actions

    def step(self, action):
        """Execute an action (placement or raise), switch turn."""
        if action < self.num_placement_actions:
            lvl, r, c = self.index_to_coords[action]
            if not self.place(lvl, r, c):
                raise ValueError(f"Illegal placement action {action}")
        else:
            src_idx, dst_idx = self.raise_action_to_pair[action]
            sl, sr, sc = self.index_to_coords[src_idx]
            dl, dr, dc = self.index_to_coords[dst_idx]
            if not self.raise_piece(sl, sr, sc, dl, dr, dc):
                raise ValueError(f"Illegal raise action {action}")
        # Handle removal for AI via greedy heuristic
        if self.check_for_removal():
            self._do_ai_removal()
        self.actions_stack.append(action)
        self.turn *= -1

    def _do_ai_removal(self):
        """Greedy removal: remove up to 2 pieces from lowest level first."""
        for _ in range(2):
            removable = self.get_removable_pieces()
            if not removable:
                break
            # Prefer removing from lowest level (cheapest to replace)
            removable.sort(key=lambda x: x[0])
            lvl, r, c = removable[0]
            self.remove(lvl, r, c)

    def undo_last_action(self):
        """Undo last action (for MCTS backpropagation). Note: does not undo removals."""
        self.turn *= -1
        action = self.actions_stack.pop()
        if action < self.num_placement_actions:
            lvl, r, c = self.index_to_coords[action]
            self.board[lvl][r, c] = 0
            self.reserves[self.turn] += 1
        else:
            src_idx, dst_idx = self.raise_action_to_pair[action]
            sl, sr, sc = self.index_to_coords[src_idx]
            dl, dr, dc = self.index_to_coords[dst_idx]
            self.board[dl][dr, dc] = 0
            self.board[sl][sr, sc] = self.turn

    def to_observation(self):
        """Encode board state as flat vector from current player's perspective."""
        obs = []
        for lvl, size in enumerate(self.LEVEL_SIZES):
            layer = self.board[lvl]
            for r in range(size):
                for c in range(size):
                    cell = layer[r, c]
                    if cell == self.turn:
                        obs.append(1.0)
                    elif cell == -self.turn:
                        obs.append(-1.0)
                    else:
                        obs.append(0.0)
        # Append normalized reserves
        obs.append(self.reserves[self.turn] / 15.0)
        obs.append(self.reserves[-self.turn] / 15.0)
        return np.array(obs, dtype=np.float32)
```

**Step 4: Run tests — verify they pass**

```bash
cd /Users/atarkian2/Documents/GitHub/Pylos-Engine && python -m pytest engine/tests/test_game.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
git add engine/game.py engine/tests/test_game.py
git commit -m "feat: core game engine with placement, support, and expanded action space"
```

---

## Task 3: Game Engine — Raise, Removal, Formations & Win Conditions

**Files:**
- Modify: `engine/tests/test_game.py` (add tests)
- Modify: `engine/game.py` (already implemented above, these tests validate it)

**Step 1: Write failing tests for raise, removal, formations**

Add to `engine/tests/test_game.py`:

```python
class TestRaise:
    def test_raise_own_piece(self):
        g = PylosGame()
        # Place 4 pieces to support level 1
        for r, c in [(0,0), (0,1), (1,0), (1,1)]:
            g.board[0][r, c] = 1
            g.reserves[1] -= 1
        # Place piece at (0,2,0) to raise
        g.board[0][2, 0] = 1
        g.reserves[1] -= 1
        assert g.raise_piece(0, 2, 0, 1, 0, 0) == True
        assert g.board[0][2, 0] == 0
        assert g.board[1][0, 0] == 1

    def test_raise_opponent_piece_fails(self):
        g = PylosGame()
        for r, c in [(0,0), (0,1), (1,0), (1,1)]:
            g.board[0][r, c] = 1
            g.reserves[1] -= 1
        g.board[0][2, 0] = -1  # opponent piece
        g.reserves[-1] -= 1
        assert g.raise_piece(0, 2, 0, 1, 0, 0) == False

    def test_raise_with_top_fails(self):
        g = PylosGame()
        for r, c in [(0,0), (0,1), (1,0), (1,1)]:
            g.board[0][r, c] = 1
            g.reserves[1] -= 1
        g.board[1][0, 0] = -1  # something on top of (0,0,0)
        assert g.raise_piece(0, 0, 0, 1, 0, 0) == False  # dst occupied too

    def test_raise_to_lower_fails(self):
        g = PylosGame()
        g.board[1][0, 0] = 1
        assert g.raise_piece(1, 0, 0, 0, 0, 0) == False

class TestFormations:
    def test_square_detection(self):
        g = PylosGame()
        g.board[0][0, 0] = 1
        g.board[0][0, 1] = 1
        g.board[0][1, 0] = 1
        g.board[0][1, 1] = 1
        g.last_move = (0, 1, 1)
        assert g.check_square(0, 1, 1) == True
        assert g.check_for_removal() == True

    def test_line_detection_level0(self):
        g = PylosGame()
        for c in range(4):
            g.board[0][0, c] = 1
        g.last_move = (0, 0, 3)
        assert g.check_line(0, 0, 3) == True
        assert g.check_for_removal() == True

    def test_line_detection_level1(self):
        g = PylosGame()
        for c in range(3):
            g.board[1][0, c] = 1
        g.last_move = (1, 0, 2)
        assert g.check_line(1, 0, 2) == True

    def test_no_formation(self):
        g = PylosGame()
        g.board[0][0, 0] = 1
        g.board[0][0, 1] = 1
        g.board[0][1, 0] = -1  # opponent breaks square
        g.board[0][1, 1] = 1
        g.last_move = (0, 1, 1)
        assert g.check_square(0, 1, 1) == False

class TestRemoval:
    def test_remove_own_piece(self):
        g = PylosGame()
        g.board[0][3, 3] = 1
        g.reserves[1] -= 1
        assert g.remove(0, 3, 3) == True
        assert g.board[0][3, 3] == 0
        assert g.reserves[1] == 15

    def test_remove_opponent_fails(self):
        g = PylosGame()
        g.board[0][3, 3] = -1
        assert g.remove(0, 3, 3) == False

    def test_remove_supporting_fails(self):
        g = PylosGame()
        for r, c in [(0,0), (0,1), (1,0), (1,1)]:
            g.board[0][r, c] = 1
        g.board[1][0, 0] = 1  # supported by the 4 below
        assert g.remove(0, 0, 0) == False  # supports level 1

class TestWinConditions:
    def test_apex_win(self):
        g = PylosGame()
        g.board[3][0, 0] = 1
        assert g.top_filled() == True
        assert g.get_result() == 1

    def test_no_moves_loses(self):
        g = PylosGame()
        g.reserves[1] = 0
        # Fill entire board with opponent pieces so no raises possible
        for lvl, layer in enumerate(g.board):
            for r in range(layer.shape[0]):
                for c in range(layer.shape[1]):
                    layer[r, c] = -1
        g.board[3][0, 0] = 0  # leave apex empty but no reserves and all pieces are opponent's
        assert g.get_result() == -1  # white (turn=1) has no moves, loses

class TestAlphaZeroInterface:
    def test_legal_actions_initial(self):
        g = PylosGame()
        actions = g.get_legal_actions()
        # Initially only level 0 placements are legal (16 positions)
        assert len(actions) == 16
        assert all(a < 30 for a in actions)

    def test_step_and_undo(self):
        g = PylosGame()
        actions = g.get_legal_actions()
        action = actions[0]
        g.step(action)
        assert g.turn == -1  # switched
        g.undo_last_action()
        assert g.turn == 1  # restored

    def test_observation_encoding(self):
        g = PylosGame()
        obs = g.to_observation()
        assert obs.shape == (32,)
        assert obs[-1] == 1.0  # opponent reserves = 15/15
        assert obs[-2] == 1.0  # own reserves = 15/15

    def test_step_raise_action(self):
        g = PylosGame()
        # Set up board for a raise
        for r, c in [(0,0), (0,1), (1,0), (1,1)]:
            g.board[0][r, c] = 1
            g.reserves[1] -= 1
        g.board[0][3, 3] = 1  # piece to raise
        g.reserves[1] -= 1
        actions = g.get_legal_actions()
        raise_actions = [a for a in actions if a >= 30]
        assert len(raise_actions) > 0
        # Find the raise from (0,3,3) to (1,0,0)
        target_src = g.coords_to_index[(0, 3, 3)]
        target_dst = g.coords_to_index[(1, 0, 0)]
        for a in raise_actions:
            src, dst = g.raise_action_to_pair[a]
            if src == target_src and dst == target_dst:
                g.step(a)
                assert g.board[0][3, 3] == 0
                assert g.board[1][0, 0] == 1
                return
        assert False, "Expected raise action not found"
```

**Step 2: Run tests**

```bash
cd /Users/atarkian2/Documents/GitHub/Pylos-Engine && python -m pytest engine/tests/test_game.py -v
```

Expected: ALL PASS (game.py already has all this logic from Step 3 of Task 2)

**Step 3: Fix any failures, iterate**

**Step 4: Commit**

```bash
git add engine/tests/test_game.py engine/game.py
git commit -m "test: comprehensive game engine tests for raises, formations, removal, win conditions"
```

---

## Task 4: Fork ImprovedTinyZero Core — MCTS, Agents, Models, Replay Buffer

**Files:**
- Create: `engine/mcts.py` (forked from ImprovedTinyZero)
- Create: `engine/agents.py` (forked)
- Create: `engine/models.py` (modified for 303 action space)
- Create: `engine/replay_buffer.py` (forked)

**Step 1: Copy and adapt mcts.py**

Fork directly from ImprovedTinyZero. No changes needed — it's action-space agnostic.

```python
# engine/mcts.py
# Forked from https://github.com/MikilFoss/ImprovedTinyZero
import math
import numpy as np
from numba import njit


class RootNode:
    def __init__(self):
        self.parent = None
        self.visits = 0
        self.children = None


class Node(RootNode):
    def __init__(self, idx, parent):
        self.idx = idx
        self.parent = parent
        self.children = None

    @property
    def visits(self):
        return self.parent.children_visits[self.idx]

    @visits.setter
    def visits(self, x):
        self.parent.children_visits[self.idx] = x

    @property
    def action(self):
        return self.parent.children_actions[self.idx]

    @property
    def value(self):
        return self.parent.children_values[self.idx]

    @value.setter
    def value(self, x):
        self.parent.children_values[self.idx] = x


@njit(fastmath=True, parallel=True)
def get_ucb_scores_jitted(children_values, children_priors, visits, children_visits, c_puct):
    return children_values + c_puct * children_priors * math.sqrt(visits) / (children_visits + 1)


def get_ucb_scores(node, c_puct):
    return get_ucb_scores_jitted(
        node.children_values, node.children_priors,
        node.visits, node.children_visits, c_puct
    )


def select(root, game, c_puct):
    current = root
    while current.children:
        ucb_scores = get_ucb_scores(current, c_puct)
        ucb_scores[current.children_visits == 0] = np.inf
        current = current.children[np.argmax(ucb_scores)]
        game.step(current.action)
    return current


def expand(leaf, children_actions, children_priors):
    leaf.children = [Node(idx, leaf) for idx, _ in enumerate(children_actions)]
    leaf.children_actions = children_actions
    leaf.children_priors = children_priors
    leaf.children_values = np.zeros_like(leaf.children_priors)
    leaf.children_visits = np.zeros_like(leaf.children_priors)


def backpropagate(leaf, game, result):
    current = leaf
    while current.parent:
        result = game.swap_result(result)
        current.value = (current.value * current.visits + result) / (current.visits + 1)
        current.visits += 1
        current = current.parent
        game.undo_last_action()
    current.visits += 1


def search(game, value_fn, policy_fn, iterations, c_puct=1.0, dirichlet_alpha=None):
    root = RootNode()
    children_actions = np.array(game.get_legal_actions())
    if len(children_actions) == 0:
        return root
    children_priors = policy_fn(game)[children_actions]
    if dirichlet_alpha:
        children_priors = 0.75 * children_priors + 0.25 * np.random.default_rng().dirichlet(
            dirichlet_alpha * np.ones_like(children_priors)
        )
    expand(root, children_actions, children_priors)

    for _ in range(iterations):
        leaf = select(root, game, c_puct)
        result = game.get_first_person_result()
        if result is None:
            children_actions = np.array(game.get_legal_actions())
            if len(children_actions) == 0:
                result = game.get_first_person_result()
                if result is None:
                    result = 0.0
            else:
                children_priors = policy_fn(game)[children_actions]
                expand(leaf, children_actions, children_priors)
                result = value_fn(game)
        backpropagate(leaf, game, result)
    return root


def play(game, agent, search_iterations, c_puct=1.0, dirichlet_alpha=None):
    root = search(
        game, agent.value_fn, agent.policy_fn, search_iterations,
        c_puct=c_puct, dirichlet_alpha=dirichlet_alpha
    )
    if root.children is None:
        return None
    return root.children_actions[np.argmax(root.children_visits)]


def pit(game, agent1, agent2, agent1_play_kwargs, agent2_play_kwargs):
    current_agent, other_agent = agent1, agent2
    current_kwargs, other_kwargs = agent1_play_kwargs, agent2_play_kwargs
    while (result := game.get_result()) is None:
        action = play(game, current_agent, **current_kwargs)
        if action is None:
            break
        game.step(action)
        current_agent, other_agent = other_agent, current_agent
        current_kwargs, other_kwargs = other_kwargs, current_kwargs
    return game.get_result()
```

**Step 2: Copy replay_buffer.py (unchanged)**

```python
# engine/replay_buffer.py
# Forked from https://github.com/MikilFoss/ImprovedTinyZero
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size):
        self.observations = deque(maxlen=max_size)
        self.actions_dist = deque(maxlen=max_size)
        self.results = deque(maxlen=max_size)

    def __len__(self):
        return len(self.observations)

    def add_sample(self, observation, actions_dist, result):
        self.observations.append(observation)
        self.actions_dist.append(actions_dist)
        self.results.append(result)

    def sample(self, batch_size):
        indices = np.random.choice(len(self), batch_size, replace=False)
        observations = np.array([self.observations[i] for i in indices], dtype=np.float32)
        actions_dist = np.array([self.actions_dist[i] for i in indices], dtype=np.float32)
        results = np.array([self.results[i] for i in indices], dtype=np.float32)
        return observations, actions_dist, results
```

**Step 3: Copy agents.py (unchanged)**

```python
# engine/agents.py
# Forked from https://github.com/MikilFoss/ImprovedTinyZero
import torch
import torch.nn.functional as F
import numpy as np
from engine.replay_buffer import ReplayBuffer
import copy
from engine.mcts import search


class ClassicMCTSAgent:
    @staticmethod
    def value_fn(game):
        game = copy.deepcopy(game)
        while (first_person_result := game.get_first_person_result()) is None:
            actions = game.get_legal_actions()
            if not actions:
                break
            game.step(np.random.choice(actions))
        return first_person_result if first_person_result is not None else 0.0

    @staticmethod
    def policy_fn(game):
        return np.ones(game.action_space) / game.action_space


class AlphaZeroAgent:
    def __init__(self, model):
        self.model = model

    def value_fn(self, game):
        observation = torch.tensor(game.to_observation(), device=self.model.device, requires_grad=False)
        value = self.model.value_forward(observation)
        return value.item()

    def policy_fn(self, game):
        observation = torch.tensor(game.to_observation(), device=self.model.device, requires_grad=False)
        policy = self.model.policy_forward(observation)
        return policy.cpu().numpy()


class AlphaZeroAgentTrainer(AlphaZeroAgent):
    def __init__(self, model, optimizer, replay_buffer_max_size):
        super().__init__(model)
        self.optimizer = optimizer
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_max_size)

    def _selfplay(self, game, search_iterations, c_puct=1.0, dirichlet_alpha=None):
        buffer = []
        while (first_person_result := game.get_first_person_result()) is None:
            root_node = search(
                game, self.value_fn, self.policy_fn, search_iterations,
                c_puct=c_puct, dirichlet_alpha=dirichlet_alpha
            )
            if root_node.children is None:
                break
            visits_dist = root_node.children_visits / root_node.children_visits.sum()
            action = root_node.children_actions[np.random.choice(len(root_node.children), p=visits_dist)]
            actions_dist = np.zeros(game.action_space, dtype=np.float32)
            actions_dist[root_node.children_actions] = visits_dist
            buffer.append((game.to_observation(), actions_dist))
            game.step(action)

        first_person_result = game.get_first_person_result()
        if first_person_result is None:
            first_person_result = 0.0
        return first_person_result, buffer

    def train_step(self, game, search_iterations, batch_size, epochs, c_puct=1.0, dirichlet_alpha=None):
        first_person_result, game_buffer = self._selfplay(
            game, search_iterations, c_puct=c_puct, dirichlet_alpha=dirichlet_alpha
        )

        result = game.swap_result(first_person_result)
        while len(game_buffer) > 0:
            observation, action_dist = game_buffer.pop()
            self.replay_buffer.add_sample(observation, action_dist, result)
            result = game.swap_result(result)

        values_losses, policies_losses = [], []
        if len(self.replay_buffer) >= batch_size:
            for _ in range(epochs):
                observations, actions_dist, results = self.replay_buffer.sample(batch_size)
                observations = torch.tensor(observations, device=self.model.device)
                actions_dist = torch.tensor(actions_dist, device=self.model.device)
                results = torch.tensor(results, device=self.model.device)

                self.optimizer.zero_grad()
                values, log_policies = self.model(observations)

                values_loss = F.mse_loss(values.squeeze(1), results)
                policies_loss = F.kl_div(log_policies, actions_dist, reduction="batchmean")

                (values_loss + policies_loss).backward()
                self.optimizer.step()

                values_losses.append(values_loss.item())
                policies_losses.append(policies_loss.item())

        return values_losses, policies_losses

    def save_training_state(self, model_out_path, optimizer_out_path):
        torch.save(self.model.state_dict(), model_out_path)
        torch.save(self.optimizer.state_dict(), optimizer_out_path)

    def load_training_state(self, model_out_path, optimizer_out_path):
        self.model.load_state_dict(torch.load(model_out_path, weights_only=True))
        self.optimizer.load_state_dict(torch.load(optimizer_out_path, weights_only=True))
```

**Step 4: Create models.py — adapted for 303 action space + 32 input**

```python
# engine/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PylosNetwork(nn.Module):
    """Neural network for Pylos with value and policy heads."""

    def __init__(self, input_shape, action_space, hidden1=512, hidden2=256):
        super().__init__()
        self.fc1 = nn.Linear(input_shape[0], hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.value_head = nn.Linear(hidden2, 1)
        self.policy_head = nn.Linear(hidden2, action_space)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def __call__(self, observations):
        self.train()
        x = F.relu(self.fc1(observations))
        x = F.relu(self.fc2(x))
        value = torch.tanh(self.value_head(x))
        log_policy = F.log_softmax(self.policy_head(x), dim=-1)
        return value, log_policy

    def value_forward(self, observation):
        self.eval()
        with torch.no_grad():
            x = F.relu(self.fc1(observation))
            x = F.relu(self.fc2(x))
            return torch.tanh(self.value_head(x))

    def policy_forward(self, observation):
        self.eval()
        with torch.no_grad():
            x = F.relu(self.fc1(observation))
            x = F.relu(self.fc2(x))
            return F.softmax(self.policy_head(x), dim=-1)
```

**Step 5: Write smoke test**

```python
# engine/tests/test_training_smoke.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgentTrainer


def test_training_smoke():
    """Run 2 self-play games to verify the full training loop works."""
    game = PylosGame()
    model = PylosNetwork(game.observation_shape, game.action_space)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    agent = AlphaZeroAgentTrainer(model, optimizer, replay_buffer_max_size=64)

    for _ in range(2):
        game.reset()
        vl, pl = agent.train_step(game, search_iterations=8, batch_size=16, epochs=1, c_puct=1.5)
    # If we get here without error, the smoke test passes
    assert True
```

**Step 6: Run smoke test**

```bash
cd /Users/atarkian2/Documents/GitHub/Pylos-Engine && python -m pytest engine/tests/test_training_smoke.py -v --timeout=120
```

Expected: PASS (may take 30-60s)

**Step 7: Commit**

```bash
git add engine/mcts.py engine/agents.py engine/models.py engine/replay_buffer.py engine/tests/test_training_smoke.py
git commit -m "feat: fork ImprovedTinyZero core (MCTS, agents, models, replay buffer) for 303-action Pylos"
```

---

## Task 5: Training Script with Checkpoint System

**Files:**
- Create: `engine/config.yaml`
- Create: `engine/train.py`
- Create: `engine/evaluate.py`

**Step 1: Create config.yaml**

```yaml
# engine/config.yaml
training:
  selfplay_games: 2000
  search_iterations: 64
  batch_size: 128
  replay_buffer_size: 512
  epochs_per_game: 3
  learning_rate: 0.001
  weight_decay: 0.0001
  c_puct: 1.5
  dirichlet_alpha: 0.3

checkpoints:
  save_every: 100        # Save every N self-play games
  eval_games: 50         # Games vs random agent for evaluation
  dir: engine/checkpoints

wandb:
  enabled: false
  project: pylos-alphazero
```

**Step 2: Create evaluate.py**

```python
# engine/evaluate.py
"""Evaluate a checkpoint against a random agent."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import torch
from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgent
from mcts import pit, play


class RandomAgent:
    @staticmethod
    def value_fn(game):
        return 0.0

    @staticmethod
    def policy_fn(game):
        return np.ones(game.action_space) / game.action_space


def evaluate_checkpoint(model_path, num_games=50, search_iterations=32):
    """Return win rate of checkpoint vs random agent."""
    game = PylosGame()
    model = PylosNetwork(game.observation_shape, game.action_space)
    model.load_state_dict(torch.load(model_path, map_location=model.device, weights_only=True))
    agent = AlphaZeroAgent(model)
    random_agent = RandomAgent()

    wins = 0
    for i in range(num_games):
        game.reset()
        # Alternate who goes first
        if i % 2 == 0:
            result = pit(
                game, agent, random_agent,
                {"search_iterations": search_iterations, "c_puct": 1.5},
                {"search_iterations": 1, "c_puct": 1.0},
            )
            if result == 1:
                wins += 1
        else:
            result = pit(
                game, random_agent, agent,
                {"search_iterations": 1, "c_puct": 1.0},
                {"search_iterations": search_iterations, "c_puct": 1.5},
            )
            if result == -1:
                wins += 1

    return wins / num_games


def assign_label(win_rate):
    if win_rate < 0.4:
        return "Beginner"
    elif win_rate < 0.6:
        return "Novice"
    elif win_rate < 0.75:
        return "Intermediate"
    elif win_rate < 0.9:
        return "Advanced"
    else:
        return "Expert"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--games", type=int, default=50)
    args = parser.parse_args()
    wr = evaluate_checkpoint(args.model_path, args.games)
    print(f"Win rate: {wr:.2%} — {assign_label(wr)}")
```

**Step 3: Create train.py with checkpoint saving**

```python
# engine/train.py
"""AlphaZero training for Pylos with periodic checkpoint saving."""
import json
import os
import sys
from datetime import datetime, timezone

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgentTrainer
from evaluate import evaluate_checkpoint, assign_label


def load_config(path="engine/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(agent, step, config, manifest_path):
    ckpt_dir = config["checkpoints"]["dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    model_file = f"checkpoint_{step:05d}.pth"
    model_path = os.path.join(ckpt_dir, model_file)
    opt_path = os.path.join(ckpt_dir, f"optimizer_{step:05d}.pth")
    agent.save_training_state(model_path, opt_path)

    # Evaluate
    eval_games = config["checkpoints"].get("eval_games", 50)
    print(f"\nEvaluating checkpoint at step {step}...")
    win_rate = evaluate_checkpoint(model_path, num_games=eval_games, search_iterations=16)
    label = assign_label(win_rate)
    print(f"  Win rate vs random: {win_rate:.2%} — {label}")

    # Update manifest
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"checkpoints": []}

    manifest["checkpoints"].append({
        "file": model_file,
        "step": step,
        "win_rate_vs_random": round(win_rate, 4),
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    config = load_config()
    tc = config["training"]
    cc = config["checkpoints"]

    game = PylosGame()
    model = PylosNetwork(game.observation_shape, game.action_space)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    agent = AlphaZeroAgentTrainer(model, optimizer, tc["replay_buffer_size"])

    ckpt_dir = cc["dir"]
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    os.makedirs(ckpt_dir, exist_ok=True)

    use_wandb = config.get("wandb", {}).get("enabled", False)
    if use_wandb:
        import wandb
        wandb.init(project=config["wandb"]["project"], name=f"run-{datetime.now():%Y%m%d-%H%M%S}")

    print(f"Starting training: {tc['selfplay_games']} self-play games")
    print(f"Action space: {game.action_space} (30 placements + 273 raises)")

    for i in tqdm(range(1, tc["selfplay_games"] + 1)):
        game.reset()
        vl, pl = agent.train_step(
            game, tc["search_iterations"], tc["batch_size"],
            tc["epochs_per_game"], c_puct=tc["c_puct"],
            dirichlet_alpha=tc["dirichlet_alpha"],
        )

        if use_wandb:
            for v, p in zip(vl, pl):
                import wandb
                wandb.log({"value_loss": v, "policy_loss": p, "game": i})

        if i % cc["save_every"] == 0:
            save_checkpoint(agent, i, config, manifest_path)

    # Final checkpoint
    save_checkpoint(agent, tc["selfplay_games"], config, manifest_path)
    print("Training complete!")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
```

**Step 4: Quick sanity test (5 games, save at game 5)**

Create a test config and run:

```bash
cd /Users/atarkian2/Documents/GitHub/Pylos-Engine && python -c "
import yaml
cfg = {
    'training': {'selfplay_games': 5, 'search_iterations': 8, 'batch_size': 16,
                 'replay_buffer_size': 64, 'epochs_per_game': 1, 'learning_rate': 0.001,
                 'weight_decay': 0.0001, 'c_puct': 1.5, 'dirichlet_alpha': 0.3},
    'checkpoints': {'save_every': 5, 'eval_games': 4, 'dir': 'engine/checkpoints'},
    'wandb': {'enabled': False}
}
with open('engine/config_test.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
```

Then run:

```bash
cd /Users/atarkian2/Documents/GitHub/Pylos-Engine && python engine/train.py
```

(Override config path in train.py to use config_test.yaml, or modify load_config to accept CLI arg)

Expected: Runs 5 games, saves checkpoint, prints eval win rate.

**Step 5: Commit**

```bash
git add engine/config.yaml engine/train.py engine/evaluate.py
git commit -m "feat: training script with checkpoint saving and evaluation"
```

---

## Task 6: FastAPI WebSocket Server

**Files:**
- Create: `engine/server.py`
- Create: `engine/tests/test_server.py`

**Step 1: Write server test**

```python
# engine/tests/test_server.py
import pytest
import json
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from server import app


def test_checkpoints_endpoint():
    client = TestClient(app)
    resp = client.get("/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert "checkpoints" in data


def test_websocket_new_game_human_vs_human():
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        data = ws.receive_json()
        assert data["type"] == "state"
        assert data["turn"] == "white"
        assert data["reserves"]["white"] == 15

        # Make a placement
        ws.send_json({"type": "move", "action": {"type": "place", "level": 0, "row": 0, "col": 0}})
        data = ws.receive_json()
        assert data["type"] == "state"
        assert data["turn"] == "black"
        assert data["reserves"]["white"] == 14
```

**Step 2: Run test — verify it fails**

```bash
cd /Users/atarkian2/Documents/GitHub/Pylos-Engine && python -m pytest engine/tests/test_server.py -v
```

Expected: FAIL — `server` module not found

**Step 3: Implement server.py**

```python
# engine/server.py
"""FastAPI WebSocket server for Pylos game sessions."""
import asyncio
import json
import os
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgent
from mcts import play as mcts_play

import torch

app = FastAPI(title="Pylos Engine")

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")


@app.get("/checkpoints")
def get_checkpoints():
    manifest_path = os.path.join(CHECKPOINTS_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {"checkpoints": []}


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


# Serve static web files
if os.path.isdir(WEB_DIR):
    app.mount("/src", StaticFiles(directory=os.path.join(WEB_DIR, "src")), name="src")


def load_ai_agent(checkpoint_file):
    game = PylosGame()
    model = PylosNetwork(game.observation_shape, game.action_space)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, checkpoint_file)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=model.device, weights_only=True))
    return AlphaZeroAgent(model)


def board_to_json(game):
    """Serialize board state for the client."""
    board = []
    for lvl, layer in enumerate(game.board):
        level_data = []
        for r in range(layer.shape[0]):
            for c in range(layer.shape[1]):
                v = layer[r, c]
                if v != 0:
                    level_data.append({
                        "row": r, "col": c,
                        "player": "white" if v == 1 else "black"
                    })
        board.append(level_data)
    return board


def legal_moves_to_json(game):
    """Serialize legal moves for the client."""
    moves = []
    for action in game.get_legal_actions():
        if action < game.num_placement_actions:
            lvl, r, c = game.index_to_coords[action]
            moves.append({"type": "place", "level": lvl, "row": r, "col": c, "action": action})
        else:
            src_idx, dst_idx = game.raise_action_to_pair[action]
            sl, sr, sc = game.index_to_coords[src_idx]
            dl, dr, dc = game.index_to_coords[dst_idx]
            moves.append({
                "type": "raise",
                "src": {"level": sl, "row": sr, "col": sc},
                "dst": {"level": dl, "row": dr, "col": dc},
                "action": action,
            })
    return moves


def make_state_msg(game):
    return {
        "type": "state",
        "board": board_to_json(game),
        "turn": "white" if game.turn == 1 else "black",
        "reserves": {"white": game.reserves[1], "black": game.reserves[-1]},
        "legal_moves": legal_moves_to_json(game),
    }


@app.websocket("/game")
async def game_ws(ws: WebSocket):
    await ws.accept()
    game = None
    agent = None
    ai_color = None
    mode = None
    search_iters = 32

    try:
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            if msg_type == "new_game":
                game = PylosGame()
                mode = msg.get("mode", "human_vs_human")
                checkpoint = msg.get("checkpoint")
                human_color = msg.get("human_color", "white")
                search_iters = msg.get("search_iterations", 32)

                if mode == "human_vs_ai" and checkpoint:
                    agent = load_ai_agent(checkpoint)
                    ai_color = -1 if human_color == "white" else 1
                elif mode == "ai_vs_ai" and checkpoint:
                    agent = load_ai_agent(checkpoint)

                await ws.send_json(make_state_msg(game))

                # If AI goes first in human_vs_ai
                if mode == "human_vs_ai" and ai_color == game.turn:
                    await _do_ai_turn(ws, game, agent, search_iters)

                # If AI vs AI, start the loop
                if mode == "ai_vs_ai":
                    await _ai_vs_ai_loop(ws, game, agent, search_iters, msg.get("delay_ms", 1500))

            elif msg_type == "move" and game and mode in ("human_vs_human", "human_vs_ai"):
                action_data = msg["action"]
                try:
                    if action_data["type"] == "place":
                        lvl, r, c = action_data["level"], action_data["row"], action_data["col"]
                        idx = game.coords_to_index.get((lvl, r, c))
                        if idx is not None and idx in game.get_legal_actions():
                            game.step(idx)
                        else:
                            await ws.send_json({"type": "error", "message": "Illegal move"})
                            continue
                    elif action_data["type"] == "raise":
                        src = tuple(action_data["src"])
                        dst = tuple(action_data["dst"])
                        src_idx = game.coords_to_index.get(src)
                        dst_idx = game.coords_to_index.get(dst)
                        # Find matching raise action
                        found = False
                        for a_idx, (si, di) in game.raise_action_to_pair.items():
                            if si == src_idx and di == dst_idx and a_idx in game.get_legal_actions():
                                game.step(a_idx)
                                found = True
                                break
                        if not found:
                            await ws.send_json({"type": "error", "message": "Illegal raise"})
                            continue
                except (KeyError, ValueError) as e:
                    await ws.send_json({"type": "error", "message": str(e)})
                    continue

                # Check game over
                result = game.get_result()
                if result is not None:
                    await ws.send_json(make_state_msg(game))
                    winner = "white" if result == 1 else "black"
                    await ws.send_json({"type": "game_over", "winner": winner})
                    continue

                await ws.send_json(make_state_msg(game))

                # AI's turn
                if mode == "human_vs_ai" and game.turn == ai_color:
                    await _do_ai_turn(ws, game, agent, search_iters)

    except WebSocketDisconnect:
        pass


async def _do_ai_turn(ws, game, agent, search_iters):
    """Execute AI move and send result."""
    import time
    t0 = time.time()
    action = mcts_play(game, agent, search_iters, c_puct=1.5)
    elapsed_ms = int((time.time() - t0) * 1000)

    if action is None:
        result = game.get_result()
        winner = "white" if result == 1 else ("black" if result == -1 else "draw")
        await ws.send_json({"type": "game_over", "winner": winner})
        return

    # Build action description
    if action < game.num_placement_actions:
        lvl, r, c = game.index_to_coords[action]
        action_desc = {"type": "place", "level": lvl, "row": r, "col": c}
    else:
        src_idx, dst_idx = game.raise_action_to_pair[action]
        sl, sr, sc = game.index_to_coords[src_idx]
        dl, dr, dc = game.index_to_coords[dst_idx]
        action_desc = {
            "type": "raise",
            "src": {"level": sl, "row": sr, "col": sc},
            "dst": {"level": dl, "row": dr, "col": dc},
        }

    game.step(action)

    await ws.send_json({"type": "ai_move", "action": action_desc, "thinking_time_ms": elapsed_ms})
    await ws.send_json(make_state_msg(game))

    result = game.get_result()
    if result is not None:
        winner = "white" if result == 1 else "black"
        await ws.send_json({"type": "game_over", "winner": winner})


async def _ai_vs_ai_loop(ws, game, agent, search_iters, delay_ms):
    """Run AI vs AI game with delays for spectating."""
    while game.get_result() is None:
        await asyncio.sleep(delay_ms / 1000)
        await _do_ai_turn(ws, game, agent, search_iters)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Step 4: Run tests**

```bash
cd /Users/atarkian2/Documents/GitHub/Pylos-Engine && python -m pytest engine/tests/test_server.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add engine/server.py engine/tests/test_server.py
git commit -m "feat: FastAPI WebSocket server with human vs AI/AI vs AI/human vs human modes"
```

---

## Task 7: Three.js 3D Visualization — Scene & Board

**Files:**
- Create: `web/index.html`
- Create: `web/src/scene.js`
- Create: `web/src/game.js`

**Step 1: Create index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pylos</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { overflow: hidden; background: #1a1a2e; font-family: 'Segoe UI', system-ui, sans-serif; color: #e0e0e0; }
    #canvas-container { width: 100vw; height: 100vh; }
    canvas { display: block; }

    /* UI Overlay */
    #ui-overlay {
      position: absolute; top: 0; left: 0; width: 100%; height: 100%;
      pointer-events: none;
    }
    #ui-overlay > * { pointer-events: auto; }

    /* Top bar */
    #top-bar {
      position: absolute; top: 20px; left: 50%; transform: translateX(-50%);
      display: flex; gap: 16px; align-items: center;
      background: rgba(0,0,0,0.7); backdrop-filter: blur(10px);
      border-radius: 12px; padding: 12px 24px;
    }
    #turn-indicator { font-size: 18px; font-weight: 600; }
    .reserves { display: flex; align-items: center; gap: 6px; font-size: 14px; }
    .reserves .dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block; }
    .dot.white { background: #f0f0f0; border: 1px solid #999; }
    .dot.black { background: #2a2a2a; border: 1px solid #555; }

    /* Side panel */
    #side-panel {
      position: absolute; top: 20px; right: 20px;
      background: rgba(0,0,0,0.7); backdrop-filter: blur(10px);
      border-radius: 12px; padding: 20px; width: 240px;
    }
    #side-panel h3 { margin-bottom: 12px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #888; }
    select, button {
      width: 100%; padding: 8px 12px; margin-bottom: 8px;
      background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
      border-radius: 8px; color: #e0e0e0; font-size: 14px; cursor: pointer;
    }
    button:hover { background: rgba(255,255,255,0.2); }
    button.primary { background: #4a6cf7; border-color: #4a6cf7; }
    button.primary:hover { background: #3a5ce7; }

    /* Status message */
    #status-bar {
      position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
      background: rgba(0,0,0,0.7); backdrop-filter: blur(10px);
      border-radius: 12px; padding: 10px 20px; font-size: 14px;
      display: none;
    }
  </style>
</head>
<body>
  <div id="canvas-container"></div>

  <div id="ui-overlay">
    <div id="top-bar">
      <div class="reserves"><span class="dot white"></span> <span id="white-reserves">15</span></div>
      <div id="turn-indicator">White's Turn</div>
      <div class="reserves"><span class="dot black"></span> <span id="black-reserves">15</span></div>
    </div>

    <div id="side-panel">
      <h3>Game Settings</h3>
      <select id="mode-select">
        <option value="human_vs_ai">Human vs AI</option>
        <option value="ai_vs_ai">AI vs AI</option>
        <option value="human_vs_human">Human vs Human</option>
      </select>
      <select id="difficulty-select">
        <option value="">Select difficulty...</option>
      </select>
      <select id="color-select">
        <option value="white">Play as White</option>
        <option value="black">Play as Black</option>
      </select>
      <button class="primary" id="new-game-btn">New Game</button>

      <h3 style="margin-top:16px">Move History</h3>
      <div id="move-history" style="max-height:200px;overflow-y:auto;font-size:12px;font-family:monospace;"></div>
    </div>

    <div id="status-bar"></div>
  </div>

  <script type="importmap">
  {
    "imports": {
      "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
    }
  }
  </script>
  <script type="module" src="/src/main.js"></script>
</body>
</html>
```

**Step 2: Create game.js — client-side state**

```javascript
// web/src/game.js
// Client-side game state representation

export const LEVEL_SIZES = [4, 3, 2, 1];

export function createEmptyBoard() {
  return LEVEL_SIZES.map(size => {
    const level = [];
    for (let r = 0; r < size; r++) {
      const row = [];
      for (let c = 0; c < size; c++) {
        row.push(null); // null = empty, "white" or "black"
      }
      level.push(row);
    }
    return level;
  });
}

export function updateBoardFromState(board, serverBoard) {
  // Reset
  for (let lvl = 0; lvl < board.length; lvl++) {
    for (let r = 0; r < board[lvl].length; r++) {
      for (let c = 0; c < board[lvl][r].length; c++) {
        board[lvl][r][c] = null;
      }
    }
  }
  // Fill from server data
  for (let lvl = 0; lvl < serverBoard.length; lvl++) {
    for (const piece of serverBoard[lvl]) {
      board[lvl][piece.row][piece.col] = piece.player;
    }
  }
  return board;
}

// Convert (level, row, col) to 3D world position
export function boardToWorld(level, row, col) {
  const baseSpacing = 1.2;
  const levelOffset = level * 1.1; // Y height per level
  const size = LEVEL_SIZES[level];
  const halfSize = (size - 1) / 2;
  return {
    x: (col - halfSize) * baseSpacing,
    y: levelOffset + 0.5, // sphere radius offset
    z: (row - halfSize) * baseSpacing,
  };
}
```

**Step 3: Create scene.js — Three.js 3D scene**

This file is long — it sets up the full 3D scene with the board, spheres, lighting, and camera. See Task 7 implementation for the complete code.

```javascript
// web/src/scene.js
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LEVEL_SIZES, boardToWorld } from './game.js';

let scene, camera, renderer, controls;
let sphereMeshes = {}; // key: "lvl-row-col" -> mesh
let positionMarkers = []; // clickable position indicators
let raycaster, mouse;

const SPHERE_RADIUS = 0.45;
const WHITE_COLOR = 0xf5f5f0;
const BLACK_COLOR = 0x2a2a2a;
const HOVER_COLOR = 0x4a6cf7;
const SELECTED_COLOR = 0x00ff88;

// Callbacks
let onPositionClick = null;
let onSphereClick = null;

export function init(container) {
  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);
  scene.fog = new THREE.FogExp2(0x1a1a2e, 0.04);

  // Camera
  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(6, 8, 6);
  camera.lookAt(0, 1.5, 0);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  container.appendChild(renderer.domElement);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, 1.5, 0);
  controls.minDistance = 4;
  controls.maxDistance = 15;
  controls.maxPolarAngle = Math.PI / 2.2;

  // Lighting
  const ambient = new THREE.AmbientLight(0x404060, 0.6);
  scene.add(ambient);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight.position.set(5, 10, 5);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 2048;
  dirLight.shadow.mapSize.height = 2048;
  dirLight.shadow.camera.near = 0.5;
  dirLight.shadow.camera.far = 30;
  dirLight.shadow.camera.left = -8;
  dirLight.shadow.camera.right = 8;
  dirLight.shadow.camera.top = 8;
  dirLight.shadow.camera.bottom = -8;
  scene.add(dirLight);

  const fillLight = new THREE.DirectionalLight(0x8888ff, 0.3);
  fillLight.position.set(-3, 5, -3);
  scene.add(fillLight);

  // Board base
  createBoard();

  // Raycaster for mouse interaction
  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  // Events
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('click', onClick);
  window.addEventListener('resize', onResize);

  // Start render loop
  animate();
}

function createBoard() {
  // Base platform
  const baseGeo = new THREE.BoxGeometry(7, 0.3, 7);
  const baseMat = new THREE.MeshStandardMaterial({
    color: 0x8B6914,
    roughness: 0.8,
    metalness: 0.1,
  });
  const base = new THREE.Mesh(baseGeo, baseMat);
  base.position.y = -0.15;
  base.receiveShadow = true;
  scene.add(base);

  // Position markers (small discs showing where spheres can go)
  createPositionMarkers();
}

function createPositionMarkers() {
  // Clear existing
  positionMarkers.forEach(m => scene.remove(m));
  positionMarkers = [];

  const markerGeo = new THREE.CylinderGeometry(0.35, 0.35, 0.02, 32);

  for (let lvl = 0; lvl < LEVEL_SIZES.length; lvl++) {
    const size = LEVEL_SIZES[lvl];
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const markerMat = new THREE.MeshStandardMaterial({
          color: 0x555555,
          roughness: 0.9,
          transparent: true,
          opacity: 0.3,
        });
        const marker = new THREE.Mesh(markerGeo, markerMat);
        const pos = boardToWorld(lvl, r, c);
        marker.position.set(pos.x, pos.y - SPHERE_RADIUS + 0.01, pos.z);
        marker.userData = { type: 'position', level: lvl, row: r, col: c };
        marker.receiveShadow = true;
        scene.add(marker);
        positionMarkers.push(marker);
      }
    }
  }
}

export function setCallbacks(posClickCb, sphereClickCb) {
  onPositionClick = posClickCb;
  onSphereClick = sphereClickCb;
}

export function updateBoard(board, legalMoves) {
  // Remove old spheres that are no longer on board
  const currentKeys = new Set();
  for (let lvl = 0; lvl < board.length; lvl++) {
    for (let r = 0; r < board[lvl].length; r++) {
      for (let c = 0; c < board[lvl][r].length; c++) {
        if (board[lvl][r][c]) {
          currentKeys.add(`${lvl}-${r}-${c}`);
        }
      }
    }
  }

  // Remove spheres not on board anymore
  for (const key of Object.keys(sphereMeshes)) {
    if (!currentKeys.has(key)) {
      scene.remove(sphereMeshes[key]);
      delete sphereMeshes[key];
    }
  }

  // Add new spheres
  for (let lvl = 0; lvl < board.length; lvl++) {
    for (let r = 0; r < board[lvl].length; r++) {
      for (let c = 0; c < board[lvl][r].length; c++) {
        const player = board[lvl][r][c];
        const key = `${lvl}-${r}-${c}`;
        if (player && !sphereMeshes[key]) {
          addSphere(lvl, r, c, player);
        }
      }
    }
  }

  // Update position marker visibility based on legal moves
  const legalPlaces = new Set();
  if (legalMoves) {
    for (const move of legalMoves) {
      if (move.type === 'place') {
        legalPlaces.add(`${move.level}-${move.row}-${move.col}`);
      }
    }
  }
  for (const marker of positionMarkers) {
    const { level, row, col } = marker.userData;
    const key = `${level}-${row}-${col}`;
    const isOccupied = board[level]?.[row]?.[col] != null;
    const isLegal = legalPlaces.has(key);
    marker.material.opacity = isOccupied ? 0 : (isLegal ? 0.5 : 0.15);
    marker.material.color.setHex(isLegal ? HOVER_COLOR : 0x555555);
  }
}

function addSphere(level, row, col, player, animated = true) {
  const geo = new THREE.SphereGeometry(SPHERE_RADIUS, 64, 64);
  const mat = new THREE.MeshStandardMaterial({
    color: player === 'white' ? WHITE_COLOR : BLACK_COLOR,
    roughness: player === 'white' ? 0.3 : 0.6,
    metalness: player === 'white' ? 0.1 : 0.2,
  });
  const mesh = new THREE.Mesh(geo, mat);
  const pos = boardToWorld(level, row, col);

  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.userData = { type: 'sphere', level, row, col, player };

  if (animated) {
    mesh.position.set(pos.x, pos.y + 5, pos.z);
    animateDropTo(mesh, pos.y);
  } else {
    mesh.position.set(pos.x, pos.y, pos.z);
  }

  scene.add(mesh);
  sphereMeshes[`${level}-${row}-${col}`] = mesh;
}

function animateDropTo(mesh, targetY) {
  const startY = mesh.position.y;
  const startTime = performance.now();
  const duration = 500;

  function update() {
    const elapsed = performance.now() - startTime;
    const t = Math.min(elapsed / duration, 1);
    // Bounce easing
    const ease = t < 0.7
      ? (t / 0.7) * (t / 0.7)
      : 1 - Math.pow(1 - t, 3) * Math.cos((t - 0.7) * Math.PI * 2);
    mesh.position.y = startY + (targetY - startY) * ease;
    if (t < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

let hoveredObject = null;

function onMouseMove(event) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const allObjects = [...positionMarkers, ...Object.values(sphereMeshes)];
  const intersects = raycaster.intersectObjects(allObjects);

  // Reset previous hover
  if (hoveredObject && hoveredObject.userData.type === 'position') {
    // Will be reset by updateBoard
  }

  hoveredObject = intersects.length > 0 ? intersects[0].object : null;
  renderer.domElement.style.cursor = hoveredObject ? 'pointer' : 'default';
}

function onClick(event) {
  if (!hoveredObject) return;

  const ud = hoveredObject.userData;
  if (ud.type === 'position' && onPositionClick) {
    onPositionClick(ud.level, ud.row, ud.col);
  } else if (ud.type === 'sphere' && onSphereClick) {
    onSphereClick(ud.level, ud.row, ud.col, ud.player);
  }
}

export function highlightSphere(level, row, col) {
  const key = `${level}-${row}-${col}`;
  const mesh = sphereMeshes[key];
  if (mesh) {
    mesh.material.emissive = new THREE.Color(SELECTED_COLOR);
    mesh.material.emissiveIntensity = 0.4;
  }
}

export function clearHighlights() {
  for (const mesh of Object.values(sphereMeshes)) {
    mesh.material.emissive = new THREE.Color(0x000000);
    mesh.material.emissiveIntensity = 0;
  }
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
```

**Step 4: Commit**

```bash
git add web/
git commit -m "feat: Three.js 3D visualization with board, spheres, lighting, and orbit camera"
```

---

## Task 8: Three.js — Network Layer & Main Game Loop

**Files:**
- Create: `web/src/network.js`
- Create: `web/src/main.js`

**Step 1: Create network.js**

```javascript
// web/src/network.js
// WebSocket communication with the Python server

let ws = null;
let messageHandler = null;

export function connect(onMessage) {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/game`);
  messageHandler = onMessage;

  ws.onopen = () => console.log('Connected to server');
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (messageHandler) messageHandler(msg);
  };
  ws.onclose = () => {
    console.log('Disconnected');
    setTimeout(() => connect(onMessage), 2000);
  };
  ws.onerror = (err) => console.error('WebSocket error:', err);
}

export function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

export async function fetchCheckpoints() {
  const resp = await fetch('/checkpoints');
  return resp.json();
}
```

**Step 2: Create main.js — ties everything together**

```javascript
// web/src/main.js
import { init as initScene, setCallbacks, updateBoard, highlightSphere, clearHighlights } from './scene.js';
import { connect, send, fetchCheckpoints } from './network.js';
import { createEmptyBoard, updateBoardFromState } from './game.js';

let board = createEmptyBoard();
let gameState = null;
let selectedSphere = null; // for raise moves: {level, row, col}

// Initialize 3D scene
initScene(document.getElementById('canvas-container'));

// Set up click handlers
setCallbacks(onPositionClick, onSphereClick);

// UI elements
const modeSelect = document.getElementById('mode-select');
const difficultySelect = document.getElementById('difficulty-select');
const colorSelect = document.getElementById('color-select');
const newGameBtn = document.getElementById('new-game-btn');
const turnIndicator = document.getElementById('turn-indicator');
const whiteReserves = document.getElementById('white-reserves');
const blackReserves = document.getElementById('black-reserves');
const statusBar = document.getElementById('status-bar');
const moveHistory = document.getElementById('move-history');

// Load checkpoints for difficulty selector
fetchCheckpoints().then(data => {
  difficultySelect.innerHTML = '';
  if (data.checkpoints && data.checkpoints.length > 0) {
    for (const ckpt of data.checkpoints) {
      const opt = document.createElement('option');
      opt.value = ckpt.file;
      opt.textContent = `${ckpt.label} (step ${ckpt.step}, ${Math.round(ckpt.win_rate_vs_random * 100)}% WR)`;
      difficultySelect.appendChild(opt);
    }
  } else {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No checkpoints (train first)';
    difficultySelect.appendChild(opt);
  }
});

// Connect WebSocket
connect(onServerMessage);

// New Game button
newGameBtn.addEventListener('click', () => {
  const mode = modeSelect.value;
  const checkpoint = difficultySelect.value;
  const humanColor = colorSelect.value;

  const msg = { type: 'new_game', mode };
  if (checkpoint) msg.checkpoint = checkpoint;
  if (mode === 'human_vs_ai') msg.human_color = humanColor;
  if (mode === 'ai_vs_ai') msg.delay_ms = 1500;

  send(msg);
  moveHistory.innerHTML = '';
  selectedSphere = null;
  clearHighlights();
  showStatus('Starting new game...');
});

// Toggle color select visibility
modeSelect.addEventListener('change', () => {
  colorSelect.style.display = modeSelect.value === 'human_vs_ai' ? 'block' : 'none';
});

function onServerMessage(msg) {
  switch (msg.type) {
    case 'state':
      gameState = msg;
      board = updateBoardFromState(board, msg.board);
      updateBoard(board, msg.legal_moves);
      turnIndicator.textContent = `${capitalize(msg.turn)}'s Turn`;
      whiteReserves.textContent = msg.reserves.white;
      blackReserves.textContent = msg.reserves.black;
      hideStatus();
      break;

    case 'ai_move':
      const desc = msg.action.type === 'place'
        ? `AI places at (${msg.action.level},${msg.action.row},${msg.action.col})`
        : `AI raises (${msg.action.src.level},${msg.action.src.row},${msg.action.src.col}) -> (${msg.action.dst.level},${msg.action.dst.row},${msg.action.dst.col})`;
      addMoveToHistory(desc, msg.thinking_time_ms);
      break;

    case 'game_over':
      showStatus(`Game over! ${capitalize(msg.winner)} wins!`);
      break;

    case 'error':
      showStatus(`Error: ${msg.message}`);
      setTimeout(hideStatus, 3000);
      break;
  }
}

function onPositionClick(level, row, col) {
  if (!gameState) return;

  if (selectedSphere) {
    // Attempting a raise
    send({
      type: 'move',
      action: {
        type: 'raise',
        src: [selectedSphere.level, selectedSphere.row, selectedSphere.col],
        dst: [level, row, col],
      },
    });
    addMoveToHistory(`Raise (${selectedSphere.level},${selectedSphere.row},${selectedSphere.col}) -> (${level},${row},${col})`);
    selectedSphere = null;
    clearHighlights();
  } else {
    // Placement
    send({
      type: 'move',
      action: { type: 'place', level, row, col },
    });
    addMoveToHistory(`Place at (${level},${row},${col})`);
  }
}

function onSphereClick(level, row, col, player) {
  if (!gameState) return;

  // Select sphere for raising
  clearHighlights();
  selectedSphere = { level, row, col };
  highlightSphere(level, row, col);
  showStatus('Click a higher position to raise this piece');
}

function addMoveToHistory(text, thinkingMs) {
  const entry = document.createElement('div');
  entry.textContent = thinkingMs ? `${text} (${thinkingMs}ms)` : text;
  entry.style.marginBottom = '4px';
  moveHistory.prepend(entry);
}

function showStatus(text) {
  statusBar.textContent = text;
  statusBar.style.display = 'block';
}

function hideStatus() {
  statusBar.style.display = 'none';
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
```

**Step 3: Test manually**

```bash
cd /Users/atarkian2/Documents/GitHub/Pylos-Engine && python engine/server.py
```

Open `http://localhost:8000` in browser. Verify:
- 3D board renders with orbit controls
- Can start Human vs Human game
- Can click positions to place spheres
- Spheres animate on placement

**Step 4: Commit**

```bash
git add web/
git commit -m "feat: complete Three.js frontend with network layer, game loop, and UI overlay"
```

---

## Task 9: Integration Testing & Polish

**Files:**
- Modify: `engine/game.py` (bug fixes from testing)
- Modify: `web/src/scene.js` (visual polish)
- Create: `engine/tests/test_integration.py`

**Step 1: Write integration test**

```python
# engine/tests/test_integration.py
"""End-to-end test: train a tiny model, save checkpoint, verify server loads it."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import json
import tempfile
from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgentTrainer


def test_full_pipeline():
    game = PylosGame()
    model = PylosNetwork(game.observation_shape, game.action_space)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    agent = AlphaZeroAgentTrainer(model, optimizer, 64)

    # Train 3 games
    for _ in range(3):
        game.reset()
        agent.train_step(game, search_iterations=4, batch_size=8, epochs=1, c_puct=1.5)

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pth")
        opt_path = os.path.join(tmpdir, "optimizer.pth")
        agent.save_training_state(model_path, opt_path)

        # Load and verify
        model2 = PylosNetwork(game.observation_shape, game.action_space)
        model2.load_state_dict(torch.load(model_path, map_location=model2.device, weights_only=True))

        # Verify inference works
        game.reset()
        obs = torch.tensor(game.to_observation(), device=model2.device)
        val = model2.value_forward(obs)
        pol = model2.policy_forward(obs)
        assert val.shape == (1,)
        assert pol.shape == (303,)
        assert abs(pol.sum().item() - 1.0) < 0.01
```

**Step 2: Run all tests**

```bash
cd /Users/atarkian2/Documents/GitHub/Pylos-Engine && python -m pytest engine/tests/ -v --timeout=120
```

Expected: ALL PASS

**Step 3: Commit**

```bash
git add engine/tests/test_integration.py
git commit -m "test: end-to-end integration test for train-save-load pipeline"
```

---

## Task 10: README & Launch Script

**Files:**
- Create: `README.md`
- Create: `run.sh`

**Step 1: Create run.sh convenience script**

```bash
#!/bin/bash
# run.sh — Launch the Pylos server
set -e
echo "Starting Pylos Engine server on http://localhost:8000"
python engine/server.py
```

**Step 2: Create README.md**

```markdown
# Pylos Engine

AlphaZero-trained AI for the board game Pylos, with a 3D web visualization.

## Quick Start

```bash
pip install -r requirements.txt
# Train the AI (takes a while — adjust config.yaml for fewer games)
python engine/train.py
# Launch the web server
bash run.sh
# Open http://localhost:8000
```

## Training

Edit `engine/config.yaml` to adjust training parameters. Checkpoints are saved to `engine/checkpoints/` and auto-evaluated against a random agent.

## Playing

Open the web UI and select:
- **Human vs AI** — pick a difficulty level (checkpoint) and play
- **AI vs AI** — watch two agents play each other
- **Human vs Human** — play locally with a friend

## Project Structure

- `engine/` — Python game logic, AlphaZero training, WebSocket server
- `web/` — Three.js 3D visualization
- `docs/plans/` — Design documents
```

**Step 3: Commit**

```bash
chmod +x run.sh
git add README.md run.sh
git commit -m "docs: README and launch script"
```

---

## Summary of Tasks

| # | Task | Key Files | Test |
|---|------|-----------|------|
| 1 | Project scaffolding | `requirements.txt`, dirs | `pip install` |
| 2 | Game engine: board + placement | `engine/game.py` | `test_game.py` |
| 3 | Game engine: raise/removal/formations | `engine/game.py` | `test_game.py` |
| 4 | Fork MCTS/agents/models/replay | `engine/mcts.py`, `agents.py`, `models.py` | `test_training_smoke.py` |
| 5 | Training script + checkpoints | `engine/train.py`, `evaluate.py`, `config.yaml` | Manual 5-game run |
| 6 | WebSocket server | `engine/server.py` | `test_server.py` |
| 7 | Three.js scene + board | `web/index.html`, `scene.js`, `game.js` | Manual browser |
| 8 | Frontend network + game loop | `web/src/network.js`, `main.js` | Manual browser |
| 9 | Integration testing | `test_integration.py` | `pytest` |
| 10 | README + launch script | `README.md`, `run.sh` | — |
