"""
Pylos game engine for AlphaZero training.

Pylos is a 2-player abstract strategy game on a 4-level pyramid:
  Level 0: 4x4 (16 positions)
  Level 1: 3x3 (9 positions)
  Level 2: 2x2 (4 positions)
  Level 3: 1x1 (1 position)
  Total: 30 positions

Players have 15 spheres each. On your turn you either place a reserve
sphere on any supported empty position, or raise one of your own
unsupported spheres to a higher level. Completing a 2x2 square or a
line (4-in-a-row on level 0, 3-in-a-row on level 1) lets you reclaim
up to 2 of your own unsupported pieces. The first player to place the
apex sphere wins. If a player has no legal moves they lose.

Action space (303 total):
  0-29:   placement actions (one per pyramid position)
  30-302: raise actions (src, dst pairs where dst is strictly higher)
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from copy import deepcopy
from engine.game_base import Game


# Level dimensions: level i has side length (4 - i)
LEVEL_SIZES = [4, 3, 2, 1]


class PylosGame(Game):
    """Pylos game engine compatible with AlphaZero MCTS framework."""

    def __init__(self, rich_obs=False):
        self._rich_obs = rich_obs

        # Pre-compute position mappings
        self.index_to_coords = []  # index -> (level, row, col)
        self.coords_to_index = {}  # (level, row, col) -> index

        idx = 0
        for level, size in enumerate(LEVEL_SIZES):
            for r in range(size):
                for c in range(size):
                    self.index_to_coords.append((level, r, c))
                    self.coords_to_index[(level, r, c)] = idx
                    idx += 1

        assert len(self.index_to_coords) == 30

        # Pre-compute raise pairs: all (src_idx, dst_idx) where dst is
        # on a strictly higher level than src
        self.raise_pairs = []
        for src_idx in range(30):
            src_level = self.index_to_coords[src_idx][0]
            for dst_idx in range(30):
                dst_level = self.index_to_coords[dst_idx][0]
                if dst_level > src_level:
                    self.raise_pairs.append((src_idx, dst_idx))

        assert len(self.raise_pairs) == 273

        # Map action index (30-302) to raise pair
        self.raise_action_to_pair = {}
        for i, pair in enumerate(self.raise_pairs):
            self.raise_action_to_pair[30 + i] = pair

        self.action_space = 303
        self.observation_shape = (92,) if rich_obs else (32,)

        # Initialize game state
        self.reset()

    # ------------------------------------------------------------------
    # Game state management
    # ------------------------------------------------------------------

    def reset(self):
        """Reset the game to initial state."""
        self.board = [
            np.zeros((4, 4), dtype=np.int8),
            np.zeros((3, 3), dtype=np.int8),
            np.zeros((2, 2), dtype=np.int8),
            np.zeros((1, 1), dtype=np.int8),
        ]
        self._turn = 1  # 1 = white, -1 = black
        self.reserves = {1: 15, -1: 15}
        self.last_move = None  # (level, row, col) of the last placed/raised piece
        self.actions_stack = []  # stack for undo

        # Interactive removal phase state
        self.pending_removal = False  # waiting for removal choices
        self.removal_count = 0       # pieces removed so far (max 2)
        self.removal_player = 0      # which player is removing (1 or -1)

    @property
    def turn(self) -> int:
        return self._turn

    @turn.setter
    def turn(self, value):
        self._turn = value

    # ------------------------------------------------------------------
    # Board queries
    # ------------------------------------------------------------------

    def is_supported(self, level, r, c):
        """Check if position (level, r, c) is supported from below."""
        if level == 0:
            return True
        below = self.board[level - 1]
        return bool(
            below[r, c] != 0
            and below[r + 1, c] != 0
            and below[r, c + 1] != 0
            and below[r + 1, c + 1] != 0
        )

    def piece_has_top(self, level, r, c):
        """Check if any piece on level+1 uses this position as support."""
        if level >= 3:
            return False
        upper = self.board[level + 1]
        upper_size = LEVEL_SIZES[level + 1]
        for dr in [-1, 0]:
            for dc in [-1, 0]:
                ur = r + dr
                uc = c + dc
                if 0 <= ur < upper_size and 0 <= uc < upper_size:
                    if upper[ur, uc] != 0:
                        return True
        return False

    def top_filled(self):
        """Check if the apex (level 3, position 0,0) is filled."""
        return bool(self.board[3][0, 0] != 0)

    # ------------------------------------------------------------------
    # Formation detection
    # ------------------------------------------------------------------

    def check_square(self, level, r, c):
        """Check if any 2x2 square containing (level, r, c) is all same color."""
        board_level = self.board[level]
        size = LEVEL_SIZES[level]
        color = board_level[r, c]
        if color == 0:
            return False

        for sr in range(max(0, r - 1), min(size - 2, r) + 1):
            for sc in range(max(0, c - 1), min(size - 2, c) + 1):
                if (
                    board_level[sr, sc] == color
                    and board_level[sr + 1, sc] == color
                    and board_level[sr, sc + 1] == color
                    and board_level[sr + 1, sc + 1] == color
                ):
                    return True
        return False

    def check_line(self, level, r, c):
        """Check for line formations containing (level, r, c)."""
        if level == 0:
            return self._check_line_n(level, r, c, 4)
        elif level == 1:
            return self._check_line_n(level, r, c, 3)
        return False

    def _check_line_n(self, level, r, c, n):
        """Check for n-in-a-row on given level containing (r, c)."""
        board_level = self.board[level]
        size = LEVEL_SIZES[level]
        color = board_level[r, c]
        if color == 0:
            return False

        if size >= n:
            row_vals = board_level[r, :]
            count = np.sum(row_vals == color)
            if count >= n:
                return True

        if size >= n:
            col_vals = board_level[:, c]
            count = np.sum(col_vals == color)
            if count >= n:
                return True

        return False

    def check_for_removal(self):
        """Check if the last move created a square or line formation."""
        if self.last_move is None:
            return False
        level, r, c = self.last_move
        return self.check_square(level, r, c) or self.check_line(level, r, c)

    def get_formation_positions(self):
        """Return the positions forming the completed pattern from the last move."""
        if self.last_move is None:
            return []
        level, r, c = self.last_move
        positions = []

        board_level = self.board[level]
        size = LEVEL_SIZES[level]
        color = board_level[r, c]
        if color != 0:
            for sr in range(max(0, r - 1), min(size - 2, r) + 1):
                for sc in range(max(0, c - 1), min(size - 2, c) + 1):
                    if (
                        board_level[sr, sc] == color
                        and board_level[sr + 1, sc] == color
                        and board_level[sr, sc + 1] == color
                        and board_level[sr + 1, sc + 1] == color
                    ):
                        for dr in range(2):
                            for dc in range(2):
                                pos = (level, sr + dr, sc + dc)
                                if pos not in positions:
                                    positions.append(pos)

        if level == 0:
            positions.extend(self._get_line_positions(level, r, c, 4))
        elif level == 1:
            positions.extend(self._get_line_positions(level, r, c, 3))

        seen = set()
        unique = []
        for p in positions:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def _get_line_positions(self, level, r, c, n):
        """Return positions forming an n-in-a-row on the given level."""
        board_level = self.board[level]
        size = LEVEL_SIZES[level]
        color = board_level[r, c]
        if color == 0:
            return []
        positions = []

        if size >= n:
            row_vals = board_level[r, :]
            if int(np.sum(row_vals == color)) >= n:
                for ci in range(size):
                    if board_level[r, ci] == color:
                        positions.append((level, r, ci))

        if size >= n:
            col_vals = board_level[:, c]
            if int(np.sum(col_vals == color)) >= n:
                for ri in range(size):
                    if board_level[ri, c] == color:
                        positions.append((level, ri, c))

        return positions

    # ------------------------------------------------------------------
    # Removable pieces
    # ------------------------------------------------------------------

    def get_removable_pieces(self):
        """Get list of current player's pieces that can be removed."""
        removable = []
        for level in range(4):
            size = LEVEL_SIZES[level]
            for r in range(size):
                for c in range(size):
                    if (
                        self.board[level][r, c] == self._turn
                        and not self.piece_has_top(level, r, c)
                    ):
                        removable.append((level, r, c))
        return removable

    # ------------------------------------------------------------------
    # Actions: place, raise, remove
    # ------------------------------------------------------------------

    def place(self, level, r, c):
        """Place a reserve sphere at (level, r, c)."""
        if self.reserves[self._turn] <= 0:
            return False
        if self.board[level][r, c] != 0:
            return False
        if not self.is_supported(level, r, c):
            return False

        self.board[level][r, c] = self._turn
        self.reserves[self._turn] -= 1
        self.last_move = (level, r, c)
        return True

    def raise_piece(self, sl, sr, sc, dl, dr, dc):
        """Raise own piece from (sl, sr, sc) to (dl, dr, dc)."""
        if self.board[sl][sr, sc] != self._turn:
            return False
        if self.piece_has_top(sl, sr, sc):
            return False
        if dl <= sl:
            return False
        if self.board[dl][dr, dc] != 0:
            return False

        self.board[sl][sr, sc] = 0
        supported = self.is_supported(dl, dr, dc)
        if not supported:
            self.board[sl][sr, sc] = self._turn
            return False

        self.board[dl][dr, dc] = self._turn
        self.last_move = (dl, dr, dc)
        return True

    def remove(self, level, r, c):
        """Remove own unsupported piece, returning it to reserves."""
        if self.board[level][r, c] != self._turn:
            return False
        if self.piece_has_top(level, r, c):
            return False

        self.board[level][r, c] = 0
        self.reserves[self._turn] += 1
        return True

    # ------------------------------------------------------------------
    # AI removal heuristic
    # ------------------------------------------------------------------

    def _do_ai_removal(self):
        """Greedy removal of up to 2 pieces after forming a square/line."""
        removed = []
        for _ in range(2):
            removable = self.get_removable_pieces()
            if not removable:
                break
            removable.sort(key=lambda x: (x[0], x[1], x[2]))
            piece = removable[0]
            self.remove(*piece)
            removed.append(piece)
        return removed

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------

    def get_legal_actions(self):
        """Get list of valid action indices."""
        legal = []

        if self.reserves[self._turn] > 0:
            for idx in range(30):
                level, r, c = self.index_to_coords[idx]
                if self.board[level][r, c] == 0 and self.is_supported(level, r, c):
                    legal.append(idx)

        for action_idx in range(30, 303):
            src_idx, dst_idx = self.raise_action_to_pair[action_idx]
            sl, sr, sc = self.index_to_coords[src_idx]
            dl, dr, dc = self.index_to_coords[dst_idx]

            if self.board[sl][sr, sc] != self._turn:
                continue
            if self.piece_has_top(sl, sr, sc):
                continue
            if self.board[dl][dr, dc] != 0:
                continue

            self.board[sl][sr, sc] = 0
            supported = self.is_supported(dl, dr, dc)
            self.board[sl][sr, sc] = self._turn

            if supported:
                legal.append(action_idx)

        return legal

    # ------------------------------------------------------------------
    # Step / Undo (AlphaZero interface)
    # ------------------------------------------------------------------

    def step(self, action, auto_remove=True):
        """Execute an action (placement or raise)."""
        self.pending_removal = False

        removed = []
        has_formation = False

        if action < 30:
            level, r, c = self.index_to_coords[action]
            success = self.place(level, r, c)
            if not success:
                raise ValueError(f"Illegal placement action {action} at ({level},{r},{c})")
            has_formation = self.check_for_removal()
            if has_formation and auto_remove:
                removed = self._do_ai_removal()
            self.actions_stack.append(('place', action, level, r, c, removed))
        else:
            src_idx, dst_idx = self.raise_action_to_pair[action]
            sl, sr, sc = self.index_to_coords[src_idx]
            dl, dr, dc = self.index_to_coords[dst_idx]
            success = self.raise_piece(sl, sr, sc, dl, dr, dc)
            if not success:
                raise ValueError(
                    f"Illegal raise action {action}: ({sl},{sr},{sc})->({dl},{dr},{dc})"
                )
            has_formation = self.check_for_removal()
            if has_formation and auto_remove:
                removed = self._do_ai_removal()
            self.actions_stack.append(('raise', action, src_idx, dst_idx, sl, sr, sc, dl, dr, dc, removed))

        if has_formation and not auto_remove:
            self.pending_removal = True
            self.removal_player = self._turn
            self.removal_count = 0

        self._turn *= -1

    def undo_last_action(self):
        """Undo the last action (pop from actions_stack)."""
        if not self.actions_stack:
            return

        entry = self.actions_stack.pop()
        self._turn *= -1

        if entry[0] == 'place':
            _, action, level, r, c, removed = entry
            for rl, rr, rc in reversed(removed):
                self.board[rl][rr, rc] = self._turn
                self.reserves[self._turn] -= 1
            self.board[level][r, c] = 0
            self.reserves[self._turn] += 1
        elif entry[0] == 'raise':
            _, action, src_idx, dst_idx, sl, sr, sc, dl, dr, dc, removed = entry
            for rl, rr, rc in reversed(removed):
                self.board[rl][rr, rc] = self._turn
                self.reserves[self._turn] -= 1
            self.board[dl][dr, dc] = 0
            self.board[sl][sr, sc] = self._turn

        self.last_move = None

    # ------------------------------------------------------------------
    # Interactive removal phase
    # ------------------------------------------------------------------

    def get_pending_removable(self):
        """Get removable pieces for the removal_player during removal phase."""
        if not self.pending_removal:
            return []
        removable = []
        for level in range(4):
            size = LEVEL_SIZES[level]
            for r in range(size):
                for c in range(size):
                    if (
                        self.board[level][r, c] == self.removal_player
                        and not self.piece_has_top(level, r, c)
                    ):
                        removable.append((level, r, c))
        return removable

    def step_removal(self, level, r, c):
        """Remove one piece during the interactive removal phase."""
        if not self.pending_removal:
            return False
        if self.removal_count >= 2:
            return False
        if self.board[level][r, c] != self.removal_player:
            return False
        if self.piece_has_top(level, r, c):
            return False

        self.board[level][r, c] = 0
        self.reserves[self.removal_player] += 1
        self.removal_count += 1

        if self.removal_count >= 2 or not self.get_pending_removable():
            self.pending_removal = False

        return True

    def skip_removal(self):
        """End the removal phase early."""
        if not self.pending_removal:
            return False
        self.pending_removal = False
        return True

    # ------------------------------------------------------------------
    # Win / result
    # ------------------------------------------------------------------

    def has_move(self):
        """Check if current player has any legal moves."""
        if self.reserves[self._turn] > 0:
            for idx in range(30):
                level, r, c = self.index_to_coords[idx]
                if self.board[level][r, c] == 0 and self.is_supported(level, r, c):
                    return True

        for action_idx in range(30, 303):
            src_idx, dst_idx = self.raise_action_to_pair[action_idx]
            sl, sr, sc = self.index_to_coords[src_idx]
            dl, dr, dc = self.index_to_coords[dst_idx]
            if self.board[sl][sr, sc] != self._turn:
                continue
            if self.piece_has_top(sl, sr, sc):
                continue
            if self.board[dl][dr, dc] != 0:
                continue
            self.board[sl][sr, sc] = 0
            supported = self.is_supported(dl, dr, dc)
            self.board[sl][sr, sc] = self._turn
            if supported:
                return True

        return False

    def get_result(self):
        """Get game result: 1 white wins, -1 black wins, None ongoing."""
        if self.top_filled():
            return self.board[3][0, 0]
        if not self.has_move():
            return -self._turn
        return None

    def get_first_person_result(self):
        """Get result from current player's perspective."""
        result = self.get_result()
        if result is None:
            return None
        return result * self._turn

    @staticmethod
    def swap_result(result):
        """Swap result perspective."""
        return -result

    # ------------------------------------------------------------------
    # Observation (AlphaZero interface)
    # ------------------------------------------------------------------

    def to_observation(self):
        """Convert game state to observation tensor."""
        if self._rich_obs:
            return self._to_observation_rich()
        return self._to_observation_basic()

    def _to_observation_basic(self):
        """Basic 32-dim observation."""
        obs = np.zeros(32, dtype=np.float32)
        for idx in range(30):
            level, r, c = self.index_to_coords[idx]
            cell = self.board[level][r, c]
            if cell != 0:
                obs[idx] = float(cell * self._turn)
        obs[30] = self.reserves[self._turn] / 15.0
        obs[31] = self.reserves[-self._turn] / 15.0
        return obs

    def _to_observation_rich(self):
        """Rich 92-dim observation with derived features."""
        obs = np.zeros(92, dtype=np.float32)
        for idx in range(30):
            level, r, c = self.index_to_coords[idx]
            cell = self.board[level][r, c]
            if cell != 0:
                obs[idx] = float(cell * self._turn)
            if self.is_supported(level, r, c):
                obs[30 + idx] = 1.0
            if cell != 0 and not self.piece_has_top(level, r, c):
                obs[60 + idx] = 1.0
        obs[90] = self.reserves[self._turn] / 15.0
        obs[91] = self.reserves[-self._turn] / 15.0
        return obs

    # ------------------------------------------------------------------
    # Server serialization (Game ABC methods)
    # ------------------------------------------------------------------

    def board_hash(self) -> bytes:
        parts = []
        for level in range(4):
            parts.append(self.board[level].tobytes())
        parts.append(bytes([self._turn + 1]))
        return b"".join(parts)

    def board_to_json(self) -> dict:
        player_names = {1: "white", -1: "black"}
        levels = []
        for level_idx in range(4):
            size = self.board[level_idx].shape[0]
            cells = []
            for r in range(size):
                for c in range(size):
                    val = int(self.board[level_idx][r, c])
                    if val != 0:
                        cells.append({
                            "row": r,
                            "col": c,
                            "player": player_names[val],
                        })
            levels.append(cells)
        return levels

    def legal_moves_to_json(self) -> list[dict]:
        legal_actions = self.get_legal_actions()
        moves = []
        for action in legal_actions:
            if action < 30:
                level, r, c = self.index_to_coords[action]
                moves.append({
                    "type": "place",
                    "level": level,
                    "row": r,
                    "col": c,
                    "action": action,
                })
            else:
                src_idx, dst_idx = self.raise_action_to_pair[action]
                sl, sr, sc = self.index_to_coords[src_idx]
                dl, dr, dc = self.index_to_coords[dst_idx]
                moves.append({
                    "type": "raise",
                    "src": {"level": sl, "row": sr, "col": sc},
                    "dst": {"level": dl, "row": dr, "col": dc},
                    "action": action,
                })
        return moves

    def action_to_json(self, action: int) -> dict:
        if action < 30:
            level, r, c = self.index_to_coords[action]
            return {"type": "place", "level": level, "row": r, "col": c}
        else:
            src_idx, dst_idx = self.raise_action_to_pair[action]
            sl, sr, sc = self.index_to_coords[src_idx]
            dl, dr, dc = self.index_to_coords[dst_idx]
            return {
                "type": "raise",
                "src": {"level": sl, "row": sr, "col": sc},
                "dst": {"level": dl, "row": dr, "col": dc},
            }

    def action_from_json(self, action_msg: dict) -> int | None:
        if "action" in action_msg:
            action = action_msg["action"]
            if action in self.get_legal_actions():
                return action
            return None

        move_type = action_msg.get("type")

        if move_type == "place":
            level = action_msg["level"]
            row = action_msg["row"]
            col = action_msg["col"]
            coords = (level, row, col)
            if coords in self.coords_to_index:
                action = self.coords_to_index[coords]
                if action in self.get_legal_actions():
                    return action
            return None

        elif move_type == "raise":
            src = action_msg["src"]
            dst = action_msg["dst"]
            if isinstance(src, list):
                src_coords = tuple(src)
            else:
                src_coords = (src["level"], src["row"], src["col"])
            if isinstance(dst, list):
                dst_coords = tuple(dst)
            else:
                dst_coords = (dst["level"], dst["row"], dst["col"])

            if src_coords not in self.coords_to_index or dst_coords not in self.coords_to_index:
                return None
            src_idx = self.coords_to_index[src_coords]
            dst_idx = self.coords_to_index[dst_coords]

            for action_idx, pair in self.raise_action_to_pair.items():
                if pair == (src_idx, dst_idx):
                    if action_idx in self.get_legal_actions():
                        return action_idx
                    return None
            return None

        return None

    def make_state_msg(self) -> dict:
        turn_names = {1: "white", -1: "black"}
        return {
            "type": "state",
            "board": self.board_to_json(),
            "turn": turn_names[self._turn],
            "reserves": {
                "white": int(self.reserves[1]),
                "black": int(self.reserves[-1]),
            },
            "legal_moves": self.legal_moves_to_json(),
        }

    def make_removal_msg(self) -> dict:
        player_names = {1: "white", -1: "black"}
        return {
            "type": "removal_phase",
            "board": self.board_to_json(),
            "removable": [
                {"level": l, "row": r, "col": c}
                for l, r, c in self.get_pending_removable()
            ],
            "formation": [
                {"level": l, "row": r, "col": c}
                for l, r, c in self.get_formation_positions()
            ],
            "removed_so_far": self.removal_count,
            "max_removals": 2,
            "player": player_names[self.removal_player],
            "reserves": {
                "white": int(self.reserves[1]),
                "black": int(self.reserves[-1]),
            },
        }

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self):
        lines = []
        symbols = {0: '.', 1: 'W', -1: 'B'}
        for level in range(4):
            size = LEVEL_SIZES[level]
            indent = '  ' * level
            lines.append(f"Level {level} ({size}x{size}):")
            for r in range(size):
                row_str = ' '.join(symbols[int(self.board[level][r, c])] for c in range(size))
                lines.append(f"  {indent}{row_str}")
        lines.append(f"Turn: {'White' if self._turn == 1 else 'Black'}")
        lines.append(f"Reserves: White={self.reserves[1]}, Black={self.reserves[-1]}")
        return '\n'.join(lines)
