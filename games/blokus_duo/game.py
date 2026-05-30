"""
Blokus Duo game engine for AlphaZero training.

Blokus Duo is a 2-player abstract strategy game on a 14x14 grid.
Each player has 21 polyomino pieces (1-5 cells each). Players take
turns placing one piece on the board. Placement rules:
  - First piece must cover the player's starting position
  - Each subsequent piece must touch a corner of a same-color piece
  - No edge-to-edge contact with same-color pieces
  - Pieces cannot overlap

The game ends when neither player can place a piece. The player
who placed more total cells wins.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from engine.game_base import Game
from games.blokus_duo.pieces import PIECES, PIECE_NAMES, NUM_PIECES


BOARD_SIZE = 14

# Starting positions for each player (diagonal corners)
START_POS = {
    1: (4, 4),    # Player 1 start
    -1: (9, 9),   # Player 2 start
}


def _build_action_table():
    """Build mapping from action index to (piece_idx, orient_idx, row, col).

    Action space = sum over all pieces of (num_orientations * BOARD_SIZE * BOARD_SIZE),
    but we only enumerate positions where the piece fits on the board.
    """
    actions = []  # list of (piece_idx, orient_idx, row, col)
    action_index = 0

    for piece_idx, (name, orientations) in enumerate(PIECES):
        for orient_idx, cells in enumerate(orientations):
            max_r = max(r for r, c in cells)
            max_c = max(c for r, c in cells)
            for row in range(BOARD_SIZE - max_r):
                for col in range(BOARD_SIZE - max_c):
                    actions.append((piece_idx, orient_idx, row, col))
                    action_index += 1

    return actions


ACTION_TABLE = _build_action_table()
ACTION_SPACE = len(ACTION_TABLE)

# Build reverse lookup: (piece_idx, orient_idx, row, col) -> action index
_ACTION_REVERSE = {}
for i, key in enumerate(ACTION_TABLE):
    _ACTION_REVERSE[key] = i


class BlokusDuoGame(Game):
    """Blokus Duo game engine compatible with AlphaZero MCTS framework."""

    def __init__(self, **kwargs):
        self.action_space = ACTION_SPACE
        # Board (14x14) + piece availability (21 per player) + turn indicator
        self.observation_shape = (BOARD_SIZE * BOARD_SIZE + NUM_PIECES * 2 + 1,)
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self._turn = 1
        # Which pieces each player still has (True = available)
        self.pieces_available = {
            1: [True] * NUM_PIECES,
            -1: [True] * NUM_PIECES,
        }
        self.move_count = {1: 0, -1: 0}
        self.actions_stack = []
        self._legal_actions_cache = None
        self._result_cache = None

    @property
    def turn(self) -> int:
        return self._turn

    @turn.setter
    def turn(self, value):
        self._turn = value

    def _invalidate_cache(self):
        self._legal_actions_cache = None
        self._result_cache = None

    def _get_piece_cells(self, piece_idx, orient_idx, row, col):
        """Get the absolute board positions for a piece placement."""
        _, orientations = PIECES[piece_idx]
        cells = orientations[orient_idx]
        return [(row + r, col + c) for r, c in cells]

    def _is_valid_placement(self, piece_idx, orient_idx, row, col, player):
        """Check if placing piece at position is valid for player."""
        cells = self._get_piece_cells(piece_idx, orient_idx, row, col)

        # Check all cells are empty and on-board
        for r, c in cells:
            if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE:
                return False
            if self.board[r, c] != 0:
                return False

        is_first_move = self.move_count[player] == 0

        if is_first_move:
            # First piece must cover the starting position
            start = START_POS[player]
            if start not in cells:
                return False
            return True

        # Check no edge-to-edge contact with same color
        has_corner_contact = False
        for r, c in cells:
            # Edge neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if self.board[nr, nc] == player:
                        return False

            # Corner neighbors
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if self.board[nr, nc] == player:
                        has_corner_contact = True

        return has_corner_contact

    def get_legal_actions(self) -> list[int]:
        if self._legal_actions_cache is not None:
            return self._legal_actions_cache

        legal = []
        player = self._turn

        for action_idx, (piece_idx, orient_idx, row, col) in enumerate(ACTION_TABLE):
            if not self.pieces_available[player][piece_idx]:
                continue
            if self._is_valid_placement(piece_idx, orient_idx, row, col, player):
                legal.append(action_idx)

        self._legal_actions_cache = legal
        return legal

    def step(self, action, **kwargs):
        piece_idx, orient_idx, row, col = ACTION_TABLE[action]
        cells = self._get_piece_cells(piece_idx, orient_idx, row, col)

        # Place piece on board
        for r, c in cells:
            self.board[r, c] = self._turn

        self.pieces_available[self._turn][piece_idx] = False
        self.move_count[self._turn] += 1
        self.actions_stack.append((action, piece_idx, cells, self._turn))
        self._turn *= -1
        self._invalidate_cache()

    def undo_last_action(self):
        if not self.actions_stack:
            return
        action, piece_idx, cells, player = self.actions_stack.pop()

        for r, c in cells:
            self.board[r, c] = 0

        self.pieces_available[player][piece_idx] = True
        self.move_count[player] -= 1
        self._turn = player
        self._invalidate_cache()

    def to_observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_shape[0], dtype=np.float32)

        # Board from current player's perspective
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                idx = r * BOARD_SIZE + c
                cell = self.board[r, c]
                if cell != 0:
                    obs[idx] = float(cell * self._turn)

        # Piece availability for current player
        base = BOARD_SIZE * BOARD_SIZE
        for i in range(NUM_PIECES):
            if self.pieces_available[self._turn][i]:
                obs[base + i] = 1.0

        # Piece availability for opponent
        base2 = base + NUM_PIECES
        for i in range(NUM_PIECES):
            if self.pieces_available[-self._turn][i]:
                obs[base2 + i] = 1.0

        # Turn indicator
        obs[-1] = 1.0 if self._turn == 1 else -1.0

        return obs

    def get_result(self):
        if self._result_cache is not None:
            return self._result_cache

        # Game continues if either player can move
        # Check current player first
        if self.get_legal_actions():
            return None

        # Current player can't move — check opponent
        self._turn *= -1
        self._invalidate_cache()
        opponent_can_move = bool(self.get_legal_actions())
        self._turn *= -1
        self._invalidate_cache()

        if opponent_can_move:
            # Current player passes, opponent still plays
            # In Blokus Duo, if you can't move you just pass.
            # Game only ends when NEITHER player can move.
            # For AlphaZero, we handle this by passing the turn.
            return None

        # Neither player can move — game over
        # Count cells placed by each player
        p1_cells = int(np.sum(self.board == 1))
        p2_cells = int(np.sum(self.board == -1))

        if p1_cells > p2_cells:
            result = 1
        elif p2_cells > p1_cells:
            result = -1
        else:
            result = 0  # draw (rare)

        self._result_cache = result
        return result

    def get_first_person_result(self):
        result = self.get_result()
        if result is None:
            return None
        if result == 0:
            return 0
        return result * self._turn

    @staticmethod
    def swap_result(result):
        return -result

    def has_move(self) -> bool:
        return len(self.get_legal_actions()) > 0

    def board_hash(self) -> bytes:
        return self.board.tobytes() + bytes([self._turn + 1])

    # ------------------------------------------------------------------
    # Server serialization
    # ------------------------------------------------------------------

    def board_to_json(self) -> dict:
        player_names = {1: "orange", -1: "purple"}
        grid = []
        for r in range(BOARD_SIZE):
            row = []
            for c in range(BOARD_SIZE):
                val = int(self.board[r, c])
                if val == 0:
                    row.append(None)
                else:
                    row.append(player_names[val])
            grid.append(row)
        return grid

    def legal_moves_to_json(self) -> list[dict]:
        moves = []
        for action_idx in self.get_legal_actions():
            piece_idx, orient_idx, row, col = ACTION_TABLE[action_idx]
            name = PIECE_NAMES[piece_idx]
            cells = self._get_piece_cells(piece_idx, orient_idx, row, col)
            moves.append({
                "type": "place",
                "piece": name,
                "piece_idx": piece_idx,
                "orientation": orient_idx,
                "row": row,
                "col": col,
                "cells": [{"row": r, "col": c} for r, c in cells],
                "action": action_idx,
            })
        return moves

    def action_to_json(self, action: int) -> dict:
        piece_idx, orient_idx, row, col = ACTION_TABLE[action]
        name = PIECE_NAMES[piece_idx]
        cells = self._get_piece_cells(piece_idx, orient_idx, row, col)
        return {
            "type": "place",
            "piece": name,
            "piece_idx": piece_idx,
            "orientation": orient_idx,
            "row": row,
            "col": col,
            "cells": [{"row": r, "col": c} for r, c in cells],
        }

    def action_from_json(self, msg: dict) -> int | None:
        if "action" in msg:
            action = msg["action"]
            if action in self.get_legal_actions():
                return action
            return None

        piece_idx = msg.get("piece_idx")
        orient_idx = msg.get("orientation")
        row = msg.get("row")
        col = msg.get("col")

        if piece_idx is not None and orient_idx is not None and row is not None and col is not None:
            key = (piece_idx, orient_idx, row, col)
            action = _ACTION_REVERSE.get(key)
            if action is not None and action in self.get_legal_actions():
                return action
        return None

    def make_state_msg(self) -> dict:
        turn_names = {1: "orange", -1: "purple"}
        return {
            "type": "state",
            "board": self.board_to_json(),
            "turn": turn_names[self._turn],
            "pieces_remaining": {
                "orange": sum(self.pieces_available[1]),
                "purple": sum(self.pieces_available[-1]),
            },
            "legal_moves": self.legal_moves_to_json(),
        }

    def __str__(self):
        symbols = {0: '.', 1: 'O', -1: 'P'}
        lines = []
        lines.append("   " + " ".join(f"{c:2d}" for c in range(BOARD_SIZE)))
        for r in range(BOARD_SIZE):
            row_str = " ".join(f" {symbols[int(self.board[r, c])]}" for c in range(BOARD_SIZE))
            lines.append(f"{r:2d} {row_str}")
        lines.append(f"Turn: {'Orange' if self._turn == 1 else 'Purple'}")
        lines.append(f"Pieces left: Orange={sum(self.pieces_available[1])}, Purple={sum(self.pieces_available[-1])}")
        return "\n".join(lines)
