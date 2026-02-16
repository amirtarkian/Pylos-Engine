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

import numpy as np
from copy import deepcopy


# Level dimensions: level i has side length (4 - i)
LEVEL_SIZES = [4, 3, 2, 1]


class PylosGame:
    """Pylos game engine compatible with AlphaZero MCTS framework."""

    def __init__(self):
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
        self.observation_shape = (32,)

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
        self.turn = 1  # 1 = white, -1 = black
        self.reserves = {1: 15, -1: 15}
        self.last_move = None  # (level, row, col) of the last placed/raised piece
        self.actions_stack = []  # stack for undo

    # ------------------------------------------------------------------
    # Board queries
    # ------------------------------------------------------------------

    def is_supported(self, level, r, c):
        """Check if position (level, r, c) is supported from below.

        Level 0 positions are always supported (on the table).
        Higher level positions need all 4 pieces below them:
          (level-1, r, c), (level-1, r+1, c), (level-1, r, c+1), (level-1, r+1, c+1)
        """
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
        """Check if any piece on level+1 uses this position as support.

        A piece at (level, r, c) supports positions on level+1 at:
          (level+1, r-1, c-1), (level+1, r-1, c), (level+1, r, c-1), (level+1, r, c)
        but only those that exist within bounds.
        """
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
        """Check if any 2x2 square containing (level, r, c) is all same color.

        The piece at (level, r, c) can be part of up to 4 different 2x2 squares.
        """
        board_level = self.board[level]
        size = LEVEL_SIZES[level]
        color = board_level[r, c]
        if color == 0:
            return False

        # Check all possible 2x2 squares that include (r, c)
        for sr in range(max(0, r - 1), min(size - 1, r) + 1):
            for sc in range(max(0, c - 1), min(size - 1, c) + 1):
                # Square with top-left at (sr, sc)
                if (
                    board_level[sr, sc] == color
                    and board_level[sr + 1, sc] == color
                    and board_level[sr, sc + 1] == color
                    and board_level[sr + 1, sc + 1] == color
                ):
                    return True
        return False

    def check_line(self, level, r, c):
        """Check for line formations containing (level, r, c).

        Level 0: 4-in-a-row (horizontal or vertical)
        Level 1: 3-in-a-row (horizontal or vertical)
        Other levels: no line formations possible.
        """
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

        # Check row (horizontal)
        if size >= n:
            row_vals = board_level[r, :]
            count = np.sum(row_vals == color)
            if count >= n:
                return True

        # Check column (vertical)
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

    # ------------------------------------------------------------------
    # Removable pieces
    # ------------------------------------------------------------------

    def get_removable_pieces(self):
        """Get list of current player's pieces that can be removed.

        A piece can be removed if:
        - It belongs to the current player
        - It has no piece on top (not supporting anything)
        """
        removable = []
        for level in range(4):
            size = LEVEL_SIZES[level]
            for r in range(size):
                for c in range(size):
                    if (
                        self.board[level][r, c] == self.turn
                        and not self.piece_has_top(level, r, c)
                    ):
                        removable.append((level, r, c))
        return removable

    # ------------------------------------------------------------------
    # Actions: place, raise, remove
    # ------------------------------------------------------------------

    def place(self, level, r, c):
        """Place a reserve sphere at (level, r, c).

        Returns True if successful, False if illegal.
        """
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
        """Raise own piece from (sl, sr, sc) to (dl, dr, dc).

        The destination must be on a strictly higher level.
        The source piece must belong to current player and have no top.
        IMPORTANT: temporarily remove source piece before checking
        destination support (the source might be supporting the dest).

        Returns True if successful, False if illegal.
        """
        # Must be own piece
        if self.board[sl][sr, sc] != self.turn:
            return False
        # Must not have a top
        if self.piece_has_top(sl, sr, sc):
            return False
        # Destination must be strictly higher
        if dl <= sl:
            return False
        # Destination must be empty
        if self.board[dl][dr, dc] != 0:
            return False

        # Temporarily remove source to check destination support
        self.board[sl][sr, sc] = 0
        supported = self.is_supported(dl, dr, dc)
        if not supported:
            # Restore source piece
            self.board[sl][sr, sc] = self.turn
            return False

        # Place at destination
        self.board[dl][dr, dc] = self.turn
        self.last_move = (dl, dr, dc)
        return True

    def remove(self, level, r, c):
        """Remove own unsupported piece, returning it to reserves.

        Returns True if successful, False if illegal.
        """
        if self.board[level][r, c] != self.turn:
            return False
        if self.piece_has_top(level, r, c):
            return False

        self.board[level][r, c] = 0
        self.reserves[self.turn] += 1
        return True

    # ------------------------------------------------------------------
    # AI removal heuristic
    # ------------------------------------------------------------------

    def _do_ai_removal(self):
        """Greedy removal of up to 2 pieces after forming a square/line.

        Removes pieces from lowest level first (to maximize strategic value).
        """
        for _ in range(2):
            removable = self.get_removable_pieces()
            if not removable:
                break
            # Sort by level ascending (lowest first), then row, then col
            removable.sort(key=lambda x: (x[0], x[1], x[2]))
            piece = removable[0]
            self.remove(*piece)

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------

    def get_legal_actions(self):
        """Get list of valid action indices.

        Placement actions (0-29): place reserve at empty, supported position.
        Raise actions (30-302): raise own unsupported piece to higher,
            empty, supported position (temporarily removing src for
            support check).
        """
        legal = []

        # Placement actions (indices 0-29)
        if self.reserves[self.turn] > 0:
            for idx in range(30):
                level, r, c = self.index_to_coords[idx]
                if self.board[level][r, c] == 0 and self.is_supported(level, r, c):
                    legal.append(idx)

        # Raise actions (indices 30-302)
        for action_idx in range(30, 303):
            src_idx, dst_idx = self.raise_action_to_pair[action_idx]
            sl, sr, sc = self.index_to_coords[src_idx]
            dl, dr, dc = self.index_to_coords[dst_idx]

            # Must be own piece, no top
            if self.board[sl][sr, sc] != self.turn:
                continue
            if self.piece_has_top(sl, sr, sc):
                continue
            # Destination must be empty
            if self.board[dl][dr, dc] != 0:
                continue

            # Temporarily remove source to check destination support
            self.board[sl][sr, sc] = 0
            supported = self.is_supported(dl, dr, dc)
            self.board[sl][sr, sc] = self.turn  # restore

            if supported:
                legal.append(action_idx)

        return legal

    # ------------------------------------------------------------------
    # Step / Undo (AlphaZero interface)
    # ------------------------------------------------------------------

    def step(self, action):
        """Execute an action (placement or raise).

        After the action, checks for formation and does greedy removal.
        Pushes state to actions_stack for undo. Switches turn.
        """
        if action < 30:
            # Placement action
            level, r, c = self.index_to_coords[action]
            success = self.place(level, r, c)
            if not success:
                raise ValueError(f"Illegal placement action {action} at ({level},{r},{c})")
            self.actions_stack.append(('place', action, level, r, c))
        else:
            # Raise action
            src_idx, dst_idx = self.raise_action_to_pair[action]
            sl, sr, sc = self.index_to_coords[src_idx]
            dl, dr, dc = self.index_to_coords[dst_idx]
            success = self.raise_piece(sl, sr, sc, dl, dr, dc)
            if not success:
                raise ValueError(
                    f"Illegal raise action {action}: ({sl},{sr},{sc})->({dl},{dr},{dc})"
                )
            self.actions_stack.append(('raise', action, src_idx, dst_idx, sl, sr, sc, dl, dr, dc))

        # Check for formation and do greedy removal
        if self.check_for_removal():
            self._do_ai_removal()

        # Switch turn
        self.turn *= -1

    def undo_last_action(self):
        """Undo the last action (pop from actions_stack).

        Reverses the placement or raise and switches turn back.
        NOTE: Does NOT undo removals -- known simplification for MCTS.
        """
        if not self.actions_stack:
            return

        entry = self.actions_stack.pop()

        # Switch turn back first
        self.turn *= -1

        if entry[0] == 'place':
            _, action, level, r, c = entry
            # Remove the placed piece and return to reserves
            self.board[level][r, c] = 0
            self.reserves[self.turn] += 1
        elif entry[0] == 'raise':
            _, action, src_idx, dst_idx, sl, sr, sc, dl, dr, dc = entry
            # Move piece back from destination to source
            self.board[dl][dr, dc] = 0
            self.board[sl][sr, sc] = self.turn

        self.last_move = None

    # ------------------------------------------------------------------
    # Win / result
    # ------------------------------------------------------------------

    def has_move(self):
        """Check if current player has any legal moves."""
        return len(self.get_legal_actions()) > 0

    def get_result(self):
        """Get game result.

        Returns:
            1 if white wins (placed apex or black has no moves)
            -1 if black wins (placed apex or white has no moves)
            None if game is ongoing
        """
        # Check if apex is filled
        if self.top_filled():
            return self.board[3][0, 0]

        # Check if current player has no moves
        if not self.has_move():
            return -self.turn  # current player loses

        return None

    def get_first_person_result(self):
        """Get result from current player's perspective.

        Returns result * self.turn, or None if game ongoing.
        """
        result = self.get_result()
        if result is None:
            return None
        return result * self.turn

    @staticmethod
    def swap_result(result):
        """Swap result perspective."""
        return -result

    # ------------------------------------------------------------------
    # Observation (AlphaZero interface)
    # ------------------------------------------------------------------

    def to_observation(self):
        """Convert game state to observation tensor.

        Returns float32 array of shape (32,):
          - 30 board cells: +1 for current player, -1 for opponent, 0 for empty
          - reserves[self.turn] / 15
          - reserves[-self.turn] / 15
        """
        obs = np.zeros(32, dtype=np.float32)

        for idx in range(30):
            level, r, c = self.index_to_coords[idx]
            cell = self.board[level][r, c]
            if cell != 0:
                obs[idx] = float(cell * self.turn)

        obs[30] = self.reserves[self.turn] / 15.0
        obs[31] = self.reserves[-self.turn] / 15.0

        return obs

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self):
        """Text display of the board."""
        lines = []
        symbols = {0: '.', 1: 'W', -1: 'B'}
        for level in range(4):
            size = LEVEL_SIZES[level]
            indent = '  ' * level
            lines.append(f"Level {level} ({size}x{size}):")
            for r in range(size):
                row_str = ' '.join(symbols[int(self.board[level][r, c])] for c in range(size))
                lines.append(f"  {indent}{row_str}")
        lines.append(f"Turn: {'White' if self.turn == 1 else 'Black'}")
        lines.append(f"Reserves: White={self.reserves[1]}, Black={self.reserves[-1]}")
        return '\n'.join(lines)
