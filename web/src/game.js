/**
 * game.js -- Client-side Pylos game state helpers.
 *
 * The Pylos board is a 4-level pyramid:
 *   Level 0: 4x4  (16 positions)
 *   Level 1: 3x3  ( 9 positions)
 *   Level 2: 2x2  ( 4 positions)
 *   Level 3: 1x1  ( 1 position) -- the apex
 *
 * Coordinate system:
 *   (level, row, col) uniquely identifies each cell.
 */

/** Number of rows/cols per level. */
export const LEVEL_SIZES = [4, 3, 2, 1];

/**
 * Create a fresh empty board.
 * Returns an array of 4 levels, each a 2D array of `null`.
 */
export function createEmptyBoard() {
  return LEVEL_SIZES.map(size => {
    const level = [];
    for (let r = 0; r < size; r++) {
      const row = [];
      for (let c = 0; c < size; c++) {
        row.push(null);
      }
      level.push(row);
    }
    return level;
  });
}

/**
 * Update a local board from the server's state message.
 *
 * @param {Array} board      - Local board (4-level array of null|"white"|"black")
 * @param {Array} serverBoard - Server board (4 arrays of {row, col, player} dicts)
 * @returns {Array} The updated board (same reference, mutated in place)
 */
export function updateBoardFromState(board, serverBoard) {
  // Clear everything first
  for (let level = 0; level < 4; level++) {
    const size = LEVEL_SIZES[level];
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        board[level][r][c] = null;
      }
    }
  }

  // Fill from server data
  for (let level = 0; level < 4; level++) {
    const cells = serverBoard[level];
    for (const cell of cells) {
      board[level][cell.row][cell.col] = cell.player; // "white" | "black"
    }
  }

  return board;
}

/**
 * Convert board coordinates to a 3D world position.
 *
 * @param {number} level - Pyramid level (0-3)
 * @param {number} row   - Row within that level
 * @param {number} col   - Column within that level
 * @returns {{x: number, y: number, z: number}}
 */
export function boardToWorld(level, row, col) {
  const baseSpacing = 1.2;
  const size = LEVEL_SIZES[level];
  const halfSize = (size - 1) / 2;

  const x = (col - halfSize) * baseSpacing;
  const y = level * 1.1 + 0.5; // 0.5 = sphere radius above ground
  const z = (row - halfSize) * baseSpacing;

  return { x, y, z };
}
