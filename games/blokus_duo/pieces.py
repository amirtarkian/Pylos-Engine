"""
Blokus Duo polyomino piece definitions.

Each piece is defined as a list of (row, col) offsets from the origin (0,0).
All 21 standard Blokus pieces from monomino (1 cell) through pentominoes (5 cells).
"""

import numpy as np

# Raw piece definitions: list of (row, col) cells
RAW_PIECES = {
    # Monomino (1)
    "I1": [(0, 0)],

    # Domino (1)
    "I2": [(0, 0), (0, 1)],

    # Triominoes (2)
    "I3": [(0, 0), (0, 1), (0, 2)],
    "V3": [(0, 0), (1, 0), (1, 1)],

    # Tetrominoes (5)
    "I4": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "O4": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "T4": [(0, 0), (0, 1), (0, 2), (1, 1)],
    "S4": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "L4": [(0, 0), (0, 1), (0, 2), (1, 0)],

    # Pentominoes (12)
    "F5": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "I5": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    "L5": [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)],
    "N5": [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)],
    "P5": [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    "T5": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    "U5": [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)],
    "V5": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    "W5": [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
    "X5": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    "Y5": [(0, 0), (1, 0), (1, 1), (2, 0), (3, 0)],
    "Z5": [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
}


def _normalize(cells):
    """Normalize piece cells so the minimum row and col are 0."""
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    normalized = tuple(sorted((r - min_r, c - min_c) for r, c in cells))
    return normalized


def _rotate_90(cells):
    """Rotate cells 90 degrees clockwise: (r, c) -> (c, -r)."""
    return [(c, -r) for r, c in cells]


def _flip(cells):
    """Flip cells horizontally: (r, c) -> (r, -c)."""
    return [(r, -c) for r, c in cells]


def _get_unique_orientations(cells):
    """Get all unique orientations (rotations + flips) of a piece."""
    orientations = set()
    current = list(cells)

    for _ in range(4):
        orientations.add(_normalize(current))
        orientations.add(_normalize(_flip(current)))
        current = _rotate_90(current)

    return [list(o) for o in sorted(orientations)]


def build_piece_table():
    """Build the complete piece table with all unique orientations.

    Returns:
        pieces: list of (piece_name, orientations) where orientations is
                a list of lists of (row, col) tuples. Each orientation is
                normalized so min row/col is 0.
        piece_names: list of piece name strings
    """
    pieces = []
    piece_names = []
    for name, cells in RAW_PIECES.items():
        orientations = _get_unique_orientations(cells)
        pieces.append((name, orientations))
        piece_names.append(name)
    return pieces, piece_names


# Pre-computed piece table
PIECES, PIECE_NAMES = build_piece_table()
NUM_PIECES = len(PIECES)  # 21

# Total unique orientations across all pieces
TOTAL_ORIENTATIONS = sum(len(orients) for _, orients in PIECES)
