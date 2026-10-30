"""
Dice Sudoku cube model ("Blind Dice Cube" V1).

A 3x3x3 twisty cube whose stickers are dice faces showing 1-9 pips. Every
center shows 5. The puzzle is solved when every face shows each value 1-9
exactly once -- the *set* of values on a face matters, not their positions,
so (unlike a normal Rubik's cube) there are many solved states.

On the physical cube corners carry odd values {1,3,7,9} and edges carry even
values {2,4,6,8}, so a solved face has each odd value on one corner and each
even value on one edge.

Facelet layout follows Kociemba's URFDLB convention so states can be handed
to a two-phase solver without translation:

             |U1 U2 U3|
             |U4 U5 U6|
             |U7 U8 U9|
    |L1 L2 L3|F1 F2 F3|R1 R2 R3|B1 B2 B3|
    |L4 L5 L6|F4 F5 F6|R4 R5 R6|B4 B5 B6|
    |L7 L8 L9|F7 F8 F9|R7 R8 R9|B7 B8 B9|
             |D1 D2 D3|
             |D4 D5 D6|
             |D7 D8 D9|

  Index = face * 9 + row * 3 + col, faces ordered U, R, F, D, L, B.

Action space (12 quarter turns):
  0-5:  U R F D L B   (clockwise, looking at that face)
  6-11: U' R' F' D' L' B'
"""

import numpy as np


FACES = "URFDLB"
ACTIONS = [f for f in FACES] + [f + "'" for f in FACES]
NUM_ACTIONS = 12

CENTER_VALUE = 5
CORNER_VALUES = (1, 3, 7, 9)
EDGE_VALUES = (2, 4, 6, 8)
PIECE_VALUES = (1, 2, 3, 4, 6, 7, 8, 9)   # values that can appear off-center
FULL_FACE_MASK = sum(1 << v for v in range(1, 10))

CORNER_POS = (0, 2, 6, 8)   # positions within a face
EDGE_POS = (1, 3, 5, 7)
CENTER_POS = 4

# Cubie -> facelet indices. Corner facelets are listed clockwise so a cyclic
# shift of a triple is a legal twist (Kociemba ordering).
U, R, F, D, L, B = (i * 9 for i in range(6))
CORNER_FACELETS = [
    (U + 8, R + 0, F + 2), (U + 6, F + 0, L + 2), (U + 0, L + 0, B + 2), (U + 2, B + 0, R + 2),
    (D + 2, F + 8, R + 6), (D + 0, L + 8, F + 6), (D + 6, B + 8, L + 6), (D + 8, R + 8, B + 6),
]
EDGE_FACELETS = [
    (U + 5, R + 1), (U + 7, F + 1), (U + 3, L + 1), (U + 1, B + 1),
    (D + 5, R + 7), (D + 1, F + 7), (D + 3, L + 7), (D + 7, B + 7),
    (F + 5, R + 3), (F + 3, L + 5), (B + 5, L + 3), (B + 3, R + 5),
]
CORNER_NAMES = ["URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB"]
EDGE_NAMES = ["UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR"]

# Outward normal of each face (x right, y up, z toward the viewer)
FACE_NORMALS = {
    "U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1),
    "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1),
}


# ----------------------------------------------------------------------
# Geometry: derive the move permutations from 3D sticker coordinates
# ----------------------------------------------------------------------

def _facelet_geometry():
    """Return (positions, normals) for all 54 facelets as int arrays."""
    pos, nrm = [], []
    for face in FACES:
        for r in range(3):
            for c in range(3):
                if face == "U":
                    p = (c - 1, 1, r - 1)
                elif face == "R":
                    p = (1, 1 - r, 1 - c)
                elif face == "F":
                    p = (c - 1, 1 - r, 1)
                elif face == "D":
                    p = (c - 1, -1, 1 - r)
                elif face == "L":
                    p = (-1, 1 - r, c - 1)
                else:  # B
                    p = (1 - c, 1 - r, -1)
                pos.append(p)
                nrm.append(FACE_NORMALS[face])
    return np.array(pos), np.array(nrm)


def _rotation_matrix(axis, quarter_turns):
    """Integer rotation matrix about a unit axis by quarter_turns * 90 deg
    (right-hand rule)."""
    ax = np.array(axis)
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    s = [0, 1, 0, -1][quarter_turns % 4]
    c = [1, 0, -1, 0][quarter_turns % 4]
    return (np.eye(3, dtype=int) * c + s * k + (1 - c) * np.outer(ax, ax)).astype(int)


def _build_move_perms():
    pos, nrm = _facelet_geometry()
    lookup = {(tuple(p), tuple(n)): i for i, (p, n) in enumerate(zip(pos, nrm))}
    perms = np.zeros((NUM_ACTIONS, 54), dtype=np.int64)
    for a in range(NUM_ACTIONS):
        face = FACES[a % 6]
        axis = FACE_NORMALS[face]
        # Clockwise when looking at the face == negative rotation about its
        # outward normal.
        rot = _rotation_matrix(axis, -1 if a < 6 else 1)
        perm = np.arange(54)
        for i in range(54):
            if pos[i] @ np.array(axis) == 1:
                j = lookup[(tuple(rot @ pos[i]), tuple(rot @ nrm[i]))]
                perm[j] = i   # sticker at i moves to j
        perms[a] = perm
    return perms


MOVE_PERMS = _build_move_perms()
INVERSE_ACTION = np.array([(a + 6) % 12 for a in range(NUM_ACTIONS)])


# ----------------------------------------------------------------------
# State helpers (states are int8 arrays of shape (54,) or (N, 54))
# ----------------------------------------------------------------------

def apply_action(state, action):
    return state[..., MOVE_PERMS[action]]


def apply_actions(state, actions):
    for a in actions:
        state = apply_action(state, a)
    return state


def all_children(states):
    """(N, 54) -> (N, 12, 54): every state after each of the 12 moves."""
    return states[:, MOVE_PERMS]


def is_solved(states):
    """True where every face holds each of 1..9 exactly once. Works on a
    single state (54,) or a batch (N, 54)."""
    s = np.asarray(states)
    faces = s.reshape(s.shape[:-1] + (6, 9)).astype(np.int64)
    masks = np.bitwise_or.reduce(np.left_shift(1, faces), axis=-1)
    counts_ok = (masks == FULL_FACE_MASK).all(axis=-1)
    return counts_ok


def face_conflicts(states):
    """Number of missing values summed over faces (0 when solved). A cheap
    hand-made heuristic, also handy for visualisation."""
    s = np.asarray(states)
    faces = s.reshape(s.shape[:-1] + (6, 9))
    one_hot = faces[..., None] == np.arange(1, 10)
    present = one_hot.any(axis=-2)
    return (~present).sum(axis=(-1, -2))


def random_solved(rng=None, design="parity"):
    """A random solved dice cube.

    design="parity": corners odd / edges even, as on the physical cube.
    design="any":    any arrangement of 1-9 per face with 5 in the middle.

    Because only the per-face value set matters, assigning each face its own
    permutation always yields a solved (and physically buildable) cube.
    """
    rng = rng or np.random.default_rng()
    state = np.zeros(54, dtype=np.int8)
    for f in range(6):
        base = f * 9
        state[base + CENTER_POS] = CENTER_VALUE
        if design == "parity":
            state[[base + p for p in CORNER_POS]] = rng.permutation(CORNER_VALUES)
            state[[base + p for p in EDGE_POS]] = rng.permutation(EDGE_VALUES)
        else:
            others = [p for p in range(9) if p != CENTER_POS]
            state[[base + p for p in others]] = rng.permutation(PIECE_VALUES)
    return state


def random_scramble_actions(n, rng=None):
    """n random quarter turns with no immediate inverse and no move repeated
    three times in a row (both are wasted moves)."""
    rng = rng or np.random.default_rng()
    seq = []
    while len(seq) < n:
        a = int(rng.integers(NUM_ACTIONS))
        if seq and a == INVERSE_ACTION[seq[-1]]:
            continue
        if len(seq) >= 2 and seq[-1] == a and seq[-2] == a:
            continue
        seq.append(a)
    return seq


def scramble(state, n, rng=None):
    actions = random_scramble_actions(n, rng)
    return apply_actions(state, actions), actions


# ----------------------------------------------------------------------
# Neural network encoding
# ----------------------------------------------------------------------

NON_CENTER = np.array([i for i in range(54) if i % 9 != CENTER_POS])
OBS_DIM = 48 * 8 + 6 * 8   # one-hot stickers + per-face value histogram
_VALUE_INDEX = np.zeros(10, dtype=np.int64)
_VALUE_INDEX[list(PIECE_VALUES)] = np.arange(8)


def to_observation(states):
    """Encode states for the network.

    - 48 non-center stickers one-hot over the 8 possible values (384)
    - per-face histogram of each value (48): makes "is this value already
      on this face?" -- the actual goal condition -- directly visible.
    """
    s = np.asarray(states)
    single = s.ndim == 1
    if single:
        s = s[None]
    n = len(s)
    idx = _VALUE_INDEX[s[:, NON_CENTER]]                      # (n, 48)
    one_hot = np.zeros((n, 48, 8), dtype=np.float32)
    flat = (np.arange(n)[:, None] * 384 + np.arange(48) * 8 + idx).ravel()
    one_hot.reshape(-1)[flat] = 1.0
    # NON_CENTER is grouped face by face (8 stickers each)
    hist = one_hot.reshape(n, 6, 8, 8).sum(axis=2)
    obs = np.concatenate([one_hot.reshape(n, -1), hist.reshape(n, -1)], axis=1)
    return obs[0] if single else obs


# ----------------------------------------------------------------------
# Notation & parsing
# ----------------------------------------------------------------------

def actions_to_notation(actions):
    """Quarter-turn actions -> compact notation, merging X X into X2."""
    out = []
    for a in actions:
        face, prime = FACES[a % 6], a >= 6
        if out and out[-1][0] == face:
            prev = out.pop()
            turns = {"": 1, "'": 3, "2": 2}[prev[1:]] + (3 if prime else 1)
            turns %= 4
            if turns:
                out.append(face + {1: "", 2: "2", 3: "'"}[turns])
            continue
        out.append(face + ("'" if prime else ""))
    return out


def notation_to_actions(moves):
    """'R U2 F'' -> list of quarter-turn actions."""
    if isinstance(moves, str):
        moves = moves.split()
    actions = []
    for m in moves:
        f = FACES.index(m[0])
        suffix = m[1:]
        if suffix == "":
            actions.append(f)
        elif suffix == "'":
            actions.append(f + 6)
        elif suffix == "2":
            actions += [f, f]
        else:
            raise ValueError(f"Bad move: {m}")
    return actions


def parse_state(text):
    """Accept 54 digits (whitespace / separators ignored) in URFDLB order."""
    digits = [int(ch) for ch in str(text) if ch.isdigit()]
    if len(digits) != 54:
        raise ValueError(f"Expected 54 digits, got {len(digits)}")
    return np.array(digits, dtype=np.int8)


def state_to_string(state):
    return " ".join("".join(str(int(v)) for v in state[f * 9:(f + 1) * 9]) for f in range(6))


def validate_state(state):
    """Return a list of human-readable problems (empty if the input looks
    like a real dice cube)."""
    problems = []
    state = np.asarray(state)
    if state.shape != (54,):
        return [f"State must have 54 stickers, got {state.shape}"]
    for f in range(6):
        if state[f * 9 + CENTER_POS] != CENTER_VALUE:
            problems.append(f"{FACES[f]} center must be 5")
    for v in PIECE_VALUES:
        n = int((state == v).sum())
        if n != 6:
            problems.append(f"Value {v} appears {n} times (expected 6, once per face)")
    bad = [i for i in NON_CENTER if state[i] not in PIECE_VALUES]
    if bad:
        problems.append(f"Off-center stickers must be 1-4 or 6-9 (bad at {bad[:5]})")
    return problems
