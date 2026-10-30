"""
Goal-state enumeration and the classical (Kociemba two-phase) solver.

A dice cube has many solved states: any arrangement where each face holds
1-9 once. A classical solver needs ONE concrete target, so we:

  1. read the physical pieces off the current state (8 corners, 12 edges),
  2. search for assignments of pieces -> slots (with orientation) such that
     no face repeats a value and the move is reachable (twist / flip /
     permutation parity),
  3. relabel each sticker with the face it must end up on, turning the
     problem into an ordinary colour cube, and hand it to Kociemba,
  4. keep the shortest solution over many candidate targets.

Candidate targets are generated "lazy pieces first": pieces that already sit
correctly are kept in place, so the targets tend to be close to the current
state.
"""

import time

import numpy as np

from dicecube.cube import CORNER_FACELETS, EDGE_FACELETS, FACES, CENTER_POS

try:
    import kociemba as _kociemba
except ImportError:  # optional native dependency
    _kociemba = None


SOLVED_COLOURS = "".join(f * 9 for f in FACES)


def kociemba_available():
    return _kociemba is not None


# ----------------------------------------------------------------------
# Pieces
# ----------------------------------------------------------------------

def read_pieces(state):
    """Sticker values of each corner / edge, indexed by the slot it is in."""
    corners = [tuple(int(state[f]) for f in fs) for fs in CORNER_FACELETS]
    edges = [tuple(int(state[f]) for f in fs) for fs in EDGE_FACELETS]
    return corners, edges


def _perm_parity(perm):
    perm = list(perm)
    seen, parity = [False] * len(perm), 0
    for i in range(len(perm)):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        parity ^= (length - 1) & 1
    return parity


# ----------------------------------------------------------------------
# Target search
# ----------------------------------------------------------------------

def iter_targets(state, rng=None, randomize=False, node_budget=200_000):
    """Yield reachable goal assignments as (corner_assign, edge_assign).

    corner_assign[slot] = (piece, ori): the piece currently in slot `piece`
    goes to `slot`, its sticker j landing on facelet (j + ori) % 3.
    edge_assign likewise with ori in {0, 1}.
    """
    rng = rng or np.random.default_rng()
    corners, edges = read_pieces(state)
    face_used = [1 << int(state[f * 9 + CENTER_POS]) for f in range(6)]
    nodes = [0]

    corner_faces = [[f // 9 for f in fs] for fs in CORNER_FACELETS]
    edge_faces = [[f // 9 for f in fs] for fs in EDGE_FACELETS]

    def options(slot, pieces, slot_faces, n_ori, used):
        opts = []
        for p in range(len(pieces)):
            if used[p]:
                continue
            for o in range(n_ori):
                ok = True
                for j, v in enumerate(pieces[p]):
                    if face_used[slot_faces[(j + o) % n_ori]] >> v & 1:
                        ok = False
                        break
                if ok:
                    # Prefer leaving pieces where they are
                    cost = (p != slot) * 2 + (o != 0)
                    opts.append((cost, rng.random() if randomize else 0.0, p, o))
        opts.sort()
        return opts

    def toggle(pieces, slot_faces, p, o, n_ori):
        for j, v in enumerate(pieces[p]):
            face_used[slot_faces[(j + o) % n_ori]] ^= (1 << v)

    corner_assign = [None] * 8
    edge_assign = [None] * 12
    c_used = [False] * 8
    e_used = [False] * 12

    def edges_dfs(slot):
        nodes[0] += 1
        if nodes[0] > node_budget:
            return
        if slot == 12:
            yield None
            return
        for _, _, p, o in options(slot, edges, edge_faces[slot], 2, e_used):
            e_used[p] = True
            toggle(edges, edge_faces[slot], p, o, 2)
            edge_assign[slot] = (p, o)
            yield from edges_dfs(slot + 1)
            toggle(edges, edge_faces[slot], p, o, 2)
            e_used[p] = False

    def corners_dfs(slot):
        nodes[0] += 1
        if nodes[0] > node_budget:
            return
        if slot == 8:
            yield None
            return
        for _, _, p, o in options(slot, corners, corner_faces[slot], 3, c_used):
            c_used[p] = True
            toggle(corners, corner_faces[slot], p, o, 3)
            corner_assign[slot] = (p, o)
            yield from corners_dfs(slot + 1)
            toggle(corners, corner_faces[slot], p, o, 3)
            c_used[p] = False

    for _ in corners_dfs(0):
        if sum(o for _, o in corner_assign) % 3 != 0:
            continue
        c_parity = _perm_parity([p for p, _ in corner_assign])
        for _ in edges_dfs(0):
            if sum(o for _, o in edge_assign) % 2 != 0:
                continue
            if _perm_parity([p for p, _ in edge_assign]) != c_parity:
                continue
            yield list(corner_assign), list(edge_assign)


def target_state(state, assignment):
    """The solved dice cube that an assignment describes."""
    corner_assign, edge_assign = assignment
    out = np.array(state, dtype=np.int8).copy()
    for slot, (p, o) in enumerate(corner_assign):
        for j in range(3):
            out[CORNER_FACELETS[slot][(j + o) % 3]] = state[CORNER_FACELETS[p][j]]
    for slot, (p, o) in enumerate(edge_assign):
        for j in range(2):
            out[EDGE_FACELETS[slot][(j + o) % 2]] = state[EDGE_FACELETS[p][j]]
    return out


def colour_string(assignment):
    """Kociemba facelet string where every sticker is coloured by the face
    it must travel to."""
    corner_assign, edge_assign = assignment
    colours = [FACES[i // 9] for i in range(54)]   # centers + default
    for slot, (p, o) in enumerate(corner_assign):
        for j in range(3):
            dest = CORNER_FACELETS[slot][(j + o) % 3]
            colours[CORNER_FACELETS[p][j]] = FACES[dest // 9]
    for slot, (p, o) in enumerate(edge_assign):
        for j in range(2):
            dest = EDGE_FACELETS[slot][(j + o) % 2]
            colours[EDGE_FACELETS[p][j]] = FACES[dest // 9]
    return "".join(colours)


# ----------------------------------------------------------------------
# Classical solver
# ----------------------------------------------------------------------

def kociemba_solve(state, max_targets=60, time_limit=5.0, seed=0):
    """Shortest Kociemba solution over many candidate goal states.

    Returns dict(moves=[...], target=np.array, targets_tried=int) or None
    when kociemba is not installed / no reachable target exists.
    """
    if _kociemba is None:
        return None
    rng = np.random.default_rng(seed)
    best, tried, start = None, 0, time.time()
    seen = set()

    # First pass keeps pieces in place where possible, later passes shuffle
    # the search order to explore different goal states.
    passes = [False] + [True] * 50
    for randomize in passes:
        per_pass = 0
        for assignment in iter_targets(state, rng=rng, randomize=randomize):
            cs = colour_string(assignment)
            if cs in seen:
                continue
            seen.add(cs)
            tried += 1
            per_pass += 1
            # kociemba returns a non-empty (identity) sequence for an
            # already-solved string, so short-circuit it
            sol = [] if cs == SOLVED_COLOURS else _kociemba.solve(cs).split()
            if best is None or len(sol) < len(best["moves"]):
                best = {"moves": sol, "target": target_state(state, assignment)}
            if tried >= max_targets or time.time() - start > time_limit or per_pass >= 8:
                break
        if tried >= max_targets or time.time() - start > time_limit:
            break
    if best is not None:
        best["targets_tried"] = tried
    return best
