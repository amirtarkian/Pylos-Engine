"""
Monte Carlo Tree Search for the dice cube (single-player AlphaZero / DeepCube).

Differences from the two-player engine/mcts.py:

  * No opponent, so no value negation on backup.
  * The network predicts cost-to-go c(s) (moves to ANY solved state). An
    edge's value is Q(s, a) = -(1 + best cost found below child a), and we
    back up the *best* (max) value rather than the mean -- one good line is
    all a puzzle needs (as in DeepCube's MCTS).
  * Every expansion checks all 12 children for "solved" immediately, so a
    solution is found the moment the search touches its parent.
  * Leaves are collected in batches with virtual loss so the network runs
    one forward pass per batch (like engine.mcts.batched_search).

Selection uses PUCT:
    a* = argmax_a  Q(s,a) - VL(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
"""

import time

import numpy as np

from dicecube.cube import INVERSE_ACTION, NUM_ACTIONS, all_children, is_solved


class Node:
    __slots__ = ("state", "parent", "action", "depth", "children", "child_states",
                 "P", "N", "W", "VL", "cost")

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = 0 if parent is None else parent.depth + 1
        self.children = None        # list[Node | None] once expanded
        self.child_states = None
        self.P = self.N = self.W = self.VL = None
        self.cost = None            # network cost-to-go estimate

    @property
    def expanded(self):
        return self.children is not None


def _path(node):
    actions = []
    while node.parent is not None:
        actions.append(int(node.action))
        node = node.parent
    return actions[::-1]


def _expand(node, cost, policy):
    node.cost = float(cost)
    node.child_states = all_children(node.state[None])[0]
    node.children = [None] * NUM_ACTIONS
    p = policy.astype(np.float64).copy()
    if node.parent is not None:
        p[INVERSE_ACTION[node.action]] = 0.0   # never undo the last move
    node.P = p / p.sum() if p.sum() > 0 else np.full(NUM_ACTIONS, 1 / NUM_ACTIONS)
    node.N = np.zeros(NUM_ACTIONS)
    # First-play urgency: an unvisited move is assumed as good as this node
    node.W = np.full(NUM_ACTIONS, -(1.0 + node.cost))
    node.VL = np.zeros(NUM_ACTIONS)


def search(state, model, max_nodes=20000, time_limit=10.0, c_puct=1.0, batch_size=32,
           virtual_loss=2.0, on_progress=None):
    """Find a move sequence from `state` to any solved state.

    Returns dict(actions, solved, nodes, seconds, root_cost, best_cost_trace).
    """
    start = time.time()
    state = np.asarray(state, dtype=np.int8)
    if is_solved(state):
        return {"actions": [], "solved": True, "nodes": 0, "seconds": 0.0,
                "root_cost": 0.0, "tree": []}

    root = Node(state)
    cost, policy = model.batched_inference(state[None])
    _expand(root, cost[0], policy[0])
    nodes = 1

    def check_children(node):
        hits = np.nonzero(is_solved(node.child_states))[0]
        return int(hits[0]) if len(hits) else None

    hit = check_children(root)
    if hit is not None:
        return {"actions": [hit], "solved": True, "nodes": 1,
                "seconds": time.time() - start, "root_cost": root.cost, "tree": []}

    while nodes < max_nodes and time.time() - start < time_limit:
        # ---- Phase 1: select a batch of leaves with virtual loss ----
        leaves = []          # (parent, action, path)
        paths = []           # every selected path, to undo virtual loss
        claimed = set()
        for _ in range(batch_size):
            node, path = root, []
            while True:
                sqrt_n = np.sqrt(node.N.sum() + 1)
                score = node.W - node.VL + c_puct * node.P * sqrt_n / (1 + node.N)
                score[node.P == 0] = -np.inf
                a = int(np.argmax(score))
                path.append((node, a))
                child = node.children[a]
                if child is None:
                    break
                node = child
            key = (id(node), a)
            for n_, a_ in path:
                n_.VL[a_] += virtual_loss
            paths.append(path)
            if key in claimed:
                continue
            claimed.add(key)
            leaves.append((node, a, path))

        # ---- Phase 2: evaluate all new leaves in one forward pass ----
        new_states = np.stack([p.child_states[a] for p, a, _ in leaves])
        costs, policies = model.batched_inference(new_states)

        found = None
        for (parent, a, path), c, pol in zip(leaves, costs, policies):
            child = Node(parent.child_states[a], parent, a)
            _expand(child, c, pol)
            parent.children[a] = child
            nodes += 1
            hit = check_children(child)
            if hit is not None and found is None:
                found = _path(child) + [hit]

            # ---- Phase 3: back up the best (lowest) cost ----
            best = child.cost
            node = child
            while node.parent is not None:
                par, act = node.parent, node.action
                par.N[act] += 1
                par.W[act] = max(par.W[act], -(1.0 + best))
                best = -par.W.max()
                node = par

        for path in paths:
            for n_, a_ in path:
                n_.VL[a_] -= virtual_loss

        if on_progress is not None:
            on_progress(nodes, -root.W.max())
        if found is not None:
            return {"actions": found, "solved": True, "nodes": nodes,
                    "seconds": time.time() - start, "root_cost": root.cost,
                    "tree": root_summary(root)}

    return {"actions": [], "solved": False, "nodes": nodes,
            "seconds": time.time() - start, "root_cost": root.cost,
            "tree": root_summary(root)}


def root_summary(root):
    """Per-move visit counts / values at the root, for the UI heatmap."""
    from dicecube.cube import ACTIONS
    total = root.N.sum() or 1
    return [{"move": ACTIONS[a], "visits": int(root.N[a]), "share": float(root.N[a] / total),
             "prior": float(root.P[a]), "value": float(root.W[a])}
            for a in range(NUM_ACTIONS)]
