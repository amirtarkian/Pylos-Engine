"""
Monte Carlo Tree Search for the dice cube (single-player AlphaZero / DeepCube).

Differences from the two-player engine/mcts.py:

  * No opponent, so no value negation on backup.
  * The network predicts cost-to-go c(s) (moves to ANY solved state). An
    edge's value is Q(s, a) = -(1 + best cost found below child a), and a
    node backs up the *best* (max) of its edges rather than the mean -- one
    good line is all a puzzle needs (as in DeepCube's MCTS).
  * Expanding a leaf scores it and all 12 children in one batch, so every
    edge starts from a one-step lookahead instead of a guess.
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


def _expand(node, policy, child_costs):
    """Attach priors and one-step-lookahead values for all 12 moves."""
    node.children = [None] * NUM_ACTIONS
    p = policy.astype(np.float64).copy()
    node.W = -(1.0 + child_costs.astype(np.float64))
    if node.parent is not None:
        undo = INVERSE_ACTION[node.action]
        p[undo] = 0.0                  # never undo the last move
        node.W[undo] = -np.inf
    node.P = p / p.sum() if p.sum() > 0 else np.full(NUM_ACTIONS, 1 / NUM_ACTIONS)
    node.N = np.zeros(NUM_ACTIONS)
    node.VL = np.zeros(NUM_ACTIONS)


def _evaluate(model, states):
    """Score each state AND its 12 children in one forward pass.

    Returns (costs (B,), policies (B, 12), child_states (B, 12, 54),
    child_costs (B, 12)) with solved children costed 0.
    """
    children = all_children(states)
    batch = np.concatenate([states, children.reshape(-1, 54)])
    costs, policies = model.batched_inference(batch)
    b = len(states)
    child_costs = costs[b:].reshape(b, NUM_ACTIONS)
    child_costs[is_solved(children.reshape(-1, 54)).reshape(b, NUM_ACTIONS)] = 0.0
    return costs[:b], policies[:b], children, child_costs


def search(state, model, max_nodes=200_000, time_limit=10.0, c_puct=0.5, batch_size=32,
           virtual_loss=0.5, on_progress=None):
    """Find a move sequence from `state` to any solved state.

    Each expansion scores the leaf and all 12 of its children in the same
    batched forward pass, so W(s, a) starts as the one-step lookahead
    -(1 + cost(child)) and is refined as the tree grows below it. A node's
    value is the best of its moves: max_a W(s, a).

    Returns dict(actions, solved, nodes, seconds, root_cost, tree); `nodes`
    counts states scored by the network.
    """
    start = time.time()
    state = np.asarray(state, dtype=np.int8)
    if is_solved(state):
        return {"actions": [], "solved": True, "nodes": 0, "seconds": 0.0,
                "root_cost": 0.0, "tree": []}

    root = Node(state)
    costs, policies, children, child_costs = _evaluate(model, state[None])
    root.cost = float(costs[0])
    root.child_states = children[0]
    _expand(root, policies[0], child_costs[0])
    nodes = 1 + NUM_ACTIONS

    hits = np.nonzero(is_solved(root.child_states))[0]
    if len(hits):
        return {"actions": [int(hits[0])], "solved": True, "nodes": nodes,
                "seconds": time.time() - start, "root_cost": root.cost,
                "tree": root_summary(root)}

    while nodes < max_nodes and time.time() - start < time_limit:
        # ---- Phase 1: select a batch of leaves with virtual loss ----
        leaves = []          # (parent, action)
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
            for n_, a_ in path:
                n_.VL[a_] += virtual_loss
            paths.append(path)
            if (id(node), a) not in claimed:
                claimed.add((id(node), a))
                leaves.append((node, a))

        # ---- Phase 2: expand all new leaves with one forward pass ----
        leaf_states = np.stack([p.child_states[a] for p, a in leaves])
        costs, policies, children, child_costs = _evaluate(model, leaf_states)
        nodes += len(leaves) * (1 + NUM_ACTIONS)

        found = None
        for i, (parent, a) in enumerate(leaves):
            child = Node(leaf_states[i], parent, a)
            child.cost = float(costs[i])
            child.child_states = children[i]
            _expand(child, policies[i], child_costs[i])
            parent.children[a] = child
            if found is None:
                hits = np.nonzero(is_solved(children[i]))[0]
                if len(hits):
                    found = _path(child) + [int(hits[0])]

            # ---- Phase 3: back up the best (lowest) cost ----
            node = child
            best = -node.W.max()
            while node.parent is not None:
                par, act = node.parent, node.action
                par.N[act] += 1
                par.W[act] = -(1.0 + best)
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
