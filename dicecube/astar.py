"""
Batch-weighted A* (DeepCubeA) guided by the same cost-to-go network.

    f(n) = weight * g(n) + h(n)

where g is moves made so far and h is the network's cost-to-go estimate.
Each iteration pops `batch_size` nodes and scores all their children in one
forward pass. weight < 1 trades optimality for speed.
"""

import heapq
import time

import numpy as np

from dicecube.cube import NUM_ACTIONS, all_children, is_solved


def search(state, model, weight=0.6, batch_size=64, max_nodes=200_000, time_limit=10.0):
    start = time.time()
    state = np.asarray(state, dtype=np.int8)
    if is_solved(state):
        return {"actions": [], "solved": True, "nodes": 0, "seconds": 0.0}

    root_key = state.tobytes()
    g = {root_key: 0}
    parent = {root_key: (None, None)}
    states = {root_key: state}
    h0, _ = model.batched_inference(state[None])
    heap = [(float(h0[0]), 0, root_key)]
    counter, generated = 1, 1

    def trace(key):
        actions = []
        while parent[key][0] is not None:
            key, a = parent[key]
            actions.append(a)
        return actions[::-1]

    while heap and generated < max_nodes and time.time() - start < time_limit:
        batch = []
        seen = set()
        while heap and len(batch) < batch_size:
            _, _, key = heapq.heappop(heap)
            if key not in seen:
                seen.add(key)
                batch.append(key)
        parents = np.stack([states[k] for k in batch])
        children = all_children(parents)                       # (b, 12, 54)
        flat = children.reshape(-1, 54)
        solved = is_solved(flat)

        new_keys, new_states, new_g = [], [], []
        for i, pk in enumerate(batch):
            for a in range(NUM_ACTIONS):
                child = children[i, a]
                ck = child.tobytes()
                cg = g[pk] + 1
                if ck in g and g[ck] <= cg:
                    continue
                g[ck] = cg
                parent[ck] = (pk, a)
                states[ck] = child
                if solved[i * NUM_ACTIONS + a]:
                    return {"actions": trace(ck), "solved": True, "nodes": generated,
                            "seconds": time.time() - start}
                new_keys.append(ck)
                new_states.append(child)
                new_g.append(cg)
        if not new_keys:
            continue
        h, _ = model.batched_inference(np.stack(new_states))
        generated += len(new_keys)
        for ck, cg, hc in zip(new_keys, new_g, h):
            heapq.heappush(heap, (weight * cg + float(hc), counter, ck))
            counter += 1

    return {"actions": [], "solved": False, "nodes": generated, "seconds": time.time() - start}
