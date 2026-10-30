"""
High-level solver: run the neural searches and the classical solver on a
dice-cube state and package the results for the UI / CLI.

    python -m dicecube.solver "394 ... (54 digits, URFDLB order)"
    python -m dicecube.solver --scramble 20
"""

import argparse
import os

import numpy as np

from dicecube import astar, mcts
from dicecube.cube import (
    actions_to_notation, apply_actions, face_conflicts, is_solved, notation_to_actions,
    parse_state, random_solved, scramble, state_to_string, validate_state,
)
from dicecube.targets import kociemba_available, kociemba_solve

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(PACKAGE_DIR, "checkpoints", "latest.pth")

_model_cache = {}


def get_model(path=DEFAULT_MODEL):
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    cached = _model_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    from dicecube.models import load_model
    model, ckpt = load_model(path, device="cpu")
    model.meta = {"step": ckpt.get("step"), "max_depth": ckpt.get("max_depth")}
    _model_cache[path] = (mtime, model)
    return model


def simplify(actions):
    """Quarter-turn actions with cancellations merged away."""
    return notation_to_actions(actions_to_notation(actions))


def _package(state, actions, model, extra):
    actions = simplify(actions)
    trace_states = [state]
    for a in actions:
        trace_states.append(apply_actions(trace_states[-1], [a]))
    trace = np.stack(trace_states)
    out = {
        "moves": actions_to_notation(actions),
        "actions": [int(a) for a in actions],
        "length_htm": len(actions_to_notation(actions)),
        "length_qtm": len(actions),
        "solved": bool(is_solved(trace[-1])),
        "final_state": state_to_string(trace[-1]),
        "missing_trace": [int(x) for x in face_conflicts(trace)],
    }
    if model is not None:
        costs, _ = model.batched_inference(trace)
        out["cost_trace"] = [round(float(c), 2) for c in costs]
    out.update(extra)
    return out


def solve(state, methods=("mcts", "astar", "kociemba"), time_limit=8.0, model_path=DEFAULT_MODEL):
    state = np.asarray(state, dtype=np.int8)
    problems = validate_state(state)
    result = {"state": state_to_string(state), "problems": problems, "results": {},
              "already_solved": bool(is_solved(state))}
    if problems:
        return result

    model = get_model(model_path)
    if model is not None:
        cost, policy = model.batched_inference(state[None])
        result["network"] = {"cost": round(float(cost[0]), 2),
                             "policy": [round(float(p), 4) for p in policy[0]],
                             **getattr(model, "meta", {})}

    if "mcts" in methods and model is not None:
        r = mcts.search(state, model, time_limit=time_limit)
        result["results"]["mcts"] = _package(state, r["actions"], model, {
            "found": r["solved"], "nodes": r["nodes"], "seconds": round(r["seconds"], 2),
            "root_moves": r.get("tree", [])})
    if "astar" in methods and model is not None:
        r = astar.search(state, model, time_limit=time_limit)
        result["results"]["astar"] = _package(state, r["actions"], model, {
            "found": r["solved"], "nodes": r["nodes"], "seconds": round(r["seconds"], 2)})
    if "kociemba" in methods and kociemba_available():
        import time
        t = time.time()
        r = kociemba_solve(state, time_limit=min(time_limit, 5.0))
        if r is not None:
            result["results"]["kociemba"] = _package(
                state, notation_to_actions(r["moves"]), model, {
                    "found": True, "targets_tried": r["targets_tried"],
                    "seconds": round(time.time() - t, 2),
                    "target": state_to_string(r["target"])})
        else:
            result["results"]["kociemba"] = {"found": False, "solved": False, "moves": [],
                                              "actions": [], "error": "No reachable goal state"}

    found = {k: v for k, v in result["results"].items() if v.get("found") and v.get("solved")}
    if found:
        result["best"] = min(found, key=lambda k: found[k]["length_htm"])
    return result


def main():
    parser = argparse.ArgumentParser(description="Solve a dice Sudoku cube")
    parser.add_argument("state", nargs="?", help="54 digits in URFDLB order")
    parser.add_argument("--scramble", type=int, default=None, help="solve a random N-move scramble")
    parser.add_argument("--time", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.scramble is not None:
        rng = np.random.default_rng(args.seed)
        state, acts = scramble(random_solved(rng), args.scramble, rng)
        print("Scramble:", " ".join(actions_to_notation(acts)))
    elif args.state:
        state = parse_state(args.state)
    else:
        parser.error("give a state or --scramble N")

    print("State:   ", state_to_string(state))
    res = solve(state, time_limit=args.time)
    for p in res["problems"]:
        print("Problem:", p)
    if "network" in res:
        print(f"Network estimates {res['network']['cost']} moves to go")
    for name, r in res["results"].items():
        status = "solved" if r.get("found") and r.get("solved") else "not found"
        extra = f"{r.get('nodes', r.get('targets_tried', ''))} nodes/targets, {r.get('seconds', '')}s"
        print(f"{name:9s} {status:9s} {r.get('length_htm', 0):3d} moves  ({extra})")
        if r.get("moves"):
            print("          " + " ".join(r["moves"]))
    if "best" in res:
        print("Best:", res["best"])


if __name__ == "__main__":
    main()
