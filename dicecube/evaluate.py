"""
Benchmark the solvers by scramble depth.

    python -m dicecube.evaluate --depths 4 8 12 16 20 30 --n 20 --time 5

Prints solve rate, mean solution length (face turns) and mean time for
MCTS, weighted A* and Kociemba, plus the network's greedy solve rate.
"""

import argparse
import time

import numpy as np

from dicecube import astar, mcts
from dicecube.cube import actions_to_notation, random_solved, scramble
from dicecube.solver import DEFAULT_MODEL, get_model
from dicecube.targets import kociemba_available, kociemba_solve
from dicecube.train import greedy_solve_rate


def run(depths, n, time_limit, seed=0, model_path=DEFAULT_MODEL):
    model = get_model(model_path)
    rng = np.random.default_rng(seed)
    rows = []
    for d in depths:
        stats = {k: {"ok": 0, "len": [], "sec": []} for k in ("mcts", "astar", "kociemba")}
        for _ in range(n):
            state, _ = scramble(random_solved(rng), d, rng)
            if model is not None:
                for name, mod in (("mcts", mcts), ("astar", astar)):
                    r = mod.search(state, model, time_limit=time_limit)
                    if r["solved"]:
                        stats[name]["ok"] += 1
                        stats[name]["len"].append(len(actions_to_notation(r["actions"])))
                    stats[name]["sec"].append(r["seconds"])
            if kociemba_available():
                t = time.time()
                r = kociemba_solve(state, time_limit=time_limit)
                stats["kociemba"]["ok"] += 1
                stats["kociemba"]["len"].append(len(r["moves"]))
                stats["kociemba"]["sec"].append(time.time() - t)
        row = {"depth": d}
        for k, s in stats.items():
            row[k] = {
                "solve_rate": s["ok"] / n,
                "mean_moves": round(float(np.mean(s["len"])), 1) if s["len"] else None,
                "mean_seconds": round(float(np.mean(s["sec"])), 2) if s["sec"] else None,
            }
        rows.append(row)
        print(_fmt(row), flush=True)
    if model is not None:
        print("greedy (no search):", greedy_solve_rate(model, depths=tuple(depths), n=200))
    return rows


def _fmt(row):
    parts = [f"depth {row['depth']:3d}"]
    for k in ("mcts", "astar", "kociemba"):
        r = row[k]
        if r["mean_seconds"] is None:
            continue
        parts.append(f"{k}: {r['solve_rate'] * 100:5.1f}% {r['mean_moves'] or '-':>5} mv {r['mean_seconds']:5.2f}s")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 8, 12, 16, 20, 30])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--time", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.depths, args.n, args.time, args.seed)


if __name__ == "__main__":
    main()
