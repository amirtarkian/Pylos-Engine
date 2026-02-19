"""Robust ELO evaluator: each checkpoint plays against a pool of anchor models.

Instead of chained vs-previous comparisons (where errors compound), this script:
1. Selects ~10 anchor checkpoints spread across training
2. Runs a round-robin tournament among anchors to establish their ELOs
3. Evaluates every checkpoint against all anchors
4. Computes ELO via maximum likelihood estimation from all matchups

Usage:
    python engine/elo_evaluator.py [--games 20] [--search-iters 32] [--anchor-spacing 10000]
"""

import sys
import os
import json
import math
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from evaluate import evaluate_vs_model

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints_v4")
MANIFEST = os.path.join(CKPT_DIR, "manifest.json")
BASE_ELO = 1000


def mle_elo(matchups, initial_guess=1200, tol=0.5):
    """Compute ELO via maximum likelihood estimation.

    matchups: list of (opponent_elo, wins, losses)
    Returns the ELO that maximizes the likelihood of observed results.
    """
    lo, hi = 0, 3000

    for _ in range(100):
        mid = (lo + hi) / 2
        # Derivative of log-likelihood
        grad = 0.0
        for opp_elo, wins, losses in matchups:
            expected = 1.0 / (1.0 + 10 ** ((opp_elo - mid) / 400))
            total = wins + losses
            if total == 0:
                continue
            # d/dR of log-likelihood
            grad += (wins - total * expected) * math.log(10) / 400

        if grad > 0:
            lo = mid
        else:
            hi = mid

        if hi - lo < tol:
            break

    return round((lo + hi) / 2)


def select_anchors(checkpoints, spacing):
    """Select anchor checkpoints spread evenly across training."""
    sorted_cps = sorted(checkpoints, key=lambda c: c["step"])
    if len(sorted_cps) <= 10:
        return sorted_cps

    anchors = [sorted_cps[0]]  # always include first
    last_step = sorted_cps[0]["step"]

    for cp in sorted_cps[1:]:
        if cp["step"] - last_step >= spacing:
            anchors.append(cp)
            last_step = cp["step"]

    # Always include the latest
    if anchors[-1]["step"] != sorted_cps[-1]["step"]:
        anchors.append(sorted_cps[-1])

    return anchors


def run_round_robin(anchors, games_per_pair, search_iters):
    """Run round-robin tournament among anchor models to establish ELOs."""
    n = len(anchors)
    # results[i][j] = win rate of anchor i vs anchor j
    results = {}

    total_pairs = n * (n - 1) // 2
    pair_num = 0

    for i in range(n):
        for j in range(i + 1, n):
            pair_num += 1
            path_i = os.path.join(CKPT_DIR, anchors[i]["file"])
            path_j = os.path.join(CKPT_DIR, anchors[j]["file"])

            print(f"  Round-robin {pair_num}/{total_pairs}: "
                  f"step {anchors[i]['step']} vs {anchors[j]['step']}...", end=" ", flush=True)

            wr = evaluate_vs_model(path_i, path_j,
                                   num_games=games_per_pair,
                                   search_iterations=search_iters)
            results[(i, j)] = wr
            results[(j, i)] = 1.0 - wr
            print(f"{wr:.1%}")

    # Iterative ELO computation: start all at BASE_ELO, update until stable
    elos = [float(BASE_ELO)] * n

    for iteration in range(50):
        new_elos = []
        for i in range(n):
            matchups = []
            for j in range(n):
                if i == j:
                    continue
                wr = results.get((i, j))
                if wr is None:
                    continue
                wins = wr * games_per_pair
                losses = (1.0 - wr) * games_per_pair
                matchups.append((elos[j], wins, losses))

            if matchups:
                new_elos.append(mle_elo(matchups, initial_guess=elos[i]))
            else:
                new_elos.append(elos[i])

        # Normalize so mean stays at BASE_ELO
        mean_elo = sum(new_elos) / len(new_elos)
        new_elos = [e - mean_elo + BASE_ELO for e in new_elos]

        max_change = max(abs(new_elos[k] - elos[k]) for k in range(n))
        elos = new_elos
        if max_change < 1:
            break

    return {anchors[i]["step"]: round(elos[i]) for i in range(n)}


def evaluate_against_anchors(cp, anchors, anchor_elos, games_per_anchor, search_iters):
    """Evaluate a single checkpoint against all anchor models."""
    cp_path = os.path.join(CKPT_DIR, cp["file"])
    matchups = []

    for anchor in anchors:
        if anchor["step"] == cp["step"]:
            continue  # skip self
        anchor_path = os.path.join(CKPT_DIR, anchor["file"])
        if not os.path.isfile(anchor_path):
            continue

        wr = evaluate_vs_model(cp_path, anchor_path,
                               num_games=games_per_anchor,
                               search_iterations=search_iters)
        wins = wr * games_per_anchor
        losses = (1.0 - wr) * games_per_anchor
        opp_elo = anchor_elos[anchor["step"]]
        matchups.append((opp_elo, wins, losses))

    if not matchups:
        return None

    return mle_elo(matchups)


def main():
    parser = argparse.ArgumentParser(description="Robust ELO evaluation for checkpoints")
    parser.add_argument("--games", type=int, default=40,
                        help="Games per matchup (default: 40)")
    parser.add_argument("--search-iters", type=int, default=32,
                        help="MCTS search iterations for eval games (default: 32)")
    parser.add_argument("--anchor-spacing", type=int, default=10000,
                        help="Step spacing between anchor models (default: 10000)")
    parser.add_argument("--round-robin-games", type=int, default=50,
                        help="Games per pair in anchor round-robin (default: 50)")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Checkpoint directory (default: checkpoints_v4)")
    args = parser.parse_args()

    if args.ckpt_dir:
        global CKPT_DIR, MANIFEST
        CKPT_DIR = args.ckpt_dir
        MANIFEST = os.path.join(CKPT_DIR, "manifest.json")

    with open(MANIFEST) as f:
        manifest = json.load(f)

    checkpoints = sorted(manifest["checkpoints"], key=lambda c: c["step"])
    print(f"Total checkpoints: {len(checkpoints)}")

    # Select anchors
    anchors = select_anchors(checkpoints, args.anchor_spacing)
    print(f"Selected {len(anchors)} anchor models: steps {[a['step'] for a in anchors]}")
    print()

    # Round-robin among anchors
    print(f"Phase 1: Round-robin tournament ({args.round_robin_games} games/pair, {args.search_iters} search iters)")
    anchor_elos = run_round_robin(anchors, args.round_robin_games, args.search_iters)
    print()
    print("Anchor ELOs:")
    for step in sorted(anchor_elos.keys()):
        print(f"  Step {step:>6}: ELO {anchor_elos[step]}")
    print()

    # Evaluate all checkpoints against anchors
    print(f"Phase 2: Evaluating all checkpoints ({args.games} games/anchor, {args.search_iters} search iters)")
    total = len(checkpoints)

    for idx, cp in enumerate(checkpoints):
        step = cp["step"]

        # Anchors already have ELOs from round-robin
        if step in anchor_elos:
            cp["elo"] = anchor_elos[step]
            cp["label"] = _label(anchor_elos[step])
            print(f"  [{idx+1}/{total}] Step {step:>6}: ELO {anchor_elos[step]} (anchor)")
            continue

        cp_path = os.path.join(CKPT_DIR, cp["file"])
        if not os.path.isfile(cp_path):
            print(f"  [{idx+1}/{total}] Step {step:>6}: checkpoint file missing, skipping")
            continue

        print(f"  [{idx+1}/{total}] Step {step:>6}: evaluating vs {len(anchors)} anchors...", end=" ", flush=True)
        elo = evaluate_against_anchors(cp, anchors, anchor_elos, args.games, args.search_iters)

        if elo is not None:
            cp["elo"] = elo
            cp["label"] = _label(elo)
            print(f"ELO {elo}")
        else:
            print(f"could not evaluate")

        # Save after each to preserve progress
        with open(MANIFEST, "w") as f:
            json.dump({"checkpoints": checkpoints}, f, indent=2)

    # Final save
    with open(MANIFEST, "w") as f:
        json.dump({"checkpoints": checkpoints}, f, indent=2)

    print()
    print("Done! Summary:")
    evaluated = [c for c in checkpoints if c.get("elo") is not None]
    for c in evaluated:
        print(f"  Step {c['step']:>6}: ELO {c['elo']}")


def _label(elo):
    if elo < 900:
        return "Beginner"
    elif elo < 1050:
        return "Novice"
    elif elo < 1200:
        return "Intermediate"
    elif elo < 1400:
        return "Advanced"
    else:
        return "Expert"


if __name__ == "__main__":
    main()
