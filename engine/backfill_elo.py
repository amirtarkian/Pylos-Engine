"""
Backfill ELO ratings for checkpoints using MLE estimation.

Evaluates each checkpoint against a set of opponents (previous checkpoints
within the same run + optionally cross-version anchors like V5 best).

Usage:
    .venv/bin/python engine/backfill_elo.py --ckpt-dir engine/checkpoints_v6 \
        --anchor engine/checkpoints_v5/checkpoint_235000.pth:1781
"""

import sys
import os
import json
import math
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from evaluate import _load_agent, _play_match, evaluate_vs_model

SEARCH_ITERS = 32     # faster than training eval (64) for backfill speed
GAMES_PER_OPPONENT = 20
RANDOM_ELO = 1000


def _mle_elo(matchups):
    """MLE ELO from list of (opponent_elo, wins, losses)."""
    lo, hi = 0.0, 3000.0
    for _ in range(100):
        mid = (lo + hi) / 2
        grad = 0.0
        for opp_elo, wins, losses in matchups:
            total = wins + losses
            if total == 0:
                continue
            expected = 1.0 / (1.0 + 10 ** ((opp_elo - mid) / 400))
            grad += (wins - total * expected) * math.log(10) / 400
        if grad > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.5:
            break
    return round((lo + hi) / 2)


def _elo_label(elo):
    if elo < 1050:
        return "Beginner"
    elif elo < 1200:
        return "Novice"
    elif elo < 1400:
        return "Intermediate"
    elif elo < 1600:
        return "Advanced"
    else:
        return "Expert"


def backfill(ckpt_dir, anchors=None, search_iters=SEARCH_ITERS,
             games_per_opponent=GAMES_PER_OPPONENT, eval_every=1):
    """Backfill ELO for all checkpoints in ckpt_dir.

    anchors: list of (path, elo) tuples for cross-version opponents
    """
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"No manifest at {manifest_path}")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    checkpoints = manifest.get("checkpoints", [])
    if not checkpoints:
        print("No checkpoints in manifest.")
        return

    # Load anchor agents once
    anchor_agents = []
    if anchors:
        for anchor_path, anchor_elo in anchors:
            if os.path.isfile(anchor_path):
                print(f"Loading anchor: {anchor_path} (ELO {anchor_elo})")
                agent = _load_agent(anchor_path)
                anchor_agents.append((agent, anchor_elo, anchor_path))
            else:
                print(f"WARNING: anchor {anchor_path} not found, skipping")

    # Select checkpoints to evaluate
    indices = list(range(0, len(checkpoints), eval_every))
    if (len(checkpoints) - 1) not in indices:
        indices.append(len(checkpoints) - 1)

    print(f"\n{len(checkpoints)} total checkpoints, evaluating {len(indices)}")
    print(f"Search iters: {search_iters}, games/opponent: {games_per_opponent}\n")

    # Track evaluated checkpoints for use as opponents
    evaluated = []  # list of (step, elo, path)

    for count, idx in enumerate(indices):
        cp = checkpoints[idx]
        filepath = os.path.join(ckpt_dir, cp["file"])
        if not os.path.isfile(filepath):
            print(f"  Skipping {cp['file']} (not found)")
            continue

        step = cp["step"]

        # Skip if already evaluated (has elo and not "Evaluating...")
        if cp.get("elo") is not None and cp.get("label") != "Evaluating...":
            evaluated.append((step, cp["elo"], filepath))
            print(f"  [{count+1}/{len(indices)}] step {step:>6}: ELO {cp['elo']} (cached)")
            continue

        print(f"  [{count+1}/{len(indices)}] step {step:>6}...", end=" ", flush=True)

        if not evaluated and not anchor_agents:
            # First checkpoint, no opponents — baseline
            cp["elo"] = RANDOM_ELO
            cp["label"] = "Baseline"
            evaluated.append((step, RANDOM_ELO, filepath))
            print(f"ELO {RANDOM_ELO} (baseline)")
        else:
            agent = _load_agent(filepath)
            matchups = []
            details = []

            # Play against anchors (cross-version)
            for anchor_agent, anchor_elo, anchor_path in anchor_agents:
                wr = _play_match(agent, anchor_agent, games_per_opponent,
                                 search_iterations=search_iters)
                wins = wr * games_per_opponent
                losses = (1.0 - wr) * games_per_opponent
                matchups.append((anchor_elo, wins, losses))
                anchor_name = os.path.basename(os.path.dirname(anchor_path))
                details.append(f"{anchor_name}({anchor_elo}): {wr:.0%}")

            # Play against previous checkpoints in this run (up to 3)
            opponents = evaluated[-3:] if len(evaluated) > 3 else evaluated[:]
            for opp_step, opp_elo, opp_path in opponents:
                wr = _play_match(agent, _load_agent(opp_path), games_per_opponent,
                                 search_iterations=search_iters)
                wins = wr * games_per_opponent
                losses = (1.0 - wr) * games_per_opponent
                matchups.append((opp_elo, wins, losses))
                details.append(f"s{opp_step}({opp_elo}): {wr:.0%}")

            elo = _mle_elo(matchups)
            label = _elo_label(elo)
            cp["elo"] = elo
            cp["label"] = label
            evaluated.append((step, elo, filepath))
            print(f"ELO {elo} ({label}) — vs [{', '.join(details)}]")

        # Save after each evaluation
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nDone! {len(indices)} checkpoints evaluated.")


def main():
    parser = argparse.ArgumentParser(description="Backfill ELO ratings for checkpoints")
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Checkpoint directory (e.g., engine/checkpoints_v6)")
    parser.add_argument("--anchor", type=str, action="append", default=[],
                        help="Cross-version anchor as path:elo (e.g., checkpoints_v5/checkpoint_235000.pth:1781)")
    parser.add_argument("--search-iters", type=int, default=SEARCH_ITERS,
                        help=f"MCTS iterations (default: {SEARCH_ITERS})")
    parser.add_argument("--games", type=int, default=GAMES_PER_OPPONENT,
                        help=f"Games per opponent (default: {GAMES_PER_OPPONENT})")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Evaluate every Nth checkpoint (default: 1)")
    args = parser.parse_args()

    # Parse anchors
    anchors = []
    for a in args.anchor:
        if ":" not in a:
            print(f"ERROR: anchor must be path:elo, got '{a}'")
            sys.exit(1)
        path, elo_str = a.rsplit(":", 1)
        anchors.append((path, int(elo_str)))

    backfill(args.ckpt_dir, anchors=anchors, search_iters=args.search_iters,
             games_per_opponent=args.games, eval_every=args.eval_every)


if __name__ == "__main__":
    main()
