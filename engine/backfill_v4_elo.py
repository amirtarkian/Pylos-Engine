"""Backfill ELO for v4 checkpoints that are stuck on 'Evaluating...'."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from evaluate import evaluate_vs_model, win_rate_to_elo, assign_label

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints_v4")
MANIFEST = os.path.join(CKPT_DIR, "manifest.json")
EVAL_GAMES = 40
SEARCH_ITER = 16


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    checkpoints = sorted(manifest["checkpoints"], key=lambda c: c["step"])

    # Find last evaluated checkpoint
    last_elo = None
    last_step = None
    for c in checkpoints:
        if c.get("elo") is not None:
            last_elo = c["elo"]
            last_step = c["step"]

    if last_elo is None:
        print("No evaluated checkpoints found, setting step 1000 as baseline")
        for c in checkpoints:
            if c["step"] == min(cp["step"] for cp in checkpoints):
                c["elo"] = 1000
                c["label"] = "Baseline"
                last_elo = 1000
                last_step = c["step"]
                break

    pending = [c for c in checkpoints if c.get("elo") is None]
    print(f"Last evaluated: step {last_step} (ELO {last_elo})")
    print(f"Pending: {len(pending)} checkpoints")
    print(f"Using {EVAL_GAMES} games, {SEARCH_ITER} search iterations\n")

    for c in pending:
        step = c["step"]
        filepath = os.path.join(CKPT_DIR, c["file"])
        prev_path = None

        # Find previous evaluated checkpoint
        for prev in reversed(checkpoints):
            if prev["step"] < step and prev.get("elo") is not None:
                prev_path = os.path.join(CKPT_DIR, prev["file"])
                prev_elo = prev["elo"]
                break

        if prev_path is None or not os.path.isfile(prev_path):
            print(f"Step {step}: no previous checkpoint, skipping")
            continue

        if not os.path.isfile(filepath):
            print(f"Step {step}: checkpoint file missing, skipping")
            continue

        print(f"Step {step}: evaluating vs step {prev['step']} (ELO {prev_elo})...", end=" ", flush=True)
        win_rate = evaluate_vs_model(filepath, prev_path, num_games=EVAL_GAMES, search_iterations=SEARCH_ITER)
        elo = round(win_rate_to_elo(win_rate, prev_elo))
        label = assign_label(win_rate)
        print(f"win_rate={win_rate:.2%}, ELO={elo} ({label})")

        c["elo"] = elo
        c["win_rate_vs_prev"] = round(win_rate, 4)
        c["label"] = label

        # Save after each evaluation so progress isn't lost
        with open(MANIFEST, "w") as f:
            json.dump({"checkpoints": checkpoints}, f, indent=2)

    print("\nDone! All checkpoints evaluated.")


if __name__ == "__main__":
    main()
