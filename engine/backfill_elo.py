"""
Backfill ELO ratings by pitting each checkpoint against the previous one.

Uses Dirichlet noise in MCTS to produce varied games (without noise, MCTS
is fully deterministic and every game between the same two models is
identical). Both players alternate white/black to cancel first-player bias.
"""

import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgent
from mcts import play
from evaluate import assign_label

SEARCH_ITERATIONS = 16
EVAL_GAMES = 20       # per matchup (10 as white, 10 as black)
EVAL_EVERY = 5        # evaluate every Nth checkpoint
BASE_ELO = 1000
K_FACTOR = 32         # standard ELO K-factor
DIRICHLET_ALPHA = 0.3 # exploration noise for varied games
MOVE_LIMIT = 200      # max moves before declaring draw


def load_agent(model_path):
    """Load a checkpoint and return an AlphaZeroAgent."""
    game = PylosGame()
    model = PylosNetwork(game.observation_shape, game.action_space)
    ckpt = torch.load(model_path, weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return AlphaZeroAgent(model)


def pit_with_limit(game, agent1, agent2, kwargs1, kwargs2, move_limit=MOVE_LIMIT):
    """Play a game with move limit. Returns 1 (white wins), -1 (black wins), or 0 (draw)."""
    current_agent, other_agent = agent1, agent2
    current_kwargs, other_kwargs = kwargs1, kwargs2
    moves = 0
    while game.get_result() is None and moves < move_limit:
        action = play(game, current_agent, **current_kwargs)
        if action is None:
            break
        game.step(action)
        current_agent, other_agent = other_agent, current_agent
        current_kwargs, other_kwargs = other_kwargs, current_kwargs
        moves += 1
    result = game.get_result()
    return result if result is not None else 0  # 0 = draw


def head_to_head(agent_a, agent_b, num_games=EVAL_GAMES):
    """Play agent_a vs agent_b, alternating colors. Returns agent_a's score (0.0-1.0).

    Draws count as 0.5 for both sides.
    """
    kwargs = {"search_iterations": SEARCH_ITERATIONS, "dirichlet_alpha": DIRICHLET_ALPHA}
    a_score = 0.0

    for i in range(num_games):
        game = PylosGame()
        game.reset()
        if i % 2 == 0:
            # A plays white
            result = pit_with_limit(game, agent_a, agent_b, kwargs, kwargs)
            if result == 1:
                a_score += 1.0
            elif result == 0:
                a_score += 0.5
        else:
            # A plays black
            result = pit_with_limit(game, agent_b, agent_a, kwargs, kwargs)
            if result == -1:
                a_score += 1.0
            elif result == 0:
                a_score += 0.5

    return a_score / num_games


def elo_update(elo_a, elo_b, score_a):
    """Update ELO rating for player A given their score (0-1) against player B."""
    expected = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    return elo_a + K_FACTOR * (score_a - expected)


def backfill_dir(ckpt_dir, label):
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"[{label}] No manifest found, skipping.")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    checkpoints = manifest.get("checkpoints", [])
    total = len(checkpoints)

    # Select which checkpoints to evaluate
    eval_indices = [i for i in range(total) if i % EVAL_EVERY == 0]
    if (total - 1) not in eval_indices:
        eval_indices.append(total - 1)

    print(f"[{label}] {total} checkpoints, evaluating {len(eval_indices)} in ELO ladder...")

    prev_agent = None
    prev_elo = BASE_ELO

    for count, idx in enumerate(eval_indices):
        cp = checkpoints[idx]
        filepath = os.path.join(ckpt_dir, cp["file"].split("/")[-1])
        if not os.path.isfile(filepath):
            print(f"  Skipping {cp['file']} (not found)")
            continue

        print(f"  [{count + 1}/{len(eval_indices)}] step {cp['step']:>6}...", end=" ", flush=True)

        agent = load_agent(filepath)

        if prev_agent is None:
            # First checkpoint gets base ELO
            cp["elo"] = BASE_ELO
            cp["win_rate_vs_prev"] = 0.5
            cp["label"] = "Novice"
            print(f"ELO={BASE_ELO} (baseline)")
        else:
            win_rate = head_to_head(agent, prev_agent)
            new_elo = elo_update(prev_elo, prev_elo, win_rate)
            # Clamp so ELO doesn't go below 100
            new_elo = max(100, new_elo)
            cp["elo"] = round(new_elo)
            cp["win_rate_vs_prev"] = round(win_rate, 4)
            cp["label"] = assign_label(win_rate)
            print(f"WR={win_rate:.0%} vs prev, ELO={cp['elo']}")
            prev_elo = new_elo

        prev_agent = agent

        # Interpolate ELO for skipped checkpoints between this and the previous eval
        if count > 0:
            prev_eval_idx = eval_indices[count - 1]
            prev_eval_elo = checkpoints[prev_eval_idx].get("elo", BASE_ELO)
            curr_elo = cp["elo"]
            gap = idx - prev_eval_idx
            if gap > 1:
                for j in range(prev_eval_idx + 1, idx):
                    frac = (j - prev_eval_idx) / gap
                    interp_elo = round(prev_eval_elo + frac * (curr_elo - prev_eval_elo))
                    checkpoints[j]["elo"] = interp_elo

        # Save after each evaluation
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"[{label}] Done.")


if __name__ == "__main__":
    engine_dir = os.path.dirname(os.path.abspath(__file__))

    v1_dir = os.path.join(engine_dir, "checkpoints")
    v2_dir = os.path.join(engine_dir, "checkpoints_v2")

    backfill_dir(v1_dir, "v1")
    backfill_dir(v2_dir, "v2")

    print("\nBackfill complete!")
