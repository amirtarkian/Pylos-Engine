"""
Evaluate a Pylos AlphaZero checkpoint against a random agent.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import torch
from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgent
from mcts import pit


class RandomAgent:
    """Agent that plays uniformly at random (no search)."""

    @staticmethod
    def value_fn(game):
        return 0.0

    @staticmethod
    def policy_fn(game):
        return np.ones(game.action_space, dtype=np.float32) / game.action_space


def evaluate_checkpoint(model_path, num_games=50, search_iterations=32):
    """Evaluate a checkpoint by playing against a RandomAgent.

    The AlphaZero agent alternates between playing first (white) and
    second (black) across the games.

    Args:
        model_path: Path to the saved model state dict (.pth file).
        num_games: Number of games to play.
        search_iterations: MCTS iterations per move for the AlphaZero agent.

    Returns:
        Win rate as a float between 0.0 and 1.0.
    """
    game = PylosGame()
    model = PylosNetwork(game.observation_shape, game.action_space)
    checkpoint = torch.load(model_path, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    az_agent = AlphaZeroAgent(model)
    random_agent = RandomAgent()

    az_kwargs = {"search_iterations": search_iterations}
    random_kwargs = {"search_iterations": 1}

    wins = 0
    for i in range(num_games):
        game.reset()
        if i % 2 == 0:
            # AlphaZero plays as white (player 1)
            result = pit(game, az_agent, random_agent, az_kwargs, random_kwargs)
            if result == 1:
                wins += 1
        else:
            # AlphaZero plays as black (player -1)
            result = pit(game, random_agent, az_agent, random_kwargs, az_kwargs)
            if result == -1:
                wins += 1

    return wins / num_games


def assign_label(win_rate):
    """Assign a skill label based on win rate vs random.

    Args:
        win_rate: Float between 0.0 and 1.0.

    Returns:
        String label: Beginner, Novice, Intermediate, Advanced, or Expert.
    """
    if win_rate < 0.4:
        return "Beginner"
    elif win_rate < 0.6:
        return "Novice"
    elif win_rate < 0.75:
        return "Intermediate"
    elif win_rate < 0.9:
        return "Advanced"
    else:
        return "Expert"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Pylos AlphaZero checkpoint")
    parser.add_argument("model_path", type=str, help="Path to the model .pth file")
    parser.add_argument("--games", type=int, default=50, help="Number of evaluation games")
    parser.add_argument(
        "--search-iterations", type=int, default=32,
        help="MCTS iterations per move"
    )
    args = parser.parse_args()

    print(f"Evaluating {args.model_path} over {args.games} games "
          f"({args.search_iterations} search iterations)...")
    win_rate = evaluate_checkpoint(
        args.model_path,
        num_games=args.games,
        search_iterations=args.search_iterations,
    )
    label = assign_label(win_rate)
    print(f"Win rate vs random: {win_rate:.2%}")
    print(f"Label: {label}")


if __name__ == "__main__":
    main()
