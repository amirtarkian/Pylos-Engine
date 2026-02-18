"""
Evaluate Pylos AlphaZero checkpoints — vs random and vs other models.
"""

import sys
import os
import math
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import torch
from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgent
from mcts import play


MAX_EVAL_MOVES = 200  # prevent infinite games during evaluation
RANDOM_ELO = 1000     # baseline ELO for random agent


class RandomAgent:
    """Agent that plays uniformly at random (no search)."""

    @staticmethod
    def value_fn(game):
        return 0.0

    @staticmethod
    def policy_fn(game):
        return np.ones(game.action_space, dtype=np.float32) / game.action_space


def _load_agent(model_path):
    """Load a checkpoint into an AlphaZeroAgent on CPU."""
    game_tmp = PylosGame()
    model = PylosNetwork(game_tmp.observation_shape, game_tmp.action_space)
    model.device = torch.device("cpu")
    model.to(model.device)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return AlphaZeroAgent(model)


def _play_match(agent_a, agent_b, num_games, search_iterations=32, eval_noise=0.15):
    """Play a match between two agents, returning agent_a's win rate.

    Alternates colors each game. Draws count as 0.5 for both.
    Uses light Dirichlet noise (eval_noise) to break determinism
    so each game plays out differently.
    """
    game = PylosGame()
    a_kwargs = {"search_iterations": search_iterations, "dirichlet_alpha": eval_noise}
    b_kwargs = {"search_iterations": search_iterations, "dirichlet_alpha": eval_noise}

    # If agent_b is RandomAgent, give it 1 iteration and no noise
    if isinstance(agent_b, RandomAgent):
        b_kwargs = {"search_iterations": 1, "dirichlet_alpha": None}

    score = 0.0
    for i in range(num_games):
        game.reset()
        if i % 2 == 0:
            agents = [agent_a, agent_b]
            kwargs = [a_kwargs, b_kwargs]
            a_color = 1
        else:
            agents = [agent_b, agent_a]
            kwargs = [b_kwargs, a_kwargs]
            a_color = -1

        cur, nxt = 0, 1
        moves = 0
        while game.get_result() is None and moves < MAX_EVAL_MOVES:
            action = play(game, agents[cur], **kwargs[cur])
            if action is None:
                break
            game.step(action)
            cur, nxt = nxt, cur
            moves += 1

        result = game.get_result()
        if result == a_color:
            score += 1.0
        elif result is None:
            score += 0.5  # draw

    return score / num_games


def win_rate_to_elo(win_rate, opponent_elo):
    """Convert a win rate against a rated opponent to an ELO rating."""
    clamped = max(0.01, min(0.99, win_rate))
    return opponent_elo + 400 * math.log10(clamped / (1 - clamped))


def evaluate_checkpoint(model_path, num_games=50, search_iterations=32):
    """Evaluate a checkpoint by playing against a RandomAgent.

    Returns win rate as a float between 0.0 and 1.0.
    """
    agent = _load_agent(model_path)
    return _play_match(agent, RandomAgent(), num_games, search_iterations)


def evaluate_vs_model(model_path_a, model_path_b, num_games=50, search_iterations=32):
    """Play two checkpoints against each other.

    Returns model_a's win rate as a float between 0.0 and 1.0.
    """
    agent_a = _load_agent(model_path_a)
    agent_b = _load_agent(model_path_b)
    return _play_match(agent_a, agent_b, num_games, search_iterations)


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
