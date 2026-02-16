"""
Training script for Pylos AlphaZero.

Runs self-play training with periodic checkpoint saving and evaluation.
"""

import sys
import os
import json
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import yaml
import torch
from tqdm import tqdm

from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgentTrainer
from evaluate import evaluate_checkpoint, assign_label


def load_config(path=None):
    """Load training configuration from a YAML file.

    Args:
        path: Path to YAML config file. Defaults to engine/config.yaml.

    Returns:
        dict with training, checkpoints, and wandb sections.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(agent, step, config, manifest_path):
    """Save a training checkpoint and update the manifest.

    Saves the model and optimizer state dicts into a single .pth file,
    evaluates the model against a random agent, and appends an entry
    to manifest.json.

    Args:
        agent: AlphaZeroAgentTrainer instance.
        step: Current training step (game number).
        config: Full config dict.
        manifest_path: Path to manifest.json.
    """
    ckpt_dir = config["checkpoints"]["dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    filename = f"checkpoint_{step:05d}.pth"
    filepath = os.path.join(ckpt_dir, filename)

    torch.save(
        {
            "model_state_dict": agent.model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "step": step,
        },
        filepath,
    )

    # Evaluate against random agent
    eval_games = config["checkpoints"]["eval_games"]
    print(f"\nEvaluating checkpoint at step {step} ({eval_games} games)...")
    win_rate = evaluate_checkpoint(filepath, num_games=eval_games, search_iterations=8)
    label = assign_label(win_rate)
    print(f"  Win rate vs random: {win_rate:.2%} ({label})")

    # Update manifest
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"checkpoints": []}

    manifest["checkpoints"].append(
        {
            "file": filename,
            "step": step,
            "win_rate_vs_random": round(win_rate, 4),
            "label": label,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Saved {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Train Pylos AlphaZero")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    ckpt_cfg = config["checkpoints"]

    # Optionally initialise wandb
    wandb_run = None
    if config.get("wandb", {}).get("enabled", False):
        import wandb

        wandb_run = wandb.init(
            project=config["wandb"].get("project", "pylos-alphazero"),
            config=config,
        )

    # Create game, model, optimizer, agent
    game = PylosGame()
    model = PylosNetwork(game.observation_shape, game.action_space)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    agent = AlphaZeroAgentTrainer(
        model, optimizer, replay_buffer_max_size=train_cfg["replay_buffer_size"]
    )

    ckpt_dir = ckpt_cfg["dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    manifest_path = os.path.join(ckpt_dir, "manifest.json")

    num_games = train_cfg["selfplay_games"]
    save_every = ckpt_cfg["save_every"]

    print(f"Starting training: {num_games} self-play games")
    print(f"  Search iterations: {train_cfg['search_iterations']}")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Checkpoints every {save_every} games -> {ckpt_dir}/")

    for step in tqdm(range(1, num_games + 1), desc="Self-play"):
        game.reset()
        value_losses, policy_losses = agent.train_step(
            game,
            search_iterations=train_cfg["search_iterations"],
            batch_size=train_cfg["batch_size"],
            epochs=train_cfg["epochs_per_game"],
            c_puct=train_cfg["c_puct"],
            dirichlet_alpha=train_cfg["dirichlet_alpha"],
        )

        # Logging
        if value_losses:
            avg_vloss = sum(value_losses) / len(value_losses)
            avg_ploss = sum(policy_losses) / len(policy_losses)
            tqdm.write(
                f"Game {step}: value_loss={avg_vloss:.4f}  policy_loss={avg_ploss:.4f}"
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "value_loss": avg_vloss,
                        "policy_loss": avg_ploss,
                        "step": step,
                    }
                )

        # Periodic checkpoint
        if step % save_every == 0:
            save_checkpoint(agent, step, config, manifest_path)

    # Final checkpoint (if not already saved)
    if num_games % save_every != 0:
        save_checkpoint(agent, num_games, config, manifest_path)

    print("\nTraining complete.")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
