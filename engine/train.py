"""
Training script for Pylos AlphaZero.

Runs parallel self-play training with periodic checkpoint saving and evaluation.
Uses multiprocessing to leverage multiple CPU cores for MCTS self-play.
"""

import sys
import os
import json
import argparse
import math
import random
import time
import threading
import multiprocessing as mp
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgentTrainer
from evaluate import evaluate_vs_model, RANDOM_ELO


# ---------------------------------------------------------------------------
# Worker process for parallel self-play
# ---------------------------------------------------------------------------

_worker_agent = None


def _init_worker(model_state_dict, obs_shape, action_space, hidden=256, num_blocks=6,
                 value_hidden=64, policy_hidden=128, rich_obs=False):
    """Initialize a worker process with its own model copy on CPU.

    Workers do single-inference MCTS where CPU is ~10x faster than MPS
    due to GPU dispatch overhead.  MPS is only used in the main process
    for batched training.
    """
    global _worker_agent
    # Limit PyTorch threads per worker to avoid oversubscription
    torch.set_num_threads(1)

    model = PylosNetwork(obs_shape, action_space, hidden=hidden,
                         num_blocks=num_blocks, value_hidden=value_hidden,
                         policy_hidden=policy_hidden)
    # Force CPU for workers regardless of what PylosNetwork auto-detects
    model.device = torch.device("cpu")
    model.to(model.device)
    model.load_state_dict(model_state_dict)
    model.eval()

    from agents import AlphaZeroAgent
    _worker_agent = AlphaZeroAgent(model, rich_obs=rich_obs)


def _board_hash(game):
    """Fast board state hash for repetition detection during self-play."""
    parts = []
    for level in range(4):
        parts.append(game.board[level].tobytes())
    parts.append(bytes([game.turn + 1]))
    return b"".join(parts)


def _batched_selfplay(model, batch_size, search_iters, c_puct, dir_alpha,
                      max_moves, rep_limit, ml_penalty, rep_penalty, temp_threshold=15,
                      rich_obs=False):
    """Run a batch of self-play games with batched MCTS on GPU/MPS.

    All games run concurrently: at each MCTS iteration, leaf evaluations
    from all active games are collected into a single batch for one GPU
    forward pass instead of N individual CPU inferences.

    Returns list of (buffer, fpr) tuples, one per game.
    """
    from mcts import batched_search
    from collections import defaultdict

    games = [PylosGame(rich_obs=rich_obs) for _ in range(batch_size)]
    buffers = [[] for _ in range(batch_size)]
    move_counts = [0] * batch_size
    state_counts = [defaultdict(int) if rep_limit else None for _ in range(batch_size)]
    results = [None] * batch_size
    active = list(range(batch_size))

    def batch_eval_fn(obs):
        return model.batched_inference(obs)

    while active:
        active_games = [games[i] for i in active]
        roots = batched_search(active_games, batch_eval_fn, search_iters,
                               c_puct=c_puct, dirichlet_alpha=dir_alpha)

        still_active = []
        for j, i in enumerate(active):
            root = roots[j]
            game = games[i]

            if root.children is None:
                fpr = game.get_first_person_result()
                results[i] = fpr if fpr is not None else 0.0
                continue

            visits = root.children_visits
            visits_dist = visits / visits.sum()
            if move_counts[i] < temp_threshold:
                # Temperature = 1: sample proportionally for exploration
                action = root.children_actions[
                    np.random.choice(len(root.children), p=visits_dist)
                ]
            else:
                # Temperature -> 0: pick strongest move
                action = root.children_actions[np.argmax(visits)]

            actions_dist = np.zeros(game.action_space, dtype=np.float32)
            actions_dist[root.children_actions] = visits_dist
            buffers[i].append((game.to_observation(), actions_dist))

            game.step(action)
            move_counts[i] += 1

            fpr = game.get_first_person_result()
            if fpr is not None:
                results[i] = fpr
                continue

            if state_counts[i] is not None:
                bh = _board_hash(game)
                state_counts[i][bh] += 1
                if state_counts[i][bh] >= rep_limit:
                    results[i] = rep_penalty
                    continue

            if max_moves and move_counts[i] >= max_moves:
                results[i] = ml_penalty
                continue

            still_active.append(i)

        active = still_active

    return [(buffers[i], results[i] if results[i] is not None else 0.0)
            for i in range(batch_size)]


def _run_selfplay(args):
    """Run a single self-play game in a worker process.

    Returns (buffer, first_person_result) where buffer is a list of
    (observation, action_distribution) tuples.

    Supports anti-draw settings:
      max_moves            -- end game after N moves (draw)
      repetition_limit     -- end game if same state seen N times (draw)
      move_limit_penalty   -- penalty for both players on move limit draw
      repetition_penalty   -- penalty for repeating player on repetition draw
    """
    search_iters, c_puct, dir_alpha, max_moves, rep_limit, ml_penalty, rep_penalty, temp_threshold, rich_obs = args
    from mcts import search
    from collections import defaultdict

    game = PylosGame(rich_obs=rich_obs)
    buffer = []
    move_count = 0
    state_counts = defaultdict(int) if rep_limit else None
    draw_reason = None

    while (fpr := game.get_first_person_result()) is None:
        root_node = search(
            game, iterations=search_iters, c_puct=c_puct,
            dirichlet_alpha=dir_alpha, eval_fn=_worker_agent.eval_fn,
        )
        if root_node.children is None:
            break

        visits = root_node.children_visits
        visits_dist = visits / visits.sum()
        if move_count < temp_threshold:
            # Temperature = 1: sample proportionally for exploration
            action = root_node.children_actions[
                np.random.choice(len(root_node.children), p=visits_dist)
            ]
        else:
            # Temperature -> 0: pick strongest move
            action = root_node.children_actions[np.argmax(visits)]
        actions_dist = np.zeros(game.action_space, dtype=np.float32)
        actions_dist[root_node.children_actions] = visits_dist
        buffer.append((game.to_observation(), actions_dist))
        game.step(action)
        move_count += 1

        # Check repetition
        if state_counts is not None:
            bh = _board_hash(game)
            state_counts[bh] += 1
            if state_counts[bh] >= rep_limit:
                draw_reason = "repetition"
                break

        # Check move limit
        if max_moves and move_count >= max_moves:
            draw_reason = "move_limit"
            break

    fpr = game.get_first_person_result()

    if draw_reason == "repetition":
        # The player who just moved caused the repetition — penalize them
        # fpr is from current player's perspective; the repeater just moved
        # so it's the opponent of current player. Penalty to repeater =
        # positive for current player (they didn't cause it), but we penalize
        # both slightly with the repeater getting more.
        fpr = rep_penalty  # negative for current player (both suffer)
    elif draw_reason == "move_limit":
        fpr = ml_penalty   # negative for both
    elif fpr is None:
        fpr = 0.0

    return buffer, fpr


# ---------------------------------------------------------------------------
# Config & checkpoint helpers
# ---------------------------------------------------------------------------

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


_manifest_lock = threading.Lock()
_eval_threads = []  # track active evaluation threads


MAX_EVAL_OPPONENTS = 5   # play against up to this many previous checkpoints
EVAL_SEARCH_ITERS = 64   # MCTS iterations for evaluation games (match training depth)


def _mle_elo(matchups):
    """Compute ELO via maximum likelihood estimation.

    matchups: list of (opponent_elo, wins, losses)
    Returns the ELO that maximizes the likelihood of observed results.
    """
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
    """Assign a skill label based on ELO rating."""
    if elo < 1050:
        return "Baseline"
    elif elo < 1150:
        return "Beginner"
    elif elo < 1300:
        return "Novice"
    elif elo < 1500:
        return "Intermediate"
    elif elo < 1700:
        return "Advanced"
    else:
        return "Expert"


def _run_eval_in_background(filepath, filename, step, eval_games, manifest_path, ckpt_dir):
    """Background thread: evaluate checkpoint vs random subset of previous models."""
    try:
        with _manifest_lock:
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
            else:
                manifest = {"checkpoints": []}

            # Skip if already evaluated
            for entry in manifest["checkpoints"]:
                if entry["step"] == step and entry.get("elo") is not None:
                    print(f"\n  [eval] Step {step}: already evaluated (ELO {entry['elo']}), skipping")
                    return

            # Collect all previously evaluated checkpoints
            evaluated = [
                e for e in manifest["checkpoints"]
                if e["step"] < step and e.get("elo") is not None
            ]

        if not evaluated:
            # First checkpoint — assign baseline ELO
            elo = RANDOM_ELO
            label = "Baseline"
            print(f"\n  [eval] Step {step}: baseline ELO {elo} (no previous checkpoint)")
        else:
            # Select random opponents (up to MAX_EVAL_OPPONENTS)
            num_opponents = min(MAX_EVAL_OPPONENTS, len(evaluated))
            opponents = random.sample(evaluated, num_opponents)
            games_per_opponent = max(4, eval_games // num_opponents)

            matchups = []
            details = []
            for opp in opponents:
                opp_path = os.path.join(ckpt_dir, opp["file"])
                if not os.path.isfile(opp_path):
                    continue
                wr = evaluate_vs_model(
                    filepath, opp_path,
                    num_games=games_per_opponent,
                    search_iterations=EVAL_SEARCH_ITERS,
                )
                wins = wr * games_per_opponent
                losses = (1.0 - wr) * games_per_opponent
                matchups.append((opp["elo"], wins, losses))
                details.append(f"step {opp['step']}({opp['elo']}): {wr:.0%}")

            if matchups:
                elo = _mle_elo(matchups)
            else:
                elo = RANDOM_ELO
            label = _elo_label(elo)
            print(f"\n  [eval] Step {step}: ELO {elo} ({label}) — vs [{', '.join(details)}]")

        # Update manifest
        with _manifest_lock:
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
            else:
                manifest = {"checkpoints": []}

            for entry in manifest["checkpoints"]:
                if entry["step"] == step:
                    entry["label"] = label
                    entry["elo"] = elo
                    break

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
    except Exception as e:
        print(f"\n  [eval] Step {step} evaluation failed: {e}")


def save_checkpoint(agent, step, config, manifest_path):
    """Save a training checkpoint and kick off background evaluation."""
    ckpt_dir = config["checkpoints"]["dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    filename = f"checkpoint_{step:05d}.pth"
    filepath = os.path.join(ckpt_dir, filename)

    model_cfg = config.get("model", {})
    torch.save(
        {
            "model_state_dict": agent.model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "step": step,
            "model_config": {
                "hidden": model_cfg.get("hidden", 256),
                "num_blocks": model_cfg.get("num_blocks", 6),
                "value_hidden": model_cfg.get("value_hidden", 64),
                "policy_hidden": model_cfg.get("policy_hidden", 128),
                "rich_obs": model_cfg.get("rich_obs", False),
            },
        },
        filepath,
    )
    print(f"  Saved {filepath}")

    # Write manifest entry immediately (evaluation pending)
    with _manifest_lock:
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        else:
            manifest = {"checkpoints": []}

        # Remove any existing entry for this step (from prior restarts)
        manifest["checkpoints"] = [
            e for e in manifest["checkpoints"] if e["step"] != step
        ]

        manifest["checkpoints"].append(
            {
                "file": filename,
                "step": step,
                "label": "Evaluating...",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # Run evaluation in background thread (CPU, separate model instance)
    eval_games = config["checkpoints"]["eval_games"]
    eval_thread = threading.Thread(
        target=_run_eval_in_background,
        args=(filepath, filename, step, eval_games, manifest_path, ckpt_dir),
        daemon=False,  # Changed: don't kill on exit
    )
    eval_thread.start()
    _eval_threads.append(eval_thread)
    print(f"  Evaluation started in background ({eval_games} games on CPU)")


_progress_prev = {}  # path -> (step, time) for rolling speed


def write_progress(path, step, total, value_loss, policy_loss, start_time, status="training"):
    """Write live training progress to a JSON file for the server to read."""
    now = time.time()
    elapsed = now - start_time

    # Rolling speed: games since last write / time since last write
    prev = _progress_prev.get(path)
    if prev and now > prev[1] and step > prev[0]:
        recent_rate = (step - prev[0]) / (now - prev[1])
    else:
        recent_rate = step / elapsed if elapsed > 0 else 0
    _progress_prev[path] = (step, now)

    eta = (total - step) / recent_rate if recent_rate > 0 else 0

    progress = {
        "status": status,
        "current_game": step,
        "total_games": total,
        "percent": round(step / total * 100, 1) if total > 0 else 0,
        "value_loss": round(value_loss, 6) if value_loss is not None else None,
        "policy_loss": round(policy_loss, 6) if policy_loss is not None else None,
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1),
        "games_per_second": round(recent_rate, 3),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def append_loss_history(ckpt_dir, step, value_loss, policy_loss):
    """Append a single loss data point to the persistent loss history file."""
    history_path = os.path.join(ckpt_dir, "loss_history.jsonl")
    with open(history_path, "a") as f:
        f.write(json.dumps({
            "step": step,
            "value_loss": round(value_loss, 6),
            "policy_loss": round(policy_loss, 6),
        }) + "\n")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Pylos AlphaZero")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel self-play workers (default: cpu_count - 2)")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    ckpt_cfg = config["checkpoints"]

    num_workers = args.workers or min(os.cpu_count() - 2, 8)

    # Optionally initialise wandb
    wandb_run = None
    if config.get("wandb", {}).get("enabled", False):
        import wandb
        wandb_run = wandb.init(
            project=config["wandb"].get("project", "pylos-alphazero"),
            config=config,
        )

    # Create game, model, optimizer, agent
    model_cfg = config.get("model", {})
    rich_obs = model_cfg.get("rich_obs", False)
    game = PylosGame(rich_obs=rich_obs)
    model = PylosNetwork(
        game.observation_shape, game.action_space,
        hidden=model_cfg.get("hidden", 256),
        num_blocks=model_cfg.get("num_blocks", 6),
        value_hidden=model_cfg.get("value_hidden", 64),
        policy_hidden=model_cfg.get("policy_hidden", 128),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    agent = AlphaZeroAgentTrainer(
        model, optimizer, replay_buffer_max_size=train_cfg["replay_buffer_size"]
    )

    # Resume from a previous checkpoint if specified
    resume_step = 0
    resume_from = ckpt_cfg.get("resume_from")
    if resume_from and os.path.isfile(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt_data = torch.load(resume_from, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt_data["model_state_dict"])
        if "optimizer_state_dict" in ckpt_data:
            optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            # Move optimizer state tensors to the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(model.device)
            print(f"  Optimizer state restored.")
        else:
            print(f"  WARNING: No optimizer state in checkpoint, starting fresh.")
        resume_step = ckpt_data.get("step", 0)
        print(f"  Resumed from step {resume_step}.")

    ckpt_dir = ckpt_cfg["dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    manifest_path = os.path.join(ckpt_dir, "manifest.json")

    num_games = train_cfg["selfplay_games"]
    save_every = ckpt_cfg["save_every"]
    search_iters = train_cfg["search_iterations"]
    c_puct = train_cfg["c_puct"]
    dir_alpha = train_cfg["dirichlet_alpha"]
    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs_per_game"]
    progress_path = os.path.join(ckpt_dir, "training_progress.json")

    # Anti-draw settings (default: disabled for backward compat)
    max_moves = train_cfg.get("max_moves", 0)
    rep_limit = train_cfg.get("repetition_limit", 0)
    ml_penalty = train_cfg.get("move_limit_draw_penalty", 0.0)
    rep_penalty = train_cfg.get("repetition_draw_penalty", 0.0)

    temp_threshold = train_cfg.get("temp_threshold", 15)
    max_grad_norm = train_cfg.get("max_grad_norm", 0)  # 0 = disabled
    value_loss_weight = train_cfg.get("value_loss_weight", 1.0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_games,
        eta_min=train_cfg.get("min_learning_rate", 1e-5),
    )

    selfplay_batch = train_cfg.get("selfplay_batch_size", 0)

    print(f"Starting training: {num_games} self-play games")
    if selfplay_batch > 0:
        print(f"  Device: {model.device} (batched self-play + training)")
        print(f"  Selfplay batch: {selfplay_batch} concurrent games")
    else:
        print(f"  Device: {model.device} (training), CPU (self-play workers)")
        print(f"  Workers: {num_workers} parallel self-play processes")
    print(f"  Search iterations: {search_iters}")
    print(f"  Batch size: {batch_size}")
    print(f"  Checkpoints every {save_every} games -> {ckpt_dir}/")
    if max_moves:
        print(f"  Anti-draw: max_moves={max_moves} (penalty={ml_penalty})")
    if rep_limit:
        print(f"  Anti-draw: repetition_limit={rep_limit} (penalty={rep_penalty})")

    start_time = time.time()
    latest_vloss = None
    latest_ploss = None
    games_done = resume_step

    # Write initial progress
    write_progress(progress_path, games_done, num_games, None, None, start_time, status="starting")

    pbar = tqdm(total=num_games, desc="Self-play")

    def _add_game_to_buffer(buffer, fpr):
        """Add one game's data to the replay buffer (no training)."""
        result = -fpr  # swap_result just negates
        while buffer:
            obs, action_dist = buffer.pop()
            agent.replay_buffer.add_sample(obs, action_dist, result)
            result = -result

    def _train_batch():
        """Run training epochs on current replay buffer."""
        nonlocal latest_vloss, latest_ploss
        if len(agent.replay_buffer) < batch_size:
            return
        model.train()
        value_losses = []
        policy_losses = []
        for _ in range(epochs):
            observations, actions_dist, results_batch = agent.replay_buffer.sample(batch_size)
            observations = torch.tensor(observations, device=model.device)
            actions_dist = torch.tensor(actions_dist, device=model.device)
            results_t = torch.tensor(results_batch, device=model.device)

            optimizer.zero_grad()
            values, log_policies = model(observations)
            vloss = F.mse_loss(values.squeeze(1), results_t)
            ploss = F.kl_div(log_policies, actions_dist, reduction="batchmean")
            (value_loss_weight * vloss + ploss).backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            value_losses.append(vloss.item())
            policy_losses.append(ploss.item())

        if value_losses:
            latest_vloss = sum(value_losses) / len(value_losses)
            latest_ploss = sum(policy_losses) / len(policy_losses)
        scheduler.step()

    if selfplay_batch > 0:
        # ── Batched MCTS self-play on GPU/MPS ─────────────────────────
        next_save = save_every
        try:
            while games_done < num_games:
                remaining = min(selfplay_batch, num_games - games_done)

                batch_results = _batched_selfplay(
                    model, remaining, search_iters, c_puct, dir_alpha,
                    max_moves, rep_limit, ml_penalty, rep_penalty, temp_threshold,
                    rich_obs=rich_obs,
                )

                for buffer, fpr in batch_results:
                    games_done += 1
                    _add_game_to_buffer(buffer, fpr)
                    pbar.update(1)

                _train_batch()
                model.eval()  # back to eval mode for next self-play batch (BatchNorm)

                if latest_vloss is not None:
                    append_loss_history(ckpt_dir, games_done, latest_vloss, latest_ploss)
                    tqdm.write(
                        f"Game {games_done}: value_loss={latest_vloss:.4f}  "
                        f"policy_loss={latest_ploss:.4f}"
                    )
                    if wandb_run:
                        wandb_run.log({
                            "value_loss": latest_vloss,
                            "policy_loss": latest_ploss,
                            "step": games_done,
                        })

                write_progress(progress_path, games_done, num_games,
                               latest_vloss, latest_ploss, start_time)

                if games_done >= next_save and games_done < num_games:
                    save_checkpoint(agent, games_done, config, manifest_path)
                    next_save += save_every
        finally:
            pbar.close()
    else:
        # ── Original multiprocessing self-play on CPU ─────────────────
        def _create_pool():
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return mp.Pool(
                num_workers,
                initializer=_init_worker,
                initargs=(cpu_state, game.observation_shape, game.action_space,
                          model_cfg.get("hidden", 256), model_cfg.get("num_blocks", 6),
                          model_cfg.get("value_hidden", 64), model_cfg.get("policy_hidden", 128),
                          rich_obs),
            )

        pool = _create_pool()
        train_interval = num_workers
        worker_refresh = train_cfg.get("worker_refresh", 200)
        games_since_refresh = 0

        try:
            while games_done < num_games:
                remaining = num_games - games_done
                args_iter = ((search_iters, c_puct, dir_alpha, max_moves, rep_limit, ml_penalty, rep_penalty, temp_threshold, rich_obs) for _ in range(remaining))
                games_since_train = 0

                for buffer, fpr in pool.imap_unordered(_run_selfplay, args_iter):
                    games_done += 1
                    games_since_train += 1
                    games_since_refresh += 1
                    _add_game_to_buffer(buffer, fpr)
                    pbar.update(1)

                    if games_since_train >= train_interval:
                        _train_batch()
                        games_since_train = 0

                        if latest_vloss is not None:
                            append_loss_history(ckpt_dir, games_done, latest_vloss, latest_ploss)
                            tqdm.write(
                                f"Game {games_done}: value_loss={latest_vloss:.4f}  "
                                f"policy_loss={latest_ploss:.4f}"
                            )
                            if wandb_run:
                                wandb_run.log({
                                    "value_loss": latest_vloss,
                                    "policy_loss": latest_ploss,
                                    "step": games_done,
                                })

                    write_progress(progress_path, games_done, num_games,
                                   latest_vloss, latest_ploss, start_time)

                    # Checkpoint save
                    if games_done % save_every == 0 and games_done < num_games:
                        if games_since_train > 0:
                            _train_batch()
                            games_since_train = 0
                        save_checkpoint(agent, games_done, config, manifest_path)
                        pool.terminate()
                        pool.join()
                        pool = _create_pool()
                        games_since_refresh = 0
                        break

                    # Refresh worker weights (without saving checkpoint)
                    if games_since_refresh >= worker_refresh:
                        if games_since_train > 0:
                            _train_batch()
                            games_since_train = 0
                        pool.terminate()
                        pool.join()
                        pool = _create_pool()
                        games_since_refresh = 0
                        tqdm.write(f"  [refresh] Workers updated at game {games_done}")
                        break
        finally:
            pool.terminate()
            pool.join()
            pbar.close()

    # Final checkpoint (if not already saved)
    if num_games % save_every != 0:
        save_checkpoint(agent, num_games, config, manifest_path)

    write_progress(progress_path, num_games, num_games,
                   latest_vloss, latest_ploss, start_time, status="complete")

    # Wait for any pending evaluations to finish
    if _eval_threads:
        print(f"Waiting for {len(_eval_threads)} pending evaluations to complete...")
        for t in _eval_threads:
            t.join(timeout=300)  # 5 min timeout per eval
        print("All evaluations complete.")

    print("\nTraining complete.")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
