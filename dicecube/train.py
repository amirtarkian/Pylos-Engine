"""
Train the dice-cube value/policy network.

A puzzle has no opponent to self-play against, so instead of AlphaZero's
game outcomes we use the single-agent analogue from DeepCube / DeepCubeA:

  * Generate states by scrambling random *solved* dice cubes k moves
    (k grows as the network gets better -- a curriculum).
  * Bellman targets from a frozen "target network" J':
        cost(s)   = 0                                   if s is solved
                  = min_a [ 1 + (0 if child_a solved else J'(child_a)) ]
        policy(s) = argmin_a of the same expression
  * Regress the online network onto those targets (MSE + cross-entropy).
  * When the loss falls below a threshold, copy online -> target and raise
    the scramble depth.

Because a state counts as solved whenever every face holds 1-9, the network
learns "distance to ANY goal", which is exactly what the puzzle asks for.

Usage:
    python -m dicecube.train                      # uses dicecube/config.yaml
    python -m dicecube.train --config my.yaml --resume
"""

import argparse
import copy
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dicecube.cube import (
    CENTER_POS, CORNER_POS, CORNER_VALUES, EDGE_POS, EDGE_VALUES, INVERSE_ACTION,
    MOVE_PERMS, NUM_ACTIONS, OBS_DIM, PIECE_VALUES, all_children, is_solved,
)
from dicecube.models import DiceCubeNetwork, load_model, save_model

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)


# ----------------------------------------------------------------------
# Batched data generation
# ----------------------------------------------------------------------

def random_solved_batch(n, rng, design="parity"):
    states = np.full((n, 6, 9), 5, dtype=np.int8)
    if design == "parity":
        states[:, :, list(CORNER_POS)] = rng.permuted(
            np.tile(np.array(CORNER_VALUES, dtype=np.int8), (n, 6, 1)), axis=2)
        states[:, :, list(EDGE_POS)] = rng.permuted(
            np.tile(np.array(EDGE_VALUES, dtype=np.int8), (n, 6, 1)), axis=2)
    else:
        others = [p for p in range(9) if p != CENTER_POS]
        states[:, :, others] = rng.permuted(
            np.tile(np.array(PIECE_VALUES, dtype=np.int8), (n, 6, 1)), axis=2)
    return states.reshape(n, 54)


def scrambled_batch(n, max_depth, rng, design="parity"):
    """n states, each scrambled a uniform random 0..max_depth moves."""
    states = random_solved_batch(n, rng, design)
    depths = rng.integers(0, max_depth + 1, size=n)
    prev = np.full(n, -1)
    rows = np.arange(n)
    for step in range(max_depth):
        acts = rng.integers(0, NUM_ACTIONS, size=n)
        undo = (prev >= 0) & (acts == INVERSE_ACTION[np.maximum(prev, 0)])
        acts = np.where(undo, (acts + 1) % NUM_ACTIONS, acts)
        moving = depths > step
        moved = states[rows[:, None], MOVE_PERMS[acts]]
        states = np.where(moving[:, None], moved, states)
        prev = np.where(moving, acts, prev)
    return states, depths


def bellman_targets(target_model, states, chunk=8192):
    """Cost-to-go and greedy-action targets for a batch of states."""
    n = len(states)
    children = all_children(states).reshape(n * NUM_ACTIONS, 54)
    child_solved = is_solved(children)
    costs = np.empty(len(children), dtype=np.float32)
    for i in range(0, len(children), chunk):
        costs[i:i + chunk], _ = target_model.batched_inference(children[i:i + chunk])
    costs[child_solved] = 0.0
    q = (1.0 + costs).reshape(n, NUM_ACTIONS)
    best = q.argmin(axis=1)
    value = q.min(axis=1)
    value[is_solved(states)] = 0.0
    return value, best


# ----------------------------------------------------------------------
# Evaluation: greedy rollouts guided by the value network
# ----------------------------------------------------------------------

def greedy_solve_rate(model, depths=(1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20), n=100,
                      seed=123, design="parity"):
    rng = np.random.default_rng(seed)
    out = {}
    for d in depths:
        states = random_solved_batch(n, rng, design)
        # exact-depth scrambles
        prev = np.full(n, -1)
        rows = np.arange(n)
        for _ in range(d):
            acts = rng.integers(0, NUM_ACTIONS, size=n)
            undo = (prev >= 0) & (acts == INVERSE_ACTION[np.maximum(prev, 0)])
            acts = np.where(undo, (acts + 1) % NUM_ACTIONS, acts)
            states = states[rows[:, None], MOVE_PERMS[acts]]
            prev = acts
        solved = is_solved(states)
        cur = states.copy()
        for _ in range(d * 2 + 5):
            active = ~solved
            if not active.any():
                break
            ch = all_children(cur[active])
            k = len(ch)
            flat = ch.reshape(k * NUM_ACTIONS, 54)
            c, _ = model.batched_inference(flat)
            c[is_solved(flat)] = -1.0
            pick = c.reshape(k, NUM_ACTIONS).argmin(axis=1)
            cur[active] = ch[np.arange(k), pick]
            solved[active] = is_solved(cur[active])
        out[str(d)] = round(float(solved.mean()), 3)
    return out


# ----------------------------------------------------------------------
# Progress files (read by the web UI)
# ----------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_config(path=None):
    path = path or os.path.join(PACKAGE_DIR, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train the dice-cube solver network")
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", action="store_true", help="continue from latest.pth")
    parser.add_argument("--steps", type=int, default=None, help="override training.steps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg, mcfg = cfg["training"], cfg.get("model", {})
    if args.steps:
        tcfg["steps"] = args.steps
    ckpt_dir = os.path.join(PROJECT_DIR, cfg["checkpoints"]["dir"])
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, "latest.pth")
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    history_path = os.path.join(ckpt_dir, "loss_history.jsonl")
    progress_path = os.path.join(ckpt_dir, "training_progress.json")

    if tcfg.get("threads"):
        torch.set_num_threads(tcfg["threads"])
    rng = np.random.default_rng(tcfg.get("seed"))
    design = tcfg.get("design", "parity")

    start_step, max_depth = 0, tcfg["start_depth"]
    if args.resume and os.path.exists(latest_path):
        model, ckpt = load_model(latest_path, device="cpu")
        model.to(model.device)
        start_step = ckpt.get("step", 0)
        max_depth = ckpt.get("max_depth", max_depth)
        print(f"Resumed from step {start_step}, depth {max_depth}")
    else:
        model = DiceCubeNetwork(OBS_DIM, NUM_ACTIONS, device="cpu", **mcfg)

    target = copy.deepcopy(model)
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"],
                                 weight_decay=tcfg.get("weight_decay", 0.0))

    manifest = {"checkpoints": []}
    if args.resume and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    start_time = time.time()
    last_update_step = start_step
    recent = []
    for step in range(start_step + 1, tcfg["steps"] + 1):
        states, _ = scrambled_batch(tcfg["batch_size"], max_depth, rng, design)
        value_t, action_t = bellman_targets(target, states)

        model.train()
        cost, log_policy = model(torch.as_tensor(states, device=model.device))
        value_loss = F.mse_loss(cost, torch.as_tensor(value_t, device=model.device))
        policy_loss = F.nll_loss(log_policy, torch.as_tensor(action_t, device=model.device))
        optimizer.zero_grad()
        (value_loss + tcfg.get("policy_weight", 1.0) * policy_loss).backward()
        optimizer.step()

        recent.append((value_loss.item(), policy_loss.item()))

        # DeepCubeA-style target update + curriculum
        check_every = tcfg["target_check_every"]
        if step % check_every == 0:
            vl = float(np.mean([r[0] for r in recent[-check_every:]]))
            pl = float(np.mean([r[1] for r in recent[-check_every:]]))
            stale = step - last_update_step >= tcfg["max_steps_per_target"]
            updated = False
            if vl < tcfg["target_loss_threshold"] or stale:
                target.load_state_dict(model.state_dict())
                last_update_step = step
                updated = True
                if max_depth < tcfg["max_depth"]:
                    max_depth += 1
            with open(history_path, "a") as f:
                f.write(json.dumps({"step": step, "value_loss": round(vl, 5),
                                    "policy_loss": round(pl, 5), "depth": max_depth}) + "\n")
            elapsed = time.time() - start_time
            rate = (step - start_step) / max(elapsed, 1e-6)
            write_json(progress_path, {
                "status": "training", "step": step, "total_steps": tcfg["steps"],
                "percent": round(step / tcfg["steps"] * 100, 1), "max_depth": max_depth,
                "value_loss": round(vl, 5), "policy_loss": round(pl, 5),
                "steps_per_second": round(rate, 2),
                "eta_seconds": round((tcfg["steps"] - step) / max(rate, 1e-6)),
                "timestamp": _now(),
            })
            print(f"step {step:6d}  depth {max_depth:2d}  value {vl:.4f}  policy {pl:.4f}"
                  f"  {'[target updated]' if updated else ''}  {rate:.1f} it/s", flush=True)

        if step % cfg["checkpoints"]["save_every"] == 0 or step == tcfg["steps"]:
            extra = {"step": step, "max_depth": max_depth}
            filename = f"checkpoint_{step:06d}.pth"
            save_model(model, latest_path, extra)
            if cfg["checkpoints"].get("keep_all", False):
                save_model(model, os.path.join(ckpt_dir, filename), extra)
            rates = greedy_solve_rate(model, design=design)
            manifest["checkpoints"].append({
                "file": filename, "step": step, "max_depth": max_depth,
                "greedy_solve_rate": rates, "timestamp": _now(),
            })
            write_json(manifest_path, manifest)
            print(f"  saved step {step}; greedy solve rate by depth: {rates}", flush=True)

    write_json(progress_path, {"status": "complete", "step": tcfg["steps"],
                               "total_steps": tcfg["steps"], "percent": 100.0,
                               "max_depth": max_depth, "timestamp": _now()})


if __name__ == "__main__":
    main()
