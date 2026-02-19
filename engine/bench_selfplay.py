"""Benchmark: CPU multiprocessing vs MPS batched self-play.

Runs a small number of complete self-play games both ways and reports
games/sec for each approach.
"""

import sys
import os
import time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from game import PylosGame
from models import PylosNetwork
from train import _run_selfplay, _init_worker, _batched_selfplay

# ── Config ────────────────────────────────────────────────────────
NUM_GAMES = 24          # total games per benchmark
SEARCH_ITERS = 64       # match v4 config
C_PUCT = 1.5
DIR_ALPHA = 0.3
MAX_MOVES = 200
REP_LIMIT = 5
ML_PENALTY = 0.0
REP_PENALTY = 0.0
TEMP_THRESHOLD = 15
BATCH_SIZES = [4, 8, 16]  # batched self-play sizes to test


def bench_cpu_workers(model, num_workers):
    """Benchmark CPU multiprocessing self-play."""
    game = PylosGame()
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}

    pool = mp.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(cpu_state, game.observation_shape, game.action_space),
    )

    args = (SEARCH_ITERS, C_PUCT, DIR_ALPHA, MAX_MOVES, REP_LIMIT,
            ML_PENALTY, REP_PENALTY, TEMP_THRESHOLD)
    args_iter = [args] * NUM_GAMES

    start = time.perf_counter()
    results = list(pool.imap_unordered(_run_selfplay, args_iter))
    elapsed = time.perf_counter() - start

    pool.terminate()
    pool.join()

    moves = sum(len(buf) for buf, _ in results)
    return elapsed, moves


def bench_batched_mps(model, batch_size):
    """Benchmark MPS/GPU batched self-play."""
    # Ensure model is on MPS
    if not torch.backends.mps.is_available():
        return None, 0

    model.device = torch.device("mps")
    model.to(model.device)
    model.eval()

    total_moves = 0
    games_done = 0

    start = time.perf_counter()
    while games_done < NUM_GAMES:
        remaining = min(batch_size, NUM_GAMES - games_done)
        results = _batched_selfplay(
            model, remaining, SEARCH_ITERS, C_PUCT, DIR_ALPHA,
            MAX_MOVES, REP_LIMIT, ML_PENALTY, REP_PENALTY, TEMP_THRESHOLD,
        )
        for buf, _ in results:
            total_moves += len(buf)
        games_done += remaining
    elapsed = time.perf_counter() - start

    return elapsed, total_moves


def bench_batched_cpu(model, batch_size):
    """Benchmark CPU batched self-play (for comparison)."""
    model.device = torch.device("cpu")
    model.to(model.device)
    model.eval()

    total_moves = 0
    games_done = 0

    start = time.perf_counter()
    while games_done < NUM_GAMES:
        remaining = min(batch_size, NUM_GAMES - games_done)
        results = _batched_selfplay(
            model, remaining, SEARCH_ITERS, C_PUCT, DIR_ALPHA,
            MAX_MOVES, REP_LIMIT, ML_PENALTY, REP_PENALTY, TEMP_THRESHOLD,
        )
        for buf, _ in results:
            total_moves += len(buf)
        games_done += remaining
    elapsed = time.perf_counter() - start

    return elapsed, total_moves


def main():
    game = PylosGame()
    model = PylosNetwork(game.observation_shape, game.action_space)

    num_workers = min(os.cpu_count() - 2, 8)

    print(f"Benchmark: {NUM_GAMES} games, {SEARCH_ITERS} search iters")
    print(f"Device: {torch.device('mps') if torch.backends.mps.is_available() else 'cpu'}")
    print(f"CPU cores: {os.cpu_count()}, workers: {num_workers}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ── CPU multiprocessing ──────────────────────────────────────
    print(f"[CPU workers x{num_workers}] Running {NUM_GAMES} games...")
    elapsed, moves = bench_cpu_workers(model, num_workers)
    cpu_gps = NUM_GAMES / elapsed
    print(f"  Time: {elapsed:.1f}s | {cpu_gps:.2f} games/s | {moves} total moves")
    print()

    # ── Batched MPS ──────────────────────────────────────────────
    for bs in BATCH_SIZES:
        print(f"[MPS batched, batch={bs}] Running {NUM_GAMES} games...")
        elapsed, moves = bench_batched_mps(model, bs)
        if elapsed is None:
            print("  MPS not available, skipping")
            continue
        mps_gps = NUM_GAMES / elapsed
        speedup = mps_gps / cpu_gps
        print(f"  Time: {elapsed:.1f}s | {mps_gps:.2f} games/s | {moves} total moves | {speedup:.2f}x vs CPU workers")
        print()

    # ── Batched CPU (for reference) ──────────────────────────────
    for bs in BATCH_SIZES:
        print(f"[CPU batched, batch={bs}] Running {NUM_GAMES} games...")
        elapsed, moves = bench_batched_cpu(model, bs)
        batched_cpu_gps = NUM_GAMES / elapsed
        speedup = batched_cpu_gps / cpu_gps
        print(f"  Time: {elapsed:.1f}s | {batched_cpu_gps:.2f} games/s | {moves} total moves | {speedup:.2f}x vs CPU workers")
        print()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
