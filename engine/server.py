"""
FastAPI WebSocket server for board game engines.

Provides:
  - GET /checkpoints        — returns checkpoint manifest
  - GET /                   — serves web/index.html
  - /src/*                  — serves web/src/ static files
  - WS  /game               — WebSocket game session
"""
from __future__ import annotations

import sys
import os
import json
import time
import asyncio

# Ensure engine modules are importable with plain imports
sys.path.insert(0, os.path.dirname(__file__))
# Ensure top-level packages (games.*) are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from collections import defaultdict

from models import GameNetwork
from agents import AlphaZeroAgent
from mcts import play as mcts_play, search as mcts_search
import copy

# ---------------------------------------------------------------------------
# Dynamic game loading
# ---------------------------------------------------------------------------

def _load_game_class(game_name):
    """Dynamically load a game class from games/{game_name}/."""
    import importlib
    mod = importlib.import_module(f"games.{game_name}")
    return mod.GAME_CLASS


# Module-level game class — set from --game CLI arg or default
_game_name = "pylos"
game_cls = _load_game_class(_game_name)


# ---------------------------------------------------------------------------
# Game loop limits
# ---------------------------------------------------------------------------

MAX_MOVES = 200            # declare draw after this many moves
REPETITION_LIMIT = 5       # declare draw if same board state occurs this many times

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ENGINE_DIR)
WEB_DIR = os.path.join(PROJECT_DIR, "web")

# Module-level: current game name, set from CLI
_game_name = "pylos"


def _find_checkpoint_dirs(game_name):
    """Scan for checkpoint directories under engine/ and games/{game_name}/."""
    dirs = []
    # Look in engine/ for legacy checkpoints
    if os.path.isdir(ENGINE_DIR):
        for name in os.listdir(ENGINE_DIR):
            if name.startswith('checkpoints'):
                path = os.path.join(ENGINE_DIR, name)
                if os.path.isdir(path):
                    dirs.append(path)
    # Look in games/{game_name}/ for new checkpoints
    game_dir = os.path.join(PROJECT_DIR, 'games', game_name)
    if os.path.isdir(game_dir):
        for name in os.listdir(game_dir):
            if name.startswith('checkpoints'):
                path = os.path.join(game_dir, name)
                if os.path.isdir(path):
                    dirs.append(path)
    return dirs


def _build_training_runs(game_name):
    """Build the list of (label, directory) training runs from checkpoint dirs."""
    runs = []
    for ckpt_dir in sorted(_find_checkpoint_dirs(game_name)):
        dirname = os.path.basename(ckpt_dir)
        # Extract version label: "checkpoints" -> "v1", "checkpoints_v2" -> "v2", etc.
        if dirname == "checkpoints":
            label = "v1"
        elif dirname.startswith("checkpoints_"):
            label = dirname.split("_", 1)[1]
        else:
            label = dirname
        runs.append((label, ckpt_dir))
    return runs


# ---------------------------------------------------------------------------
# Hint system: shared checkpoint path helper + cached hint agent
# ---------------------------------------------------------------------------

def _resolve_checkpoint_path(label: str, ckpt_dir: str, filename: str) -> str:
    if label == "v1":
        return filename
    return os.path.basename(ckpt_dir) + "/" + filename


def _find_best_checkpoint() -> str | None:
    best_elo, best_file = -1, None
    for label, ckpt_dir in TRAINING_RUNS:
        manifest_path = os.path.join(ckpt_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path) as f:
            data = json.load(f)
        for cp in data.get("checkpoints", []):
            if not os.path.isfile(os.path.join(ckpt_dir, cp["file"])):
                continue
            if cp.get("elo", 0) > best_elo:
                best_elo = cp["elo"]
                best_file = _resolve_checkpoint_path(label, ckpt_dir, cp["file"])
    return best_file


_hint_agent: AlphaZeroAgent | None = None
_hint_agent_lock = asyncio.Lock()
_hint_cache: dict[str, list] = {}
_HINT_CACHE_MAX = 256
_HINT_SEARCH_ITERATIONS = 128

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()

# Mount static files for the frontend (only if the directory exists)
if os.path.isdir(os.path.join(WEB_DIR, "src")):
    app.mount("/src", StaticFiles(directory=os.path.join(WEB_DIR, "src")), name="static")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/checkpoints")
async def get_checkpoints():
    """Return merged checkpoint manifests from all training runs."""
    training_runs = _build_training_runs(_game_name)
    all_checkpoints = []

    for label, ckpt_dir in training_runs:
        manifest_path = os.path.join(ckpt_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r") as f:
                data = json.load(f)
                for cp in data.get("checkpoints", []):
                    if not os.path.isfile(os.path.join(ckpt_dir, cp["file"])):
                        continue
                    cp["version"] = label
                    cp["file"] = _resolve_checkpoint_path(label, ckpt_dir, cp["file"])
                    all_checkpoints.append(cp)

    return {"checkpoints": all_checkpoints}


import time as _time
from collections import deque

_speed_samples = {}  # run_label -> deque of (step, timestamp), last 5 samples


@app.get("/training/status")
async def get_training_status():
    """Return training progress for all runs, with rolling average speed."""
    training_runs = _build_training_runs(_game_name)
    runs = {}

    for label, ckpt_dir in training_runs:
        path = os.path.join(ckpt_dir, "training_progress.json")
        if os.path.isfile(path):
            with open(path, "r") as f:
                data = json.load(f)

            now = _time.time()
            step = data.get("current_game", 0)
            total = data.get("total_games", 0)

            if label not in _speed_samples:
                _speed_samples[label] = deque(maxlen=6)  # 6 samples = 5 intervals

            samples = _speed_samples[label]
            # Only record if step advanced
            if not samples or step > samples[-1][0]:
                samples.append((step, now))

            # Compute average speed over the window
            if len(samples) >= 2:
                oldest = samples[0]
                newest = samples[-1]
                dt = newest[1] - oldest[1]
                dg = newest[0] - oldest[0]
                if dt > 0 and dg > 0:
                    avg_rate = dg / dt
                    data["games_per_second"] = round(avg_rate, 3)
                    if avg_rate > 0:
                        data["eta_seconds"] = round((total - step) / avg_rate, 1)

            runs[label] = data
        else:
            runs[label] = {"status": "idle"}

    return runs


@app.get("/training/loss-history")
async def get_loss_history():
    """Return persistent loss history for all training runs, downsampled for charts."""
    training_runs = _build_training_runs(_game_name)
    max_points = 500  # enough for chart resolution
    result = {}
    for label, ckpt_dir in training_runs:
        history_path = os.path.join(ckpt_dir, "loss_history.jsonl")
        entries = []
        if os.path.isfile(history_path):
            with open(history_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        # Downsample if too many points
        if len(entries) > max_points:
            step = len(entries) / max_points
            sampled = [entries[int(i * step)] for i in range(max_points - 1)]
            sampled.append(entries[-1])  # always include latest
            entries = sampled
        result[label] = entries
    return result


@app.get("/")
async def index():
    """Serve the frontend page for the active game."""
    if _game_name == "blokus_duo":
        path = os.path.join(WEB_DIR, "blokus.html")
    else:
        path = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(path):
        return FileResponse(path)
    return JSONResponse({"error": "frontend not found"}, status_code=404)


@app.get("/blokus")
async def blokus_page():
    """Serve the Blokus Duo frontend."""
    path = os.path.join(WEB_DIR, "blokus.html")
    if os.path.isfile(path):
        return FileResponse(path)
    return JSONResponse({"error": "blokus.html not found"}, status_code=404)


@app.get("/training")
async def training_dashboard():
    """Serve the dedicated training dashboard page."""
    path = os.path.join(WEB_DIR, "training.html")
    if os.path.isfile(path):
        return FileResponse(path)
    return JSONResponse({"error": "training.html not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Helper: load AI agent
# ---------------------------------------------------------------------------

def load_ai_agent(checkpoint_file: str) -> AlphaZeroAgent:
    """Load a GameNetwork from a checkpoint file and return an AlphaZeroAgent.

    Reads model architecture from checkpoint metadata (V5+).
    Falls back to default architecture for older checkpoints.
    """
    checkpoints_base = os.path.join(ENGINE_DIR, "checkpoints")

    if "/" in checkpoint_file:
        # v2+ path like "checkpoints_v2/checkpoint_00500.pth"
        checkpoint_path = os.path.join(ENGINE_DIR, checkpoint_file)
    else:
        checkpoint_path = os.path.join(checkpoints_base, checkpoint_file)

    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Handle both formats: raw state_dict or {"model_state_dict": ..., "model_config": ...}
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
        model_cfg = checkpoint_data.get("model_config", {})
    else:
        state_dict = checkpoint_data
        model_cfg = {}

    # Read model architecture from checkpoint metadata, or use defaults
    hidden = model_cfg.get("hidden", 256)
    num_blocks = model_cfg.get("num_blocks", 6)
    value_hidden = model_cfg.get("value_hidden", 64)
    policy_hidden = model_cfg.get("policy_hidden", 128)
    rich_obs = model_cfg.get("rich_obs", False)

    game_tmp = game_cls(rich_obs=rich_obs)
    model = GameNetwork(
        input_shape=game_tmp.observation_shape,
        action_space=game_tmp.action_space,
        hidden=hidden,
        num_blocks=num_blocks,
        value_hidden=value_hidden,
        policy_hidden=policy_hidden,
    )
    model.device = torch.device("cpu")
    model.to(model.device)
    model.load_state_dict(state_dict)
    model.eval()
    return AlphaZeroAgent(model, rich_obs=rich_obs)


# ---------------------------------------------------------------------------
# Helper: AI turn
# ---------------------------------------------------------------------------

async def _do_ai_turn(
    ws: WebSocket,
    game,
    agent: AlphaZeroAgent,
    search_iters: int,
) -> bool:
    """Run MCTS, apply the move, and send ai_move + state messages.

    Returns True if the game is over after this move.
    """
    t0 = time.monotonic()
    action = await asyncio.to_thread(
        mcts_play, game, agent, search_iters
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if action is None:
        # No legal moves -- game over (current player loses)
        result = game.get_result()
        winner = "white" if result == 1 else "black"
        await ws.send_json({
            "type": "game_over",
            "winner": winner,
            "reason": "no_legal_moves",
        })
        return True

    action_json = game.action_to_json(action)

    # Step WITHOUT auto-removal so we can capture the formation positions
    # before any pieces are removed (removal could break the pattern visually).
    # Games without removal support simply ignore the kwarg.
    game.step(action, auto_remove=False)

    # Capture formation positions and perform greedy removal manually
    # (only for games that support a removal phase, e.g. Pylos)
    formation_json = []
    removed_json = []
    if hasattr(game, 'pending_removal') and game.pending_removal:
        # Capture the completed pattern (square/line) for orange highlight
        if hasattr(game, 'get_formation_positions'):
            for fl, fr, fc in game.get_formation_positions():
                formation_json.append({"level": fl, "row": fr, "col": fc})

        # Greedy AI removal: remove up to 2 pieces (lowest level first)
        for _ in range(2):
            removable = game.get_pending_removable()
            if not removable:
                break
            removable.sort(key=lambda x: (x[0], x[1], x[2]))
            rl, rr, rc = removable[0]
            game.step_removal(rl, rr, rc)
            removed_json.append({"level": rl, "row": rr, "col": rc})

        # End removal phase if still pending (chose fewer than max)
        if game.pending_removal:
            game.skip_removal()

    ai_move_msg = {
        "type": "ai_move",
        "action": action_json,
        "thinking_time_ms": elapsed_ms,
    }
    if removed_json:
        ai_move_msg["removed"] = removed_json
    if formation_json:
        ai_move_msg["formation"] = formation_json

    await ws.send_json(ai_move_msg)

    # Always send the board state so the final piece animates
    await ws.send_json(game.make_state_msg())

    # Check game over
    result = game.get_result()
    if result is not None:
        winner = "white" if result == 1 else "black"
        if hasattr(game, 'top_filled') and game.top_filled():
            reason = "apex_placed"
        else:
            reason = "game_over"
        await ws.send_json({
            "type": "game_over",
            "winner": winner,
            "reason": reason,
        })
        return True

    return False


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/game")
async def game_ws(ws: WebSocket):
    await ws.accept()

    game = None
    agent: AlphaZeroAgent | None = None
    agent_white: AlphaZeroAgent | None = None
    agent_black: AlphaZeroAgent | None = None
    mode: str | None = None
    search_iters: int = 32
    human_color: int = 1  # 1=white, -1=black
    ai_vs_ai_task: asyncio.Task | None = None
    paused = asyncio.Event()
    paused.set()  # starts unpaused (set = running)
    step_event = asyncio.Event()  # triggers one AI move while paused
    shared_delay = [1500]  # mutable container so AI loop sees updates

    # Per-session action history for take-back.
    action_stack: list = []

    # Hint debounce: prevents concurrent MCTS hint computations per session
    hint_computing = False

    try:
        while True:
            raw = await ws.receive_json()
            msg_type = raw.get("type")

            # ----------------------------------------------------------
            # new_game
            # ----------------------------------------------------------
            if msg_type == "new_game":
                # Cancel any running ai_vs_ai loop
                if ai_vs_ai_task is not None and not ai_vs_ai_task.done():
                    ai_vs_ai_task.cancel()
                    ai_vs_ai_task = None

                game = game_cls()
                action_stack = []
                mode = raw.get("mode", "human_vs_human")
                agent = None

                if mode == "human_vs_ai":
                    checkpoint = raw.get("checkpoint")
                    if checkpoint:
                        try:
                            agent = load_ai_agent(checkpoint)
                        except Exception as e:
                            await ws.send_json({"type": "error", "message": f"Failed to load checkpoint: {e}"})
                            continue
                    search_iters = raw.get("search_iterations", 32)
                    hc = raw.get("human_color", "white")
                    human_color = 1 if hc == "white" else -1

                elif mode == "ai_vs_ai":
                    # Support dual checkpoints: one for white, one for black
                    ckpt_white = raw.get("checkpoint_white") or raw.get("checkpoint")
                    ckpt_black = raw.get("checkpoint_black") or raw.get("checkpoint")
                    try:
                        if ckpt_white:
                            agent_white = load_ai_agent(ckpt_white)
                        if ckpt_black:
                            agent_black = load_ai_agent(ckpt_black)
                    except Exception as e:
                        await ws.send_json({"type": "error", "message": f"Failed to load checkpoint: {e}"})
                        continue
                    # Fallback: single agent for both sides
                    agent = agent_white or agent_black
                    search_iters = raw.get("search_iterations", 32)
                    shared_delay[0] = raw.get("delay_ms", 1500)

                # Send initial state
                await ws.send_json(game.make_state_msg())

                # If human_vs_ai and AI goes first
                if mode == "human_vs_ai" and agent is not None and game.turn != human_color:
                    await _do_ai_turn(ws, game, agent, search_iters)

                # If ai_vs_ai, start the async loop
                if mode == "ai_vs_ai" and (agent_white or agent_black):
                    async def _ai_vs_ai_loop(
                        _ws=ws, _game=game,
                        _agent_w=agent_white, _agent_b=agent_black,
                        _agent_fallback=agent,
                        _iters=search_iters, _delay_ref=shared_delay,
                        _paused=paused, _step=step_event,
                    ):
                        move_count = 0
                        state_counts = defaultdict(int)
                        try:
                            while True:
                                # Wait for either resume or single-step
                                while not _paused.is_set() and not _step.is_set():
                                    await asyncio.sleep(0.05)
                                was_step = _step.is_set()
                                _step.clear()
                                if not was_step:
                                    await asyncio.sleep(_delay_ref[0] / 1000.0)
                                # Pick agent based on whose turn it is
                                if _game.turn == 1:  # white
                                    cur_agent = _agent_w or _agent_fallback
                                else:  # black
                                    cur_agent = _agent_b or _agent_fallback
                                done = await _do_ai_turn(_ws, _game, cur_agent, _iters)
                                if done:
                                    break

                                move_count += 1
                                bh = _game.board_hash()
                                state_counts[bh] += 1

                                # Draw by repetition
                                if state_counts[bh] >= REPETITION_LIMIT:
                                    await _ws.send_json({
                                        "type": "game_over",
                                        "winner": "draw",
                                        "reason": "repetition",
                                    })
                                    break

                                # Draw by move limit
                                if move_count >= MAX_MOVES:
                                    await _ws.send_json({
                                        "type": "game_over",
                                        "winner": "draw",
                                        "reason": "move_limit",
                                    })
                                    break
                        except asyncio.CancelledError:
                            pass

                    ai_vs_ai_task = asyncio.create_task(_ai_vs_ai_loop())

            # ----------------------------------------------------------
            # move (human)
            # ----------------------------------------------------------
            elif msg_type == "move":
                if game is None:
                    await ws.send_json({"type": "error", "message": "No active game"})
                    continue

                # Block moves while removal phase is active
                if hasattr(game, 'pending_removal') and game.pending_removal:
                    await ws.send_json(game.make_removal_msg())
                    continue

                action_msg = raw.get("action", {})
                action = game.action_from_json(action_msg)

                if action is None:
                    await ws.send_json({"type": "error", "message": "Illegal move"})
                    continue

                game.step(action, auto_remove=False)
                action_stack.append(int(action))

                # Check for interactive removal phase BEFORE game-over,
                # since removing pieces may free up moves for the opponent
                if hasattr(game, 'pending_removal') and game.pending_removal:
                    await ws.send_json(game.make_removal_msg())
                    continue

                # Always send state so the final piece animates
                await ws.send_json(game.make_state_msg())

                # Check game over
                result = game.get_result()
                if result is not None:
                    winner = "white" if result == 1 else "black"
                    if hasattr(game, 'top_filled') and game.top_filled():
                        reason = "apex_placed"
                    else:
                        reason = "game_over"
                    await ws.send_json({
                        "type": "game_over",
                        "winner": winner,
                        "reason": reason,
                    })
                    continue

                # If human_vs_ai and it's now the AI's turn
                if mode == "human_vs_ai" and agent is not None and game.turn != human_color:
                    await _do_ai_turn(ws, game, agent, search_iters)

            # ----------------------------------------------------------
            # remove (during removal phase)
            # ----------------------------------------------------------
            elif msg_type == "remove":
                if game is None or not getattr(game, 'pending_removal', False):
                    await ws.send_json({"type": "error", "message": "Not in removal phase"})
                    continue

                level = raw.get("level")
                row = raw.get("row")
                col = raw.get("col")

                if not game.step_removal(level, row, col):
                    await ws.send_json({"type": "error", "message": "Cannot remove that piece"})
                    continue
                action_stack.append(f"remove:{level},{row},{col}")

                # Still in removal phase?
                if game.pending_removal:
                    await ws.send_json(game.make_removal_msg())
                    continue

                # Removal phase ended — send state, then check game over
                await ws.send_json(game.make_state_msg())

                result = game.get_result()
                if result is not None:
                    winner = "white" if result == 1 else "black"
                    if hasattr(game, 'top_filled') and game.top_filled():
                        reason = "apex_placed"
                    else:
                        reason = "game_over"
                    await ws.send_json({
                        "type": "game_over",
                        "winner": winner,
                        "reason": reason,
                    })
                    continue

                # If human_vs_ai and it's now the AI's turn
                if mode == "human_vs_ai" and agent is not None and game.turn != human_color:
                    await _do_ai_turn(ws, game, agent, search_iters)

            # ----------------------------------------------------------
            # skip_removal (end removal phase early)
            # ----------------------------------------------------------
            elif msg_type == "skip_removal":
                if game is None or not getattr(game, 'pending_removal', False):
                    await ws.send_json({"type": "error", "message": "Not in removal phase"})
                    continue

                game.skip_removal()
                action_stack.append("skip_removal")

                await ws.send_json(game.make_state_msg())

                # Check game over after removal phase ends
                result = game.get_result()
                if result is not None:
                    winner = "white" if result == 1 else "black"
                    if hasattr(game, 'top_filled') and game.top_filled():
                        reason = "apex_placed"
                    else:
                        reason = "game_over"
                    await ws.send_json({
                        "type": "game_over",
                        "winner": winner,
                        "reason": reason,
                    })
                    continue

                # If human_vs_ai and it's now the AI's turn
                if mode == "human_vs_ai" and agent is not None and game.turn != human_color:
                    await _do_ai_turn(ws, game, agent, search_iters)

            # ----------------------------------------------------------
            # pause / resume
            # ----------------------------------------------------------
            elif msg_type == "pause":
                paused.clear()  # clear = paused (wait() will block)
                await ws.send_json({"type": "paused", "paused": True})

            elif msg_type == "resume":
                paused.set()  # set = running (wait() passes through)
                await ws.send_json({"type": "paused", "paused": False})

            elif msg_type == "step":
                # Play one AI move while keeping the game paused
                if not paused.is_set():
                    step_event.set()

            elif msg_type == "set_delay":
                shared_delay[0] = max(0, int(raw.get("delay_ms", 1500)))

            elif msg_type == "take_back":
                if game is None:
                    await ws.send_json({"type": "error", "message": "No active game"})
                    continue
                if mode != "human_vs_human":
                    await ws.send_json({"type": "error", "message": "Take-back is only available in Human vs Human"})
                    continue
                if not action_stack:
                    await ws.send_json({"type": "error", "message": "Nothing to take back"})
                    continue

                # Pop trailing remove/skip_removal entries, then the placement/raise that started the turn.
                while action_stack and isinstance(action_stack[-1], str):
                    action_stack.pop()
                if action_stack:
                    action_stack.pop()  # the placement/raise action int

                # Replay remaining actions from a fresh game.
                game = game_cls()
                for entry in action_stack:
                    if isinstance(entry, int):
                        game.step(entry, auto_remove=False)
                    elif isinstance(entry, str) and entry.startswith("remove:"):
                        _, coords = entry.split(":", 1)
                        l, r, c = (int(x) for x in coords.split(","))
                        assert game.step_removal(l, r, c), f"replay failed at {entry!r}"
                    elif entry == "skip_removal":
                        assert game.skip_removal(), f"replay failed at {entry!r}"
                    else:
                        raise ValueError(f"unknown action_stack entry: {entry!r}")

                state = game.make_state_msg()
                state["take_back"] = True
                await ws.send_json(state)

            # ----------------------------------------------------------
            # hint
            # ----------------------------------------------------------
            elif msg_type == "hint":
                global _hint_agent
                if game is None:
                    await ws.send_json({"type": "error", "message": "No active game"})
                    continue
                if game.pending_removal:
                    await ws.send_json({"type": "error", "message": "Hints not available during removal"})
                    continue
                if hint_computing:
                    await ws.send_json({"type": "error", "message": "Hint already computing"})
                    continue

                hint_computing = True
                t0 = time.monotonic()
                try:
                    async with _hint_agent_lock:
                        if _hint_agent is None:
                            best_ckpt = _find_best_checkpoint()
                            if best_ckpt is None:
                                await ws.send_json({"type": "error", "message": "No checkpoints available"})
                                continue
                            _hint_agent = load_ai_agent(best_ckpt)

                    board_hash = _board_hash(game)
                    if board_hash in _hint_cache:
                        all_moves = _hint_cache[board_hash]
                    else:
                        game_copy = copy.deepcopy(game)
                        root = await asyncio.to_thread(
                            mcts_search, game_copy,
                            eval_fn=_hint_agent.eval_fn,
                            iterations=_HINT_SEARCH_ITERATIONS,
                        )
                        if root.children is None:
                            await ws.send_json({
                                "type": "hint_result", "moves": [],
                                "thinking_time_ms": int((time.monotonic() - t0) * 1000),
                            })
                            continue

                        visits = root.children_visits
                        scores = visits / visits.sum()
                        all_moves = []
                        for i, action in enumerate(root.children_actions):
                            move_json = _action_to_json(game, action)
                            move_json["score"] = float(scores[i])
                            move_json["visits"] = int(visits[i])
                            move_json["win_rate"] = float(root.children_values[i])
                            move_json["action"] = int(action)
                            all_moves.append(move_json)
                        all_moves.sort(key=lambda m: m["score"], reverse=True)

                        if len(_hint_cache) >= _HINT_CACHE_MAX:
                            oldest_key = next(iter(_hint_cache))
                            del _hint_cache[oldest_key]
                        _hint_cache[board_hash] = all_moves

                    await ws.send_json({
                        "type": "hint_result",
                        "moves": all_moves,
                        "thinking_time_ms": int((time.monotonic() - t0) * 1000),
                    })
                finally:
                    hint_computing = False

            else:
                await ws.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        pass
    finally:
        if ai_vs_ai_task is not None and not ai_vs_ai_task.done():
            ai_vs_ai_task.cancel()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Board game WebSocket server")
    parser.add_argument("--game", default="pylos", help="Game to load (e.g. pylos, blokus_duo)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    _game_name = args.game
    game_cls = _load_game_class(args.game)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
