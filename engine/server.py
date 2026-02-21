"""
FastAPI WebSocket server for the Pylos board game engine.

Provides:
  - GET /checkpoints        — returns checkpoint manifest
  - GET /                   — serves web/index.html
  - /src/*                  — serves web/src/ static files
  - WS  /game               — WebSocket game session
"""

import sys
import os
import json
import time
import asyncio

# Ensure engine modules are importable with plain imports
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from collections import defaultdict

from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgent
from mcts import play as mcts_play

# ---------------------------------------------------------------------------
# Game loop limits
# ---------------------------------------------------------------------------

MAX_MOVES = 200            # declare draw after this many moves
REPETITION_LIMIT = 5       # declare draw if same board state occurs this many times


def _board_hash(game: PylosGame) -> str:
    """Return a hashable string representing the current board + turn."""
    parts = []
    for level in range(4):
        parts.append(game.board[level].tobytes())
    parts.append(bytes([game.turn + 1]))  # 0 for black, 2 for white
    return b"".join(parts)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ENGINE_DIR)
WEB_DIR = os.path.join(PROJECT_DIR, "web")
CHECKPOINTS_DIR = os.path.join(ENGINE_DIR, "checkpoints")
CHECKPOINTS_V2_DIR = os.path.join(ENGINE_DIR, "checkpoints_v2")
CHECKPOINTS_V3_DIR = os.path.join(ENGINE_DIR, "checkpoints_v3")
CHECKPOINTS_V4_DIR = os.path.join(ENGINE_DIR, "checkpoints_v4")
CHECKPOINTS_V5_DIR = os.path.join(ENGINE_DIR, "checkpoints_v5")

# All training runs: (label, directory)
TRAINING_RUNS = [
    ("v1", CHECKPOINTS_DIR),
    ("v2", CHECKPOINTS_V2_DIR),
    ("v3", CHECKPOINTS_V3_DIR),
    ("v4", CHECKPOINTS_V4_DIR),
    ("v5", CHECKPOINTS_V5_DIR),
]

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
    all_checkpoints = []

    for label, ckpt_dir in TRAINING_RUNS:
        manifest_path = os.path.join(ckpt_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r") as f:
                data = json.load(f)
                for cp in data.get("checkpoints", []):
                    cp["version"] = label
                    # Prefix non-v1 paths so loader can resolve them
                    if label != "v1":
                        dirname = os.path.basename(ckpt_dir)
                        cp["file"] = dirname + "/" + cp["file"]
                    all_checkpoints.append(cp)

    return {"checkpoints": all_checkpoints}


import time as _time
from collections import deque

_speed_samples = {}  # run_label -> deque of (step, timestamp), last 5 samples


@app.get("/training/status")
async def get_training_status():
    """Return training progress for all runs, with rolling average speed."""
    runs = {}

    for label, ckpt_dir in TRAINING_RUNS:
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
    max_points = 500  # enough for chart resolution
    result = {}
    for label, ckpt_dir in TRAINING_RUNS:
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
    """Serve the frontend index.html."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse({"error": "index.html not found"}, status_code=404)


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
    """Load a PylosNetwork from a checkpoint file and return an AlphaZeroAgent.

    Reads model architecture from checkpoint metadata (V5+).
    Falls back to default architecture for older checkpoints.
    """
    if "/" in checkpoint_file:
        # v2+ path like "checkpoints_v2/checkpoint_00500.pth"
        checkpoint_path = os.path.join(ENGINE_DIR, checkpoint_file)
    else:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_file)

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

    game_tmp = PylosGame(rich_obs=rich_obs)
    model = PylosNetwork(
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
# Helper: serialization
# ---------------------------------------------------------------------------

def board_to_json(game: PylosGame) -> list:
    """Serialize the board as a list of 4 lists (one per level).

    Each level contains dicts for occupied cells:
        {"row": r, "col": c, "player": "white" | "black"}
    """
    player_names = {1: "white", -1: "black"}
    levels = []
    for level_idx in range(4):
        size = game.board[level_idx].shape[0]
        cells = []
        for r in range(size):
            for c in range(size):
                val = int(game.board[level_idx][r, c])
                if val != 0:
                    cells.append({
                        "row": r,
                        "col": c,
                        "player": player_names[val],
                    })
        levels.append(cells)
    return levels


def legal_moves_to_json(game: PylosGame) -> list:
    """Serialize legal moves as a JSON-friendly list.

    Placement moves:
        {"type": "place", "level": L, "row": R, "col": C, "action": idx}
    Raise moves:
        {"type": "raise", "src": {"level": L, "row": R, "col": C},
         "dst": {"level": L, "row": R, "col": C}, "action": idx}
    """
    legal_actions = game.get_legal_actions()
    moves = []
    for action in legal_actions:
        if action < 30:
            level, r, c = game.index_to_coords[action]
            moves.append({
                "type": "place",
                "level": level,
                "row": r,
                "col": c,
                "action": action,
            })
        else:
            src_idx, dst_idx = game.raise_action_to_pair[action]
            sl, sr, sc = game.index_to_coords[src_idx]
            dl, dr, dc = game.index_to_coords[dst_idx]
            moves.append({
                "type": "raise",
                "src": {"level": sl, "row": sr, "col": sc},
                "dst": {"level": dl, "row": dr, "col": dc},
                "action": action,
            })
    return moves


def make_state_msg(game: PylosGame) -> dict:
    """Build the full state message to send to the client."""
    turn_names = {1: "white", -1: "black"}
    return {
        "type": "state",
        "board": board_to_json(game),
        "turn": turn_names[game.turn],
        "reserves": {
            "white": int(game.reserves[1]),
            "black": int(game.reserves[-1]),
        },
        "legal_moves": legal_moves_to_json(game),
    }


def make_removal_msg(game: PylosGame) -> dict:
    """Build the removal phase message to send to the client."""
    player_names = {1: "white", -1: "black"}
    return {
        "type": "removal_phase",
        "board": board_to_json(game),
        "removable": [
            {"level": l, "row": r, "col": c}
            for l, r, c in game.get_pending_removable()
        ],
        "formation": [
            {"level": l, "row": r, "col": c}
            for l, r, c in game.get_formation_positions()
        ],
        "removed_so_far": game.removal_count,
        "max_removals": 2,
        "player": player_names[game.removal_player],
        "reserves": {
            "white": int(game.reserves[1]),
            "black": int(game.reserves[-1]),
        },
    }


def _action_to_json(game: PylosGame, action: int) -> dict:
    """Convert an action index to a JSON-friendly dict."""
    if action < 30:
        level, r, c = game.index_to_coords[action]
        return {"type": "place", "level": level, "row": r, "col": c}
    else:
        src_idx, dst_idx = game.raise_action_to_pair[action]
        sl, sr, sc = game.index_to_coords[src_idx]
        dl, dr, dc = game.index_to_coords[dst_idx]
        return {
            "type": "raise",
            "src": {"level": sl, "row": sr, "col": sc},
            "dst": {"level": dl, "row": dr, "col": dc},
        }


# ---------------------------------------------------------------------------
# Helper: resolve client move to action index
# ---------------------------------------------------------------------------

def _resolve_action(game: PylosGame, action_msg: dict) -> int | None:
    """Resolve a client action message to an internal action index.

    Returns the action index if valid and legal, or None otherwise.
    """
    # If the client sends the action index directly, use it
    if "action" in action_msg:
        action = action_msg["action"]
        if action in game.get_legal_actions():
            return action
        return None

    move_type = action_msg.get("type")

    if move_type == "place":
        level = action_msg["level"]
        row = action_msg["row"]
        col = action_msg["col"]
        coords = (level, row, col)
        if coords in game.coords_to_index:
            action = game.coords_to_index[coords]
            if action in game.get_legal_actions():
                return action
        return None

    elif move_type == "raise":
        src = action_msg["src"]
        dst = action_msg["dst"]
        # src/dst can be [level, row, col] arrays or dicts
        if isinstance(src, list):
            src_coords = tuple(src)
        else:
            src_coords = (src["level"], src["row"], src["col"])
        if isinstance(dst, list):
            dst_coords = tuple(dst)
        else:
            dst_coords = (dst["level"], dst["row"], dst["col"])

        if src_coords not in game.coords_to_index or dst_coords not in game.coords_to_index:
            return None
        src_idx = game.coords_to_index[src_coords]
        dst_idx = game.coords_to_index[dst_coords]

        # Find matching raise action
        for action_idx, pair in game.raise_action_to_pair.items():
            if pair == (src_idx, dst_idx):
                if action_idx in game.get_legal_actions():
                    return action_idx
                return None
        return None

    return None


# ---------------------------------------------------------------------------
# Helper: AI turn
# ---------------------------------------------------------------------------

async def _do_ai_turn(
    ws: WebSocket,
    game: PylosGame,
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

    action_json = _action_to_json(game, action)
    game.step(action)  # auto_remove=True by default (AI uses greedy removal)

    # Extract auto-removed pieces from the actions stack
    removed_json = []
    if game.actions_stack:
        last_entry = game.actions_stack[-1]
        removed_pieces = last_entry[-1]  # last element is the removed list
        if removed_pieces:
            player_names = {1: "white", -1: "black"}
            for rl, rr, rc in removed_pieces:
                removed_json.append({"level": rl, "row": rr, "col": rc})

    ai_move_msg = {
        "type": "ai_move",
        "action": action_json,
        "thinking_time_ms": elapsed_ms,
    }
    if removed_json:
        ai_move_msg["removed"] = removed_json

    await ws.send_json(ai_move_msg)

    # Always send the board state so the final piece animates
    await ws.send_json(make_state_msg(game))

    # Check game over
    result = game.get_result()
    if result is not None:
        winner = "white" if result == 1 else "black"
        reason = "apex_placed" if game.top_filled() else "no_legal_moves"
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

    game: PylosGame | None = None
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

                game = PylosGame()
                mode = raw.get("mode", "human_vs_human")
                agent = None

                if mode == "human_vs_ai":
                    checkpoint = raw.get("checkpoint")
                    if checkpoint:
                        agent = load_ai_agent(checkpoint)
                    search_iters = raw.get("search_iterations", 32)
                    hc = raw.get("human_color", "white")
                    human_color = 1 if hc == "white" else -1

                elif mode == "ai_vs_ai":
                    # Support dual checkpoints: one for white, one for black
                    ckpt_white = raw.get("checkpoint_white") or raw.get("checkpoint")
                    ckpt_black = raw.get("checkpoint_black") or raw.get("checkpoint")
                    if ckpt_white:
                        agent_white = load_ai_agent(ckpt_white)
                    if ckpt_black:
                        agent_black = load_ai_agent(ckpt_black)
                    # Fallback: single agent for both sides
                    agent = agent_white or agent_black
                    search_iters = raw.get("search_iterations", 32)
                    shared_delay[0] = raw.get("delay_ms", 1500)

                # Send initial state
                await ws.send_json(make_state_msg(game))

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
                                bh = _board_hash(_game)
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
                if game.pending_removal:
                    await ws.send_json(make_removal_msg(game))
                    continue

                action_msg = raw.get("action", {})
                action = _resolve_action(game, action_msg)

                if action is None:
                    await ws.send_json({"type": "error", "message": "Illegal move"})
                    continue

                game.step(action, auto_remove=False)

                # Check for interactive removal phase BEFORE game-over,
                # since removing pieces may free up moves for the opponent
                if game.pending_removal:
                    await ws.send_json(make_removal_msg(game))
                    continue

                # Always send state so the final piece animates
                await ws.send_json(make_state_msg(game))

                # Check game over
                result = game.get_result()
                if result is not None:
                    winner = "white" if result == 1 else "black"
                    reason = "apex_placed" if game.top_filled() else "no_legal_moves"
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
                if game is None or not game.pending_removal:
                    await ws.send_json({"type": "error", "message": "Not in removal phase"})
                    continue

                level = raw.get("level")
                row = raw.get("row")
                col = raw.get("col")

                if not game.step_removal(level, row, col):
                    await ws.send_json({"type": "error", "message": "Cannot remove that piece"})
                    continue

                # Still in removal phase?
                if game.pending_removal:
                    await ws.send_json(make_removal_msg(game))
                    continue

                # Removal phase ended — send state, then check game over
                await ws.send_json(make_state_msg(game))

                result = game.get_result()
                if result is not None:
                    winner = "white" if result == 1 else "black"
                    reason = "apex_placed" if game.top_filled() else "no_legal_moves"
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
                if game is None or not game.pending_removal:
                    await ws.send_json({"type": "error", "message": "Not in removal phase"})
                    continue

                game.skip_removal()

                await ws.send_json(make_state_msg(game))

                # Check game over after removal phase ends
                result = game.get_result()
                if result is not None:
                    winner = "white" if result == 1 else "black"
                    reason = "apex_placed" if game.top_filled() else "no_legal_moves"
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
