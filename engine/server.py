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

from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgent
from mcts import play as mcts_play

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ENGINE_DIR)
WEB_DIR = os.path.join(PROJECT_DIR, "web")
CHECKPOINTS_DIR = os.path.join(ENGINE_DIR, "checkpoints")

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
    """Return the checkpoint manifest, or an empty list if not found."""
    manifest_path = os.path.join(CHECKPOINTS_DIR, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {"checkpoints": []}


@app.get("/")
async def index():
    """Serve the frontend index.html."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse({"error": "index.html not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Helper: load AI agent
# ---------------------------------------------------------------------------

def load_ai_agent(checkpoint_file: str) -> AlphaZeroAgent:
    """Load a PylosNetwork from a checkpoint file and return an AlphaZeroAgent."""
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_file)
    game_tmp = PylosGame()
    model = PylosNetwork(
        input_shape=game_tmp.observation_shape,
        action_space=game_tmp.action_space,
    )
    state_dict = torch.load(checkpoint_path, map_location=model.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return AlphaZeroAgent(model)


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
    game.step(action)

    await ws.send_json({
        "type": "ai_move",
        "action": action_json,
        "thinking_time_ms": elapsed_ms,
    })

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

    await ws.send_json(make_state_msg(game))
    return False


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/game")
async def game_ws(ws: WebSocket):
    await ws.accept()

    game: PylosGame | None = None
    agent: AlphaZeroAgent | None = None
    mode: str | None = None
    search_iters: int = 32
    human_color: int = 1  # 1=white, -1=black
    ai_vs_ai_task: asyncio.Task | None = None

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
                    checkpoint = raw.get("checkpoint")
                    if checkpoint:
                        agent = load_ai_agent(checkpoint)
                    search_iters = raw.get("search_iterations", 32)
                    delay_ms = raw.get("delay_ms", 1500)

                # Send initial state
                await ws.send_json(make_state_msg(game))

                # If human_vs_ai and AI goes first
                if mode == "human_vs_ai" and agent is not None and game.turn != human_color:
                    await _do_ai_turn(ws, game, agent, search_iters)

                # If ai_vs_ai, start the async loop
                if mode == "ai_vs_ai" and agent is not None:
                    async def _ai_vs_ai_loop(
                        _ws=ws, _game=game, _agent=agent,
                        _iters=search_iters, _delay=delay_ms,
                    ):
                        try:
                            while True:
                                await asyncio.sleep(_delay / 1000.0)
                                done = await _do_ai_turn(_ws, _game, _agent, _iters)
                                if done:
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

                action_msg = raw.get("action", {})
                action = _resolve_action(game, action_msg)

                if action is None:
                    await ws.send_json({"type": "error", "message": "Illegal move"})
                    continue

                game.step(action)

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

                await ws.send_json(make_state_msg(game))

                # If human_vs_ai and it's now the AI's turn
                if mode == "human_vs_ai" and agent is not None and game.turn != human_color:
                    await _do_ai_turn(ws, game, agent, search_iters)

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
