"""
FastAPI server for the dice-cube solver.

  GET  /                 -- web/dice.html (3D visualiser + solver UI)
  GET  /api/geometry     -- facelet layout + move permutations for the client
  GET  /api/scramble     -- random solved cube scrambled N moves
  POST /api/validate     -- check a typed-in state
  POST /api/solve        -- run MCTS / A* / Kociemba on a state
  GET  /api/training     -- training progress, loss history, eval manifest

Run:  python -m dicecube.server   (or bash run_dice.sh) -> http://localhost:8001
"""

import asyncio
import json
import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dicecube.cube import (
    ACTIONS, CORNER_FACELETS, EDGE_FACELETS, FACES, MOVE_PERMS, _facelet_geometry,
    actions_to_notation, face_conflicts, is_solved, parse_state, random_solved,
    scramble, state_to_string, validate_state,
)
from dicecube.solver import get_model, solve
from dicecube.targets import kociemba_available

WEB_DIR = os.path.join(PROJECT_DIR, "web")
CKPT_DIR = os.path.join(PACKAGE_DIR, "checkpoints")

app = FastAPI()
app.mount("/src", StaticFiles(directory=os.path.join(WEB_DIR, "src")), name="static")

_solve_lock = asyncio.Lock()


class StateRequest(BaseModel):
    state: str
    methods: list[str] = ["mcts", "astar", "kociemba"]
    time_limit: float = 8.0


def _state_info(state):
    info = {
        "state": state_to_string(state),
        "solved": bool(is_solved(state)),
        "missing": int(face_conflicts(state)),
        "problems": validate_state(state),
        "net_cost": None,
    }
    model = get_model()
    if model is not None and not info["problems"]:
        cost, _ = model.batched_inference(state[None])
        info["net_cost"] = 0.0 if info["solved"] else round(float(cost[0]), 2)
    return info


@app.get("/")
async def index():
    return FileResponse(os.path.join(WEB_DIR, "dice.html"))


@app.get("/api/geometry")
async def geometry():
    pos, nrm = _facelet_geometry()
    return {
        "faces": FACES,
        "actions": ACTIONS,
        "positions": pos.tolist(),
        "normals": nrm.tolist(),
        "move_perms": MOVE_PERMS.tolist(),
        "corners": [list(c) for c in CORNER_FACELETS],
        "edges": [list(e) for e in EDGE_FACELETS],
        "kociemba": kociemba_available(),
        "model": get_model() is not None,
    }


@app.get("/api/scramble")
async def api_scramble(n: int = 20, seed: int | None = None):
    rng = np.random.default_rng(seed)
    solved = random_solved(rng)
    state, actions = scramble(solved, max(0, min(n, 100)), rng)
    return {**_state_info(state), "solved_state": state_to_string(solved),
            "scramble": actions_to_notation(actions)}


@app.post("/api/validate")
async def api_validate(req: StateRequest):
    try:
        state = parse_state(req.state)
    except ValueError as e:
        return JSONResponse({"problems": [str(e)]}, status_code=400)
    return _state_info(state)


@app.post("/api/solve")
async def api_solve(req: StateRequest):
    try:
        state = parse_state(req.state)
    except ValueError as e:
        return JSONResponse({"problems": [str(e)]}, status_code=400)
    time_limit = max(0.5, min(req.time_limit, 30.0))
    async with _solve_lock:
        result = await asyncio.to_thread(solve, state, tuple(req.methods), time_limit)
    return result


@app.get("/api/training")
async def api_training(max_points: int = 400):
    out = {"progress": None, "history": [], "manifest": {"checkpoints": []}}
    path = os.path.join(CKPT_DIR, "training_progress.json")
    if os.path.isfile(path):
        with open(path) as f:
            out["progress"] = json.load(f)
    path = os.path.join(CKPT_DIR, "manifest.json")
    if os.path.isfile(path):
        with open(path) as f:
            out["manifest"] = json.load(f)
    path = os.path.join(CKPT_DIR, "loss_history.jsonl")
    if os.path.isfile(path):
        with open(path) as f:
            # keyed by step: a resumed run re-logs steps after its last checkpoint
            by_step = {e["step"]: e for e in (json.loads(line) for line in f if line.strip())}
        entries = [by_step[k] for k in sorted(by_step)]
        if len(entries) > max_points:
            stride = len(entries) / max_points
            entries = [entries[int(i * stride)] for i in range(max_points - 1)] + [entries[-1]]
        out["history"] = entries
    return out


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    print(f"Dice cube solver on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
