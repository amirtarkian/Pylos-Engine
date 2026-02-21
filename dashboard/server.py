"""FastAPI dashboard server for training metrics.

Receives metrics via HTTP POST and serves them to the frontend via REST + SSE.
"""

import asyncio
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from dashboard.state import TrainingState

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DASHBOARD_DIR)
WEB_DIR = os.path.join(PROJECT_DIR, "web")

# ---------------------------------------------------------------------------
# App + state
# ---------------------------------------------------------------------------

app = FastAPI()
state = TrainingState()

# Mount static files for the frontend
if os.path.isdir(os.path.join(WEB_DIR, "src")):
    app.mount("/src", StaticFiles(directory=os.path.join(WEB_DIR, "src")), name="static")


# ---------------------------------------------------------------------------
# POST endpoints — receive data from training pipeline
# ---------------------------------------------------------------------------

@app.post("/api/metrics")
async def post_metrics(request: Request):
    """Receive a training progress update."""
    data = await request.json()
    state.update_progress(data)
    return JSONResponse({"status": "ok"})


@app.post("/api/checkpoint")
async def post_checkpoint(request: Request):
    """Receive a checkpoint event."""
    data = await request.json()
    state.add_checkpoint(data)
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# GET endpoints — serve data to dashboard frontend
# ---------------------------------------------------------------------------

@app.get("/api/metrics")
async def get_metrics():
    """Return current training state."""
    return JSONResponse(state.get_progress())


@app.get("/api/loss-history")
async def get_loss_history():
    """Return full loss history array."""
    return JSONResponse(state.get_loss_history())


@app.get("/api/checkpoints")
async def get_checkpoints():
    """Return checkpoint list."""
    return JSONResponse(state.get_checkpoints())


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------

@app.get("/api/events")
async def sse_events():
    """Stream SSE events for real-time dashboard updates."""
    q = state.subscribe()

    async def event_stream():
        try:
            while True:
                event, data = await q.get()
                payload = json.dumps(data)
                yield f"event: {event}\ndata: {payload}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            state.unsubscribe(q)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    """Serve the dashboard HTML."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse({"error": "index.html not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Pylos Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
