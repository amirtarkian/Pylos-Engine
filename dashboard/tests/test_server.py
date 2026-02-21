"""Integration tests for the dashboard server."""

import threading

import pytest
from httpx import ASGITransport, AsyncClient

from dashboard.state import TrainingState


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset server state between tests."""
    import dashboard.server as srv
    original = srv.state
    srv.state = TrainingState()
    yield
    srv.state = original


@pytest.fixture
async def client():
    from dashboard.server import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_post_and_get_metrics(client):
    """POST metrics then GET them back."""
    payload = {"current_game": 50, "total_games": 100, "value_loss": 0.5}
    resp = await client.post("/api/metrics", json=payload)
    assert resp.status_code == 200

    resp = await client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["current_game"] == 50
    assert data["total_games"] == 100


@pytest.mark.anyio
async def test_post_and_get_checkpoint(client):
    """POST a checkpoint then GET the list."""
    payload = {"file": "checkpoint_00100.pth", "step": 100, "elo": 1200}
    resp = await client.post("/api/checkpoint", json=payload)
    assert resp.status_code == 200

    resp = await client.get("/api/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["file"] == "checkpoint_00100.pth"
    assert data[0]["elo"] == 1200


@pytest.mark.anyio
async def test_loss_history_endpoint(client):
    """GET /api/loss-history returns an array."""
    resp = await client.get("/api/loss-history")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.anyio
async def test_sse_receives_metrics_event(client):
    """SSE queue receives an event after POST."""
    import dashboard.server as srv

    q = srv.state.subscribe()
    await client.post("/api/metrics", json={"current_game": 75, "total_games": 200})

    event, data = q.get_nowait()
    assert event == "metrics"
    assert data["current_game"] == 75

    srv.state.unsubscribe(q)


@pytest.mark.anyio
async def test_sse_receives_checkpoint_event(client):
    """SSE queue receives a checkpoint event."""
    import dashboard.server as srv

    q = srv.state.subscribe()
    await client.post("/api/checkpoint", json={"file": "ckpt_200.pth", "step": 200, "elo": 1350})

    event, data = q.get_nowait()
    assert event == "checkpoint"
    assert data["step"] == 200

    srv.state.unsubscribe(q)


@pytest.mark.anyio
async def test_sse_endpoint_content_type():
    """GET /api/events returns text/event-stream content type."""
    from dashboard.server import app as _app, sse_events

    # Call the endpoint directly to verify it returns a StreamingResponse
    resp = await sse_events()
    assert resp.media_type == "text/event-stream"


@pytest.mark.anyio
async def test_index_serves_html(client):
    """GET / returns the dashboard HTML."""
    resp = await client.get("/")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_get_metrics_empty(client):
    """GET /api/metrics with no data returns empty dict."""
    resp = await client.get("/api/metrics")
    assert resp.status_code == 200
    assert resp.json() == {}


@pytest.mark.anyio
async def test_get_checkpoints_empty(client):
    """GET /api/checkpoints with no data returns empty list."""
    resp = await client.get("/api/checkpoints")
    assert resp.status_code == 200
    assert resp.json() == []


def test_training_state_concurrent_writes():
    """TrainingState handles concurrent writes safely."""
    ts = TrainingState()
    errors = []

    def writer(n):
        try:
            for i in range(100):
                ts.update_progress({"step": n * 100 + i})
                ts.add_checkpoint({"file": f"ckpt_{n}_{i}.pth", "step": n * 100 + i})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(ts.get_checkpoints()) == 400
