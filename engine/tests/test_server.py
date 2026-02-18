"""Tests for the FastAPI WebSocket server."""

import pytest
from fastapi.testclient import TestClient
from server import app


# ======================================================================
# REST endpoint tests
# ======================================================================

def test_checkpoints_endpoint():
    client = TestClient(app)
    resp = client.get("/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert "checkpoints" in data


# ======================================================================
# WebSocket: human vs human
# ======================================================================

def test_websocket_new_game_human_vs_human():
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        data = ws.receive_json()
        assert data["type"] == "state"
        assert data["turn"] == "white"
        assert data["reserves"]["white"] == 15
        assert data["reserves"]["black"] == 15
        assert len(data["legal_moves"]) == 16  # initial placements on 4x4

        # Make a valid placement
        move = data["legal_moves"][0]
        ws.send_json({"type": "move", "action": move})
        data = ws.receive_json()
        assert data["type"] == "state"
        assert data["turn"] == "black"
        assert data["reserves"]["white"] == 14


def test_websocket_illegal_move():
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        ws.receive_json()  # initial state

        # Try placing on level 1 (unsupported at game start)
        ws.send_json({
            "type": "move",
            "action": {"type": "place", "level": 1, "row": 0, "col": 0},
        })
        data = ws.receive_json()
        assert data["type"] == "error"
        assert "Illegal" in data["message"]


def test_websocket_multiple_moves():
    """Play a few valid moves in sequence."""
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        data = ws.receive_json()
        assert data["type"] == "state"

        # White plays
        move = data["legal_moves"][0]
        ws.send_json({"type": "move", "action": move})
        data = ws.receive_json()
        assert data["type"] == "state"
        assert data["turn"] == "black"

        # Black plays
        move = data["legal_moves"][0]
        ws.send_json({"type": "move", "action": move})
        data = ws.receive_json()
        assert data["type"] == "state"
        assert data["turn"] == "white"
        assert data["reserves"]["white"] == 14
        assert data["reserves"]["black"] == 14


def test_websocket_move_without_game():
    """Sending a move before starting a game should return an error."""
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({
            "type": "move",
            "action": {"type": "place", "level": 0, "row": 0, "col": 0},
        })
        data = ws.receive_json()
        assert data["type"] == "error"
        assert "No active game" in data["message"]


def test_websocket_unknown_message_type():
    """Unknown message types should return an error."""
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "foobar"})
        data = ws.receive_json()
        assert data["type"] == "error"


def test_websocket_board_structure():
    """Verify the board structure in the state message."""
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        data = ws.receive_json()
        board = data["board"]
        # 4 levels
        assert len(board) == 4
        # Initially all empty
        for level_cells in board:
            assert level_cells == []


def test_websocket_board_after_placement():
    """After a placement the board should contain the placed piece."""
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        data = ws.receive_json()

        # Place at the first legal move
        move = data["legal_moves"][0]
        ws.send_json({"type": "move", "action": move})
        data = ws.receive_json()
        board = data["board"]

        # Level 0 should have exactly one piece
        assert len(board[0]) == 1
        piece = board[0][0]
        assert piece["player"] == "white"
        assert piece["row"] == move["row"]
        assert piece["col"] == move["col"]


def test_websocket_legal_moves_format():
    """Legal moves should have correct format."""
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        data = ws.receive_json()
        for move in data["legal_moves"]:
            assert "type" in move
            assert move["type"] == "place"
            assert "level" in move
            assert "row" in move
            assert "col" in move
            assert "action" in move


def test_websocket_new_game_resets():
    """Starting a new game should reset the state."""
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        # Start first game and make a move
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        data = ws.receive_json()
        move = data["legal_moves"][0]
        ws.send_json({"type": "move", "action": move})
        ws.receive_json()

        # Start a new game
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        data = ws.receive_json()
        assert data["type"] == "state"
        assert data["turn"] == "white"
        assert data["reserves"]["white"] == 15
        assert data["reserves"]["black"] == 15
        assert len(data["legal_moves"]) == 16
