"""Tests for the server's take_back action (HvH only)."""

from fastapi.testclient import TestClient
from server import app


def _make_hvh_session():
    client = TestClient(app)
    ws = client.websocket_connect("/game").__enter__()
    ws.send_json({"type": "new_game", "mode": "human_vs_human"})
    initial = ws.receive_json()
    assert initial["type"] == "state"
    assert initial["turn"] == "white"
    return client, ws, initial


def _legal_place_action(state_msg, level, row, col):
    """Find the action index for placing at (level,row,col) in legal_moves."""
    for m in state_msg["legal_moves"]:
        if (
            m["type"] == "place"
            and m["level"] == level
            and m["row"] == row
            and m["col"] == col
        ):
            return m["action"]
    raise AssertionError(f"No legal place at L{level}({row},{col})")


def test_take_back_rejected_without_game():
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "take_back"})
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_take_back_rejected_in_human_vs_ai():
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_ai", "human_color": "white"})
        _ = ws.receive_json()  # initial state
        ws.send_json({"type": "take_back"})
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_take_back_rejected_with_empty_stack():
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        _ = ws.receive_json()
        ws.send_json({"type": "take_back"})
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_take_back_one_placement_returns_to_initial_state():
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        initial = ws.receive_json()
        action = _legal_place_action(initial, 0, 0, 0)

        ws.send_json({"type": "move", "action": {"action": action}})
        after_move = ws.receive_json()
        assert after_move["type"] == "state"
        assert after_move["turn"] == "black"
        # Reserves: white spent one
        assert after_move["reserves"]["white"] == initial["reserves"]["white"] - 1

        ws.send_json({"type": "take_back"})
        rolled_back = ws.receive_json()
        assert rolled_back["type"] == "state"
        assert rolled_back.get("take_back") is True
        assert rolled_back["turn"] == initial["turn"]
        assert rolled_back["reserves"] == initial["reserves"]
        assert rolled_back["board"] == initial["board"]


def test_take_back_rewinds_only_one_turn():
    """White places, black places, take_back -> black should be back on the clock."""
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        s0 = ws.receive_json()
        a1 = _legal_place_action(s0, 0, 0, 0)
        ws.send_json({"type": "move", "action": {"action": a1}})
        s1 = ws.receive_json()
        assert s1["turn"] == "black"

        a2 = _legal_place_action(s1, 0, 1, 1)
        ws.send_json({"type": "move", "action": {"action": a2}})
        s2 = ws.receive_json()
        assert s2["turn"] == "white"

        ws.send_json({"type": "take_back"})
        rolled = ws.receive_json()
        assert rolled["type"] == "state"
        assert rolled.get("take_back") is True
        assert rolled["turn"] == "black"
        # White piece from move 1 is still on the board
        assert rolled["reserves"]["white"] == s0["reserves"]["white"] - 1
        assert rolled["reserves"]["black"] == s0["reserves"]["black"]


def test_take_back_unwinds_full_turn_including_removal():
    """White completes a square and removes a black piece, then take_back rewinds both."""
    client = TestClient(app)
    with client.websocket_connect("/game") as ws:
        ws.send_json({"type": "new_game", "mode": "human_vs_human"})
        s0 = ws.receive_json()
        assert s0["type"] == "state"

        # Drive into a square: white plays L0(0,0), L0(0,1), L0(1,0); black plays
        # filler positions. White's 4th placement at L0(1,1) completes the 2x2
        # square on level 0 and triggers a removal phase for white.
        moves = [
            (0, 0, 0),  # W
            (0, 3, 3),  # B (filler, far corner)
            (0, 0, 1),  # W
            (0, 3, 0),  # B (filler)
            (0, 1, 0),  # W
            (0, 3, 1),  # B (filler)
            (0, 1, 1),  # W -- completes square
        ]
        latest = s0
        for (level, row, col) in moves[:-1]:
            a = _legal_place_action(latest, level, row, col)
            ws.send_json({"type": "move", "action": {"action": a}})
            latest = ws.receive_json()
            assert latest["type"] == "state"

        # Final placement -- expect removal_phase, not state.
        level, row, col = moves[-1]
        a = _legal_place_action(latest, level, row, col)
        white_reserves_before_final = latest["reserves"]["white"]
        black_reserves_before_final = latest["reserves"]["black"]
        ws.send_json({"type": "move", "action": {"action": a}})
        msg = ws.receive_json()
        assert msg["type"] == "removal_phase", f"expected removal_phase, got {msg['type']}"
        assert msg["removable"], "expected at least one removable piece"

        # White removes one black piece (the first removable).
        target = msg["removable"][0]
        ws.send_json({"type": "remove", "level": target["level"], "row": target["row"], "col": target["col"]})
        after_remove = ws.receive_json()
        # Could be removal_phase (if white can still remove another) or state.
        assert after_remove["type"] in ("removal_phase", "state")

        # Take back -- should fully unwind: white's L0(1,1) placement AND the removal.
        ws.send_json({"type": "take_back"})
        rolled = ws.receive_json()
        assert rolled["type"] == "state", f"expected state, got {rolled['type']}"
        assert rolled.get("take_back") is True
        # White is back on the clock with the same reserves as before the final placement.
        assert rolled["turn"] == "white"
        assert rolled["reserves"]["white"] == white_reserves_before_final
        # Removed black piece is back on the board: black reserves restored.
        assert rolled["reserves"]["black"] == black_reserves_before_final
