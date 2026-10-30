import pytest
from fastapi.testclient import TestClient

from dicecube.server import app

client = TestClient(app)


def test_geometry():
    g = client.get("/api/geometry").json()
    assert len(g["move_perms"]) == 12
    assert len(g["positions"]) == 54


def test_scramble_and_validate():
    r = client.get("/api/scramble?n=10&seed=1").json()
    assert r["problems"] == []
    assert not r["solved"]
    v = client.post("/api/validate", json={"state": r["solved_state"]}).json()
    assert v["solved"]
    bad = client.post("/api/validate", json={"state": "12"})
    assert bad.status_code == 400


def test_solve_kociemba():
    pytest.importorskip("kociemba")
    r = client.get("/api/scramble?n=8&seed=2").json()
    s = client.post("/api/solve", json={"state": r["state"], "methods": ["kociemba"],
                                        "time_limit": 3}).json()
    assert s["results"]["kociemba"]["solved"]


def test_index_and_training():
    assert client.get("/").status_code == 200
    assert "progress" in client.get("/api/training").json()
