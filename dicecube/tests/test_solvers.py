import numpy as np
import pytest

from dicecube import astar, mcts
from dicecube.cube import (
    NUM_ACTIONS, apply_actions, is_solved, random_solved, scramble, state_to_string,
)
from dicecube.targets import colour_string, iter_targets, target_state


class UniformModel:
    """Stand-in network: no knowledge, uniform policy."""

    def batched_inference(self, states):
        n = len(states)
        return np.zeros(n, dtype=np.float32), np.full((n, NUM_ACTIONS), 1 / NUM_ACTIONS)


def test_first_target_of_solved_cube_is_identity():
    s = random_solved(np.random.default_rng(0))
    corners, edges = next(iter_targets(s))
    assert corners == [(i, 0) for i in range(8)]
    assert edges == [(i, 0) for i in range(12)]
    assert colour_string((corners, edges)) == "".join(f * 9 for f in "URFDLB")


def test_targets_are_solved_states():
    rng = np.random.default_rng(1)
    s, _ = scramble(random_solved(rng), 25, rng)
    for i, assignment in enumerate(iter_targets(s, rng=rng, randomize=True)):
        t = target_state(s, assignment)
        assert is_solved(t)
        assert sorted(t) == sorted(s)          # same stickers, rearranged
        if i == 5:
            break


def test_kociemba_solves_random_scrambles():
    pytest.importorskip("kociemba")
    from dicecube.targets import kociemba_solve
    from dicecube.cube import notation_to_actions
    rng = np.random.default_rng(2)
    for _ in range(3):
        s, _ = scramble(random_solved(rng), 30, rng)
        r = kociemba_solve(s, max_targets=10)
        end = apply_actions(s, notation_to_actions(r["moves"]))
        assert is_solved(end)
        assert (end == r["target"]).all()


@pytest.mark.parametrize("module", [mcts, astar])
def test_neural_searches_solve_short_scrambles(module):
    rng = np.random.default_rng(3)
    model = UniformModel()
    for depth in (1, 2):
        s, _ = scramble(random_solved(rng), depth, rng)
        r = module.search(s, model, time_limit=20)
        assert r["solved"]
        assert is_solved(apply_actions(s, r["actions"]))
        assert len(r["actions"]) <= depth


def test_search_on_solved_state_is_empty():
    s = random_solved(np.random.default_rng(4))
    assert mcts.search(s, UniformModel())["actions"] == []
    assert astar.search(s, UniformModel())["actions"] == []


def test_solver_packages_results():
    pytest.importorskip("kociemba")
    from dicecube.solver import solve
    rng = np.random.default_rng(5)
    s, _ = scramble(random_solved(rng), 12, rng)
    r = solve(s, methods=("kociemba",), model_path="/nonexistent.pth")
    k = r["results"]["kociemba"]
    assert k["solved"] and k["found"]
    assert k["missing_trace"][-1] == 0
    assert r["best"] == "kociemba"
