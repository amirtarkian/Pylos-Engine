import numpy as np
import pytest

from dicecube.cube import (
    ACTIONS, CORNER_FACELETS, EDGE_FACELETS, FACES, INVERSE_ACTION, MOVE_PERMS, OBS_DIM,
    actions_to_notation, apply_action, apply_actions, face_conflicts, is_solved,
    notation_to_actions, parse_state, random_solved, scramble, state_to_string,
    to_observation, validate_state,
)

SOLVED_COLOURS = np.repeat(np.arange(6), 9)


def test_moves_are_permutations_of_order_four():
    for a in range(12):
        assert sorted(MOVE_PERMS[a]) == list(range(54))
        s = SOLVED_COLOURS
        for _ in range(4):
            s = apply_action(s, a)
        assert (s == SOLVED_COLOURS).all()


def test_inverse_moves_cancel():
    rng = np.random.default_rng(0)
    s = random_solved(rng)
    for a in range(12):
        assert (apply_action(apply_action(s, a), INVERSE_ACTION[a]) == s).all()


def test_centers_never_move():
    for a in range(12):
        for f in range(6):
            assert MOVE_PERMS[a][f * 9 + 4] == f * 9 + 4


def test_sexy_move_has_order_six():
    seq = notation_to_actions("R U R' U'")
    s = SOLVED_COLOURS
    for i in range(1, 7):
        s = apply_actions(s, seq)
        assert (s == SOLVED_COLOURS).all() == (i == 6)


def test_moves_match_kociemba_convention():
    kociemba = pytest.importorskip("kociemba")
    for a in range(12):
        s = apply_action(SOLVED_COLOURS, a)
        text = "".join(FACES[i] for i in s)
        assert kociemba.solve(text) == ACTIONS[INVERSE_ACTION[a]]


def test_cubie_facelets_cover_every_non_center_once():
    used = [f for c in CORNER_FACELETS for f in c] + [f for e in EDGE_FACELETS for f in e]
    assert sorted(used) == [i for i in range(54) if i % 9 != 4]


@pytest.mark.parametrize("design", ["parity", "any"])
def test_random_solved_is_solved(design):
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = random_solved(rng, design)
        assert is_solved(s)
        assert face_conflicts(s) == 0
        assert validate_state(s) == []


def test_parity_design_puts_odd_on_corners():
    s = random_solved(np.random.default_rng(2))
    for c in CORNER_FACELETS:
        assert all(s[f] % 2 == 1 for f in c)
    for e in EDGE_FACELETS:
        assert all(s[f] % 2 == 0 for f in e)


def test_scramble_unsolves_and_batch_is_solved():
    rng = np.random.default_rng(3)
    s = random_solved(rng)
    t, _ = scramble(s, 1, rng)
    assert not is_solved(t)
    assert face_conflicts(t) > 0
    assert is_solved(np.stack([s, t])).tolist() == [True, False]


def test_notation_round_trip():
    assert actions_to_notation(notation_to_actions("R U2 F' D D' B")) == ["R", "U2", "F'", "B"]
    assert actions_to_notation([0, 0, 0]) == ["U'"]
    assert actions_to_notation([1, 0, 6, 7]) == []


def test_parse_and_format():
    s = random_solved(np.random.default_rng(4))
    assert (parse_state(state_to_string(s)) == s).all()
    with pytest.raises(ValueError):
        parse_state("123")


def test_validate_catches_bad_input():
    s = random_solved(np.random.default_rng(5))
    s[0], s[1] = 9, 9 if s[0] != 9 else 1
    problems = validate_state(s)
    assert problems
    s2 = random_solved(np.random.default_rng(5))
    s2[4] = 3
    assert any("center" in p for p in validate_state(s2))


def test_observation_shape_and_histogram():
    s = random_solved(np.random.default_rng(6))
    obs = to_observation(s)
    assert obs.shape == (OBS_DIM,)
    hist = obs[384:].reshape(6, 8)
    assert (hist == 1).all()     # solved: every value once per face
    assert to_observation(np.stack([s, s])).shape == (2, OBS_DIM)
