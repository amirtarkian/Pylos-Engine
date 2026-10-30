import numpy as np
import torch

from dicecube.cube import MOVE_PERMS, OBS_DIM, is_solved, to_observation
from dicecube.models import DiceCubeNetwork, load_model, save_model
from dicecube.train import bellman_targets, random_solved_batch, scrambled_batch


def test_torch_encoding_matches_numpy_reference():
    rng = np.random.default_rng(0)
    states, _ = scrambled_batch(64, 10, rng)
    net = DiceCubeNetwork(OBS_DIM, device="cpu", hidden=32, num_blocks=1)
    enc = net.encode(torch.as_tensor(states)).numpy()
    assert np.array_equal(enc, to_observation(states))


def test_batches_are_valid_cubes():
    rng = np.random.default_rng(1)
    assert is_solved(random_solved_batch(50, rng)).all()
    states, depths = scrambled_batch(200, 5, rng)
    assert states.shape == (200, 54)
    assert is_solved(states[depths == 0]).all()
    # every value still appears 6 times
    for v in (1, 2, 3, 4, 6, 7, 8, 9):
        assert ((states == v).sum(axis=1) == 6).all()


def test_bellman_targets_one_move_from_goal():
    rng = np.random.default_rng(2)
    solved = random_solved_batch(8, rng)
    one_away = solved[:, MOVE_PERMS[0]]
    net = DiceCubeNetwork(OBS_DIM, device="cpu", hidden=32, num_blocks=1)
    value, best = bellman_targets(net, np.concatenate([solved, one_away]))
    assert (value[:8] == 0).all()
    assert np.allclose(value[8:], 1.0)          # U' finishes it
    assert (best[8:] == 6).all()


def test_save_load_round_trip(tmp_path):
    net = DiceCubeNetwork(OBS_DIM, device="cpu", hidden=32, num_blocks=1)
    path = str(tmp_path / "m.pth")
    save_model(net, path, {"step": 7})
    loaded, ckpt = load_model(path)
    assert ckpt["step"] == 7
    x = random_solved_batch(4, np.random.default_rng(3))
    a, _ = net.batched_inference(x)
    b, _ = loaded.batched_inference(x)
    assert np.allclose(a, b)
