import torch
from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgentTrainer


def test_training_smoke():
    """Run 2 self-play games to verify the full training loop works."""
    game = PylosGame()
    model = PylosNetwork(game.observation_shape, game.action_space)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    agent = AlphaZeroAgentTrainer(model, optimizer, replay_buffer_max_size=64)

    for _ in range(2):
        game.reset()
        vl, pl = agent.train_step(game, search_iterations=8, batch_size=16, epochs=1, c_puct=1.5)
    assert True  # If we got here, the pipeline works


def test_model_output_shapes():
    """Verify model produces correct output shapes."""
    game = PylosGame()
    model = PylosNetwork(game.observation_shape, game.action_space)
    obs = torch.tensor(game.to_observation(), device=model.device)

    # Training mode (batch of 2 required for BatchNorm)
    batch = obs.unsqueeze(0).repeat(2, 1)
    value, log_policy = model(batch)
    assert value.shape == (2, 1)
    assert log_policy.shape == (2, 303)

    # Inference mode
    val = model.value_forward(obs)
    pol = model.policy_forward(obs)
    assert val.shape == (1,)
    assert pol.shape == (303,)
    assert abs(pol.sum().item() - 1.0) < 0.01
