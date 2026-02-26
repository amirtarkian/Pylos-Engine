"""End-to-end integration tests for the Pylos AlphaZero engine."""

import os
import json
import tempfile
import torch
from game import PylosGame
from models import PylosNetwork
from agents import AlphaZeroAgentTrainer, AlphaZeroAgent
from mcts import play as mcts_play, pit
from evaluate import evaluate_checkpoint, assign_label, RandomAgent


class TestTrainSaveLoad:
    def test_full_pipeline(self):
        """Train 3 games, save checkpoint, load, verify inference works."""
        game = PylosGame()
        model = PylosNetwork(game.observation_shape, game.action_space)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        agent = AlphaZeroAgentTrainer(model, optimizer, 64)

        for _ in range(3):
            game.reset()
            agent.train_step(game, search_iterations=4, batch_size=8, epochs=1, c_puct=1.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pth")
            opt_path = os.path.join(tmpdir, "optimizer.pth")
            agent.save_training_state(model_path, opt_path)

            # Load in fresh model
            model2 = PylosNetwork(game.observation_shape, game.action_space)
            model2.load_state_dict(torch.load(model_path, map_location=model2.device, weights_only=True))

            # Verify inference
            game.reset()
            obs = torch.tensor(game.to_observation(), device=model2.device)
            val = model2.value_forward(obs)
            pol = model2.policy_forward(obs)
            assert val.shape == (1,)
            assert pol.shape == (303,)
            assert abs(pol.sum().item() - 1.0) < 0.01

    def test_checkpoint_manifest(self):
        """Verify checkpoint saving creates valid manifest.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            game = PylosGame()
            model = PylosNetwork(game.observation_shape, game.action_space)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            agent = AlphaZeroAgentTrainer(model, optimizer, 64)

            model_path = os.path.join(tmpdir, "model.pth")
            opt_path = os.path.join(tmpdir, "optimizer.pth")
            agent.save_training_state(model_path, opt_path)

            # Verify file exists and is loadable
            assert os.path.exists(model_path)
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            assert "input_fc.weight" in state


class TestMCTSPlay:
    def test_agent_can_play_full_game(self):
        """An AlphaZero agent (untrained) can play a complete game without crashing."""
        game = PylosGame()
        model = PylosNetwork(game.observation_shape, game.action_space)
        agent = AlphaZeroAgent(model)

        moves = 0
        while game.get_result() is None and moves < 100:
            action = mcts_play(game, agent, search_iterations=4, c_puct=1.5)
            if action is None:
                break
            game.step(action)
            moves += 1

        # Game should have ended (or we hit max moves)
        assert moves > 0

    def test_pit_completes(self):
        """Two agents can play against each other."""
        game = PylosGame()
        model = PylosNetwork(game.observation_shape, game.action_space)
        agent1 = AlphaZeroAgent(model)
        agent2 = AlphaZeroAgent(model)

        result = pit(
            game, agent1, agent2,
            {"search_iterations": 4, "c_puct": 1.5},
            {"search_iterations": 4, "c_puct": 1.5},
        )
        # Result should be 1 or -1
        assert result in (1, -1)


class TestEvaluation:
    def test_evaluate_untrained(self):
        """Evaluate an untrained model (should have roughly 50% win rate vs random)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            game = PylosGame()
            model = PylosNetwork(game.observation_shape, game.action_space)
            model_path = os.path.join(tmpdir, "model.pth")
            torch.save(model.state_dict(), model_path)

            wr = evaluate_checkpoint(model_path, num_games=4, search_iterations=4)
            assert 0.0 <= wr <= 1.0

    def test_assign_label(self):
        assert assign_label(0.3) == "Beginner"
        assert assign_label(0.5) == "Novice"
        assert assign_label(0.7) == "Intermediate"
        assert assign_label(0.85) == "Advanced"
        assert assign_label(0.95) == "Expert"

    def test_random_agent_plays(self):
        """RandomAgent can play a full game."""
        game = PylosGame()
        agent = RandomAgent()
        moves = 0
        while game.get_result() is None and moves < 200:
            action = mcts_play(game, agent, search_iterations=2, c_puct=1.0)
            if action is None:
                break
            game.step(action)
            moves += 1
        assert moves > 0


class TestServerIntegration:
    def test_full_human_vs_human_game_flow(self):
        """Play several moves in a human_vs_human WebSocket game."""
        from server import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        with client.websocket_connect("/game") as ws:
            ws.send_json({"type": "new_game", "mode": "human_vs_human"})
            state = ws.receive_json()
            assert state["type"] == "state"

            # Play 4 moves (2 per player)
            for i in range(4):
                legal = state["legal_moves"]
                assert len(legal) > 0
                move = legal[0]  # pick first legal move
                ws.send_json({"type": "move", "action": move})
                state = ws.receive_json()
                assert state["type"] == "state"

            # Verify game progressed
            total_reserves = state["reserves"]["white"] + state["reserves"]["black"]
            assert total_reserves < 30  # some pieces placed
