# Pylos Engine

AlphaZero-trained AI for the board game [Pylos](https://www.ultraboardgames.com/pylos/game-rules.php), with a 3D web visualization.

Built on top of [ImprovedTinyZero](https://github.com/MikilFoss/ImprovedTinyZero) — an AlphaZero-style framework using MCTS + neural network self-play.

## Quick Start

```bash
# Set up
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train the AI (adjust config.yaml for fewer games to test)
python engine/train.py

# Launch the web server
bash run.sh
# Open http://localhost:8000
```

## Training

The AI trains via AlphaZero self-play: MCTS with a neural network for value/policy estimation.

Edit `engine/config.yaml` to adjust:
- `selfplay_games` — number of training games (default: 2000)
- `search_iterations` — MCTS depth per move (default: 64)
- `checkpoints.save_every` — save frequency (default: every 100 games)

Checkpoints are saved to `engine/checkpoints/` with a `manifest.json` tracking each snapshot's win rate vs a random agent.

For a quick test run:
```bash
python engine/train.py --config engine/config.yaml  # or create a smaller config
```

## Playing

Open `http://localhost:8000` after starting the server. Three game modes:

- **Human vs AI** — pick a difficulty level (checkpoint) and color
- **AI vs AI** — watch two agents play with animated moves
- **Human vs Human** — play locally with a friend

The difficulty selector shows checkpoints from training, labeled Beginner through Expert based on win rate vs random.

### Controls

- **Click** an empty position to place a sphere
- **Click** your own sphere, then click a higher position to raise it
- **Orbit** the camera by dragging, **zoom** with scroll

## Project Structure

```
engine/
  game.py          — Pylos game logic (303-action space: 30 placements + 273 raises)
  mcts.py          — Monte Carlo Tree Search
  agents.py        — ClassicMCTS, AlphaZero, and AlphaZeroTrainer agents
  models.py        — PylosNetwork (MLP with value + policy heads)
  replay_buffer.py — Experience replay for training
  train.py         — Self-play training loop with checkpoint saving
  evaluate.py      — Evaluate checkpoints vs random agent
  server.py        — FastAPI + WebSocket server
  config.yaml      — Training hyperparameters
  checkpoints/     — Saved model snapshots

web/
  index.html       — Main page with UI overlay
  src/scene.js     — Three.js 3D scene (board, spheres, lighting)
  src/game.js      — Client-side game state
  src/network.js   — WebSocket communication
  src/main.js      — Game loop and UI logic
```

## Rules

Pylos is a 2-player abstract strategy game on a 4-level pyramid. Players alternate placing or raising spheres. Completing a square (2x2) or line of your color lets you reclaim up to 2 pieces. The first player to place the apex sphere wins.

## Dice Sudoku Cube Solver

`dicecube/` reuses the same recipe (residual value/policy network + tree search)
to solve the dice "Blind Dice Cube": a 3x3x3 whose stickers are 1-9 pip dice faces
with 5 in every center. It is solved when **every face shows 1-9 once**, so there
are hundreds of thousands of solved states, not just one.

```bash
bash run_dice.sh          # http://localhost:8001: 3D cube, sticker editor, solver, playback
python -m dicecube.solver --scramble 20          # CLI
python -m dicecube.solver "<54 digits, URFDLB>"  # solve your own cube
python -m dicecube.train                         # retrain the network (dicecube/config.yaml)
python -m dicecube.evaluate                      # benchmark by scramble depth
```

Three solvers run side by side: AlphaZero-style **MCTS** and **weighted A\*** guided
by the trained network, plus **Kociemba two-phase** over many candidate goal
states as a guaranteed fallback. See
[docs/plans/2026-10-19-dice-cube-solver.md](docs/plans/2026-10-19-dice-cube-solver.md)
for how it works.

## Testing

```bash
source .venv/bin/activate
python -m pytest engine/tests/ -v
python -m pytest dicecube/tests/ -v
```
