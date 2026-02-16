# Pylos Engine Design

## Overview

A Pylos board game engine with AlphaZero-style AI training and a 3D web-based visualization. Forked from [ImprovedTinyZero](https://github.com/MikilFoss/ImprovedTinyZero) and extended with a complete action space, advanced rules, checkpoint-based difficulty levels, and an interactive Three.js frontend.

## Architecture

Monorepo with two main components:

```
pylos-engine/
├── engine/              # Python: game logic, training, AI server
│   ├── game.py          # Pylos game with full action space
│   ├── train.py         # AlphaZero self-play training
│   ├── server.py        # FastAPI + WebSocket server
│   ├── config.yaml      # Training hyperparameters
│   ├── agents.py        # Forked from ImprovedTinyZero
│   ├── mcts.py          # Forked from ImprovedTinyZero
│   ├── models.py        # Neural network (expanded action space)
│   ├── replay_buffer.py # Forked from ImprovedTinyZero
│   ├── evaluate.py      # Checkpoint evaluation (vs random agent)
│   └── checkpoints/     # Saved model snapshots + manifest.json
├── web/                 # Three.js 3D visualization
│   ├── index.html
│   └── src/
│       ├── scene.js     # 3D board, spheres, lighting, camera
│       ├── game.js      # Client-side game state + move validation
│       ├── network.js   # WebSocket communication
│       ├── ui.js        # HTML overlay (mode selector, difficulty, etc.)
│       └── animations.js # Sphere placement/raise/removal animations
└── requirements.txt
```

## Game Engine

### Rules (Advanced Variant)

- 2-player game on a 4-level pyramid (4x4, 3x3, 2x2, 1x1 = 30 positions)
- Each player has 15 spheres in reserve
- On your turn: place a reserve sphere on any supported empty position, OR raise one of your own unsupported spheres to a higher supported empty position
- Completing a **square** (2x2 of your color on any level) or a **line** (4-in-a-row on level 0, 3-in-a-row on level 1) triggers a removal phase: reclaim up to 2 of your own unsupported pieces
- First player to place the apex sphere (level 3) wins
- If a player has no legal moves, they lose

### Action Space

The existing ImprovedTinyZero Pylos implementation only encodes placement moves (30 actions). We expand to:

| Action type | Index range | Count | Description |
|-------------|-------------|-------|-------------|
| Placement   | 0-29        | 30    | Place reserve sphere at position i |
| Raise       | 30-N        | ~variable | Move own sphere from position i to higher position j |

All valid (source, destination) raise pairs are pre-computed at init time. Only positions where `dest_level > src_level` are included. At runtime, legal action masking filters to currently valid moves.

**Removal sub-phase:** After completing a square/line, the player enters a removal sub-turn. This is handled separately from the MCTS action space — the current player selects 0-2 of their own unsupported pieces to reclaim. For AI, removal is handled by a simple greedy heuristic (or a secondary small search).

### Observation Encoding

Flat vector of length 30 (one per pyramid position):
- `+1.0` = current player's sphere
- `-1.0` = opponent's sphere
- `0.0` = empty

Plus additional features: current player's reserves (normalized), opponent's reserves (normalized).

## Training Pipeline

### Algorithm

AlphaZero-style self-play (forked from ImprovedTinyZero):
1. **MCTS search** with neural network value + policy guidance
2. **Self-play** generates (observation, MCTS visit distribution, game outcome) tuples
3. **Training** via MSE loss (value head) + KL divergence (policy head)
4. **Replay buffer** stores recent game data for batch training

### Model

`LinearNetwork` with increased capacity for the larger action space:
- Input: observation vector (30 positions + 2 reserve features = 32)
- Hidden: 512 → 256 fully connected + ReLU
- Value head: Linear → tanh (outputs scalar in [-1, 1])
- Policy head: Linear → log_softmax (outputs distribution over all actions)

### Hyperparameters (config.yaml)

- Self-play games: 2000+
- MCTS search iterations: 64
- c_puct: 1.5
- Dirichlet alpha: 0.3
- Batch size: 128
- Replay buffer: 512 samples
- Training epochs per game: 3
- Learning rate: 1e-3 (AdamW)
- Weight decay: 1e-4

### Checkpoint System

```
engine/checkpoints/
├── checkpoint_0050.pth
├── checkpoint_0200.pth
├── checkpoint_0500.pth
├── checkpoint_1000.pth
├── checkpoint_2000.pth
├── checkpoint_final.pth
└── manifest.json
```

`manifest.json`:
```json
{
  "checkpoints": [
    {
      "file": "checkpoint_0050.pth",
      "step": 50,
      "win_rate_vs_random": 0.55,
      "label": "Beginner",
      "timestamp": "2026-02-15T12:00:00Z"
    },
    ...
  ]
}
```

Save frequency is configurable. Each checkpoint is auto-evaluated against a random agent (100 games) to assign a win rate and human-friendly label.

## WebSocket Server

FastAPI application with:

- `GET /checkpoints` — returns manifest.json
- `WS /game` — real-time game session

### WebSocket Protocol

**Client → Server:**
```json
{"type": "new_game", "mode": "human_vs_ai", "checkpoint": "checkpoint_0500.pth", "human_color": "white"}
{"type": "move", "action": {"type": "place", "level": 0, "row": 2, "col": 3}}
{"type": "move", "action": {"type": "raise", "src": [0, 1, 2], "dst": [1, 0, 1]}}
{"type": "remove", "pieces": [[0, 3, 1]]}
{"type": "skip_removal"}
```

**Server → Client:**
```json
{"type": "state", "board": [...], "turn": "white", "reserves": {"white": 12, "black": 13}, "legal_moves": [...]}
{"type": "ai_move", "action": {...}, "thinking_time_ms": 340}
{"type": "removal_phase", "removable_pieces": [[0,1,2], [0,3,1]]}
{"type": "game_over", "winner": "black", "reason": "apex_placed"}
```

### Game Modes

- **Human vs AI:** Human plays one color, AI (loaded from selected checkpoint) plays the other
- **AI vs AI:** Server plays both sides with configurable delay (1-2s) for spectating
- **Human vs Human:** Server validates moves but no AI involvement

## 3D Visualization

### Scene

- Wooden board base with subtle grid markings for each pyramid level
- Position markers (slight depressions or circles) showing where spheres can be placed
- Ambient light + directional light with soft shadows
- Orbit camera controls (drag to rotate, scroll to zoom)

### Spheres

- White marble / dark wood appearance (or clean white/black with subtle shading)
- Legal placement positions glow on hover
- Selected sphere (for raising) highlighted with outline

### Animations

- **Placement:** Sphere drops from above with bounce easing
- **Raise:** Sphere arcs from source to destination
- **Removal:** Sphere fades out and floats upward

### UI Overlay (HTML/CSS)

- Mode selector: Human vs AI / AI vs AI / Human vs Human
- Difficulty slider: populated from checkpoint manifest ("Beginner" → "Expert")
- Reserve counters for both players (visual sphere stacks)
- Turn indicator
- Move history panel (collapsible)
- "New Game" button

### Interaction Flow

1. Click empty legal position → place sphere (if reserves > 0)
2. Click own sphere → highlight it → click higher legal position → raise
3. After square/line completion: removable pieces pulse/glow, click to remove (up to 2), click "Done" to end removal phase

### Tech Stack

Vanilla Three.js served as static files. No build tools required — ES modules in the browser. Single `index.html` entry point.

## Testing Strategy

- **Game logic:** Unit tests for placement, raising, removal, square/line detection, win conditions, legal action generation, undo
- **Training:** Smoke test (10 self-play games) to verify training loop runs without errors
- **Server:** WebSocket integration tests for game flow
- **3D:** Manual testing in browser

## Dependencies

### Python (engine/)
- torch
- numpy
- numba
- fastapi
- uvicorn
- websockets
- pyyaml
- tqdm
- wandb (optional)

### Frontend (web/)
- Three.js (via CDN or vendored)
- No build tools
