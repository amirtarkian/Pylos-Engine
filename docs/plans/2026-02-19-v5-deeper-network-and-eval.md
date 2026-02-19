# V5: Deeper Network & Richer Evaluation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a significantly more capable V5 model by deepening the residual network, enriching the board observation encoding, and increasing evaluation reliability — addressing the ELO plateau at ~1050-1095 seen in V4.

**Architecture:** V5 uses 10 residual blocks (up from 6) with richer 92-dim observations (up from 32-dim) that include piece support/freedom info, bigger value/policy heads, and gradient clipping. The richer input representation lets the network focus on strategy rather than re-deriving spatial properties. Evaluation uses 2x more games per matchup for statistically reliable ELO estimates.

**Tech Stack:** Python, PyTorch, NumPy, YAML

---

## Why Value Loss Should Decrease With a Deeper Network

The value head predicts who will win from a given board position. Its output is in [-1, +1] and the target is the actual game outcome ({-1, 0, +1}). The MSE loss measures how accurately the model evaluates positions:

```
value_loss = E[(predicted_value - actual_outcome)²]
```

### Current V4 State: Value Loss ≈ 0.256

With value loss at 0.256, the model's average prediction error is √0.256 ≈ 0.506 on a [-1, +1] scale. This means the model is barely better than guessing — it knows basic heuristics like "more reserves = better" but can't evaluate complex positions.

### Why a Deeper Network Helps

**Layer-by-layer feature composition:** Each residual block learns to transform features at increasing abstraction levels:

| Layers | V4 (6 blocks) | V5 (10 blocks) |
|--------|---------------|-----------------|
| 1-2 | Piece positions, adjacency | Same |
| 3-4 | Partial squares, support chains | Same |
| 5-6 | Near-complete formations, piece mobility | Same |
| 7-8 | (doesn't exist) | Multi-step removal opportunities |
| 9-10 | (doesn't exist) | Strategic position evaluation (piece economy, blocking) |

V4 maxes out at ~6 layers of feature composition. It can detect squares and count pieces, but it cannot reason about multi-step consequences like "completing this square → reclaiming 2 pieces → gaining piece advantage → winning". That requires 8-10 layers of reasoning.

**Richer observations accelerate learning:** The current 32-dim input forces the network to spend its early layers re-deriving what positions are supported and which pieces are free to move — information that's trivially computable from the board. By providing these as input features (92 dims), the network can spend ALL its layers on strategic reasoning.

### Expected Value Loss Trajectory

| Version | Architecture | Value Loss Plateau | Why |
|---------|-------------|-------------------|-----|
| V3 | 2-layer MLP (290K params) | 0.33 | Can only learn piece count heuristics |
| V4 | 6 blocks × 256 (893K params) | 0.25 | Can detect formations, but limited strategic depth |
| V5 | 10 blocks × 256 (1.5M params) + rich obs | Target: < 0.15 | Full formation reasoning + multi-step evaluation |
| Theory minimum | Perfect evaluator | ~0.05-0.08 | Irreducible noise: some positions are genuinely uncertain |

The irreducible minimum is NOT zero because many positions are genuinely ambiguous — either player could win depending on play. But 0.15 would represent a model that correctly evaluates most non-ambiguous positions.

---

## Task 1: Add Rich Observation Encoding to PylosGame

**Files:**
- Modify: `engine/game.py:34,67,661-680`

**Why:** The current 32-dim observation (30 board cells + 2 reserves) makes the network rediscover spatial properties. A 92-dim observation adds precomputed support and freedom features, freeing network capacity for strategy.

**Step 1: Add `rich_obs` parameter to `PylosGame.__init__`**

In `engine/game.py`, modify `__init__` to accept a `rich_obs` parameter and update `observation_shape`:

```python
class PylosGame:
    def __init__(self, rich_obs=False):
        self._rich_obs = rich_obs
        # Pre-compute position mappings
        self.index_to_coords = []
        self.coords_to_index = {}
        # ... (existing code unchanged) ...

        self.action_space = 303
        self.observation_shape = (92,) if rich_obs else (32,)

        self.reset()
```

**Step 2: Replace `to_observation` with a dispatcher**

Replace the existing `to_observation` method with:

```python
def to_observation(self):
    """Convert game state to observation tensor.

    Basic mode (32 dims): 30 board cells + 2 reserves.
    Rich mode (92 dims): 30 board cells + 30 support + 30 freedom + 2 reserves.
    """
    if self._rich_obs:
        return self._to_observation_rich()
    return self._to_observation_basic()

def _to_observation_basic(self):
    """Original 32-dim observation: board cells + reserves."""
    obs = np.zeros(32, dtype=np.float32)
    for idx in range(30):
        level, r, c = self.index_to_coords[idx]
        cell = self.board[level][r, c]
        if cell != 0:
            obs[idx] = float(cell * self.turn)
    obs[30] = self.reserves[self.turn] / 15.0
    obs[31] = self.reserves[-self.turn] / 15.0
    return obs

def _to_observation_rich(self):
    """Rich 92-dim observation: board + support + freedom + reserves.

    Channels (each 30 dims, one per pyramid position):
      0-29:  Board cells — +1 current player, -1 opponent, 0 empty
      30-59: Supported — 1.0 if position is supported from below, 0.0 otherwise
      60-89: Free — 1.0 if piece has no piece above it (can be raised/removed), 0.0 otherwise
      90:    Current player reserves / 15
      91:    Opponent reserves / 15
    """
    obs = np.zeros(92, dtype=np.float32)

    for idx in range(30):
        level, r, c = self.index_to_coords[idx]
        cell = self.board[level][r, c]

        # Board cell
        if cell != 0:
            obs[idx] = float(cell * self.turn)

        # Supported from below
        if self.is_supported(level, r, c):
            obs[30 + idx] = 1.0

        # Free (no piece above) — only meaningful for occupied positions
        if cell != 0 and not self.piece_has_top(level, r, c):
            obs[60 + idx] = 1.0

    obs[90] = self.reserves[self.turn] / 15.0
    obs[91] = self.reserves[-self.turn] / 15.0
    return obs
```

**Step 3: Run quick sanity check**

Run: `.venv/bin/python -c "from engine.game import PylosGame; g = PylosGame(rich_obs=True); print(f'shape: {g.observation_shape}'); obs = g.to_observation(); print(f'obs shape: {obs.shape}, sum: {obs.sum():.1f}'); g2 = PylosGame(); print(f'basic shape: {g2.observation_shape}, obs: {g2.to_observation().shape}')"`

Expected: `shape: (92,)`, `obs shape: (92,)`, basic shape: `(32,)`.

**Step 4: Commit**

```bash
git add engine/game.py
git commit -m "feat: add rich 92-dim observation encoding for V5"
```

---

## Task 2: Deepen the Network and Enlarge Heads

**Files:**
- Modify: `engine/models.py:26,37-44`

**Why:** V4's 6 blocks × 256 with 64-dim value head and 128-dim policy head saturated at ELO ~1095. V5 uses 10 blocks for deeper strategic reasoning and larger heads for more expressive value/policy predictions. The `hidden` and `num_blocks` params are already in the constructor — we just change the defaults and enlarge the heads.

**Step 1: Update `PylosNetwork` default params and head sizes**

In `engine/models.py`, update `__init__`:

```python
class PylosNetwork(nn.Module):
    def __init__(self, input_shape, action_space, hidden=256, num_blocks=6,
                 value_hidden=64, policy_hidden=128):
        super().__init__()

        # Input projection
        self.input_fc = nn.Linear(input_shape[0], hidden)
        self.input_bn = nn.BatchNorm1d(hidden)

        # Residual tower
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(num_blocks)])

        # Value head
        self.value_fc1 = nn.Linear(hidden, value_hidden)
        self.value_bn = nn.BatchNorm1d(value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        # Policy head
        self.policy_fc1 = nn.Linear(hidden, policy_hidden)
        self.policy_bn = nn.BatchNorm1d(policy_hidden)
        self.policy_fc2 = nn.Linear(policy_hidden, action_space)

        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)
```

This is backward-compatible: V1-V4 checkpoints were created with `hidden=256, num_blocks=6, value_hidden=64, policy_hidden=128` (the defaults).

V5 will create the model with: `hidden=256, num_blocks=10, value_hidden=128, policy_hidden=256`.

**Step 2: Verify model creation and parameter count**

Run: `.venv/bin/python -c "from engine.models import PylosNetwork; m = PylosNetwork((92,), 303, hidden=256, num_blocks=10, value_hidden=128, policy_hidden=256); print(f'V5 params: {sum(p.numel() for p in m.parameters()):,}'); m4 = PylosNetwork((32,), 303); print(f'V4 params: {sum(p.numel() for p in m4.parameters()):,}')"`

Expected: V5 ~1.5M params, V4 ~893K params.

**Step 3: Commit**

```bash
git add engine/models.py
git commit -m "feat: add configurable value/policy head sizes for V5"
```

---

## Task 3: Save Model Architecture in Checkpoints

**Files:**
- Modify: `engine/train.py:314-330` (save_checkpoint function)
- Modify: `engine/evaluate.py:36-48` (_load_agent function)

**Why:** V5 uses a different input dimension (92 vs 32) and different head sizes. When loading checkpoints for evaluation, we need to know which architecture was used. Storing `model_config` in the checkpoint file solves this cleanly.

**Step 1: Store model config when saving checkpoints**

In `engine/train.py`, modify `save_checkpoint` to include model metadata:

```python
def save_checkpoint(agent, step, config, manifest_path):
    ckpt_dir = config["checkpoints"]["dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    filename = f"checkpoint_{step:05d}.pth"
    filepath = os.path.join(ckpt_dir, filename)

    # Read model architecture from config
    model_cfg = config.get("model", {})

    torch.save(
        {
            "model_state_dict": agent.model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "step": step,
            "model_config": {
                "hidden": model_cfg.get("hidden", 256),
                "num_blocks": model_cfg.get("num_blocks", 6),
                "value_hidden": model_cfg.get("value_hidden", 64),
                "policy_hidden": model_cfg.get("policy_hidden", 128),
                "rich_obs": model_cfg.get("rich_obs", False),
            },
        },
        filepath,
    )
    # ... rest of function unchanged
```

**Step 2: Update `_load_agent` in evaluate.py to read model config**

In `engine/evaluate.py`, replace `_load_agent`:

```python
def _load_agent(model_path):
    """Load a checkpoint into an AlphaZeroAgent on CPU.

    Reads model architecture from checkpoint metadata (V5+).
    Falls back to V4 defaults for older checkpoints.
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    # Read model config from checkpoint, or use V4 defaults
    model_cfg = checkpoint.get("model_config", {})
    hidden = model_cfg.get("hidden", 256)
    num_blocks = model_cfg.get("num_blocks", 6)
    value_hidden = model_cfg.get("value_hidden", 64)
    policy_hidden = model_cfg.get("policy_hidden", 128)
    rich_obs = model_cfg.get("rich_obs", False)

    game_tmp = PylosGame(rich_obs=rich_obs)
    model = PylosNetwork(
        game_tmp.observation_shape, game_tmp.action_space,
        hidden=hidden, num_blocks=num_blocks,
        value_hidden=value_hidden, policy_hidden=policy_hidden,
    )
    model.device = torch.device("cpu")
    model.to(model.device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return AlphaZeroAgent(model, rich_obs=rich_obs)
```

**Step 3: Update `AlphaZeroAgent` to use the right observation**

In `engine/agents.py`, add `rich_obs` to agent:

```python
class AlphaZeroAgent:
    def __init__(self, model, rich_obs=False):
        self.model = model
        self.model.eval()
        self._rich_obs = rich_obs

    def eval_fn(self, game):
        """Combined value + policy in a single forward pass."""
        # Temporarily set game's observation mode to match our model
        orig_rich = game._rich_obs
        game._rich_obs = self._rich_obs
        observation = torch.tensor(game.to_observation(), device=self.model.device, requires_grad=False)
        game._rich_obs = orig_rich
        return self.model.inference(observation)
```

Also update `value_fn` and `policy_fn` similarly.

**Step 4: Commit**

```bash
git add engine/train.py engine/evaluate.py engine/agents.py
git commit -m "feat: save/load model architecture metadata in checkpoints"
```

---

## Task 4: Add Gradient Clipping and Update Training for V5

**Files:**
- Modify: `engine/train.py:536-563` (_train_batch function)
- Modify: `engine/train.py:442-450` (model creation)

**Why:** A deeper network with more parameters is more susceptible to gradient explosion. Gradient clipping (max_norm=1.0) prevents training instabilities. We also need the training script to read model architecture from config.

**Step 1: Read model config from YAML and create model accordingly**

In `engine/train.py`, after loading config in `main()`, create the model using config-driven architecture:

```python
# Create game, model, optimizer, agent
model_cfg = config.get("model", {})
rich_obs = model_cfg.get("rich_obs", False)
game = PylosGame(rich_obs=rich_obs)
model = PylosNetwork(
    game.observation_shape, game.action_space,
    hidden=model_cfg.get("hidden", 256),
    num_blocks=model_cfg.get("num_blocks", 6),
    value_hidden=model_cfg.get("value_hidden", 64),
    policy_hidden=model_cfg.get("policy_hidden", 128),
)
```

**Step 2: Add gradient clipping to `_train_batch`**

In `_train_batch()`, add `torch.nn.utils.clip_grad_norm_` before `optimizer.step()`:

```python
def _train_batch():
    nonlocal latest_vloss, latest_ploss
    if len(agent.replay_buffer) < batch_size:
        return
    model.train()
    max_grad_norm = train_cfg.get("max_grad_norm", 0)  # 0 = disabled
    value_losses = []
    policy_losses = []
    for _ in range(epochs):
        observations, actions_dist, results_batch = agent.replay_buffer.sample(batch_size)
        observations = torch.tensor(observations, device=model.device)
        actions_dist = torch.tensor(actions_dist, device=model.device)
        results_t = torch.tensor(results_batch, device=model.device)

        optimizer.zero_grad()
        values, log_policies = model(observations)
        vloss = F.mse_loss(values.squeeze(1), results_t)
        ploss = F.kl_div(log_policies, actions_dist, reduction="batchmean")
        (vloss + ploss).backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        value_losses.append(vloss.item())
        policy_losses.append(ploss.item())

    if value_losses:
        latest_vloss = sum(value_losses) / len(value_losses)
        latest_ploss = sum(policy_losses) / len(policy_losses)
    scheduler.step()
```

**Step 3: Update worker initialization for rich observations**

In `_init_worker`, the worker creates a `PylosGame` for the observation shape. Update to pass `rich_obs`:

```python
def _init_worker(model_state_dict, obs_shape, action_space, hidden, num_blocks,
                 value_hidden, policy_hidden):
    global _worker_agent
    torch.set_num_threads(1)
    model = PylosNetwork(obs_shape, action_space, hidden=hidden,
                         num_blocks=num_blocks, value_hidden=value_hidden,
                         policy_hidden=policy_hidden)
    model.device = torch.device("cpu")
    model.to(model.device)
    model.load_state_dict(model_state_dict)
    model.eval()
    from agents import AlphaZeroAgent
    _worker_agent = AlphaZeroAgent(model)
```

Update `_create_pool` accordingly:

```python
def _create_pool():
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return mp.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(cpu_state, game.observation_shape, game.action_space,
                  model_cfg.get("hidden", 256), model_cfg.get("num_blocks", 6),
                  model_cfg.get("value_hidden", 64), model_cfg.get("policy_hidden", 128)),
    )
```

**Step 4: Update `_run_selfplay` to use the right observation**

The worker process creates `PylosGame()` inside `_run_selfplay`. We need to pass `rich_obs` to the worker. Add it to the args tuple:

```python
def _run_selfplay(args):
    search_iters, c_puct, dir_alpha, max_moves, rep_limit, ml_penalty, rep_penalty, temp_threshold, rich_obs = args
    from mcts import search
    from collections import defaultdict

    game = PylosGame(rich_obs=rich_obs)
    # ... rest unchanged
```

And update the args_iter in the training loop:

```python
args_iter = ((search_iters, c_puct, dir_alpha, max_moves, rep_limit, ml_penalty, rep_penalty, temp_threshold, rich_obs)
             for _ in range(remaining))
```

**Step 5: Commit**

```bash
git add engine/train.py
git commit -m "feat: config-driven model architecture + gradient clipping for V5"
```

---

## Task 5: Create V5 Config

**Files:**
- Create: `engine/config_v5.yaml`

**Step 1: Create the V5 config**

```yaml
# ─────────────────────────────────────────────────────────────────
# Pylos AlphaZero Training Config — v5
#
# Deeper residual network (10 blocks × 256) with rich 92-dim
# observations. Fresh training — incompatible with V4 checkpoints.
#
# Run:
#   caffeinate -dims .venv/bin/python -m engine.train --config engine/config_v5.yaml
# ─────────────────────────────────────────────────────────────────

model:
  hidden: 256
  num_blocks: 10
  value_hidden: 128
  policy_hidden: 256
  rich_obs: true

training:
  selfplay_games: 500000
  search_iterations: 64
  batch_size: 256
  replay_buffer_size: 65536
  epochs_per_game: 4
  learning_rate: 0.001
  weight_decay: 0.0001
  c_puct: 1.5
  dirichlet_alpha: 0.3
  selfplay_batch_size: 0

  min_learning_rate: 0.00005
  temp_threshold: 15
  max_grad_norm: 1.0

  # ── Draw limits ──────────────────────────────────────────────
  max_moves: 200
  repetition_limit: 5
  move_limit_draw_penalty: 0.0
  repetition_draw_penalty: 0.0

checkpoints:
  save_every: 1000
  eval_games: 40
  dir: engine/checkpoints_v5

wandb:
  enabled: false
  project: pylos-alphazero
```

Key V5 parameter changes from V4:
- `model.num_blocks: 10` (was 6) — 67% more depth for strategic reasoning
- `model.value_hidden: 128` (was 64) — 2x more capacity in value evaluation
- `model.policy_hidden: 256` (was 128) — 2x more capacity in action selection
- `model.rich_obs: true` (was implicit false) — 92-dim observations
- `replay_buffer_size: 65536` (was 32768) — more diverse training data for deeper model
- `learning_rate: 0.001` (was 0.002) — slightly lower for larger model stability
- `max_grad_norm: 1.0` (new) — gradient clipping prevents explosion
- `selfplay_games: 500000` — can be extended if model keeps improving

**Step 2: Commit**

```bash
git add engine/config_v5.yaml
git commit -m "feat: add V5 config with deeper network and rich observations"
```

---

## Task 6: Increase ELO Evaluation Games

**Files:**
- Modify: `engine/elo_evaluator.py:167-174` (argparse defaults)

**Why:** The current defaults (20 games/anchor, 30 games/pair in round-robin) give noisy ELO estimates. A 50% ± 15% confidence interval isn't enough to distinguish close models. Doubling the games narrows CIs to ± 10%, making trends clearer.

**Step 1: Update default game counts in elo_evaluator.py**

```python
parser.add_argument("--games", type=int, default=40,
                    help="Games per matchup (default: 40)")
parser.add_argument("--search-iters", type=int, default=32,
                    help="MCTS search iterations for eval games (default: 32)")
parser.add_argument("--anchor-spacing", type=int, default=10000,
                    help="Step spacing between anchor models (default: 10000)")
parser.add_argument("--round-robin-games", type=int, default=50,
                    help="Games per pair in anchor round-robin (default: 50)")
```

**Step 2: Also update the CKPT_DIR to be configurable**

Add a `--ckpt-dir` argument so the evaluator can be used for any training version:

```python
parser.add_argument("--ckpt-dir", type=str, default=None,
                    help="Checkpoint directory (default: auto-detect from manifest)")
```

And at the top of `main()`:

```python
global CKPT_DIR, MANIFEST
if args.ckpt_dir:
    CKPT_DIR = args.ckpt_dir
    MANIFEST = os.path.join(CKPT_DIR, "manifest.json")
```

**Step 3: Commit**

```bash
git add engine/elo_evaluator.py
git commit -m "feat: increase ELO eval games and add configurable checkpoint dir"
```

---

## Task 7: Update Web UI and Server for V5 Tracking

**Files:**
- Modify: `engine/server.py` (TRAINING_RUNS dict)
- Modify: `web/index.html` (if V5 option needed in dropdown)

**Step 1: Add V5 to TRAINING_RUNS in server.py**

Find the `TRAINING_RUNS` dict in `engine/server.py` and add the v5 entry:

```python
"v5": {
    "dir": os.path.join(ENGINE_DIR, "checkpoints_v5"),
    "config": os.path.join(ENGINE_DIR, "config_v5.yaml"),
},
```

**Step 2: Commit**

```bash
git add engine/server.py
git commit -m "feat: add V5 training run to web dashboard"
```

---

## Task 8: Start V5 Training

**Step 1: Create checkpoints directory**

```bash
mkdir -p engine/checkpoints_v5
```

**Step 2: Verify the model creates correctly**

```bash
.venv/bin/python -c "
from engine.game import PylosGame
from engine.models import PylosNetwork
g = PylosGame(rich_obs=True)
m = PylosNetwork(g.observation_shape, g.action_space, hidden=256, num_blocks=10, value_hidden=128, policy_hidden=256)
print(f'Obs shape: {g.observation_shape}')
print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
print(f'Device: {m.device}')
import torch
m.train()
x = torch.randn(4, 92, device=m.device)
v, p = m(x)
print(f'Train: value={v.shape}, policy={p.shape}')
m.eval()
val, pol = m.inference(torch.randn(92, device=m.device))
print(f'Inference: value={val:.3f}, policy_sum={pol.sum():.3f}')
print('All checks passed!')
"
```

Expected: ~1.5M params, obs shape (92,), shapes correct, policy sums to 1.0.

**Step 3: Start V5 training (keep V4 running)**

```bash
caffeinate -dims .venv/bin/python -m engine.train --config engine/config_v5.yaml
```

**Step 4: Monitor first 5K games**

After ~10-15 minutes, check:
```bash
tail -5 engine/checkpoints_v5/loss_history.jsonl
cat engine/checkpoints_v5/training_progress.json
```

Expected: value_loss should start around 0.7-0.9 and drop faster than V4 did (V4 was at 0.25 by ~100K games; V5 should reach 0.25 by ~50K games).

---

## Summary

| Change | V4 | V5 | Why |
|--------|----|----|-----|
| Residual blocks | 6 | 10 | Deeper strategic reasoning (multi-step consequences) |
| Value head | 256→64→1 | 256→128→1 | More nuanced position evaluation |
| Policy head | 256→128→303 | 256→256→303 | Better action pattern recognition |
| Observation | 32 dims (board + reserves) | 92 dims (board + support + freedom + reserves) | Free up layers for strategy vs spatial derivation |
| Total params | ~893K | ~1.5M | 1.7x more capacity |
| Gradient clipping | None | max_norm=1.0 | Prevent gradient explosion in deeper network |
| Replay buffer | 32K | 64K | More diverse training data for larger model |
| Learning rate | 0.002 | 0.001 | Lower for larger model stability |
| ELO eval games/anchor | 20 | 40 | Narrower confidence intervals for reliable ELO |
| ELO round-robin games | 30 | 50 | More reliable anchor calibration |
