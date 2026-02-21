# Deeper Network & Evaluation Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the shallow 2-layer MLP with a deeper residual network that can learn Pylos spatial patterns, and fix the evaluation to produce meaningful results with more games and stochastic play.

**Architecture:** The new network uses residual blocks (skip connections) so gradients flow through many layers without vanishing. The observation stays at 32 dimensions (no game.py changes needed) — the deeper network learns spatial features internally. Evaluation adds Dirichlet noise so games aren't deterministic, and increases to 200 games for statistical significance. All changes are backward-compatible with checkpoint loading (new model, fresh training).

**Tech Stack:** Python, PyTorch, NumPy

---

## Why This Fixes the Value Loss Plateau

The current 2-layer MLP (32→512→256) has ~290K parameters and can only compose 2 levels of features. It learned "who has more pieces" (value loss 1.0→0.33) but cannot learn:

1. **2x2 square detection** — requires correlating 4 specific board positions per pattern, across multiple possible squares. Needs multiple layers to first detect pairs, then detect squares.
2. **Support chains** — whether a piece can be raised depends on what's above AND below it. Requires reasoning across levels.
3. **Reclamation strategy** — completing a square lets you reclaim 2 pieces. Evaluating this requires: detect near-complete squares → estimate reclamation value → combine with positional score.

A 6-block residual network (32→256→[6 ResBlocks of 256]→heads) gives the model ~12 layers of feature composition with ~600K parameters — enough to learn these patterns while staying fast for CPU inference during MCTS.

Expected impact: value loss should drop from 0.33 → below 0.15 within 50K-100K games, as the network can now represent the spatial features that dominate Pylos strategy.

---

### Task 1: Replace PylosNetwork with a deeper residual architecture

**Files:**
- Modify: `engine/models.py`

**Step 1: Write the new PylosNetwork with residual blocks**

Replace the entire `engine/models.py` with:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-activation residual block: BN → ReLU → Linear → BN → ReLU → Linear + skip."""

    def __init__(self, dim):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(x))
        out = self.fc1(out)
        out = F.relu(self.bn2(out))
        out = self.fc2(out)
        return out + residual


class PylosNetwork(nn.Module):
    def __init__(self, input_shape, action_space, hidden=256, num_blocks=6):
        super().__init__()

        # Input projection
        self.input_fc = nn.Linear(input_shape[0], hidden)
        self.input_bn = nn.BatchNorm1d(hidden)

        # Residual tower
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(num_blocks)])

        # Value head
        self.value_fc1 = nn.Linear(hidden, 64)
        self.value_bn = nn.BatchNorm1d(64)
        self.value_fc2 = nn.Linear(64, 1)

        # Policy head
        self.policy_fc1 = nn.Linear(hidden, 128)
        self.policy_bn = nn.BatchNorm1d(128)
        self.policy_fc2 = nn.Linear(128, action_space)

        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def _backbone(self, x):
        """Shared residual backbone."""
        x = F.relu(self.input_bn(self.input_fc(x)))
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, observations):
        x = self._backbone(observations)
        # Value head
        v = F.relu(self.value_bn(self.value_fc1(x)))
        value = torch.tanh(self.value_fc2(v))
        # Policy head
        p = F.relu(self.policy_bn(self.policy_fc1(x)))
        log_policy = F.log_softmax(self.policy_fc2(p), dim=-1)
        return value, log_policy

    def inference(self, observation):
        """Combined value + policy in a single forward pass."""
        self.eval()
        with torch.no_grad():
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            x = self._backbone(observation)
            v = F.relu(self.value_bn(self.value_fc1(x)))
            value = torch.tanh(self.value_fc2(v))
            p = F.relu(self.policy_bn(self.policy_fc1(x)))
            policy = F.softmax(self.policy_fc2(p), dim=-1)
            return value.squeeze(0).item(), policy.squeeze(0).cpu().numpy()

    def batched_inference(self, observations_np):
        """Batch inference for multiple observations."""
        self.eval()
        with torch.no_grad():
            obs = torch.tensor(observations_np, device=self.device)
            x = self._backbone(obs)
            v = F.relu(self.value_bn(self.value_fc1(x)))
            values = torch.tanh(self.value_fc2(v)).squeeze(-1)
            p = F.relu(self.policy_bn(self.policy_fc1(x)))
            policies = F.softmax(self.policy_fc2(p), dim=-1)
            return values.cpu().numpy(), policies.cpu().numpy()

    def value_forward(self, observation):
        self.eval()
        with torch.no_grad():
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            x = self._backbone(observation)
            v = F.relu(self.value_bn(self.value_fc1(x)))
            return torch.tanh(self.value_fc2(v)).squeeze(0)

    def policy_forward(self, observation):
        self.eval()
        with torch.no_grad():
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            x = self._backbone(observation)
            p = F.relu(self.policy_bn(self.policy_fc1(x)))
            return F.softmax(self.policy_fc2(p), dim=-1).squeeze(0)
```

Key design decisions:
- **BatchNorm1d** — normalizes activations, stabilizes training at depth. Requires batch dim even for single inference, hence the `unsqueeze(0)` in inference methods.
- **Pre-activation residual blocks** — BN→ReLU before linear, proven to train better at depth than post-activation.
- **6 blocks × 256 hidden** — ~600K params. Deep enough to learn spatial patterns, small enough for fast CPU MCTS inference (~10 games/sec with 8 workers).
- **Separate backbone method** — avoids duplicating the residual tower logic across forward/inference/batched_inference.

**Step 2: Verify model creates and runs**

Run: `.venv/bin/python -c "from engine.models import PylosNetwork; import torch; m = PylosNetwork((32,), 303); print(f'Params: {sum(p.numel() for p in m.parameters()):,}'); x = torch.randn(4, 32, device=m.device); m.train(); v, p = m(x); print(f'Train: v={v.shape}, p={p.shape}'); m.eval(); val, pol = m.inference(torch.randn(32, device=m.device)); print(f'Inference: v={val:.3f}, policy sum={pol.sum():.3f}')"`

Expected: Params ~600K, shapes correct, policy sums to 1.0.

**Step 3: Commit**

```bash
git add engine/models.py
git commit -m "feat: deeper residual network (6 blocks × 256 hidden) for better learning"
```

---

### Task 2: Add stochastic evaluation with more games

**Files:**
- Modify: `engine/evaluate.py:51-92`

**Why:** Currently evaluation is deterministic (no noise, argmax move selection). Two similar models play identical games every time, producing win rates quantized to {0, 0.25, 0.5, 0.75, 1.0}. Adding Dirichlet noise to eval MCTS makes each game unique, so 200 games produces a continuous win rate with meaningful statistical power.

**Step 1: Add Dirichlet noise to evaluation play**

In `engine/evaluate.py`, modify `_play_match` to pass `dirichlet_alpha` to the `play` function:

```python
def _play_match(agent_a, agent_b, num_games, search_iterations=32, eval_noise=0.15):
    """Play a match between two agents, returning agent_a's win rate.

    Alternates colors each game. Draws count as 0.5 for both.
    Uses light Dirichlet noise (eval_noise) to break determinism
    so each game plays out differently.
    """
    game = PylosGame()
    a_kwargs = {"search_iterations": search_iterations, "dirichlet_alpha": eval_noise}
    b_kwargs = {"search_iterations": search_iterations, "dirichlet_alpha": eval_noise}

    # If agent_b is RandomAgent, give it 1 iteration and no noise
    if isinstance(agent_b, RandomAgent):
        b_kwargs = {"search_iterations": 1, "dirichlet_alpha": None}

    score = 0.0
    for i in range(num_games):
        game.reset()
        if i % 2 == 0:
            agents = [agent_a, agent_b]
            kwargs = [a_kwargs, b_kwargs]
            a_color = 1
        else:
            agents = [agent_b, agent_a]
            kwargs = [b_kwargs, a_kwargs]
            a_color = -1

        cur, nxt = 0, 1
        moves = 0
        while game.get_result() is None and moves < MAX_EVAL_MOVES:
            action = play(game, agents[cur], **kwargs[cur])
            if action is None:
                break
            game.step(action)
            cur, nxt = nxt, cur
            moves += 1

        result = game.get_result()
        if result == a_color:
            score += 1.0
        elif result is None:
            score += 0.5  # draw

    return score / num_games
```

**Step 2: Update config to use 200 eval games**

In `engine/config_v3.yaml` (or the new v4 config), change:
```yaml
eval_games: 200
```

With 200 games and noise, a true 55% win rate would show as 55%±7% (95% CI), which is distinguishable from 50%. Without noise, 200 games would still produce only 2 distinct outcomes.

**Step 3: Commit**

```bash
git add engine/evaluate.py engine/config_v3.yaml
git commit -m "fix: add noise to eval games and increase to 200 for statistical significance"
```

---

### Task 3: Create v4 config for fresh training with new architecture

**Files:**
- Create: `engine/config_v4.yaml`

**Why:** The new deeper network has different weights and can't load old checkpoints (architecture mismatch). We need a fresh config that starts from scratch. The hyperparameters are also tuned for the deeper model.

**Step 1: Create the v4 config**

```yaml
# ─────────────────────────────────────────────────────────────────
# Pylos AlphaZero Training Config — v4
#
# Fresh training with deeper residual network (6 blocks × 256).
# Trains from scratch — incompatible with v3 checkpoints.
#
# Run:
#   caffeinate -dims .venv/bin/python -m engine.train --config engine/config_v4.yaml
# ─────────────────────────────────────────────────────────────────

training:
  selfplay_games: 500000
  search_iterations: 64
  batch_size: 256
  replay_buffer_size: 32768
  epochs_per_game: 4
  learning_rate: 0.002
  weight_decay: 0.0001
  c_puct: 1.5
  dirichlet_alpha: 0.3
  selfplay_batch_size: 0

  min_learning_rate: 0.0001
  temp_threshold: 15

  # ── Draw limits ──────────────────────────────────────────────
  max_moves: 200
  repetition_limit: 5
  move_limit_draw_penalty: 0.0
  repetition_draw_penalty: 0.0

checkpoints:
  save_every: 1000
  eval_games: 200
  dir: engine/checkpoints_v4

  # No resume — fresh training with new architecture
  # resume_from:

wandb:
  enabled: false
  project: pylos-alphazero
```

Key parameter changes from v3:
- `batch_size: 256` (was 128) — larger batches stabilize training with BatchNorm
- `replay_buffer_size: 32768` (was 16384) — more diverse training data
- `epochs_per_game: 4` (was 10) — less overfitting per batch, more diverse gradients
- `learning_rate: 0.002` (was 0.001) — slightly higher to compensate for BatchNorm's stabilizing effect
- `min_learning_rate: 0.0001` (was 0.00001) — don't decay too low
- `eval_games: 200` (was 50) — meaningful evaluation
- `selfplay_games: 500000` — sufficient for the deeper model to converge

**Step 2: Commit**

```bash
git add engine/config_v4.yaml
git commit -m "feat: add v4 config for deeper network training"
```

---

### Task 4: Fix train.py to handle BatchNorm properly during self-play

**Files:**
- Modify: `engine/train.py`

**Why:** BatchNorm behaves differently in train vs eval mode. During self-play, the model must be in `eval()` mode (use running stats, not batch stats) because we infer one position at a time. The current code already calls `model.eval()` in workers and `model.train()` in `_train_batch()`, but we need to ensure `_batched_selfplay` also uses eval mode for inference.

**Step 1: Verify _batched_selfplay calls model.eval()**

In `engine/train.py`, the `batched_inference` method already calls `self.eval()`. Check that `_batched_selfplay` doesn't call `model.train()` anywhere. If it does, remove it.

Also: after `_train_batch()` completes (which sets `model.train()`), we should set `model.eval()` before the next self-play batch when using the batched path. Add after the `_train_batch()` call in the batched selfplay loop:

```python
_train_batch()
model.eval()  # back to eval mode for next self-play batch
```

**Step 2: Commit**

```bash
git add engine/train.py
git commit -m "fix: ensure model in eval mode during self-play with BatchNorm"
```

---

### Task 5: Stop v3 training, start v4 training

**Step 1: Stop the currently running v3 training**

Check if running and stop it:
```bash
ps aux | grep "engine.train" | grep -v grep
# kill the PID if running
```

**Step 2: Start v4 training**

```bash
caffeinate -dims .venv/bin/python -m engine.train --config engine/config_v4.yaml
```

**Step 3: Monitor for first 5K games**

After ~5 minutes, check:
```bash
tail -5 engine/checkpoints_v4/loss_history.jsonl
```

Expected: value_loss should start around 0.8-1.0 and drop rapidly. By 5K games it should be below 0.5 (already better trajectory than v3).

---

## Summary

| Change | What | Why |
|--------|------|-----|
| Deeper model | 2-layer MLP → 6-block ResNet | Break the 0.33 value loss ceiling |
| BatchNorm | Added to all layers | Stabilize deep network training |
| Residual connections | Skip connections in each block | Enable gradient flow through 12+ layers |
| Eval noise | Dirichlet α=0.15 in eval | Break deterministic games, get real win rates |
| More eval games | 50 → 200 | Statistical significance (±7% at 95% CI) |
| Larger batch | 128 → 256 | Better gradient estimates, BatchNorm needs it |
| Larger buffer | 16K → 32K | More diverse training data |
| Fewer epochs | 10 → 4 | Less overfitting, more generalization |
| Fresh start | No resume | New architecture incompatible with old checkpoints |
