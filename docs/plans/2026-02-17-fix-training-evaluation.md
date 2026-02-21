# Fix Training & Evaluation Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the AlphaZero training pipeline so the model actually improves, and fix the evaluation pipeline so we can measure progress accurately.

**Architecture:** Multiple bugs compound to produce the "stuck at 50%" symptom. The evaluation only compares model-vs-model (never vs random), most evals never complete (daemon threads killed on restart), the optimizer state isn't restored on resume (destroying Adam momentum), and the model class overrides `__call__` instead of defining `forward()`. We fix evaluation first (so we can measure), then training (so improvement happens).

**Tech Stack:** Python, PyTorch, NumPy, MCTS

---

## Root Cause Analysis

Investigation found **6 issues** contributing to the "50% win rate" symptom:

| # | Issue | Impact | Severity |
|---|-------|--------|----------|
| 1 | Evaluation only runs model-vs-model, never vs random | UI shows 50% because similar models tie; we have no signal for absolute strength | **Critical** |
| 2 | Daemon eval threads killed on process restart | Most evaluations from step 32K+ never complete, showing "Evaluating..." forever | **High** |
| 3 | Optimizer state not restored on resume | AdamW momentum/variance reset — training regresses after every restart | **Critical** |
| 4 | Replay buffer empty on resume | Model trains on tiny dataset initially, causing instability | **Medium** |
| 5 | `PylosNetwork.__call__` overrides `nn.Module.__call__` instead of defining `forward()` | Bypasses PyTorch hook/module machinery; `self.train()` called every forward pass during training | **Medium** |
| 6 | Value loss plateaued at ~0.48 (model predicts ~0 for everything) | Confirms model isn't learning — value head is useless | **Symptom** |

Evidence:
- Loss history: value_loss ~0.48-0.53 from step 33K to 52K (no decrease)
- Manifest: `win_rate_vs_random` is `null` for all checkpoints after step 31K
- Manifest: Most entries from step 32K+ stuck at `"label": "Evaluating..."`
- Config: `resume_from` loads only `model_state_dict`, not optimizer

---

### Task 1: Fix PylosNetwork to use `forward()` instead of `__call__`

**Files:**
- Modify: `engine/models.py:22-28`
- Test: Manual — run existing training for a few steps

**Why:** Overriding `__call__` on an `nn.Module` bypasses PyTorch's module call machinery (hooks, grad mode context, etc.). Also, `self.train()` is called on every forward pass during training, which is wasteful and could interfere with `eval()` mode if called from the wrong context.

**Step 1: Rename `__call__` to `forward` and remove `self.train()` call**

In `engine/models.py`, change:

```python
# BEFORE (line 22-28):
def __call__(self, observations):
    self.train()
    x = F.relu(self.fc1(observations))
    x = F.relu(self.fc2(x))
    value = torch.tanh(self.value_head(x))
    log_policy = F.log_softmax(self.policy_head(x), dim=-1)
    return value, log_policy
```

To:

```python
# AFTER:
def forward(self, observations):
    x = F.relu(self.fc1(observations))
    x = F.relu(self.fc2(x))
    value = torch.tanh(self.value_head(x))
    log_policy = F.log_softmax(self.policy_head(x), dim=-1)
    return value, log_policy
```

**Step 2: Ensure callers set train/eval mode explicitly**

In `engine/train.py`, the `_train_batch` function already calls `model(observations)` which will now go through `nn.Module.__call__` → `forward()`. Add explicit `model.train()` before the training loop.

In `engine/train.py`, add `model.train()` before the training epoch loop inside `_train_batch()`:

```python
def _train_batch():
    nonlocal latest_vloss, latest_ploss
    if len(agent.replay_buffer) < batch_size:
        return
    model.train()  # ADD THIS LINE
    value_losses = []
    ...
```

**Step 3: Run a quick smoke test**

Run: `cd /Users/atarkian2/Documents/GitHub/Pylos-Engine && python -c "from engine.models import PylosNetwork; import torch; m = PylosNetwork((32,), 303); x = torch.randn(2, 32, device=m.device); m.train(); v, p = m(x); print('forward ok, value shape:', v.shape, 'policy shape:', p.shape)"`

Expected: `forward ok, value shape: torch.Size([2, 1]) policy shape: torch.Size([2, 303])`

**Step 4: Commit**

```
git add engine/models.py engine/train.py
git commit -m "fix: use forward() instead of __call__ in PylosNetwork"
```

---

### Task 2: Fix evaluation to include vs-random and complete reliably

**Files:**
- Modify: `engine/evaluate.py:236-293` (the `_run_eval_in_background` function in `engine/train.py`)
- Modify: `engine/train.py:236-293`

**Why:** The `_run_eval_in_background` function only evaluates model-vs-previous-model. It never computes `win_rate_vs_random`, so the manifest field stays `null`. Additionally, daemon threads are killed when the process exits, so evaluations often don't complete.

**Step 1: Add vs-random evaluation to `_run_eval_in_background`**

In `engine/train.py`, modify `_run_eval_in_background` to also run `evaluate_checkpoint` (vs random). Add this after the vs-prev evaluation block, before the manifest update:

```python
def _run_eval_in_background(filepath, filename, step, eval_games, manifest_path, ckpt_dir):
    """Background thread: evaluate checkpoint vs random and vs previous model."""
    try:
        from evaluate import evaluate_checkpoint

        win_rate_vs_random = None
        win_rate_vs_prev = None
        prev_path = None
        prev_elo = None

        # ── Evaluate vs random (always) ──────────────────────────────
        win_rate_vs_random = evaluate_checkpoint(
            filepath, num_games=eval_games, search_iterations=32,
        )
        label = assign_label(win_rate_vs_random)

        # ── Evaluate vs previous model (if available) ────────────────
        with _manifest_lock:
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
            else:
                manifest = {"checkpoints": []}

            evaluated = [
                e for e in manifest["checkpoints"]
                if e["step"] < step and e.get("elo") is not None
            ]
            if evaluated:
                prev_entry = max(evaluated, key=lambda e: e["step"])
                prev_path = os.path.join(ckpt_dir, prev_entry["file"])
                prev_elo = prev_entry["elo"]

        if prev_path and os.path.isfile(prev_path):
            win_rate_vs_prev = evaluate_vs_model(
                filepath, prev_path, num_games=eval_games, search_iterations=32,
            )
            elo = round(win_rate_to_elo(win_rate_vs_prev, prev_elo))
            print(f"\n  [eval] Step {step}: vs random {win_rate_vs_random:.2%}, "
                  f"vs prev {win_rate_vs_prev:.2%}, ELO {elo} ({label})")
        else:
            elo = RANDOM_ELO
            print(f"\n  [eval] Step {step}: vs random {win_rate_vs_random:.2%}, "
                  f"baseline ELO {elo} ({label})")

        # ── Update manifest ──────────────────────────────────────────
        with _manifest_lock:
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
            else:
                manifest = {"checkpoints": []}

            for entry in manifest["checkpoints"]:
                if entry["step"] == step:
                    entry["label"] = label
                    entry["elo"] = elo
                    entry["win_rate_vs_random"] = round(win_rate_vs_random, 4)
                    if win_rate_vs_prev is not None:
                        entry["win_rate_vs_prev"] = round(win_rate_vs_prev, 4)
                    break

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
    except Exception as e:
        print(f"\n  [eval] Step {step} evaluation failed: {e}")
```

**Step 2: Use fewer eval games for faster completion**

The current config uses `eval_games: 300` which takes a long time. Reduce to a faster default for vs-random (which is easier to distinguish), keep higher for vs-prev. Or simply reduce overall:

In `engine/config_v3.yaml`, change:
```yaml
eval_games: 50   # was 300; faster evals complete before next checkpoint
```

**Step 3: Fix label assignment to use vs-random win rate**

The `assign_label` function is designed for vs-random win rates. The current code (before our fix) was passing `win_rate_vs_prev` to `assign_label`, which is meaningless. Our fix above already passes `win_rate_vs_random` to `assign_label`. Verify this is correct in the updated code.

**Step 4: Commit**

```
git add engine/train.py engine/config_v3.yaml
git commit -m "fix: evaluate vs random (not just vs-prev) and use vs-random for labels"
```

---

### Task 3: Fix optimizer state restoration on resume

**Files:**
- Modify: `engine/train.py:436-443`

**Why:** When resuming from a checkpoint, only `model_state_dict` is loaded. The AdamW optimizer starts fresh, losing all momentum and variance tracking. This causes the effective learning rate to spike (Adam bias correction) and training to destabilize, potentially undoing all learned improvements.

**Step 1: Load optimizer state on resume**

In `engine/train.py`, modify the resume block (around line 436):

```python
# BEFORE:
resume_from = ckpt_cfg.get("resume_from")
if resume_from and os.path.isfile(resume_from):
    print(f"Resuming from checkpoint: {resume_from}")
    ckpt_data = torch.load(resume_from, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt_data["model_state_dict"])
    resume_step = ckpt_data.get("step", 0)
    print(f"  Model weights loaded successfully (step {resume_step}).")
```

```python
# AFTER:
resume_from = ckpt_cfg.get("resume_from")
if resume_from and os.path.isfile(resume_from):
    print(f"Resuming from checkpoint: {resume_from}")
    ckpt_data = torch.load(resume_from, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt_data["model_state_dict"])
    if "optimizer_state_dict" in ckpt_data:
        optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        # Move optimizer state to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(model.device)
        print(f"  Optimizer state restored.")
    else:
        print(f"  WARNING: No optimizer state in checkpoint, starting fresh.")
    resume_step = ckpt_data.get("step", 0)
    print(f"  Resumed from step {resume_step}.")
```

**Step 2: Commit**

```
git add engine/train.py
git commit -m "fix: restore optimizer state on resume to preserve Adam momentum"
```

---

### Task 4: Add learning rate scheduler for long training runs

**Files:**
- Modify: `engine/train.py` (training loop)
- Modify: `engine/config_v3.yaml`

**Why:** A constant learning rate of 0.001 over 1M games may be too high for later stages of training, causing oscillation around the optimum. AlphaZero uses learning rate annealing.

**Step 1: Add cosine annealing LR scheduler**

In `engine/train.py`, after the optimizer creation (around line 430), add:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_games,
    eta_min=train_cfg.get("min_learning_rate", 1e-5),
)
```

**Step 2: Step the scheduler after each training batch**

In `_train_batch()`, after the training loop, add:

```python
def _train_batch():
    nonlocal latest_vloss, latest_ploss
    if len(agent.replay_buffer) < batch_size:
        return
    model.train()
    ...
    # After the epoch loop:
    scheduler.step()
```

**Step 3: Add `min_learning_rate` to config**

In `engine/config_v3.yaml`, under `training:`:
```yaml
min_learning_rate: 0.00001
```

**Step 4: Commit**

```
git add engine/train.py engine/config_v3.yaml
git commit -m "feat: add cosine annealing LR scheduler for training stability"
```

---

### Task 5: Add temperature schedule for self-play exploration

**Files:**
- Modify: `engine/train.py` (self-play functions)
- Modify: `engine/config_v3.yaml`

**Why:** Standard AlphaZero uses a temperature parameter for action selection during self-play: high temperature early in the game (explore diverse openings) and temperature → 0 after N moves (play strongest). The current code always samples proportionally to visit counts (temperature=1), which may produce lower-quality training data.

**Step 1: Add temperature-based action selection to `_run_selfplay`**

In `engine/train.py`, modify `_run_selfplay` to accept and use a temperature threshold:

```python
def _run_selfplay(args):
    search_iters, c_puct, dir_alpha, max_moves, rep_limit, ml_penalty, rep_penalty, temp_threshold = args
    from mcts import search
    from collections import defaultdict

    game = PylosGame()
    buffer = []
    move_count = 0
    state_counts = defaultdict(int) if rep_limit else None
    draw_reason = None

    while (fpr := game.get_first_person_result()) is None:
        root_node = search(
            game, iterations=search_iters, c_puct=c_puct,
            dirichlet_alpha=dir_alpha, eval_fn=_worker_agent.eval_fn,
        )
        if root_node.children is None:
            break

        visits = root_node.children_visits
        if move_count < temp_threshold:
            # Temperature = 1: sample proportionally
            visits_dist = visits / visits.sum()
            action = root_node.children_actions[
                np.random.choice(len(root_node.children), p=visits_dist)
            ]
        else:
            # Temperature → 0: pick best move
            visits_dist = visits / visits.sum()
            action = root_node.children_actions[np.argmax(visits)]

        actions_dist = np.zeros(game.action_space, dtype=np.float32)
        actions_dist[root_node.children_actions] = visits_dist
        buffer.append((game.to_observation(), actions_dist))
        game.step(action)
        move_count += 1

        # ... rest unchanged (repetition check, move limit check) ...
```

Update the args tuple where `_run_selfplay` is called:

```python
temp_threshold = train_cfg.get("temp_threshold", 15)
args_iter = ((search_iters, c_puct, dir_alpha, max_moves, rep_limit,
              ml_penalty, rep_penalty, temp_threshold) for _ in range(remaining))
```

**Step 2: Do the same for `_batched_selfplay`**

Apply the same temperature logic in `_batched_selfplay` for the action selection step.

**Step 3: Add config parameter**

In `engine/config_v3.yaml`:
```yaml
temp_threshold: 15   # moves before switching to greedy play
```

**Step 4: Commit**

```
git add engine/train.py engine/config_v3.yaml
git commit -m "feat: add temperature schedule for self-play action selection"
```

---

### Task 6: Backfill evaluations for existing checkpoints

**Files:**
- Modify: `engine/backfill_elo.py` (already exists as untracked)

**Why:** Many checkpoints from step 32K+ have `win_rate_vs_random: null` and `label: "Evaluating..."`. We need to backfill these evaluations to understand the true training trajectory.

**Step 1: Check existing backfill script and update it**

Read `engine/backfill_elo.py` and ensure it:
1. Iterates over all checkpoints in the manifest
2. Runs `evaluate_checkpoint` (vs random) for any entry where `win_rate_vs_random` is null
3. Updates the manifest with results

**Step 2: Run the backfill**

```bash
cd /Users/atarkian2/Documents/GitHub/Pylos-Engine
python engine/backfill_elo.py --config engine/config_v3.yaml
```

**Step 3: Commit updated manifest**

```
git add engine/backfill_elo.py
git commit -m "fix: backfill missing vs-random evaluations for existing checkpoints"
```

---

### Task 7: Retrain from a clean start with all fixes applied

**Files:**
- Modify: `engine/config_v3.yaml` (update resume_from or start fresh)

**Why:** After fixing the model architecture, optimizer restoration, temperature schedule, and LR scheduler, the existing checkpoints were trained with bugs. A fresh training run (or resume from the best evaluated checkpoint) will benefit from all fixes.

**Step 1: Decide starting point**

Option A: Start fresh (no resume) — cleanest but loses all progress.
Option B: Resume from best checkpoint but with fixed optimizer — some progress preserved.

Recommendation: **Option B** — resume from the highest-ELO checkpoint with `win_rate_vs_random >= 0.75` (likely around step 28000, which showed Expert/1.0 vs random).

Update `engine/config_v3.yaml`:
```yaml
resume_from: engine/checkpoints_v3/checkpoint_28000.pth
```

**Step 2: Run training with all fixes**

```bash
caffeinate -dims .venv/bin/python -m engine.train --config engine/config_v3.yaml
```

**Step 3: Monitor progress**

After 5K-10K games, check:
- Is value loss decreasing? (Should drop below 0.3)
- Is win rate vs random increasing? (Should exceed 75%)
- Are evaluations completing? (Labels should not be stuck at "Evaluating...")

---

## Summary of Expected Impact

| Fix | Expected Effect |
|-----|----------------|
| `forward()` instead of `__call__` | Proper PyTorch module behavior |
| Eval vs random | Accurate absolute strength measurement |
| Optimizer restoration | Stable training after resume, no regression |
| LR scheduler | Prevents late-training oscillation |
| Temperature schedule | Higher-quality training data, better exploration |
| Backfill evaluations | Understand true training trajectory |
