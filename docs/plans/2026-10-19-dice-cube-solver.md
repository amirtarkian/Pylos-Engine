# Dice Sudoku Cube Solver

A solver and 3D visualiser for the "Blind Dice Cube" (V1): a 3x3x3 twisty
cube whose stickers are raised dice faces showing 1-9 pips. It uses the same
approach as the Pylos engine: a residual value/policy network trained without
human data, plus tree search.

## The puzzle

- Every center shows **5**. Centers never move, so the cube has no fixed
  "colour" for any face.
- From the photos of the real cube: **corners carry odd values (1, 3, 7, 9)**
  and **edges carry even values (2, 4, 6, 8)**. Each value appears 6 times
  (once per face in the solved state).
- **Solved = every face shows each of 1-9 exactly once.** Only the *set* of
  values on a face matters, not where they sit on it.

That last rule is the big difference from a normal Rubik's cube. A coloured
cube has exactly one solved state. Here any arrangement that puts {1,3,7,9}
on each face's corners and {2,4,6,8} on its edges counts. A depth-first
enumeration over one cube's pieces found **more than 700,000 reachable solved
states** before hitting its search budget. The solver therefore aims for the
*nearest* goal, not a fixed one.

Mechanically it is still a 3x3x3, so the usual group rules apply to any
target: corner twist sums to 0 mod 3, edge flips sum to 0 mod 2, and corner
and edge permutation parities match.

## Code layout

```
dicecube/
  cube.py      -- state (54 pip values, Kociemba URFDLB facelet order), the 12
                  quarter-turn permutations (derived from 3D sticker geometry),
                  goal test, scrambles, encoding
  models.py    -- DiceCubeNetwork: Pylos ResidualBlock backbone, cost-to-go
                  value head (softplus) + 12-way policy head
  train.py     -- value iteration with a curriculum (DeepCubeA-style)
  mcts.py      -- single-player AlphaZero MCTS (batched, virtual loss)
  astar.py     -- batch-weighted A* using the same network as heuristic
  targets.py   -- goal-state enumeration + Kociemba two-phase bridge
  solver.py    -- runs everything, packages traces for the UI (also a CLI)
  evaluate.py  -- benchmark by scramble depth
  server.py    -- FastAPI app (port 8001)
web/dice.html, web/src/dice/*.js -- Three.js visualiser
```

## How it learns ("same style as Pylos")

Pylos uses AlphaZero: self-play games give win/loss targets for a value head
and MCTS visit counts give targets for a policy head. A one-player puzzle has
no opponent, so the training signal comes from the Bellman equation instead
(the DeepCube / DeepCubeA recipe):

1. Generate a batch of states by scrambling **random solved dice cubes**
   0..k moves. Random designs mean the network doesn't depend on the exact
   piece set of any one physical cube.
2. For every state, look one move ahead with a frozen *target network* J':

   ```
   cost(s)   = 0                                  if s is solved
             = min over 12 moves a of  1 + J'(a(s))   (J' = 0 for solved children)
   policy(s) = the argmin move
   ```

3. Train the online network on those targets (MSE for cost, cross-entropy for
   policy).
4. Every 50 steps, if the loss is low (or 300 steps have passed), copy the
   online network into the target network and **raise k by one**. This is a
   curriculum from 1-move scrambles up to 30.

Because "solved" is the per-face set test, the network learns the distance to
the *nearest* of the many goal states, which is what the puzzle asks for.

The input is the 48 movable stickers one-hot encoded over the 8 possible
values, plus a per-face histogram of values. The histogram shows the goal
condition ("is this value already on this face?") directly.

## How it searches

- **MCTS** (`mcts.py`) is the AlphaZero search adapted to one player. It uses
  PUCT selection with the policy head as priors. Q-values are
  `-(1 + best cost found below)` and back up the *best* value, not the mean,
  since one good line is all a puzzle needs. Each expansion checks all 12
  children against the goal. Leaves are batched with virtual loss, as in
  `engine.mcts.batched_search`. The visit counts at the root are shown in the
  UI.
- **Weighted A\*** (`astar.py`) expands the lowest `0.6*g + h(state)` first,
  scoring 64 nodes per network call.
- **Kociemba two-phase** (`targets.py`) is the classical safety net. It reads
  the physical pieces off the state and enumerates concrete goal assignments
  (piece → slot + orientation, with no repeated value on any face, and parity
  checked). It recolours every sticker by the face it must travel to, which
  turns the problem into an ordinary colour cube, and solves that. It tries
  up to 60 goal states and keeps the shortest result. The first candidates
  keep pieces in place, later ones explore randomly.

The UI runs all three and highlights the shortest result. Neural search
finds optimal or near-optimal solutions for shallow scrambles. Kociemba
solves anything in about 17-20 moves.

## Solving your physical cube

1. `pip install -r requirements.txt`, then `bash run_dice.sh` and open
   http://localhost:8001.
2. Hold the cube with any face toward you (**F**) and any face up (**U**).
   Keep this grip for the whole solve.
3. Enter the stickers in the net editor: click a cell and type its pip count.
   The "How to hold the cube" box explains how to read U, R, B, L and D
   consistently. The editor flags any value count that isn't exactly 6.
4. Press **Solve**, then step through the moves. The 3D cube animates each
   turn. Orange pips mean a value is repeated on its face, green means the
   face is complete.

Move notation is standard: U R F D L B turn the top/right/front/bottom/left/
back face clockwise *as you look at that face*. A `'` means counter-clockwise
and `2` means a half turn.

## Results

Filled in from `python -m dicecube.evaluate` after training (see below).
