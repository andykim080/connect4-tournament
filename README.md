# connect4-tournament

A clean, frozen copy of the final Connect 4 agent — just the pieces needed
to play live games on tournament day. Active development happens in a
separate repo (`/Users/andykim/Downloads/RL/connect4-rl/`); nothing here
should change during the tournament.

## What the agent is doing

On each of its turns, the agent:

1. Asks a trained policy network (`andy_pg_final.keras`) for prior
   probabilities over the 7 columns — its "gut feel" for which moves look
   promising.
2. Runs **MCTS** (Monte Carlo Tree Search) for 200 simulations, using those
   priors to focus its search and a value head to estimate how good the
   resulting positions are.
3. If the position is close enough to the end of the game, switches to an
   **exact endgame minimax search** (depth 8) instead of trusting MCTS —
   this guarantees the agent never misses a forced win or loss within the
   horizon.

The result is a move that combines learned intuition (the network) with
deliberate lookahead (search) and exact play near the end of the game.

## Locked tournament config

These are baked into `tools/play_tournament.py` and **must not be changed**
on tournament day:

| param              | value | meaning |
| ------------------ | ----- | ------- |
| `n_simulations`    | 200   | MCTS rollouts per move |
| `c_puct`           | 2.0   | exploration constant (higher = explore more) |
| `β` (prior weight) | 1.0   | how much MCTS trusts the policy network's priors |
| `use_endgame`      | True  | switch to exact search near end of game |
| `endgame_depth`    | 8     | plies of exact lookahead in endgame mode |

## Layout

```
checkpoints/        andy_pg_final.keras      — the trained policy/value network
src/
  engine.py           board representation, legal-move generation, win detection
  mcts.py             Monte Carlo Tree Search
  endgame.py          exact minimax for late-game positions
  model_io.py         loads the .keras file and runs inference
tools/
  play_tournament.py  terminal game runner (the official tournament entrypoint)
  play_ui.py          Flask web UI wrapping the same agent — nicer to demo with
  templates/          HTML for the web UI
  static/             CSS/JS for the web UI
tests/
  test_play_tournament.py, test_mcts.py
tournament_games/   game logs + auto-saved _in_progress.pkl (gitignored)
```

## Running a game

You have two options — both use the exact same agent and locked config.
The web UI just wraps the terminal runner with a nicer interface.

### Option 1 — Terminal (official tournament entrypoint)

```
python tools/play_tournament.py
```

What happens:

1. **Sanity check.** The script loads the model and runs MCTS on an empty
   board with `n_sims=30` (a quick warmup). It should print
   `empty board → col 3 (~1-2s, n_sims=30)`. If you don't see that, stop
   and investigate before the round starts.
2. **Pick your side.** It asks whether *you* are playing first (X) or
   second (O). The agent takes the other color.
3. **Play.** On the agent's turn it runs MCTS and prints the column it
   chose. On your (the opponent's) turn, type the column number 1–7 to
   record what your human opponent played.

Controls during the game:

- `1`–`7` — record the opponent's move in that column
- `u` — undo the last two plies (one agent move + one opponent move)
- `q` — quit and save an `_in_progress.pkl` so the game can be resumed

### Option 2 — Web UI (same agent, prettier face)

```
python tools/play_ui.py
```

This starts a local Flask server and auto-opens
`http://localhost:5050` in your browser. Click a column to play; the
agent plays back automatically. Useful for demoing or for anyone who
prefers clicking to typing.

## Testing

```
pytest tests/ -v
```

Expected: 25 passing tests (17 in `test_play_tournament.py`, 8 in
`test_mcts.py`). Run this before tournament day to confirm the agent
still behaves the same after any environment changes.
