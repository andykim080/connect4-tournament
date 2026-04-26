# connect4-tournament

Clean, segregated folder for tournament-day play of the final Connect 4 agent.
Development repo lives separately at `/Users/andykim/Downloads/RL/connect4-rl/`.
This folder contains only what is needed to run a live game and verify the
agent behaves as expected.

## Final agent

MCTS + endgame minimax, with `andy_pg_final.keras` as the policy prior.

## Locked tournament config

| param          | value |
| -------------- | ----- |
| `n_simulations` | 200   |
| `c_puct`       | 2.0   |
| `β` (prior weight) | 1.0 |
| `use_endgame`  | True  |
| `endgame_depth` | 8    |

These values are baked into `tools/play_tournament.py` and must not be changed
on tournament day.

## Layout

```
checkpoints/        andy_pg_final.keras (the policy prior)
src/                engine.py, model_io.py, mcts.py, endgame.py
tools/              play_tournament.py  (live game runner)
tests/              test_play_tournament.py, test_mcts.py
tournament_games/   game logs + _in_progress.pkl recovery file (gitignored)
```

## Running

Run a tournament game:

```
python tools/play_tournament.py
```

The script first runs a sanity check (load model, MCTS on empty board with
`n_sims=30`) and prints `empty board → col 3 (~1-2s, n_sims=30)`. Then it
asks whether you're playing first (X) or second (O), runs MCTS on your turn,
and prompts for the opponent's column on theirs. `u` undoes the last two
plies, `q` quits and saves an in-progress pickle.

## Testing

```
pytest tests/ -v
```

Expected: 25 passing tests (17 in `test_play_tournament.py`, 8 in
`test_mcts.py`).
