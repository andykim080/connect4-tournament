#!/usr/bin/env python3
"""Tournament game runner — designed for live Zoom-call operation.

Locked tournament config (per Saturday 2026-04-25 work):
  MCTS + endgame minimax,  prior = andy_pg_final.keras
  n_simulations = 200,  c_puct = 2.0,  β = 1.0,  endgame_depth = 8

Usage
-----
  python tools/play_tournament.py [--no-color]

The script will:
  1. Run a pre-game sanity check (load model, MCTS on empty board).
  2. Ask whether you're playing first (X) or second (O).
  3. Display the board after every move.
  4. On your turn: run MCTS, show wall-clock time, show top-3 visit
     counts, and play the move automatically.
  5. On opponent's turn: prompt for the column they played.
  6. Detect win/draw, save the game log, ask whether to play another.

Special inputs at any opponent prompt:
  u   — undo the last two plies (their move + your response, or just
        the last opponent move if it's the most recent action)
  q   — quit gracefully (saves game-in-progress)

Files written:
  tournament_games/_in_progress.pkl   updated every move; recovers a
                                       game if the script crashes
  tournament_games/game_<timestamp>.txt   one per finished game
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from engine import update_board, check_for_win, find_legal, look_for_win
from model_io import ModelPlayer

# ---------------------------------------------------------------------------
# Locked tournament config
# ---------------------------------------------------------------------------

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PG_PATH    = os.path.join(ROOT, 'checkpoints', 'andy_pg_final.keras')
GAMES_DIR  = os.path.join(ROOT, 'tournament_games')
IN_PROG    = os.path.join(GAMES_DIR, '_in_progress.pkl')

N_SIMS         = 200
C_PUCT         = 2.0
PRIOR_TEMP     = 1.0
ENDGAME_DEPTH  = 8

# ANSI colors (disabled with --no-color)
COLOR_X = '\033[1;31m'   # bold red
COLOR_O = '\033[1;34m'   # bold blue
COLOR_RESET = '\033[0m'
USE_COLOR = True


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    board:        np.ndarray              # current 6×7 board
    move_number:  int        = 0          # next move number to play (1-indexed)
    turn_color:   str        = 'plus'     # whose turn it is now
    my_color:     str        = 'plus'     # which side WE are playing
    history:     list        = field(default_factory=list)
    # history entries: dicts with keys
    #   move_number, color, col, board_after (np.ndarray copy)
    game_log:    list        = field(default_factory=list)
    started_at:  str         = ''         # ISO timestamp


def new_state(my_color: str) -> GameState:
    return GameState(
        board=np.zeros((6, 7), dtype=np.float32),
        move_number=1,
        turn_color='plus',          # plus always starts (engine convention)
        my_color=my_color,
        history=[],
        game_log=[],
        started_at=datetime.now().isoformat(timespec='seconds'),
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def char_for(value: float) -> str:
    """1 → X, −1 → O, 0 → '.'  with optional ANSI color."""
    if value == 1:
        return f'{COLOR_X}X{COLOR_RESET}' if USE_COLOR else 'X'
    if value == -1:
        return f'{COLOR_O}O{COLOR_RESET}' if USE_COLOR else 'O'
    return '.'


def render_board(board: np.ndarray) -> str:
    rows = [' 0 1 2 3 4 5 6']
    for r in range(6):
        rows.append(' ' + ' '.join(char_for(board[r, c]) for c in range(7)))
    return '\n'.join(rows)


def turn_label(state: GameState) -> str:
    """\"Your turn (X)\" / \"Opponent's turn (O)\"."""
    side_letter = 'X' if state.turn_color == 'plus' else 'O'
    is_me = (state.turn_color == state.my_color)
    who = 'Your turn' if is_me else "Opponent's turn"
    return f'{who} ({side_letter})'


def show_state(state: GameState) -> None:
    header = f'\nMove {state.move_number} — {turn_label(state)}'
    print(header)
    print(render_board(state.board))


# ---------------------------------------------------------------------------
# Move application + checkpointing
# ---------------------------------------------------------------------------

def apply_move(state: GameState, col: int) -> str:
    """Apply `col` for the current turn_color. Returns the engine result
    string ('nobody', 'h-plus', 'v-minus', etc.). Updates history / log /
    flips turn_color and increments move_number on non-terminal moves."""
    if col not in find_legal(state.board):
        raise ValueError(f'illegal move: column {col} is full or invalid')

    color = state.turn_color
    state.board = update_board(state.board, color, col)
    state.history.append({
        'move_number':  state.move_number,
        'color':        color,
        'col':          int(col),
        'board_after':  state.board.copy(),
    })
    state.game_log.append((state.move_number, color, int(col)))
    result = check_for_win(state.board, col)
    state.move_number += 1
    state.turn_color = 'minus' if color == 'plus' else 'plus'
    save_in_progress(state)
    return result


def undo(state: GameState) -> None:
    """Pop the most recent move. If the most recent move is theirs (and we
    just played), pop two — so we're back to where they should re-input."""
    if not state.history:
        print('  (nothing to undo)')
        return
    last = state.history.pop()
    state.move_number = last['move_number']
    state.turn_color = last['color']
    state.board = (state.history[-1]['board_after'].copy()
                   if state.history else np.zeros((6, 7), dtype=np.float32))
    if state.game_log:
        state.game_log.pop()
    save_in_progress(state)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_in_progress(state: GameState) -> None:
    os.makedirs(GAMES_DIR, exist_ok=True)
    try:
        with open(IN_PROG, 'wb') as f:
            pickle.dump(state, f)
    except OSError as e:
        print(f'  ⚠ could not save in-progress: {e}')


def load_in_progress() -> GameState | None:
    if not os.path.exists(IN_PROG):
        return None
    try:
        with open(IN_PROG, 'rb') as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError) as e:
        print(f'  ⚠ in-progress file corrupt ({e}); starting fresh')
        return None


def save_finished_game(state: GameState, winner: str) -> str:
    os.makedirs(GAMES_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(GAMES_DIR, f'game_{ts}.txt')
    with open(path, 'w') as f:
        f.write(f'started_at: {state.started_at}\n')
        f.write(f'finished_at: {datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'my_color: {state.my_color}\n')
        f.write(f'winner: {winner}\n')
        f.write(f'\nmove sequence:\n')
        for n, color, col in state.game_log:
            f.write(f'  {n:2d}  {color:<5s}  col {col}\n')
        f.write('\nfinal board:\n')
        # Strip color from saved render
        global USE_COLOR
        prev = USE_COLOR
        USE_COLOR = False
        f.write(render_board(state.board) + '\n')
        USE_COLOR = prev
    if os.path.exists(IN_PROG):
        try:
            os.remove(IN_PROG)
        except OSError:
            pass
    return path


# ---------------------------------------------------------------------------
# My move (tactical wrapper + endgame + MCTS, with visit-count exposure)
# ---------------------------------------------------------------------------

def compute_my_move(state: GameState, pg: ModelPlayer) -> tuple[int, str, dict]:
    """Returns (col, decision_path, visits_dict).

    decision_path ∈ {'own_win', 'block', 'endgame', 'mcts'}.
    visits_dict is empty for non-MCTS branches; otherwise maps col → visit count.
    """
    color = state.my_color
    board = state.board

    own = look_for_win(board, color)
    if own >= 0:
        return own, 'own_win', {}
    opp = 'minus' if color == 'plus' else 'plus'
    block = look_for_win(board, opp)
    if block >= 0:
        return block, 'block', {}

    # Endgame override
    from endgame import in_endgame, solve_endgame
    if in_endgame(board):
        col, score = solve_endgame(board, color, max_depth=ENDGAME_DEPTH)
        if col is not None:
            return col, f'endgame ({score:+.2f})', {}

    # Full PUCT-MCTS
    from mcts import _mcts_search
    root = _mcts_search(board, color, pg.model, N_SIMS, C_PUCT,
                        mode=pg.mode, prior_temperature=PRIOR_TEMP)
    visits = {c: child.N for c, child in root.children.items()}
    if not visits:                # defensive — shouldn't happen if legal exists
        return int(np.random.choice(find_legal(board))), 'fallback', {}
    best = max(visits, key=visits.get)
    return best, 'mcts', visits


def format_top3(visits: dict) -> str:
    """\"col3=120, col4=45, col2=20\" — top-3 by visit count."""
    if not visits:
        return ''
    top = sorted(visits.items(), key=lambda kv: -kv[1])[:3]
    return ', '.join(f'col{c}={n}' for c, n in top)


# ---------------------------------------------------------------------------
# Opponent prompt — returns int col, 'u', or 'q'
# ---------------------------------------------------------------------------

def opponent_prompt(state: GameState) -> int | str:
    legal = set(find_legal(state.board))
    while True:
        try:
            raw = input("  Opponent's move (column 0-6, 'u' undo, "
                        "'q' quit): ").strip().lower()
        except EOFError:
            return 'q'
        if raw in ('q', 'quit', 'exit'):
            return 'q'
        if raw in ('u', 'undo'):
            return 'u'
        if not raw:
            continue
        try:
            col = int(raw)
        except ValueError:
            print(f'  ⚠ unrecognized input "{raw}". Type 0-6, u, or q.')
            continue
        if col < 0 or col > 6:
            print(f'  ⚠ column {col} out of range 0-6')
            continue
        if col not in legal:
            print(f'  ⚠ column {col} is full')
            continue
        return col


# ---------------------------------------------------------------------------
# Pre-game sanity check
# ---------------------------------------------------------------------------

def sanity_check(pg: ModelPlayer, n_sims: int = 30) -> None:
    """Verify the model loads and MCTS picks a sensible move on empty board.
    A small n_sims keeps this fast (few seconds)."""
    print('Pre-game sanity check ...')
    from mcts import mcts_move
    board = np.zeros((6, 7))
    t0 = time.time()
    col = mcts_move(board, 'plus', pg.model,
                    n_simulations=n_sims, c_puct=C_PUCT, mode=pg.mode,
                    prior_temperature=PRIOR_TEMP, use_endgame=True,
                    endgame_depth=ENDGAME_DEPTH)
    elapsed = time.time() - t0
    print(f'  empty board → col {col}  ({elapsed:.1f}s, n_sims={n_sims})')
    if col not in (2, 3, 4):
        raise RuntimeError(f'sanity check FAILED: empty board → col {col} '
                            '(expected 2/3/4 — central column preference broken)')
    print('  ✓ sanity check passed')


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def setup_game() -> GameState:
    """Resume an in-progress game if present, else ask whether the user is
    first or second."""
    existing = load_in_progress()
    if existing is not None:
        print(f'\nFound in-progress game from {existing.started_at}.')
        print(render_board(existing.board))
        ans = input("Resume it? (y/n): ").strip().lower()
        if ans.startswith('y'):
            return existing
        # else fall through to a fresh game
        try:
            os.remove(IN_PROG)
        except OSError:
            pass

    while True:
        ans = input("\nAre you playing first (X) or second (O)? "
                    "[1=first, 2=second]: ").strip()
        if ans in ('1', 'first', 'x', 'X'):
            return new_state('plus')
        if ans in ('2', 'second', 'o', 'O'):
            return new_state('minus')
        print('  please enter 1 or 2')


def play_one_game(pg: ModelPlayer) -> str:
    state = setup_game()
    print(f'\nYou are {"FIRST (X)" if state.my_color == "plus" else "SECOND (O)"}')
    show_state(state)

    while True:
        if state.turn_color == state.my_color:
            # ── My turn ───────────────────────────────────────────────────
            print('  Computing ...', flush=True)
            t0 = time.time()
            col, path, visits = compute_my_move(state, pg)
            elapsed = time.time() - t0
            extra = f' (visits: {format_top3(visits)})' if visits else ''
            print(f'  → played col {col}  via {path}  '
                  f'({elapsed:.1f}s){extra}')
            result = apply_move(state, col)
            show_state(state)
            if result != 'nobody':
                return announce_winner(state, result)
            if not find_legal(state.board):
                return announce_winner(state, 'nobody')

        else:
            # ── Opponent's turn ───────────────────────────────────────────
            inp = opponent_prompt(state)
            if inp == 'q':
                print('\n  quitting — game saved as in-progress for later resume')
                return 'quit'
            if inp == 'u':
                undo(state)
                # If we just undid an opponent move, keep undoing until
                # control is back to opponent (since we don't want to
                # immediately re-play our own move).
                while state.history and state.turn_color == state.my_color:
                    undo(state)
                show_state(state)
                continue
            result = apply_move(state, inp)
            show_state(state)
            if result != 'nobody':
                return announce_winner(state, result)
            if not find_legal(state.board):
                return announce_winner(state, 'nobody')


def announce_winner(state: GameState, result: str) -> str:
    if result == 'nobody':
        winner = 'tie'
        print('\n=== DRAW ===')
    else:
        winner_color = 'plus' if result[2:] == 'plus' else 'minus'
        winner = 'me' if winner_color == state.my_color else 'opponent'
        side_letter = 'X' if winner_color == 'plus' else 'O'
        if winner == 'me':
            print(f'\n=== I WON ({side_letter}) ===')
        else:
            print(f'\n=== OPPONENT WON ({side_letter}) ===')
    path = save_finished_game(state, winner)
    print(f'  game log saved to {os.path.relpath(path, ROOT)}')
    return winner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-color', action='store_true',
                    help='disable ANSI color codes')
    args = ap.parse_args()
    if args.no_color:
        global USE_COLOR
        USE_COLOR = False

    print('=' * 56)
    print('Connect 4 tournament runner — locked config')
    print('  MCTS+endgame  n_sims=200  c_puct=2.0  β=1.0  depth=8')
    print('  prior: andy_pg_final.keras')
    print('=' * 56)

    print('\nLoading prior ...')
    pg = ModelPlayer(PG_PATH)
    print(f'  ✓ loaded {os.path.basename(PG_PATH)}')

    sanity_check(pg)

    while True:
        try:
            outcome = play_one_game(pg)
        except KeyboardInterrupt:
            print('\n  interrupted — game saved as in-progress for later resume')
            return
        if outcome == 'quit':
            return
        ans = input('\nPlay another game? (y/n): ').strip().lower()
        if not ans.startswith('y'):
            print('Done. Good luck.')
            return


if __name__ == '__main__':
    main()
