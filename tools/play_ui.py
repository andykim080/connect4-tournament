#!/usr/bin/env python3
"""Flask web UI for the Connect 4 tournament agent.

Wraps tools/play_tournament.py — same agent, same locked config (MCTS+endgame,
n_sims=200, c_puct=2.0, β=1.0, depth=8, prior=andy_pg_final.keras), prettier face.

Run:    python tools/play_ui.py
Open:   http://localhost:5050
"""

from __future__ import annotations

import os
import sys
import time
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, request, jsonify

import play_tournament as pt
from engine import find_legal
from model_io import ModelPlayer

# ---------------------------------------------------------------------------
# Globals — single-user, single-game session at a time
# ---------------------------------------------------------------------------

app = Flask(__name__)

PG_MODEL: ModelPlayer | None = None
STATE: pt.GameState | None = None
LAST_AGENT_MOVE: dict | None = None
GAME_OVER: bool = False
WINNER: str | None = None        # 'me' | 'opponent' | 'tie'
WIN_CELLS: list = []             # list of [r, c] for highlight (computed on win)
LOCK = threading.Lock()          # serialize TF inference + state mutation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _winning_cells(board, last_col: int) -> list:
    """Recompute the 4-in-a-row cells for highlighting (engine returns only a
    type string). Returns [] if no win at last_col."""
    rows, cols = 6, 7
    # Find row of the most-recently-placed piece in last_col
    row = None
    for r in range(rows):
        if board[r, last_col] != 0:
            row = r
            break
    if row is None:
        return []
    player = board[row, last_col]

    def line(dr, dc):
        cells = [(row, last_col)]
        rr, cc = row + dr, last_col + dc
        while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player:
            cells.append((rr, cc))
            rr += dr; cc += dc
        rr, cc = row - dr, last_col - dc
        while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player:
            cells.insert(0, (rr, cc))
            rr -= dr; cc -= dc
        return cells

    for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        ln = line(dr, dc)
        if len(ln) >= 4:
            return [[r, c] for r, c in ln[:4]]
    return []


def state_to_json() -> dict:
    if STATE is None:
        return {"initialized": False}
    return {
        "initialized": True,
        "board": STATE.board.astype(int).tolist(),
        "my_color": STATE.my_color,
        "turn_color": STATE.turn_color,
        "move_number": STATE.move_number,
        "history": [
            {"n": h["move_number"], "color": h["color"], "col": h["col"]}
            for h in STATE.history
        ],
        "legal_cols": list(find_legal(STATE.board)),
        "last_agent_move": LAST_AGENT_MOVE,
        "game_over": GAME_OVER,
        "winner": WINNER,
        "win_cells": WIN_CELLS,
        "config": {
            "n_sims": pt.N_SIMS,
            "c_puct": pt.C_PUCT,
            "beta": pt.PRIOR_TEMP,
            "endgame_depth": pt.ENDGAME_DEPTH,
        },
    }


def _check_terminal(result: str, last_col: int) -> None:
    """If engine says game ended, set GAME_OVER/WINNER/WIN_CELLS, save log."""
    global GAME_OVER, WINNER, WIN_CELLS
    if result != 'nobody':
        winner_color = 'plus' if result[2:] == 'plus' else 'minus'
        WINNER = 'me' if winner_color == STATE.my_color else 'opponent'
        WIN_CELLS = _winning_cells(STATE.board, last_col)
        GAME_OVER = True
        try:
            pt.save_finished_game(STATE, WINNER)
        except Exception as e:
            print(f"  ⚠ could not save finished game: {e}")
    elif not find_legal(STATE.board):
        WINNER = 'tie'
        WIN_CELLS = []
        GAME_OVER = True
        try:
            pt.save_finished_game(STATE, 'tie')
        except Exception as e:
            print(f"  ⚠ could not save finished game: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/state')
def get_state():
    return jsonify(state_to_json())


@app.route('/new_game', methods=['POST'])
def new_game():
    global STATE, LAST_AGENT_MOVE, GAME_OVER, WINNER, WIN_CELLS
    data = request.get_json(silent=True) or {}
    my_color = data.get('my_color', 'plus')
    if my_color not in ('plus', 'minus'):
        return jsonify({"error": "my_color must be 'plus' or 'minus'"}), 400
    with LOCK:
        STATE = pt.new_state(my_color=my_color)
        LAST_AGENT_MOVE = None
        GAME_OVER = False
        WINNER = None
        WIN_CELLS = []
    return jsonify(state_to_json())


@app.route('/agent_move', methods=['POST'])
def agent_move():
    global LAST_AGENT_MOVE
    if STATE is None:
        return jsonify({"error": "no game in progress"}), 400
    if GAME_OVER:
        return jsonify({"error": "game is over"}), 400
    if STATE.turn_color != STATE.my_color:
        return jsonify({"error": "not agent's turn"}), 400

    with LOCK:
        t0 = time.time()
        col, path, visits = pt.compute_my_move(STATE, PG_MODEL)
        elapsed = time.time() - t0
        result = pt.apply_move(STATE, col)
        top3 = sorted(visits.items(), key=lambda kv: -kv[1])[:3] if visits else []
        LAST_AGENT_MOVE = {
            "col": int(col),
            "path": path,
            "visits": [{"col": int(c), "n": int(n)} for c, n in top3],
            "elapsed": round(elapsed, 1),
        }
        _check_terminal(result, col)
    return jsonify(state_to_json())


@app.route('/opponent_move', methods=['POST'])
def opponent_move():
    if STATE is None:
        return jsonify({"error": "no game in progress"}), 400
    if GAME_OVER:
        return jsonify({"error": "game is over"}), 400
    if STATE.turn_color == STATE.my_color:
        return jsonify({"error": "not opponent's turn"}), 400

    data = request.get_json(silent=True) or {}
    col = data.get('col')
    if not isinstance(col, int) or col < 0 or col > 6:
        return jsonify({"error": "col must be int 0-6"}), 400
    if col not in find_legal(STATE.board):
        return jsonify({"error": f"column {col} is full"}), 400

    with LOCK:
        result = pt.apply_move(STATE, col)
        _check_terminal(result, col)
    return jsonify(state_to_json())


@app.route('/undo', methods=['POST'])
def undo():
    """Pop until it's opponent's turn — matches play_tournament.py logic."""
    global LAST_AGENT_MOVE, GAME_OVER, WINNER, WIN_CELLS
    if STATE is None or not STATE.history:
        return jsonify({"error": "nothing to undo"}), 400

    with LOCK:
        pt.undo(STATE)
        # Keep undoing until control is back to the opponent so we don't
        # immediately re-play our own move.
        while STATE.history and STATE.turn_color == STATE.my_color:
            pt.undo(STATE)
        LAST_AGENT_MOVE = None
        GAME_OVER = False
        WINNER = None
        WIN_CELLS = []
    return jsonify(state_to_json())


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

def boot() -> None:
    global PG_MODEL
    print('=' * 60)
    print('Connect 4 tournament UI — locked config')
    print(f'  MCTS+endgame  n_sims={pt.N_SIMS}  c_puct={pt.C_PUCT}  '
          f'β={pt.PRIOR_TEMP}  depth={pt.ENDGAME_DEPTH}')
    print(f'  prior: {os.path.basename(pt.PG_PATH)}')
    print('=' * 60)
    print('\nLoading prior ...')
    PG_MODEL = ModelPlayer(pt.PG_PATH)
    print(f'  ✓ loaded {os.path.basename(pt.PG_PATH)}')
    pt.sanity_check(PG_MODEL)
    print('\nServer ready. Open http://localhost:5050 in your browser.')
    print('=' * 60)


if __name__ == '__main__':
    boot()
    # Open browser tab once the server is actually accepting connections.
    # Use WERKZEUG_RUN_MAIN guard so the reloader (if ever enabled) doesn't double-open.
    if not os.environ.get('WERKZEUG_RUN_MAIN'):
        threading.Timer(1.0, lambda: webbrowser.open('http://127.0.0.1:5050')).start()
    # threaded=True is fine — LOCK serializes TF inference + state mutation.
    # Port 5050 to dodge macOS AirPlay Receiver on 5000.
    app.run(host='127.0.0.1', port=5050, debug=False, threaded=True)
