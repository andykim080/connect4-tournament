"""Tests for tools/play_tournament.py — the live tournament runner.

Covers all five user-requested cases, plus a couple of guards I want
in place before relying on this Monday:

  1. Empty-board init: state shape + render content
  2. Move application + display: a single move advances state
  3. Win detection: a 3-in-a-row + winning move flips result correctly
  4. Undo: state returns to the prior position
  5. Recovery from in-progress file: pickle round-trip preserves state

Run:  pytest tests/test_play_tournament.py -v
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import numpy as np
import pytest

import play_tournament as pt
from engine import update_board


# ---------------------------------------------------------------------------
# 1. Empty board initialization
# ---------------------------------------------------------------------------

def test_empty_board_init():
    s = pt.new_state(my_color='plus')
    assert s.board.shape == (6, 7)
    assert (s.board == 0).all()
    assert s.move_number == 1
    assert s.turn_color == 'plus'      # plus always starts (engine convention)
    assert s.my_color == 'plus'
    assert s.history == []
    assert s.game_log == []
    rendered = pt.render_board(s.board)
    assert '0 1 2 3 4 5 6' in rendered
    assert rendered.count('.') == 42   # 42 empty cells


def test_empty_board_init_second_player():
    s = pt.new_state(my_color='minus')
    assert s.my_color == 'minus'
    # turn_color is still 'plus' — opponent moves first when we play second
    assert s.turn_color == 'plus'


# ---------------------------------------------------------------------------
# 2. Move application + display
# ---------------------------------------------------------------------------

def test_move_application_basic(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, 'GAMES_DIR', str(tmp_path))
    monkeypatch.setattr(pt, 'IN_PROG', str(tmp_path / '_in_progress.pkl'))

    s = pt.new_state(my_color='plus')
    result = pt.apply_move(s, 3)
    assert result == 'nobody'
    assert s.board[5, 3] == 1                     # bottom row, col 3, plus
    assert s.move_number == 2
    assert s.turn_color == 'minus'
    assert len(s.history) == 1
    assert s.history[0]['col'] == 3
    assert s.history[0]['color'] == 'plus'
    assert s.game_log == [(1, 'plus', 3)]


def test_move_application_alternates(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, 'GAMES_DIR', str(tmp_path))
    monkeypatch.setattr(pt, 'IN_PROG', str(tmp_path / '_in_progress.pkl'))

    s = pt.new_state(my_color='plus')
    pt.apply_move(s, 3)
    pt.apply_move(s, 3)
    assert s.board[5, 3] == 1                # plus on bottom
    assert s.board[4, 3] == -1               # minus stacked above
    assert s.turn_color == 'plus'
    assert s.move_number == 3


def test_render_contains_player_chars(monkeypatch):
    # Disable color so the assertion is on plain X/O
    monkeypatch.setattr(pt, 'USE_COLOR', False)
    b = np.zeros((6, 7))
    b = update_board(b, 'plus',  3)
    b = update_board(b, 'minus', 4)
    rendered = pt.render_board(b)
    assert 'X' in rendered
    assert 'O' in rendered


def test_apply_move_rejects_illegal(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, 'GAMES_DIR', str(tmp_path))
    monkeypatch.setattr(pt, 'IN_PROG', str(tmp_path / '_in_progress.pkl'))
    s = pt.new_state(my_color='plus')
    # Fill column 0 (6 plies), then try one more
    for _ in range(3):
        pt.apply_move(s, 0)
        pt.apply_move(s, 0)
    with pytest.raises(ValueError):
        pt.apply_move(s, 0)


# ---------------------------------------------------------------------------
# 3. Win detection
# ---------------------------------------------------------------------------

def test_win_detection_horizontal(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, 'GAMES_DIR', str(tmp_path))
    monkeypatch.setattr(pt, 'IN_PROG', str(tmp_path / '_in_progress.pkl'))

    s = pt.new_state(my_color='plus')
    # Build: plus plays cols 0,1,2 / minus plays col 6 thrice / plus wins col 3
    sequence = [(0,'plus'),(6,'minus'),(1,'plus'),(6,'minus'),
                (2,'plus'),(6,'minus')]
    for col, _ in sequence:
        pt.apply_move(s, col)
    assert s.turn_color == 'plus'
    result = pt.apply_move(s, 3)
    assert result.startswith('h-')
    assert result.endswith('plus')


def test_win_detection_vertical(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, 'GAMES_DIR', str(tmp_path))
    monkeypatch.setattr(pt, 'IN_PROG', str(tmp_path / '_in_progress.pkl'))

    s = pt.new_state(my_color='plus')
    # plus plays col 0 four times, minus plays elsewhere
    pt.apply_move(s, 0)         # plus row 5
    pt.apply_move(s, 1)         # minus
    pt.apply_move(s, 0)         # plus row 4
    pt.apply_move(s, 1)         # minus
    pt.apply_move(s, 0)         # plus row 3
    pt.apply_move(s, 1)         # minus
    result = pt.apply_move(s, 0)  # plus row 2 — VERTICAL WIN
    assert result.startswith('v-')
    assert result.endswith('plus')


# ---------------------------------------------------------------------------
# 4. Undo
# ---------------------------------------------------------------------------

def test_undo_single_move(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, 'GAMES_DIR', str(tmp_path))
    monkeypatch.setattr(pt, 'IN_PROG', str(tmp_path / '_in_progress.pkl'))

    s = pt.new_state(my_color='plus')
    pt.apply_move(s, 3)
    assert s.move_number == 2
    assert s.turn_color == 'minus'
    pt.undo(s)
    assert s.move_number == 1
    assert s.turn_color == 'plus'
    assert (s.board == 0).all()
    assert s.history == []
    assert s.game_log == []


def test_undo_after_two_moves(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, 'GAMES_DIR', str(tmp_path))
    monkeypatch.setattr(pt, 'IN_PROG', str(tmp_path / '_in_progress.pkl'))

    s = pt.new_state(my_color='plus')
    pt.apply_move(s, 3)
    pt.apply_move(s, 4)
    pt.undo(s)
    assert s.move_number == 2
    assert s.turn_color == 'minus'
    assert s.board[5, 3] == 1
    assert s.board[5, 4] == 0


def test_undo_on_empty_is_safe(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pt, 'GAMES_DIR', str(tmp_path))
    monkeypatch.setattr(pt, 'IN_PROG', str(tmp_path / '_in_progress.pkl'))

    s = pt.new_state(my_color='plus')
    pt.undo(s)              # should NOT raise
    out = capsys.readouterr().out
    assert 'nothing to undo' in out


# ---------------------------------------------------------------------------
# 5. Recovery from in-progress file
# ---------------------------------------------------------------------------

def test_in_progress_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, 'GAMES_DIR', str(tmp_path))
    monkeypatch.setattr(pt, 'IN_PROG', str(tmp_path / '_in_progress.pkl'))

    s = pt.new_state(my_color='minus')
    pt.apply_move(s, 3)
    pt.apply_move(s, 3)
    pt.apply_move(s, 4)

    loaded = pt.load_in_progress()
    assert loaded is not None
    assert np.array_equal(loaded.board, s.board)
    assert loaded.move_number == s.move_number
    assert loaded.turn_color == s.turn_color
    assert loaded.my_color == s.my_color
    assert len(loaded.history) == len(s.history)


def test_load_in_progress_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, 'GAMES_DIR', str(tmp_path))
    monkeypatch.setattr(pt, 'IN_PROG', str(tmp_path / 'does_not_exist.pkl'))
    assert pt.load_in_progress() is None


def test_load_in_progress_corrupt_returns_none(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pt, 'GAMES_DIR', str(tmp_path))
    bad = tmp_path / '_in_progress.pkl'
    bad.write_bytes(b'this is not a valid pickle')
    monkeypatch.setattr(pt, 'IN_PROG', str(bad))
    out = pt.load_in_progress()
    assert out is None
    captured = capsys.readouterr().out
    assert 'corrupt' in captured


# ---------------------------------------------------------------------------
# Smoke checks on the helpers used in the live loop
# ---------------------------------------------------------------------------

def test_format_top3_orders_descending():
    s = pt.format_top3({0: 5, 3: 120, 4: 45, 2: 20})
    # Expected: col3 first (120), then col4 (45), then col2 (20)
    assert s.startswith('col3=120')
    parts = s.split(', ')
    assert parts[1].startswith('col4=')
    assert parts[2].startswith('col2=')


def test_format_top3_empty():
    assert pt.format_top3({}) == ''


def test_save_finished_game_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, 'GAMES_DIR', str(tmp_path))
    monkeypatch.setattr(pt, 'IN_PROG', str(tmp_path / '_in_progress.pkl'))
    monkeypatch.setattr(pt, 'USE_COLOR', False)

    s = pt.new_state(my_color='plus')
    pt.apply_move(s, 3)
    pt.apply_move(s, 4)
    path = pt.save_finished_game(s, 'me')
    assert os.path.exists(path)
    with open(path) as f:
        contents = f.read()
    assert 'winner: me' in contents
    assert 'col 3' in contents
    assert 'col 4' in contents
    # In-progress file removed after finished game saved
    assert not os.path.exists(str(tmp_path / '_in_progress.pkl'))
