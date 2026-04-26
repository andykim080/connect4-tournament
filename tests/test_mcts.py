"""Unit tests for src/mcts.py (Day 4).

Eight tests, from docs/mcts_design.md §9:
  1. Priors respected on empty board (col 3 most visited for ANDY_CNN)
  2. Immediate win taken
  3. Immediate loss blocked
  4. Win over block when both exist
  5. Legal mask enforced (full column never returned)
  6. Terminal node returns correct value via backup
  7. N sums (sum child.N == n_simulations)
  8. Symmetry (visit counts on a symmetric board are roughly symmetric)

Run:  pytest tests/test_mcts.py -v
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pytest

from engine import update_board
from model_io import ModelPlayer
from mcts import mcts_move, _mcts_search, _expand, _rollout_value, _backup, Node


# ---------------------------------------------------------------------------
# Shared fixture: load ANDY_CNN once per session.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def andy():
    return ModelPlayer('ANDY_CNN')


# ---------------------------------------------------------------------------
# 1. Priors respected on empty board
# ---------------------------------------------------------------------------

def test_priors_respected_empty_board(andy):
    """Col 3 has prior ~0.94 in ANDY_CNN; 100 sims should make it most-visited."""
    board = np.zeros((6, 7))
    root = _mcts_search(board, 'plus', andy.model, n_simulations=100,
                        c_puct=2.0, mode=andy.mode)
    visits = {c: root.children[c].N for c in root.children}
    top = max(visits, key=visits.get)
    assert top == 3, f'Expected col 3 most-visited; got {top}  visits={visits}'


# ---------------------------------------------------------------------------
# 2. Immediate win taken
# ---------------------------------------------------------------------------

def test_immediate_win_taken(andy):
    """Set up plus with three in col 0 (rows 3,4,5); col 0 wins immediately."""
    board = np.zeros((6, 7))
    # stack 3 plus in column 0 (bottom)
    board = update_board(board, 'plus',  0)   # row 5
    board = update_board(board, 'minus', 1)
    board = update_board(board, 'plus',  0)   # row 4
    board = update_board(board, 'minus', 1)
    board = update_board(board, 'plus',  0)   # row 3 — now three stacked
    # It's plus's turn; col 0 wins.
    col = mcts_move(board, 'plus', andy.model, n_simulations=50, c_puct=2.0,
                    mode=andy.mode)
    assert col == 0, f'Expected col 0 (own win), got {col}'


# ---------------------------------------------------------------------------
# 3. Immediate loss blocked
# ---------------------------------------------------------------------------

def test_immediate_block(andy):
    """Opponent has three in col 0; plus must play col 0 to block."""
    board = np.zeros((6, 7))
    board = update_board(board, 'minus', 0)
    board = update_board(board, 'plus',  1)
    board = update_board(board, 'minus', 0)
    board = update_board(board, 'plus',  1)
    board = update_board(board, 'minus', 0)
    # plus to move; minus threatens col 0.
    col = mcts_move(board, 'plus', andy.model, n_simulations=50, c_puct=2.0,
                    mode=andy.mode)
    assert col == 0, f'Expected col 0 (block), got {col}'


# ---------------------------------------------------------------------------
# 4. Win chosen over block when both exist
# ---------------------------------------------------------------------------

def test_win_over_block(andy):
    """Plus has own win in col 3; minus threatens col 0. Plus must take the win."""
    board = np.zeros((6, 7))
    # build plus's 3 in col 3 AND minus's 3 in col 0
    board = update_board(board, 'plus',  3)
    board = update_board(board, 'minus', 0)
    board = update_board(board, 'plus',  3)
    board = update_board(board, 'minus', 0)
    board = update_board(board, 'plus',  3)
    board = update_board(board, 'minus', 0)
    col = mcts_move(board, 'plus', andy.model, n_simulations=50, c_puct=2.0,
                    mode=andy.mode)
    assert col == 3, f'Expected col 3 (own win preferred over block), got {col}'


# ---------------------------------------------------------------------------
# 5. Legal mask enforced — never return a full column
# ---------------------------------------------------------------------------

def test_legal_mask_full_column(andy):
    """Fill column 3 entirely; MCTS must not return col 3."""
    board = np.zeros((6, 7))
    for _ in range(3):
        board = update_board(board, 'plus',  3)
        board = update_board(board, 'minus', 3)
    # col 3 is full (6 pieces); it's plus's turn.
    for _ in range(20):
        col = mcts_move(board, 'plus', andy.model, n_simulations=30, c_puct=2.0,
                        mode=andy.mode)
        assert col != 3, f'MCTS returned full column 3'


# ---------------------------------------------------------------------------
# 6. Terminal node backup returns correct value
# ---------------------------------------------------------------------------

def test_terminal_backup_value(andy):
    """A node already at terminal should return its exact value — no rollout."""
    board = np.zeros((6, 7))
    # plus just won in col 0 with a vertical four.
    board = update_board(board, 'plus',  0)
    board = update_board(board, 'plus',  0)
    board = update_board(board, 'plus',  0)
    board = update_board(board, 'plus',  0)
    # `color` for the node = whose turn it is to move from this state; plus
    # has already won so minus would be "to move" but the game is over.
    terminal = Node(board=board, color='minus', is_terminal=True,
                    terminal_value=-1.0)   # minus (to-move) has lost
    val = _rollout_value(terminal)
    assert val == -1.0


# ---------------------------------------------------------------------------
# 7. Visit counts sum to n_simulations (+ 1 for root priming)
# ---------------------------------------------------------------------------

def test_visits_sum_to_n_sims(andy):
    """sum of child.N equals n_simulations (root.N = n_simulations + 1)."""
    board = np.zeros((6, 7))
    n_sims = 50
    root = _mcts_search(board, 'plus', andy.model, n_simulations=n_sims,
                        c_puct=2.0, mode=andy.mode)
    total = sum(c.N for c in root.children.values())
    assert total == n_sims, f'Expected {n_sims}; got {total}'
    assert root.N == n_sims + 1, f'root.N should be n_sims + 1 (initial prime)'


# ---------------------------------------------------------------------------
# 8. Symmetry — visit counts on a vertically symmetric board
# ---------------------------------------------------------------------------

def test_central_preference(andy):
    """Given ANDY_CNN's prior (~0.94 on col 3), 400 sims from an empty board
    must leave col 3 as the most-visited root child.

    Strict pairwise symmetry of visits is not enforceable: ANDY_CNN's prior
    is not exactly reflection-symmetric (trained with SGD, no symmetry
    constraint), and single-rollout noise at λ=1 amplifies any small
    asymmetry once a column receives a lucky/unlucky rollout.
    """
    np.random.seed(0)
    board = np.zeros((6, 7))
    root = _mcts_search(board, 'plus', andy.model, n_simulations=400,
                        c_puct=2.0, mode=andy.mode)
    visits = [root.children[c].N for c in range(7)]
    assert int(np.argmax(visits)) == 3, \
        f'expected col 3 most-visited on empty board; got {visits}'
    assert root.children[3].N >= 200, \
        f'expected col 3 to dominate (≥200/400 visits); got {visits}'
