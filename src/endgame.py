"""Endgame override for MCTS.

Activates when the board has ≤ ENDGAME_REMAINING empty cells. Runs a
depth-bounded α-β minimax search; if the search reaches terminal nodes
the value is exact, otherwise a simple heuristic (threat count + center
control) scores the leaf.

Public API
----------
  pieces_remaining(board)            → int
  in_endgame(board)                  → bool
  solve_endgame(board, color, max_depth=8) → (col, score)
                                          col is None if no legal moves
"""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from engine import update_board, check_for_win, find_legal, look_for_win

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENDGAME_REMAINING = 12      # activate when ≤ 12 empty cells (≥ 30 pieces played)
DEFAULT_DEPTH      = 8

# Center-out column ordering — concentrates α-β cuts on stronger lines first.
COL_ORDER = [3, 2, 4, 1, 5, 0, 6]


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def pieces_remaining(board: np.ndarray) -> int:
    return 42 - int(np.abs(board).sum())


def in_endgame(board: np.ndarray) -> bool:
    return pieces_remaining(board) <= ENDGAME_REMAINING


# ---------------------------------------------------------------------------
# Heuristic evaluator (frontier scores; not used at terminal nodes)
# ---------------------------------------------------------------------------

# All 4-in-a-row lines (69 total): 6 cols × 3 vertical + 4 cols × 7 horiz +
# 12 down-right diag + 12 down-left diag. Pre-computed once.
def _all_lines() -> list[list[tuple[int, int]]]:
    lines = []
    # vertical
    for c in range(7):
        for r in range(3):
            lines.append([(r + i, c) for i in range(4)])
    # horizontal
    for r in range(6):
        for c in range(4):
            lines.append([(r, c + i) for i in range(4)])
    # diagonal down-right
    for r in range(3):
        for c in range(4):
            lines.append([(r + i, c + i) for i in range(4)])
    # diagonal down-left
    for r in range(3):
        for c in range(3, 7):
            lines.append([(r + i, c - i) for i in range(4)])
    return lines


_LINES = _all_lines()


def count_threats(board: np.ndarray, color: str) -> int:
    """Number of 4-cell lines containing exactly 3 of `color` and 1 empty.
    These are immediate winning threats — color wins next turn if the empty
    cell is reachable. (We don't filter on reachability — that's expensive
    and the heuristic is fine being a loose upper bound.)
    """
    val = 1 if color == 'plus' else -1
    n = 0
    for line in _LINES:
        own = empty = 0
        for (r, c) in line:
            v = board[r, c]
            if v == val:
                own += 1
            elif v == 0:
                empty += 1
        if own == 3 and empty == 1:
            n += 1
    return n


def center_control(board: np.ndarray, color: str) -> int:
    val = 1 if color == 'plus' else -1
    return int((board[:, 3] == val).sum())


def heuristic_eval(board: np.ndarray, color: str) -> float:
    """Score from `color`'s perspective, in [-1, 1].

    Coarse weighting: each own threat = +0.05, each opp threat = −0.05;
    center-column piece advantage = +0.02 per net piece. Bounded ±1.
    """
    opp = 'minus' if color == 'plus' else 'plus'
    s = 0.05 * (count_threats(board, color) - count_threats(board, opp))
    s += 0.02 * (center_control(board, color) - center_control(board, opp))
    return float(np.clip(s, -1.0, 1.0))


# ---------------------------------------------------------------------------
# α-β minimax with center-out move ordering and a simple transposition table
# ---------------------------------------------------------------------------

def _ordered_legal(board: np.ndarray) -> list[int]:
    legal_set = set(find_legal(board))
    return [c for c in COL_ORDER if c in legal_set]


def _ab(board: np.ndarray, color: str, depth: int,
        alpha: float, beta: float, tt: dict) -> float:
    key = (board.tobytes(), color, depth)
    if key in tt:
        return tt[key]

    # Tactical short-circuit
    win_col = look_for_win(board, color)
    if win_col >= 0:
        # one ply to terminal: value = +1 from `color`'s perspective
        tt[key] = 1.0
        return 1.0

    legal = _ordered_legal(board)
    if not legal:
        tt[key] = 0.0
        return 0.0
    if depth == 0:
        v = heuristic_eval(board, color)
        tt[key] = v
        return v

    opp = 'minus' if color == 'plus' else 'plus'
    best = -math.inf
    for c in legal:
        child = update_board(board, color, c)
        # Did we just win?
        if check_for_win(child, c) != 'nobody':
            best = 1.0
            tt[key] = 1.0
            return 1.0
        # Otherwise: opponent's turn — recurse with negated bounds.
        v = -_ab(child, opp, depth - 1, -beta, -alpha, tt)
        if v > best:
            best = v
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    tt[key] = best
    return best


def solve_endgame(board: np.ndarray, color: str,
                  max_depth: int = DEFAULT_DEPTH) -> tuple[int | None, float]:
    """Pick the best move via α-β.

    Returns (col, score) where `col` is the chosen column (None if no
    legal moves, e.g. board full) and `score` is the evaluation from
    `color`'s perspective in [−1, 1].
    """
    legal = _ordered_legal(board)
    if not legal:
        return None, 0.0

    # Take a tactical move at root if available (cheap; identical to MCTS).
    own_win = look_for_win(board, color)
    if own_win >= 0:
        return own_win, 1.0
    opp = 'minus' if color == 'plus' else 'plus'
    block = look_for_win(board, opp)
    if block >= 0:
        # Forced block — only one sensible move.
        return block, 0.0    # value uncertain post-block; defer to caller

    tt: dict = {}
    best_col   = legal[0]
    best_score = -math.inf
    for c in legal:
        child = update_board(board, color, c)
        if check_for_win(child, c) != 'nobody':
            return c, 1.0
        v = -_ab(child, opp, max_depth - 1,
                 -math.inf, -best_score, tt)
        if v > best_score:
            best_score = v
            best_col   = c
    return best_col, best_score
