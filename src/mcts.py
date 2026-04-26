"""PUCT-MCTS for Connect 4 — Day 4.

Public API
----------
  mcts_move(board, color, model, n_simulations=500, c_puct=2.0, mode=None, add_noise=False) -> int
  mcts_policy(board, color, model, n_simulations=500, c_puct=2.0, mode=None, temperature=1.0) -> np.ndarray

Design locked in docs/mcts_design.md. Summary:
  • PUCT(s,a) = Q(s,a) + c_puct · P(s,a) · sqrt(N(s)) / (1 + N(s,a))
  • Expansion creates all legal children at once with policy-net priors.
  • Leaf eval uses a single uniform rollout (λ=1); terminal nodes return their
    exact value without rolling out.
  • Backup walks to root flipping sign at each level.
  • Root selection: most-visited child. Tactical wrapper (own-win / block-win)
    is applied BEFORE search so trivially correct moves are never missed.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from engine import (
    update_board, check_for_win, find_legal, look_for_win, rollout,
)
from model_io import detect_mode, get_policy


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """Tree node.

    `board`     — (6,7) state at THIS node.
    `color`     — whose turn it is to move FROM this state.
    `parent`    — parent Node, or None for root.
    `action`    — column played to REACH this node (None for root).
    `prior`     — P(s,a) from the parent's policy net (0.0 for root).
    `children`  — dict col → Node; populated on expansion.
    `N`, `W`    — visit count and accumulated backed-up value.
    `is_expanded` — True once children are created.
    `is_terminal` — True if this node is a terminal position.
    `terminal_value` — value from THIS node's color's perspective if terminal.
    """
    board:          np.ndarray
    color:          str
    parent:         'Node | None' = None
    action:         int | None    = None
    prior:          float         = 0.0
    children:       dict          = field(default_factory=dict)
    N:              int           = 0
    W:              float         = 0.0
    is_expanded:    bool          = False
    is_terminal:    bool          = False
    terminal_value: float         = 0.0

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


# ---------------------------------------------------------------------------
# PUCT scoring
# ---------------------------------------------------------------------------

def puct_score(parent: Node, child: Node, c_puct: float) -> float:
    """Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)).

    NOTE on the sign of Q: `_backup` flips sign at each level, so `child.W`
    (and therefore `child.Q`) is accumulated from the CHILD's perspective —
    the opposite of the parent's perspective. Since the parent picks the
    child that is best for the PARENT's player, we negate child.Q here.
    (A tempting argmax of child.Q without the negation picks moves that
    are good for the OPPONENT, which is what vanilla AlphaZero-pseudocode
    readers occasionally get wrong.)
    """
    u = c_puct * child.prior * math.sqrt(parent.N) / (1 + child.N)
    return -child.Q + u


# ---------------------------------------------------------------------------
# Selection (descend tree to a leaf)
# ---------------------------------------------------------------------------

def _select(root: Node, c_puct: float) -> Node:
    node = root
    while node.is_expanded and not node.is_terminal:
        # Pick child with highest PUCT.
        best_child = max(node.children.values(),
                         key=lambda c: puct_score(node, c, c_puct))
        node = best_child
    return node


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

def _expand(leaf: Node, model, mode: str,
            add_noise: bool = False,
            dirichlet_alpha: float = 0.3,
            noise_eps: float = 0.25,
            prior_temperature: float = 1.0) -> None:
    """Create children for every legal move; set priors from the policy net.

    `prior_temperature` (β): softmax temperature applied to PG's priors
    before storing in nodes. priors = softmax(log(raw + 1e-8) / β),
    renormalized over LEGAL moves only. β=1.0 is identity (no change),
    β<1 sharpens (more peaked), β>1 softens (flatter). AlphaGo uses 0.67.
    """
    legal = find_legal(leaf.board)
    if not legal:
        # terminal by exhaustion (tie) — caller handled elsewhere; defensive.
        leaf.is_terminal   = True
        leaf.terminal_value = 0.0
        return

    probs = get_policy(model, leaf.board, leaf.color, mode)   # (7,)
    if prior_temperature == 1.0:
        masked = np.zeros(7, dtype=np.float64)
        for c in legal:
            masked[c] = max(float(probs[c]), 1e-8)
        masked /= masked.sum()
    else:
        # softmax(log(p)/β) over legal moves only; subtract max for stability
        legal_log = np.array([math.log(max(float(probs[c]), 1e-8))
                              for c in legal]) / prior_temperature
        legal_log -= legal_log.max()
        legal_probs = np.exp(legal_log)
        legal_probs /= legal_probs.sum()
        masked = np.zeros(7, dtype=np.float64)
        for i, c in enumerate(legal):
            masked[c] = legal_probs[i]

    if add_noise:
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
        for i, c in enumerate(legal):
            masked[c] = (1 - noise_eps) * masked[c] + noise_eps * noise[i]

    child_color = 'minus' if leaf.color == 'plus' else 'plus'
    for c in legal:
        child_board = update_board(leaf.board, leaf.color, c)
        child = Node(
            board  = child_board,
            color  = child_color,
            parent = leaf,
            action = c,
            prior  = float(masked[c]),
        )
        # Detect terminality at creation — cheap and avoids re-rolling.
        result = check_for_win(child_board, c)
        if result != 'nobody':
            child.is_terminal = True
            # winner just moved (leaf.color); from child's perspective its player
            # (child_color) has lost.
            child.terminal_value = -1.0
        elif not find_legal(child_board):
            child.is_terminal   = True
            child.terminal_value = 0.0
        leaf.children[c] = child

    leaf.is_expanded = True


# ---------------------------------------------------------------------------
# Leaf evaluation — λ=1 rollout (or terminal value)
# ---------------------------------------------------------------------------

def _rollout_value(leaf: Node) -> float:
    """Return value from the leaf's color's perspective."""
    if leaf.is_terminal:
        return leaf.terminal_value
    # engine.rollout takes (board, next_player_color). Pure uniform playout.
    result = rollout(leaf.board, leaf.color)
    if result == 'tie':
        return 0.0
    winner_color = 'plus' if result[2:] == 'plus' else 'minus'
    return 1.0 if winner_color == leaf.color else -1.0


# ---------------------------------------------------------------------------
# Backup — walk to root flipping sign each step
# ---------------------------------------------------------------------------

def _backup(leaf: Node, value: float) -> None:
    node = leaf
    sign = 1.0
    while node is not None:
        node.N += 1
        node.W += sign * value
        sign   *= -1.0
        node    = node.parent


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def _mcts_search(
    board:             np.ndarray,
    color:             str,
    model,
    n_simulations:     int,
    c_puct:            float,
    mode:              str | None = None,
    add_noise:         bool       = False,
    prior_temperature: float      = 1.0,
    use_fork_filter:   bool       = False,
) -> Node:
    """Run n_simulations of PUCT-MCTS; return the root (for inspection)."""
    if mode is None:
        mode = detect_mode(model)

    root = Node(board=board.copy(), color=color)
    _expand(root, model, mode, add_noise=add_noise,
            prior_temperature=prior_temperature)
    # Task F (root-only fork filter): if enabled, deprioritize moves that
    # let the opponent immediately create a fork.
    if use_fork_filter:
        try:
            from fork_filter import apply_fork_filter
            apply_fork_filter(root, color)
        except ImportError:
            pass    # fork_filter module not yet implemented (Task F stub)
    # "Prime" root.N so sqrt(N) in PUCT isn't zero for the first pick.
    root.N = 1

    for _ in range(n_simulations):
        leaf = _select(root, c_puct)

        if not leaf.is_expanded and not leaf.is_terminal:
            _expand(leaf, model, mode, add_noise=False,
                    prior_temperature=prior_temperature)

        value = _rollout_value(leaf)
        _backup(leaf, value)

    return root


# ---------------------------------------------------------------------------
# Public: mcts_move
# ---------------------------------------------------------------------------

def mcts_move(
    board:             np.ndarray,
    color:             str,
    model,
    n_simulations:     int        = 500,
    c_puct:            float      = 2.0,
    mode:              str | None = None,
    add_noise:         bool       = False,
    prior_temperature: float      = 1.0,
    use_endgame:       bool       = False,
    endgame_depth:     int        = 8,
    use_opening_book:  bool       = False,
    use_fork_filter:   bool       = False,
) -> int:
    """Return the best legal column via PUCT-MCTS with NN priors.

    Override stack at the root, in priority order:
      0. Opening book (if `use_opening_book` and position is in book).
      1. Endgame minimax (if `use_endgame` and ≤ ENDGAME_REMAINING empties).
      2. Immediate own-win — play it.
      3. Block opponent's immediate win — play it.
      4. PUCT-MCTS with optional fork-filter prior adjustment at root.

    `use_opening_book` and `use_fork_filter` are stubs until Tasks E and F
    are implemented; they currently no-op when False (default) and pass
    through to the (yet-unimplemented) modules when True.
    """
    if use_opening_book:
        try:
            from opening_book import lookup_opening
            col = lookup_opening(board, color)
            if col is not None:
                return col
        except ImportError:
            pass    # book module not yet implemented

    if use_endgame:
        from endgame import in_endgame, solve_endgame
        if in_endgame(board):
            col, _ = solve_endgame(board, color, max_depth=endgame_depth)
            if col is not None:
                return col

    col = look_for_win(board, color)
    if col >= 0:
        return col
    opp = 'minus' if color == 'plus' else 'plus'
    col = look_for_win(board, opp)
    if col >= 0:
        return col

    # Pass fork-filter flag down to the search; _expand checks it at the
    # root level only (per Task F spec: "Hard filter at root only,
    # not recursive into the tree").
    root = _mcts_search(board, color, model, n_simulations, c_puct,
                        mode=mode, add_noise=add_noise,
                        prior_temperature=prior_temperature,
                        use_fork_filter=use_fork_filter)
    return max(root.children, key=lambda c: root.children[c].N)


# ---------------------------------------------------------------------------
# Public: mcts_policy — for PG training with MCTS targets
# ---------------------------------------------------------------------------

def mcts_policy(
    board:         np.ndarray,
    color:         str,
    model,
    n_simulations: int        = 500,
    c_puct:        float      = 2.0,
    mode:          str | None = None,
    temperature:   float      = 1.0,
) -> np.ndarray:
    """Return (7,) visit-count distribution over columns.

    temperature → 0  ⇒ one-hot on most-visited (deterministic)
    temperature = 1  ⇒ visit counts / total (proportional)
    """
    root = _mcts_search(board, color, model, n_simulations, c_puct, mode=mode)
    counts = np.zeros(7, dtype=np.float64)
    for c, child in root.children.items():
        counts[c] = child.N
    if counts.sum() == 0:
        return counts
    if temperature <= 1e-6:
        out = np.zeros(7)
        out[int(np.argmax(counts))] = 1.0
        return out
    scaled = counts ** (1.0 / temperature)
    return scaled / scaled.sum()
