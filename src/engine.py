import numpy as np
import random


def update_board(board, color, column):
    board = board.copy()
    colsum = int(abs(board[:, column]).sum())
    row = 5 - colsum
    if row >= 0:
        board[row, column] = 1 if color == 'plus' else -1
    return board


def check_for_win(board, col):
    colsum = int(abs(board[:, col]).sum())
    row = 6 - colsum
    val = board[row, col]
    if val == 0:
        return 'nobody'

    # vertical
    if row + 3 < 6:
        if board[row, col] + board[row+1, col] + board[row+2, col] + board[row+3, col] == 4 * val:
            return 'v-plus' if val == 1 else 'v-minus'

    # horizontal — check all windows containing this column
    for c_start in range(max(0, col - 3), min(col + 1, 4)):
        s = board[row, c_start] + board[row, c_start+1] + board[row, c_start+2] + board[row, c_start+3]
        if s == 4:
            return 'h-plus'
        if s == -4:
            return 'h-minus'

    # diagonal down-right (top-left to bottom-right)
    for d in range(-3, 1):
        r0, c0 = row + d, col + d
        if 0 <= r0 <= 2 and 0 <= c0 <= 3:
            s = board[r0, c0] + board[r0+1, c0+1] + board[r0+2, c0+2] + board[r0+3, c0+3]
            if s == 4:
                return 'd-plus'
            if s == -4:
                return 'd-minus'

    # diagonal down-left (top-right to bottom-left)
    for d in range(-3, 1):
        r0, c0 = row + d, col - d
        if 0 <= r0 <= 2 and 3 <= c0 <= 6:
            s = board[r0, c0] + board[r0+1, c0-1] + board[r0+2, c0-2] + board[r0+3, c0-3]
            if s == 4:
                return 'd-plus'
            if s == -4:
                return 'd-minus'

    return 'nobody'


def find_legal(board):
    return [c for c in range(7) if abs(board[0, c]) < 0.1]


def look_for_win(board, color):
    for col in find_legal(board):
        b = update_board(board, color, col)
        if check_for_win(b, col) != 'nobody':
            return col
    return -1


def find_all_nonlosers(board, color):
    opp = 'minus' if color == 'plus' else 'plus'
    legal = find_legal(board)
    allowed = []
    for col in legal:
        b = update_board(board, color, col)
        # check if opponent can win immediately from any reply
        opp_wins = any(
            check_for_win(update_board(b, opp, j), j) != 'nobody'
            for j in find_legal(b)
        )
        if not opp_wins:
            allowed.append(col)
    return allowed


def rollout(board, next_player):
    player = next_player
    while True:
        legal = find_legal(board)
        if not legal:
            return 'tie'
        col = random.choice(legal)
        board = update_board(board, player, col)
        result = check_for_win(board, col)
        if result != 'nobody':
            return result
        player = 'minus' if player == 'plus' else 'plus'


def _back_prop(winner, path, color0, mcts_dict):
    for i, state in enumerate(path):
        mcts_dict[state][0] += 1
        if winner[2] == color0[0]:
            mcts_dict[state][1] += 1 if i % 2 == 1 else -1
        elif winner != 'tie':
            mcts_dict[state][1] += -1 if i % 2 == 1 else 1


def mcts(board, color0, nsteps):
    win_col = look_for_win(board, color0)
    if win_col >= 0:
        return win_col

    legal0 = find_all_nonlosers(board, color0)
    if not legal0:
        legal0 = find_legal(board)

    mcts_dict = {tuple(board.ravel()): [0, 0]}

    for _ in range(nsteps):
        color = color0
        board_mcts = board.copy()
        path = [tuple(board_mcts.ravel())]
        winner = 'nobody'

        while winner == 'nobody':
            legal = find_legal(board_mcts)
            if not legal:
                winner = 'tie'
                _back_prop(winner, path, color0, mcts_dict)
                break

            children = [tuple(update_board(board_mcts, color, c).ravel()) for c in legal]
            for child in children:
                if child not in mcts_dict:
                    mcts_dict[child] = [0, 0]

            parent_n = mcts_dict[path[-1]][0]
            ucb1 = np.zeros(len(legal))
            for i, child in enumerate(children):
                n, w = mcts_dict[child]
                ucb1[i] = 10 * nsteps if n == 0 else w / n + 2 * np.sqrt(np.log(max(parent_n, 1)) / n)

            best = int(np.argmax(ucb1))
            board_mcts = update_board(board_mcts, color, legal[best])
            path.append(children[best])
            winner = check_for_win(board_mcts, legal[best])

            if winner != 'nobody':
                _back_prop(winner, path, color0, mcts_dict)
                break

            color = 'minus' if color == 'plus' else 'plus'

            if mcts_dict[children[best]][0] == 0:
                winner = rollout(board_mcts, color)
                _back_prop(winner, path, color0, mcts_dict)
                break

    best_col, best_val = -1, -np.inf
    for col in legal0:
        child = tuple(update_board(board, color0, col).ravel())
        n, w = mcts_dict.get(child, [0, 0])
        val = w / n if n > 0 else -np.inf
        if val > best_val:
            best_val, best_col = val, col
    return best_col


def display_board(board):
    line = '-' * (7 * 5 + 8)
    blank = ('|' + ' ' * 5) * 7 + '|'
    print('   0     1     2     3     4     5     6')
    print(line)
    for row in range(6):
        print(blank)
        row_str = '|'
        for col in range(7):
            v = board[row, col]
            row_str += ('  X  |' if v == 1 else '  O  |' if v == -1 else '     |')
        print(row_str)
        print(blank)
        print(line)
    print('   0     1     2     3     4     5     6')
