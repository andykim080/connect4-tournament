import numpy as np
import os
import keras

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ---------------------------------------------------------------------------
# Custom Keras layers required by transformer models
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PositionalIndex(keras.layers.Layer):
    """Learned positional embedding added element-wise to a token sequence."""

    def build(self, input_shape):
        seq_len, d_model = input_shape[-2], input_shape[-1]
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(seq_len, d_model),
            initializer='zeros',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_embedding

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable()
class ClassTokenIndex(keras.layers.Layer):
    """Outputs a zero-index tensor (batch, 1) for CLS token embedding lookup."""

    def call(self, x):
        import tensorflow as tf
        return tf.zeros((tf.shape(x)[0], 1), dtype=tf.int32)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def get_config(self):
        return super().get_config()


# ---------------------------------------------------------------------------
# Board → model input converters
# ---------------------------------------------------------------------------

def board_to_xy2(board: np.ndarray, color: str) -> np.ndarray:
    """Return (6, 7, 2) float32 array with current-player-first channel order."""
    out = np.zeros((6, 7, 2), dtype=np.float32)
    if color == 'plus':
        out[:, :, 0] = (board == 1).astype(np.float32)
        out[:, :, 1] = (board == -1).astype(np.float32)
    else:
        out[:, :, 0] = (board == -1).astype(np.float32)
        out[:, :, 1] = (board == 1).astype(np.float32)
    return out


def board_to_flat42x2(board: np.ndarray, color: str) -> np.ndarray:
    return board_to_xy2(board, color).ravel()


def board_to_patch12x32(board: np.ndarray, color: str) -> np.ndarray:
    """Convert board to (12, 32) token sequence for transformer models.

    Layout: 6 row tokens (current-player perspective) + 6 row tokens
    (opponent perspective), each padded to 32 features.
    """
    xy2 = board_to_xy2(board, color)          # (6, 7, 2)
    cur_rows = xy2[:, :, 0]                    # (6, 7) current player's pieces
    opp_rows = xy2[:, :, 1]                    # (6, 7) opponent's pieces
    tokens = np.concatenate([cur_rows, opp_rows], axis=0)  # (12, 7)
    padded = np.zeros((12, 32), dtype=np.float32)
    padded[:, :7] = tokens
    return padded


_CONVERTERS = {
    'xy2':        board_to_xy2,
    'flat42x2':   board_to_flat42x2,
    'patch12x32': board_to_patch12x32,
}

# ---------------------------------------------------------------------------
# Model loading & mode detection
# ---------------------------------------------------------------------------

_CUSTOM_OBJECTS = {'PositionalIndex': PositionalIndex, 'ClassTokenIndex': ClassTokenIndex}


def load_model(name_or_path: str) -> keras.Model:
    path = (name_or_path if os.path.isabs(name_or_path)
            else os.path.join(MODELS_DIR, name_or_path))
    if not path.endswith('.keras'):
        path += '.keras'
    return keras.models.load_model(
        path, custom_objects=_CUSTOM_OBJECTS, compile=False, safe_mode=False
    )


def detect_mode(model: keras.Model) -> str:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    dims = tuple(shape[1:])
    if dims == (6, 7, 2):
        return 'xy2'
    if dims == (84,):
        return 'flat42x2'
    if len(dims) == 2 and dims[1] == 32:
        return 'patch12x32'
    raise ValueError(f"Unrecognised input shape {shape} — add a converter.")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def get_policy(model: keras.Model, board: np.ndarray, color: str,
               mode: str | None = None) -> np.ndarray:
    """Return raw policy probabilities for all 7 columns (shape (7,))."""
    if mode is None:
        mode = detect_mode(model)
    x = _CONVERTERS[mode](board, color)[np.newaxis]   # (1, ...)
    out = model(x, training=False)
    # handle dual-output models (policy + value)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.numpy()[0]  # (7,)


def greedy_move(model: keras.Model, board: np.ndarray, color: str,
                mode: str | None = None) -> int:
    """Return the legal column with the highest policy probability."""
    legal = [c for c in range(7) if abs(board[0, c]) < 0.1]
    probs = get_policy(model, board, color, mode)
    probs_masked = np.full(7, -np.inf)
    for c in legal:
        probs_masked[c] = probs[c]
    return int(np.argmax(probs_masked))


def stochastic_move(model: keras.Model, board: np.ndarray, color: str,
                    temperature: float = 1.0,
                    mode: str | None = None) -> int:
    """Sample a legal column from the temperature-scaled policy distribution."""
    legal = [c for c in range(7) if abs(board[0, c]) < 0.1]
    probs = get_policy(model, board, color, mode)
    # temperature scaling in log space, then softmax over legal moves only
    log_p = np.log(probs + 1e-8) / temperature
    log_p -= log_p.max()
    masked = np.zeros(7)
    for c in legal:
        masked[c] = np.exp(log_p[c])
    masked /= masked.sum()
    return int(np.random.choice(7, p=masked))


# ---------------------------------------------------------------------------
# Game helper — returns trajectory for PG training
# ---------------------------------------------------------------------------

def play_game(model_a, model_b, temperature: float = 1.0,
              random_openings: int = 0,
              mode_a: str | None = None, mode_b: str | None = None):
    """Play one game between model_a and model_b.

    Who plays plus is chosen randomly each call (50/50).

    random_openings — number of *total* ply at the start that are played
    randomly (not by the neural network) and excluded from trajectories.

    Returns
    -------
    winner   : 'a', 'b', or 'tie'
    move_seq : list of (color, col)
    traj_a   : list of (board_before_move, andy_color, col)  — NN decisions only
    traj_b   : list of (board_before_move, opp_color,  col)  — NN decisions only
    """
    from engine import update_board, check_for_win, find_legal, look_for_win

    # randomly assign roles
    a_is_plus = np.random.rand() < 0.5
    if a_is_plus:
        plus_model,  minus_model  = model_a, model_b
        plus_mode,   minus_mode   = mode_a,  mode_b
        plus_label,  minus_label  = 'a',     'b'
    else:
        plus_model,  minus_model  = model_b, model_a
        plus_mode,   minus_mode   = mode_b,  mode_a
        plus_label,  minus_label  = 'b',     'a'

    a_color = 'plus' if a_is_plus else 'minus'
    b_color = 'minus' if a_is_plus else 'plus'

    board = np.zeros((6, 7))
    color = 'plus'
    models = {'plus': plus_model, 'minus': minus_model}
    modes  = {'plus': plus_mode,  'minus': minus_mode}

    move_seq = []
    traj_a, traj_b = [], []
    ply = 0

    while True:
        legal = find_legal(board)
        if not legal:
            winner = 'tie'
            break

        if ply < random_openings:
            col = int(np.random.choice(legal))
        else:
            model = models[color]
            mode  = modes[color]
            board_before = board.copy()
            if temperature == 0:
                col = greedy_move(model, board, color, mode)
            else:
                col = stochastic_move(model, board, color, temperature, mode)
            # record trajectory for the player who just decided
            if (color == a_color):
                traj_a.append((board_before, a_color, col))
            else:
                traj_b.append((board_before, b_color, col))

        move_seq.append((color, col))
        board = update_board(board, color, col)
        result = check_for_win(board, col)
        ply += 1

        if result != 'nobody':
            w_color = 'plus' if result[2:] == 'plus' else 'minus'
            winner = plus_label if w_color == 'plus' else minus_label
            break

        color = 'minus' if color == 'plus' else 'plus'

    return winner, move_seq, traj_a, traj_b


# ---------------------------------------------------------------------------
# Convenience wrapper that caches mode detection
# ---------------------------------------------------------------------------

class ModelPlayer:
    def __init__(self, name_or_path: str):
        self.name = os.path.splitext(os.path.basename(name_or_path))[0]
        self.model = load_model(name_or_path)
        self.mode = detect_mode(self.model)

    def move(self, board: np.ndarray, color: str) -> int:
        return greedy_move(self.model, board, color, self.mode)

    def stochastic_move(self, board: np.ndarray, color: str,
                        temperature: float = 1.0) -> int:
        return stochastic_move(self.model, board, color, temperature, self.mode)

    def policy(self, board: np.ndarray, color: str) -> np.ndarray:
        return get_policy(self.model, board, color, self.mode)
