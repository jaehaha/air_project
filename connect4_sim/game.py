"""Connect 4 rules on a (ROWS, COLS) int array; row 0 is the bottom row."""
import numpy as np

from .config import COLS, EMPTY, ROWS


def legal_columns(board: np.ndarray) -> list[int]:
    return [c for c in range(COLS) if board[ROWS - 1, c] == EMPTY]


def winner(board: np.ndarray) -> int:
    """Returns the player with four in a row, or EMPTY if there is none."""
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r, c]
            if p == EMPTY:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                rr, cc = r + 3 * dr, c + 3 * dc
                if 0 <= rr < ROWS and 0 <= cc < COLS and all(board[r + k * dr, c + k * dc] == p for k in range(4)):
                    return int(p)
    return EMPTY


def to_text(board: np.ndarray) -> str:
    """Top row first, as seen from the robot side: '.' empty, 'R' robot, 'Y' human."""
    return "\n".join(" ".join(".RY"[v] for v in row) for row in board[::-1])
