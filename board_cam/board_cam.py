"""Reads the current board from the board camera and builds the LLM prompt for the robot's next move.

    MUJOCO_GL=egl python board_cam/board_cam.py --random-moves 9    # sim board after 9 random moves
    python board_cam/board_cam.py --image frame.png                 # a saved board_cam frame

Prints the board state, then the prompt. Give the prompt to an LLM and pass its answer to
parse_column() to get the column to play.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path[0] = str(Path(__file__).resolve().parents[1])  # project root, instead of this directory
from board_cam.px_avg.reader import read_board
from connect4_sim import game
from connect4_sim.config import COLS, EMPTY, HUMAN, ROBOT, ROWS

PROMPT = """You are playing Connect 4 as {me}. Your opponent is {opponent}.

Rules: the board has {rows} rows and {cols} columns. Players take turns dropping one disc into a \
column; the disc falls to the lowest empty cell of that column. The first player to line up four \
of their own discs horizontally, vertically or diagonally wins.

Current board (top row first, columns numbered 1 to {cols} from left to right):
{board}

Legend: . = empty, R = Red, Y = Yellow
Discs on the board: Red {n_red}, Yellow {n_yellow}
Lowest empty row in each column (row 1 = bottom): {heights}
Columns you can play: {legal}

It is your turn ({me}). Choose the best column. Priorities:
1. If you can complete four in a row this move, do it.
2. Otherwise, if {opponent} could complete four in a row on their next move, block it.
3. Otherwise, build toward your own four in a row, prefer the center columns, and do not play a \
move that lets {opponent} win by playing on top of your disc.

Answer with one line and nothing else, in exactly this format:
column: <number>"""

NAMES = {ROBOT: "Red (R)", HUMAN: "Yellow (Y)"}


def board_text(board: np.ndarray) -> str:
    """The board with column numbers, top row first, as seen from the robot side."""
    return " ".join(str(c + 1) for c in range(COLS)) + "\n" + game.to_text(board)


def build_prompt(board: np.ndarray, player: int = ROBOT) -> str:
    """Prompt asking an LLM for `player`'s next column on `board` (env.board_state() layout)."""
    legal = game.legal_columns(board)
    if game.winner(board) != EMPTY or not legal:
        raise ValueError("the game is already over")
    heights = ", ".join(f"column {c + 1}: " + (f"row {int((board[:, c] != EMPTY).sum()) + 1}" if c in legal else "full")
                        for c in range(COLS))
    return PROMPT.format(
        me=NAMES[player], opponent=NAMES[HUMAN if player == ROBOT else ROBOT], rows=ROWS, cols=COLS,
        board=board_text(board), n_red=int((board == ROBOT).sum()), n_yellow=int((board == HUMAN).sum()),
        heights=heights, legal=", ".join(str(c + 1) for c in legal))


def parse_column(answer: str, board: np.ndarray) -> int | None:
    """Column index (0 = robot's left) from the LLM's answer, or None if it names no legal column."""
    match = re.search(r"column\s*:?\s*(\d+)", answer, re.IGNORECASE) or re.search(r"\d+", answer)
    if match is None:
        return None
    col = int(match.group(1) if match.re.groups else match.group()) - 1
    return col if col in game.legal_columns(board) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="saved board_cam frame; default is a fresh simulated board")
    parser.add_argument("--random-moves", type=int, default=0, help="random moves to play first (sim only)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.image:
        from PIL import Image
        image = np.asarray(Image.open(args.image).convert("RGB"))
    else:
        from connect4_sim.env import Connect4Env, run
        rng = np.random.default_rng(args.seed)
        env = Connect4Env()
        env.reset()
        player = HUMAN
        for _ in range(args.random_moves):
            board = env.board_state()
            if game.winner(board) != EMPTY or not game.legal_columns(board):
                break
            run(env.drop_disc(player, int(rng.choice(game.legal_columns(board)))))
            player = ROBOT if player == HUMAN else HUMAN
        image = env.render()

    board = read_board(image)
    print(board_text(board), "\n")
    print(build_prompt(board))


if __name__ == "__main__":
    main()
