"""Play Connect 4 against the simulated OMX arm in the MuJoCo viewer.

    python scripts/play.py          # type a column (1-7, 1 = robot's left) + Enter in the terminal
    python scripts/play.py --auto   # the human side plays random moves
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from connect4_sim import game
from connect4_sim.config import COLS, EMPTY, HUMAN, ROBOT
from connect4_sim.env import Connect4Env


def choose_column(env: Connect4Env, rng: np.random.Generator) -> int:
    """Robot move selection. Placeholder: replace with the VLM/LLM (env.render() is the board_cam image)."""
    return int(rng.choice(game.legal_columns(env.board_state())))


def read_columns(moves: queue.Queue):
    for line in sys.stdin:
        if line.strip().isdigit() and 1 <= int(line) <= COLS:
            moves.put(int(line) - 1)
        else:
            print(f"type a column number 1-{COLS}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="human side plays random moves")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = Connect4Env()
    moves = queue.Queue()
    if not args.auto:
        threading.Thread(target=read_columns, args=(moves,), daemon=True).start()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        def animate(steps):
            for _ in steps:
                start = time.perf_counter()
                viewer.sync()
                if not viewer.is_running():
                    sys.exit()
                time.sleep(max(0.0, env.control_dt - (time.perf_counter() - start)))

        def finished() -> bool:
            board = env.board_state()
            print(game.to_text(board), "\n")
            won = game.winner(board)
            if won != EMPTY or not game.legal_columns(board):
                print({ROBOT: "robot wins", HUMAN: "human wins", EMPTY: "draw"}[won])
                return True
            return False

        print(f"your move: column 1-{COLS} + Enter" if not args.auto else "auto play")
        over = False
        while viewer.is_running():
            animate(env.settle(0.1))
            if over:
                continue
            legal = game.legal_columns(env.board_state())
            if args.auto:
                col = int(rng.choice(legal))
            elif not moves.empty():
                col = moves.get()
                if col not in legal:
                    print("column is full")
                    continue
            else:
                continue
            animate(env.human_move(col))
            over = finished()
            if not over:
                animate(env.robot_move(choose_column(env, rng)))
                over = finished()


if __name__ == "__main__":
    main()
