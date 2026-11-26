"""Shared pieces of the board-reading experiments: random test boards and accuracy metrics."""
import numpy as np

from connect4_sim import game
from connect4_sim.config import COLS, EMPTY, HUMAN, ROBOT, ROWS
from connect4_sim.env import Connect4Env, run

CLASS_NAMES = {EMPTY: "empty", ROBOT: "red", HUMAN: "yellow"}


def random_position(env: Connect4Env, rng: np.random.Generator):
    env.reset()
    player = int(rng.choice([ROBOT, HUMAN]))
    for _ in range(int(rng.integers(0, ROWS * COLS + 1))):
        run(env.drop_disc(player, int(rng.choice(game.legal_columns(env.board_state())))))
        player = HUMAN if player == ROBOT else ROBOT


def random_boards(n_boards: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(board_cam image, ground-truth board) pairs; the same seed always gives the same boards."""
    rng = np.random.default_rng(seed)
    env = Connect4Env()
    boards = []
    for _ in range(n_boards):
        random_position(env, rng)
        boards.append((env.render(), env.board_state()))
    return boards


def make_record(index: int, truth: np.ndarray, pred: np.ndarray | None, seconds: float, **extra) -> dict:
    return {"index": index, "n_discs": int((truth != EMPTY).sum()), "seconds": seconds,
            "truth": truth.tolist(), "pred": None if pred is None else pred.tolist(),
            "cell_accuracy": None if pred is None else float((pred == truth).mean()), **extra}


def summarize(records: list[dict]) -> dict:
    parsed = [r for r in records if r["pred"] is not None]
    if not parsed:
        return {"boards": len(records), "parsed_boards": 0}
    truth = np.array([r["truth"] for r in parsed])
    pred = np.array([r["pred"] for r in parsed])
    confusion = np.array([[int(((truth == t) & (pred == p)).sum()) for p in CLASS_NAMES] for t in CLASS_NAMES])
    by_fill = {}
    for label, lo, hi in (("0-13 discs", 0, 13), ("14-27 discs", 14, 27), ("28-42 discs", 28, 42)):
        group = [r for r in parsed if lo <= r["n_discs"] <= hi]
        if group:
            by_fill[label] = {"boards": len(group), "cell_accuracy": float(np.mean([r["cell_accuracy"] for r in group]))}
    names = list(CLASS_NAMES.values())
    return {
        "boards": len(records),
        "parsed_boards": len(parsed),
        "exact_match_boards": sum(r["cell_accuracy"] == 1.0 for r in parsed),
        "cell_accuracy": float((truth == pred).mean()),
        "recall": {name: float(confusion[i, i] / max(confusion[i].sum(), 1)) for i, name in enumerate(names)},
        "precision": {name: float(confusion[i, i] / max(confusion[:, i].sum(), 1)) for i, name in enumerate(names)},
        "confusion_rows_truth_cols_pred": {"labels": names, "matrix": confusion.tolist()},
        "accuracy_by_fill": by_fill,
        "mean_seconds_per_image": float(np.mean([r["seconds"] for r in records])),
    }
