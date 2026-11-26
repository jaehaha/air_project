"""Compares the pixel-average board reader with the simulator's ground-truth board.

    MUJOCO_GL=egl python board_cam/px_avg/test/eval_px_avg.py --n-boards 50

Uses the same random boards as the VLM experiment (same seed -> same images). Also reports how
the accuracy holds up when the sampling grid is shifted or the image brightness changes.
Writes board_cam/px_avg/test/results/{results,summary}.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from board_cam.evaluation import make_record, random_boards, summarize
from board_cam.px_avg.reader import cell_colors, read_board

SHIFTS_PX = (4, 8, 12, 16, 20)
BRIGHTNESS = (0.4, 0.6, 0.8, 1.2, 1.5)


def cell_accuracy(boards, transform=lambda image: image, offset_px=(0, 0)) -> float:
    return float(np.mean([(read_board(transform(image), offset_px) == truth).mean() for image, truth in boards]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boards", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    boards = random_boards(args.n_boards, args.seed)

    records = []
    for i, (image, truth) in enumerate(boards):
        start = time.perf_counter()
        pred = read_board(image)
        records.append(make_record(i, truth, pred, time.perf_counter() - start))

    # How far apart the three classes are in the features the classifier thresholds on.
    colors = np.concatenate([cell_colors(image)[::-1, ::-1].reshape(-1, 3) for image, _ in boards])
    labels = np.concatenate([truth.reshape(-1) for _, truth in boards])
    saturation = (colors.max(1) - colors.min(1)) / colors.max(1)
    green_red = colors[:, 1] / colors[:, 0]
    margins = {
        "saturation_empty_max": float(saturation[labels == 0].max()),
        "saturation_disc_min": float(saturation[labels != 0].min()),
        "green_red_ratio_red_max": float(green_red[labels == 1].max()),
        "green_red_ratio_yellow_min": float(green_red[labels == 2].min()),
    }

    def scale(factor):
        return lambda image: np.clip(image.astype(float) * factor, 0, 255).astype(np.uint8)

    robustness = {
        "grid_shift_px_worst_direction": {
            str(shift): min(cell_accuracy(boards, offset_px=(dx * shift, dy * shift))
                            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)))
            for shift in SHIFTS_PX},
        "brightness_scale": {str(factor): cell_accuracy(boards, transform=scale(factor)) for factor in BRIGHTNESS},
    }

    summary = {"method": "px_avg", "seed": args.seed, "image_size": list(boards[0][0].shape[:2]),
               **summarize(records), "class_margins": margins, "robustness_cell_accuracy": robustness}
    (out / "results.json").write_text(json.dumps({"summary": summary, "boards": records}, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
