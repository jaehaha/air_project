"""Compares the VLM's reading of board_cam images with the simulator's ground-truth board.

    MUJOCO_GL=egl python board_cam/qwen4B/test/eval_vlm.py --n-boards 50

Boards are random positions (alternating random legal moves, 0-42 discs). Every reader mode
(full image / board crop / one crop per row) sees the same boards. Writes the images, per-board
predictions and summary metrics to board_cam/qwen4B/test/results/<model>/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from board_cam.evaluation import make_record, random_boards, summarize
from board_cam.qwen4B.vlm import DEFAULT_MODEL, MODES, BoardReader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boards", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    args = parser.parse_args()

    out = Path(__file__).parent / "results" / args.model.split("/")[-1]
    (out / "images").mkdir(parents=True, exist_ok=True)
    boards = random_boards(args.n_boards, args.seed)
    for i, (image, _) in enumerate(boards):
        Image.fromarray(image).save(out / "images" / f"board_{i:03d}.png")

    reader = BoardReader(args.model)
    summaries = {}
    for mode in args.modes:
        reader.mode = mode
        records = []
        for i, (image, truth) in enumerate(boards):
            start = time.perf_counter()
            pred, answer = reader.read(image)
            records.append(make_record(i, truth, pred, time.perf_counter() - start, answer=answer))
            print(f"[{mode} {i + 1}/{args.n_boards}] discs={records[-1]['n_discs']:2d} "
                  f"cell_acc={'malformed' if pred is None else format(records[-1]['cell_accuracy'], '.3f')}", flush=True)
        summaries[mode] = {"model": args.model, "mode": mode, "seed": args.seed, **summarize(records)}
        (out / f"{mode}.json").write_text(json.dumps({"summary": summaries[mode], "boards": records}, indent=1))

    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
