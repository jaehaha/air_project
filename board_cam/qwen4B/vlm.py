"""Reads the Connect 4 board state from a board_cam image with a local VLM."""
import re

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from connect4_sim.config import COLS, EMPTY, HUMAN, ROBOT, ROWS

from board_cam.camera import crop_grid, image_grid_to_board

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
SYMBOLS = {"W": EMPTY, "R": ROBOT, "Y": HUMAN}
MODES = ("full", "crop", "rows")  # whole camera image / board crop / one crop + query per row

BOARD_PROMPT = f"""This image shows a blue Connect 4 board with {ROWS} rows and {COLS} columns of round holes.
Each hole is white (empty), red (red disc) or yellow (yellow disc).

Read the board row by row, from the top row to the bottom row. For each row, list its {COLS} holes \
from left to right using one letter per hole: W = white, R = red, Y = yellow.
Answer with exactly {ROWS} lines in this format, and nothing else:
row 1: W W W W W W W"""

ROW_PROMPT = f"""This image shows one row of {COLS} round holes of a blue Connect 4 board.
Each hole is white (empty), red (red disc) or yellow (yellow disc).
List the {COLS} holes from left to right using one letter per hole: W = white, R = red, Y = yellow.
Answer with exactly {COLS} letters separated by spaces, and nothing else."""


def parse_rows(text: str, n_rows: int) -> np.ndarray | None:
    """Parses the answer into an image-oriented grid (row 0 = top, col 0 = image left)."""
    rows = []
    for line in text.upper().splitlines():
        letters = re.findall(r"\b[WRY]\b", line.split(":")[-1])
        if letters:
            rows.append(letters)
    if len(rows) != n_rows or any(len(row) != COLS for row in rows):
        return None
    return np.array([[SYMBOLS[ch] for ch in row] for row in rows])


class BoardReader:
    def __init__(self, model_name: str = DEFAULT_MODEL, mode: str = "crop", device: str = "cuda"):
        assert mode in MODES
        self.mode = mode
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()

    @torch.inference_mode()
    def ask(self, image: np.ndarray, prompt: str, max_new_tokens: int = 128) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(image)},
                                                 {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def read(self, image: np.ndarray) -> tuple[np.ndarray | None, str]:
        """`image` is the full board_cam frame. Returns (board in env.board_state() layout, or None
        if the answer was malformed; raw answer)."""
        if self.mode == "rows":
            answers = [self.ask(crop_grid(image, rows=slice(r, r + 1)), ROW_PROMPT, 24) for r in range(ROWS)]
            answer = "\n".join(a.replace("\n", " ") for a in answers)
        else:
            answer = self.ask(image if self.mode == "full" else crop_grid(image), BOARD_PROMPT)
        grid = parse_rows(answer, ROWS)
        return (None if grid is None else image_grid_to_board(grid)), answer
