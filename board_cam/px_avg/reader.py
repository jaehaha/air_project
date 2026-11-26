"""Reads the Connect 4 board from a board_cam image by averaging the pixels at each hole.

The camera and the board are fixed, so every hole is at a known place in the image
(board_cam.camera.GRID_BOX). No model, no training.
"""
import numpy as np

from board_cam.camera import GRID_BOX, image_grid_to_board
from connect4_sim.config import COLS, EMPTY, HUMAN, ROBOT, ROWS

PATCH = 0.4            # side of the averaged square, as a fraction of the cell size
MIN_SATURATION = 0.35  # below this the hole is white paper, i.e. empty
YELLOW_MIN_GREEN = 0.5  # green/red ratio: red discs are well below, yellow discs well above


def cell_colors(image: np.ndarray, offset_px: tuple[float, float] = (0, 0)) -> np.ndarray:
    """Mean RGB (0-1) of the center patch of each hole; shape (ROWS, COLS, 3), row 0 = top,
    col 0 = image left. `offset_px` shifts the sampling grid (x, y), to test misalignment."""
    height, width = image.shape[:2]
    left, top, right, bottom = GRID_BOX
    cell_w, cell_h = (right - left) * width / COLS, (bottom - top) * height / ROWS
    half_w, half_h = PATCH * cell_w / 2, PATCH * cell_h / 2
    colors = np.zeros((ROWS, COLS, 3))
    for r in range(ROWS):
        for c in range(COLS):
            cx = left * width + (c + 0.5) * cell_w + offset_px[0]
            cy = top * height + (r + 0.5) * cell_h + offset_px[1]
            patch = image[round(cy - half_h):round(cy + half_h) + 1, round(cx - half_w):round(cx + half_w) + 1]
            colors[r, c] = patch.reshape(-1, 3).mean(axis=0) / 255
    return colors


def classify(colors: np.ndarray) -> np.ndarray:
    """Mean RGB per hole -> EMPTY / ROBOT (red) / HUMAN (yellow)."""
    red, green = colors[..., 0], colors[..., 1]
    brightest, darkest = colors.max(axis=-1), colors.min(axis=-1)
    saturation = (brightest - darkest) / np.maximum(brightest, 1e-6)
    disc = np.where(green / np.maximum(red, 1e-6) > YELLOW_MIN_GREEN, HUMAN, ROBOT)
    return np.where(saturation < MIN_SATURATION, EMPTY, disc)


def read_board(image: np.ndarray, offset_px: tuple[float, float] = (0, 0)) -> np.ndarray:
    """Full board_cam frame -> board in env.board_state() layout (row 0 = bottom, col 0 = robot's left)."""
    return image_grid_to_board(classify(cell_colors(image, offset_px)))
