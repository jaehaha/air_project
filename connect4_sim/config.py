"""Geometry of the Connect 4 set and workspace layout (meters).

The board values are estimated from the product listing (21 x 15 x 16 cm set);
measure the real set with calipers and update them before sim-to-real work.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OMX_XML = ROOT / "assets/open_manipulator_description/mujoco/omx/omx.xml"

ROWS, COLS = 6, 7
EMPTY, ROBOT, HUMAN = 0, 1, 2  # cell values; robot plays red, human plays yellow


@dataclass(frozen=True)
class BoardConfig:
    disc_diameter: float = 0.024
    disc_thickness: float = 0.005
    disc_mass: float = 0.003
    col_pitch: float = 0.028       # column center-to-center distance
    divider_thickness: float = 0.002
    slot_depth: float = 0.007      # inner gap between the two panels
    panel_thickness: float = 0.002
    hole_diameter: float = 0.020
    floor_height: float = 0.020    # where the lowest disc rests, above the table
    top_margin: float = 0.002      # panel height above the top row
    backing_paper: bool = True     # white paper covering the robot-side face of the board
    tray_size: tuple = (0.15, 0.21, 0.015)  # depth (x), width (y), rim height

    @property
    def width(self) -> float:
        return COLS * self.col_pitch + self.divider_thickness

    @property
    def top(self) -> float:
        return self.floor_height + ROWS * self.disc_diameter + self.top_margin


@dataclass(frozen=True)
class Layout:
    """Poses in the robot base frame (x forward, y left, z up; table at z=0)."""
    riser_height: float = 0.10        # robot base is mounted this high above the table
    board_xy: tuple = (0.14, 0.0)     # board center; board plane faces the robot
    feeder_xy: tuple = (0.06, -0.16)  # where the robot picks up its next disc
    stash_xy: tuple = (0.30, -0.42)   # unused discs are parked here
