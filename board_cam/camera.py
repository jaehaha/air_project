"""Board camera: sits on the human side of the table and looks at the board face.

The pose was picked in the MuJoCo viewer ("Copy camera"); it is expressed in the robot base frame.
"""
import mujoco

NAME = "board_cam"
POS = [0.691, -0.018, 0.119]
XYAXES = [0.026, 1.000, -0.000, -0.169, 0.004, 0.986]
FOVY = 45.0
HEIGHT, WIDTH = 960, 1280  # render size used for VLM input


def add_board_camera(spec: mujoco.MjSpec):
    camera = spec.worldbody.add_camera(name=NAME, pos=POS, fovy=FOVY)
    camera.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    camera.alt.xyaxes = XYAXES
    return camera


# Where the 6x7 hole grid appears in the image for this camera pose, as fractions of the image
# size: (left, top, right, bottom). Re-measure if the camera or the board moves.
GRID_BOX = (0.340, 0.190, 0.670, 0.5125)


def crop_grid(image, margin: float = 0.15, rows: slice = slice(None)):
    """Crops the hole grid (optionally only some rows, top row = 0). `margin` is in cell units."""
    height, width = image.shape[:2]
    left, top, right, bottom = GRID_BOX
    cell_w, cell_h = (right - left) / 7, (bottom - top) / 6
    first, last = rows.indices(6)[:2]
    box = (left - margin * cell_w, top + (first - margin) * cell_h,
           right + margin * cell_w, top + (last + margin) * cell_h)
    x0, y0, x1, y1 = (round(v * s) for v, s in zip(box, (width, height, width, height)))
    return image[max(y0, 0):y1, max(x0, 0):x1]


def image_grid_to_board(grid):
    """Image-oriented grid (row 0 = top, col 0 = image left) -> env.board_state() layout
    (row 0 = bottom, col 0 = robot's left). The camera faces the robot, so the image is mirrored
    left-right relative to the robot's view."""
    return grid[::-1, ::-1].copy()
