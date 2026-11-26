"""Connect 4 + OMX simulation environment."""
import mujoco
import numpy as np

from board_cam import camera as board_camera

from . import game
from .config import COLS, EMPTY, HUMAN, ROBOT, ROWS, BoardConfig, Layout
from .controller import ARM_JOINTS, GRIPPER_OPEN, REST_QPOS, ScriptedPlacer
from .scene import DISCS_PER_PLAYER, build_model, column_y, row_z, stash_pose

UPRIGHT = np.array([np.sqrt(0.5), 0, np.sqrt(0.5), 0])  # disc axis along world x
PLAYER_NAMES = {ROBOT: "robot", HUMAN: "human"}
FINGER_COVER = 0.010     # how much of the disc the fingertips overlap when pinching
PINCH_TO_TIP = 0.005     # pinch site is this far behind the fingertips


class Connect4Env:
    def __init__(self, cfg: BoardConfig = BoardConfig(), layout: Layout = Layout(), control_hz: int = 50):
        self.cfg, self.layout = cfg, layout
        self.model = build_model(cfg, layout)
        self.data = mujoco.MjData(self.model)
        self.n_substeps = round(1 / control_hz / self.model.opt.timestep)
        self.control_dt = self.n_substeps * self.model.opt.timestep
        self.placer = ScriptedPlacer(build_model(cfg, layout, robot_only=True), self.control_dt)
        self._arm_qadr = np.array([self.model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self._gripper_qadr = self.model.joint("gripper_joint_1").qposadr[0]
        self._renderer = None
        self.reset()

    # ---------------------------------------------------------------- basics
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._arm_qadr] = REST_QPOS
        self.data.qpos[self._gripper_qadr] = GRIPPER_OPEN
        self.data.qpos[self._gripper_qadr + 1] = -GRIPPER_OPEN
        self.data.ctrl[:] = np.append(REST_QPOS, GRIPPER_OPEN)
        self._used = {ROBOT: 0, HUMAN: 0}
        for idx, player in enumerate((ROBOT, HUMAN)):
            for i in range(DISCS_PER_PLAYER):
                self._set_disc(player, i, stash_pose(idx, i, self.cfg, self.layout))
        self._load_feeder()
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl=None):
        """One control step. `ctrl` = position targets for joint1..5 and the gripper (rad)."""
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, self.n_substeps)

    def joint_positions(self) -> np.ndarray:
        """Measured joint1..5 and gripper angles (same layout as the action)."""
        return np.append(self.data.qpos[self._arm_qadr], self.data.qpos[self._gripper_qadr])

    def render(self, camera: str = board_camera.NAME, height: int = board_camera.HEIGHT,
               width: int = board_camera.WIDTH) -> np.ndarray:
        if self._renderer is None or (self._renderer.height, self._renderer.width) != (height, width):
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    # ----------------------------------------------------------------- discs
    def _disc_joint(self, player: int, i: int):
        return self.model.joint(f"{PLAYER_NAMES[player]}_disc{i}")

    def _set_disc(self, player: int, i: int, pose):
        joint = self._disc_joint(player, i)
        self.data.qpos[joint.qposadr[0]:joint.qposadr[0] + 7] = pose
        self.data.qvel[joint.dofadr[0]:joint.dofadr[0] + 6] = 0

    def _load_feeder(self):
        """Stands the robot's next disc upright in the feeder pocket."""
        if self._used[ROBOT] < DISCS_PER_PLAYER:
            pos = [*self.layout.feeder_xy, self.cfg.disc_diameter / 2]
            self._set_disc(ROBOT, self._used[ROBOT], [*pos, *UPRIGHT])

    def _slot_xy(self, col: int) -> np.ndarray:
        return np.array([self.layout.board_xy[0], self.layout.board_xy[1] + column_y(col, self.cfg)])

    def board_state(self) -> np.ndarray:
        """Ground-truth (ROWS, COLS) grid read from the simulated disc poses; row 0 = bottom."""
        board = np.full((ROWS, COLS), EMPTY, dtype=int)
        cfg = self.cfg
        for player in (ROBOT, HUMAN):
            for i in range(DISCS_PER_PLAYER):
                x, y, z = self.data.body(f"{PLAYER_NAMES[player]}_disc{i}").xpos
                dx = x - self.layout.board_xy[0]
                col = (COLS - 1) / 2 - (y - self.layout.board_xy[1]) / cfg.col_pitch
                row = (z - cfg.floor_height) / cfg.disc_diameter - 0.5
                r, c = round(row), round(col)
                if abs(dx) < cfg.slot_depth / 2 and 0 <= r < ROWS and 0 <= c < COLS and abs(row - r) < 0.3:
                    board[r, c] = player
        return board

    # ------------------------------------------------------------ game moves
    def _check_legal(self, col: int):
        if col not in game.legal_columns(self.board_state()):
            raise ValueError(f"column {col} is not a legal move")

    def settle(self, seconds: float = 0.6):
        for _ in range(int(seconds / self.control_dt)):
            self.step()
            yield

    def drop_disc(self, player: int, col: int):
        """Drops one of `player`'s discs into `col` (teleported above the slot, no arm motion).
        Generator: one item per control step."""
        self._check_legal(col)
        z = self.cfg.top + self.cfg.disc_diameter / 2 + 0.002
        self._set_disc(player, self._used[player], [*self._slot_xy(col), z, *UPRIGHT])
        self._used[player] += 1
        if player == ROBOT:
            self._load_feeder()
        yield from self.settle()

    def human_move(self, col: int):
        """Drops a yellow disc into `col`. Generator: one item per control step."""
        yield from self.drop_disc(HUMAN, col)

    def robot_move(self, col: int):
        """Scripted pick from the feeder and drop into `col`. Generator: one item per control step."""
        self._check_legal(col)
        cfg = self.cfg
        grip_offset = PINCH_TO_TIP - FINGER_COVER  # pinch site height relative to the disc top
        pick = [*self.layout.feeder_xy, cfg.disc_diameter + grip_offset]
        place = [*self._slot_xy(col), cfg.top + 0.002 + PINCH_TO_TIP]
        clearance = cfg.top + cfg.disc_diameter + 0.02
        for ctrl in self.placer.pick_and_place(self.data.qpos[self._arm_qadr].copy(), pick, place, clearance):
            self.step(ctrl)
            yield
        self._used[ROBOT] += 1
        self._load_feeder()
        yield from self.settle(0.3)


def run(steps):
    """Consumes a move generator without visualization."""
    for _ in steps:
        pass
