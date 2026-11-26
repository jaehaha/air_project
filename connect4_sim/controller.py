"""Scripted pick-and-place for the OMX arm (IK + joint-space waypoints).

Serves as the sim "expert": its position targets are the same kind of action
an ACT policy outputs (5 arm joints + gripper).
"""
import mujoco
import numpy as np

ARM_JOINTS = [f"joint{i}" for i in range(1, 6)]
GRIPPER_OPEN, GRIPPER_CLOSED = 0.5, 0.0
REST_QPOS = np.array([-1.57, 0.0, 0.0, 1.2, 0.0])  # turned toward the feeder
MAX_JOINT_SPEED = 1.5  # rad/s


class ArmKinematics:
    """IK and gravity terms, computed on a robot-only copy of the model."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site("pinch").id
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.lower, self.upper = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS]].T
        self.kp = model.actuator_gainprm[[model.actuator(j).id for j in ARM_JOINTS], 0]

    def solve(self, pos, closing_axis, seed=None, tol=2e-4) -> np.ndarray:
        """Joint angles that put the pinch site at `pos`, fingers pointing straight down and
        closing along the horizontal direction `closing_axis`."""
        m, d = self.model, self.data
        y = np.array([closing_axis[0], closing_axis[1], 0.0])
        y /= np.linalg.norm(y)
        x = np.array([0.0, 0.0, -1.0])
        target = np.stack([x, y, np.cross(x, y)], axis=1)
        seeds = [] if seed is None else [np.asarray(seed, dtype=float)]
        pan = np.arctan2(pos[1], pos[0])
        seeds += [np.array([pan, j2, 0.3, np.pi / 2 - 0.3 - j2, roll]) for j2 in (-0.3, 0.3, -0.9) for roll in (1.5, -1.5)]
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for q in seeds:
            q = q.copy()
            for _ in range(200):
                d.qpos[self.qadr] = q
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                rot = d.site_xmat[self.site].reshape(3, 3)
                rot_err = 0.5 * sum(np.cross(rot[:, i], target[:, i]) for i in range(3))
                err = np.concatenate([pos - d.site_xpos[self.site], rot_err])
                if np.linalg.norm(err[:3]) < tol and np.linalg.norm(err[3:]) < 1e-3:
                    return q
                mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
                jac = np.vstack([jacp, jacr])[:, self.dadr]
                dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), err)
                q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lower, self.upper)
        raise ValueError(f"IK failed for target {np.round(pos, 3)}")

    def gravity_feedforward(self, q: np.ndarray) -> np.ndarray:
        """Position-target offset that cancels the steady-state sag of the P controller."""
        d = self.data
        d.qpos[self.qadr] = q
        d.qvel[:] = 0
        mujoco.mj_forward(self.model, d)
        return d.qfrc_bias[self.dadr] / self.kp


class ScriptedPlacer:
    """Generates control targets (5 arm joints + gripper) at the env control rate."""

    def __init__(self, model: mujoco.MjModel, control_dt: float):
        self.kin = ArmKinematics(model)
        self.dt = control_dt

    def _move(self, q0, q1, gripper, speed=MAX_JOINT_SPEED):
        steps = max(int(np.abs(q1 - q0).max() / speed / self.dt), int(0.3 / self.dt))
        for i in range(1, steps + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / steps)
            q = q0 + s * (q1 - q0)
            yield np.append(q + self.kin.gravity_feedforward(q), gripper)

    def _hold(self, q, gripper, seconds):
        ctrl = np.append(q + self.kin.gravity_feedforward(q), gripper)
        for _ in range(int(seconds / self.dt)):
            yield ctrl

    def _line(self, q0, p0, p1, axis, gripper, n=6):
        """Straight Cartesian segment, returns the final joint angles via StopIteration."""
        q = q0
        for s in np.linspace(0, 1, n + 1)[1:]:
            q_next = self.kin.solve(p0 + s * (p1 - p0), axis, seed=q)
            yield from self._move(q, q_next, gripper, speed=0.6)
            q = q_next
        return q

    def pick_and_place(self, q_start, pick_pos, place_pos, clearance_z, axis=(1.0, 0.0)):
        """`pick_pos`/`place_pos` are pinch-site positions at grasp/release; the disc axis is `axis`."""
        pick_pos, place_pos = np.asarray(pick_pos), np.asarray(place_pos)
        above_pick = np.array([pick_pos[0], pick_pos[1], clearance_z])
        above_place = np.array([place_pos[0], place_pos[1], clearance_z])
        q_above_pick = self.kin.solve(above_pick, axis)
        q_above_place = self.kin.solve(above_place, axis, seed=q_above_pick)

        yield from self._move(np.asarray(q_start), q_above_pick, GRIPPER_OPEN)
        q = yield from self._line(q_above_pick, above_pick, pick_pos, axis, GRIPPER_OPEN)
        yield from self._hold(q, GRIPPER_CLOSED, 0.5)
        q = yield from self._line(q, pick_pos, above_pick, axis, GRIPPER_CLOSED)
        yield from self._move(q, q_above_place, GRIPPER_CLOSED)
        yield from self._hold(q_above_place, GRIPPER_CLOSED, 0.3)
        q = yield from self._line(q_above_place, above_place, place_pos, axis, GRIPPER_CLOSED)
        yield from self._hold(q, GRIPPER_OPEN, 0.5)
        q = yield from self._line(q, place_pos, above_place, axis, GRIPPER_OPEN, n=2)
        yield from self._move(q, REST_QPOS, GRIPPER_OPEN)
