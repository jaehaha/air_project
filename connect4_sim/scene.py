"""Builds the MuJoCo scene: OMX-F arm + Connect 4 board + discs + cameras."""
import mujoco
import numpy as np

from board_cam.camera import add_board_camera

from .config import COLS, ROWS, BoardConfig, Layout, OMX_XML

BLUE = [0.05, 0.2, 0.75, 1]
RED = [0.85, 0.08, 0.08, 1]
YELLOW = [0.98, 0.8, 0.05, 1]
DISCS_PER_PLAYER = ROWS * COLS // 2
# Discs are tiny and fall fast: default contact softness lets them sink through the thin board walls.
TIMESTEP = 0.001
STIFF_CONTACT = [0.004, 1]
JOINT_LIMITS_DEG = {"joint1": (-270, 360), "joint2": (-120, 90), "joint3": (-120, 90), "joint4": (-100, 100),
                    "joint5": (-270, 270)}


def column_y(col: int, cfg: BoardConfig) -> float:
    """Board-frame y of a column. Column 0 is on the robot's left (+y)."""
    return ((COLS - 1) / 2 - col) * cfg.col_pitch


def row_z(row: int, cfg: BoardConfig) -> float:
    """Height of a resting disc's center. Row 0 is the bottom row."""
    return cfg.floor_height + (row + 0.5) * cfg.disc_diameter


def _panel_mesh(cfg: BoardConfig, segments: int = 32):
    """Triangle mesh of a board panel (in the y-z plane) with one round hole per cell."""
    r = cfg.hole_diameter / 2
    hx = cfg.panel_thickness / 2
    hy, hz = cfg.col_pitch / 2, cfg.disc_diameter / 2
    corners = np.arctan2([hz, hz, -hz, -hz], [hy, -hy, -hy, hy]) % (2 * np.pi)
    angles = np.unique(np.concatenate([np.linspace(0, 2 * np.pi, segments, endpoint=False), corners]))
    c, s = np.cos(angles), np.sin(angles)
    edge = np.minimum(hy / np.maximum(np.abs(c), 1e-9), hz / np.maximum(np.abs(s), 1e-9))
    n = len(angles)
    verts, faces = [], []
    for row in range(ROWS):
        for col in range(COLS):
            y0, z0 = column_y(col, cfg), row_z(row, cfg)
            base = len(verts)
            # vertex rings: hole/front, edge/front, hole/back, edge/back
            for x, rad in ((hx, r), (hx, edge), (-hx, r), (-hx, edge)):
                rad = np.broadcast_to(rad, n)
                verts.extend(np.stack([np.full(n, x), y0 + rad * c, z0 + rad * s], axis=1))
            for i in range(n):
                j = (i + 1) % n
                hf, ef, hb, eb = (base + k * n for k in range(4))
                faces += [[hf + i, ef + i, ef + j], [hf + i, ef + j, hf + j]]  # front face
                faces += [[hb + i, eb + j, eb + i], [hb + i, hb + j, eb + j]]  # back face
                faces += [[hf + i, hf + j, hb + j], [hf + i, hb + j, hb + i]]  # hole wall
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def _add_board(spec: mujoco.MjSpec, cfg: BoardConfig, layout: Layout):
    vert, face = _panel_mesh(cfg)
    mesh = spec.add_mesh(name="board_panel", uservert=vert.flatten(), userface=face.flatten())
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL

    board = spec.worldbody.add_body(name="board", pos=[*layout.board_xy, 0])
    box = mujoco.mjtGeom.mjGEOM_BOX

    def solid(name, pos, size, visible=True):
        return board.add_geom(name=name, type=box, pos=pos, size=size, solref=STIFF_CONTACT,
                              rgba=BLUE if visible else [0, 0, 0, 0], group=0 if visible else 3)

    half_w = cfg.width / 2
    grid_h = ROWS * cfg.disc_diameter
    z_mid = cfg.floor_height + grid_h / 2
    panel_x = cfg.slot_depth / 2 + cfg.panel_thickness / 2
    # Panels: collision is a plain slab, the visual is the mesh with holes.
    for side, sx in (("front", -1), ("back", 1)):
        solid(f"board_{side}_col", [sx * panel_x, 0, z_mid],
              [cfg.panel_thickness / 2, half_w, grid_h / 2], visible=False)
        board.add_geom(name=f"board_{side}_vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="board_panel",
                       pos=[sx * panel_x, 0, 0], rgba=BLUE, contype=0, conaffinity=0, group=0)
        solid(f"board_{side}_top", [sx * panel_x, 0, cfg.top - cfg.top_margin / 2],
              [cfg.panel_thickness / 2, half_w, cfg.top_margin / 2])
    for i in range(COLS + 1):
        y = (i - COLS / 2) * cfg.col_pitch
        solid(f"board_divider{i}", [0, y, z_mid + cfg.top_margin / 2],
              [cfg.slot_depth / 2, cfg.divider_thickness / 2, (grid_h + cfg.top_margin) / 2])
    outer_x = cfg.slot_depth / 2 + cfg.panel_thickness
    if cfg.backing_paper:
        # White sheet taped over the robot-side face, so empty holes look white from the human side.
        board.add_geom(name="board_paper", type=box, pos=[-outer_x - 0.0003, 0, z_mid + cfg.top_margin / 2],
                       size=[0.0002, half_w, (grid_h + cfg.top_margin) / 2], rgba=[1, 1, 1, 1],
                       contype=0, conaffinity=0)
    solid("board_floor", [0, 0, cfg.floor_height / 2], [outer_x, half_w, cfg.floor_height / 2])

    # Tray the board stands in.
    tx, ty, rim = cfg.tray_size[0] / 2, cfg.tray_size[1] / 2, cfg.tray_size[2]
    wall = 0.002
    solid("tray_bottom", [0, 0, wall / 2], [tx, ty, wall / 2])
    for name, pos, size in (("tray_front", [-tx, 0, rim / 2], [wall / 2, ty, rim / 2]),
                            ("tray_back", [tx, 0, rim / 2], [wall / 2, ty, rim / 2]),
                            ("tray_left", [0, ty, rim / 2], [tx, wall / 2, rim / 2]),
                            ("tray_right", [0, -ty, rim / 2], [tx, wall / 2, rim / 2])):
        solid(name, pos, size)


def _add_feeder(spec: mujoco.MjSpec, cfg: BoardConfig, layout: Layout):
    """A pocket that holds one disc upright (disc axis along x) for the gripper to pinch."""
    feeder = spec.worldbody.add_body(name="feeder", pos=[*layout.feeder_xy, 0])
    wall, height, play = 0.004, 0.008, 0.0005
    hx = cfg.disc_thickness / 2 + play
    hy = cfg.disc_diameter / 2 + play
    box = mujoco.mjtGeom.mjGEOM_BOX
    for name, pos, size in (("feeder_front", [-hx - wall / 2, 0, height / 2], [wall / 2, hy + wall, height / 2]),
                            ("feeder_back", [hx + wall / 2, 0, height / 2], [wall / 2, hy + wall, height / 2]),
                            ("feeder_left", [0, hy + wall / 2, height / 2], [hx, wall / 2, height / 2]),
                            ("feeder_right", [0, -hy - wall / 2, height / 2], [hx, wall / 2, height / 2])):
        feeder.add_geom(name=name, type=box, pos=pos, size=size, rgba=[0.3, 0.3, 0.3, 1], solref=STIFF_CONTACT)


def _add_discs(spec: mujoco.MjSpec, cfg: BoardConfig, layout: Layout):
    for player, rgba in (("robot", RED), ("human", YELLOW)):
        for i in range(DISCS_PER_PLAYER):
            body = spec.worldbody.add_body(name=f"{player}_disc{i}")
            body.add_freejoint(name=f"{player}_disc{i}")
            # Cylinder axis is the body z axis.
            body.add_geom(name=f"{player}_disc{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          size=[cfg.disc_diameter / 2, cfg.disc_thickness / 2, 0], rgba=rgba,
                          mass=cfg.disc_mass, solref=STIFF_CONTACT, friction=[1.0, 0.005, 0.0001], condim=4)


def stash_pose(player_idx: int, i: int, cfg: BoardConfig, layout: Layout) -> np.ndarray:
    """qpos (pos + quat) of an unused disc lying flat in the stash area."""
    step = cfg.disc_diameter + 0.006
    x = layout.stash_xy[0] + player_idx * step * 4 + (i % 3) * step
    y = layout.stash_xy[1] + (i // 3) * step
    return np.array([x, y, cfg.disc_thickness / 2, 1, 0, 0, 0])


def build_spec(cfg: BoardConfig = BoardConfig(), layout: Layout = Layout(), robot_only: bool = False) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(OMX_XML))
    spec.modelname = "connect4_omx"
    spec.option.timestep = TIMESTEP
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]

    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.85, 0.82, 0.76], rgb2=[0.8, 0.77, 0.71], width=256, height=256)
    table = spec.add_material(name="table")
    table.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    table.texrepeat = [10, 10]
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], material="table",
                            solref=STIFF_CONTACT)
    spec.worldbody.add_light(pos=[0.1, 0, 1.2], dir=[0, 0, -1])

    # The description files carry no joint limits; these are the hardware limits ROBOTIS documents
    # for the OMX follower (as used in robotis_mujoco_menagerie).
    for name, limits in JOINT_LIMITS_DEG.items():
        joint = spec.joint(name)
        joint.range = np.deg2rad(limits)
        joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE

    # The arm sits on a riser so the gripper can reach over the board pointing straight down.
    base = spec.body("link0")
    base.pos = [0, 0, layout.riser_height]
    spec.worldbody.add_geom(name="riser", type=mujoco.mjtGeom.mjGEOM_BOX, rgba=[0.45, 0.45, 0.48, 1],
                            pos=[-0.02, 0, layout.riser_height / 2], size=[0.06, 0.06, layout.riser_height / 2])

    # Pinch point between the fingertips: x along the fingers, y along the closing direction.
    spec.body("link5").add_site(name="pinch", pos=[0.0895, -0.0016, 0], size=[0.002, 0, 0], rgba=[0, 1, 0, 0.5],
                                group=4)
    if robot_only:
        return spec

    _add_board(spec, cfg, layout)
    _add_feeder(spec, cfg, layout)
    _add_discs(spec, cfg, layout)

    target = spec.worldbody.add_body(name="board_center", pos=[*layout.board_xy, (cfg.floor_height + cfg.top) / 2])
    add_board_camera(spec)
    spec.worldbody.add_camera(name="overview", pos=[-0.25, -0.50, 0.50], fovy=45,
                              mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody=target.name)
    return spec


def build_model(cfg: BoardConfig = BoardConfig(), layout: Layout = Layout(), robot_only: bool = False) -> mujoco.MjModel:
    return build_spec(cfg, layout, robot_only).compile()
