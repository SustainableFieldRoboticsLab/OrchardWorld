from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class LSystemParams:

    a: float = 18.95


    d1: float = 137.5
    d2: float = 137.5
    lr: float = 1.109
    vr: float = 1.732
    n: int = 8

    F0: float = 0.05
    w0: float = 1.0
    roll0: float = 45.0
    trunk_length_factor: float = 4.0


    radius_model: str = "pipe"
    tip_radius: float = 0.0015


    pipe_beta: float = 2.3


    target_height: float = 2.4


    base_radius: float = 0.035


    shape_jitter: float = 0.0


    kind: str = "ternary"


    ap_scaffolds: tuple = (3, 6)
    ap_trunk_internodes: tuple = (2, 4)


    ap_scaffold_pitch: tuple = (42.0, 65.0)
    ap_lateral_pitch: tuple = (32.0, 58.0)

    ap_divergence: tuple = (128.0, 148.0)
    ap_droop: tuple = (4.0, 13.0)
    ap_len_decay: tuple = (0.66, 0.82)
    ap_children: tuple = (2, 3)
    ap_prune: tuple = (0.08, 0.40)
    ap_wobble: tuple = (5.0, 16.0)
    ap_internodes_per_axis: tuple = (2, 4)
    ap_leader_decay: float = 0.82
    ap_form_droop_bias: tuple = (0.6, 1.6)


def preset(name: str) -> LSystemParams:
    name = name.lower()
    table = {

        "ta": (18.95, 94.74, 132.63, 1.109, 1.732, 10),
        "tb": (18.95, 137.5, 137.5, 1.109, 1.732, 8),
        "tc": (22.5, 112.5, 157.5, 1.790, 1.732, 8),
        "td": (36.0, 180.0, 252.0, 1.070, 1.732, 6),
    }
    if name == "apple":

        return LSystemParams(kind="apple", n=4, target_height=2.6, base_radius=0.055,
                             tip_radius=0.004, pipe_beta=2.2)
    if name not in table:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(table) + ['apple']}")
    a, d1, d2, lr, vr, n = table[name]
    return LSystemParams(a=a, d1=d1, d2=d2, lr=lr, vr=vr, n=n)


class StiffnessModel(str, Enum):
    BEAM = "beam"
    RUDIMENTARY = "rudimentary"
    RIGID = "rigid"


@dataclass
class PhysicsParams:


    wood_density: float = 850.0


    youngs_modulus: float = 7.0e9


    beam_factor: float = 3.14159265 / 4.0


    damping_ratio: float = 0.1


    phi_upper: float = 50.0
    phi_lower: float = 0.05
    phi_decay: float = 0.45

    model: StiffnessModel = StiffnessModel.BEAM

    max_stiffness: float = 5.0e5
    min_stiffness: float = 1.0e-4
    joint_limit: float = 2.0
    use_limits: bool = False
    armature: float = 0.0


    joint_type: str = "d6"


    dynamics_jitter: float = 0.0

    gravity: float = -9.81


    linear_drag: float = 1.5
    angular_drag: float = 2.5
    ground_z: float = 0.0


    ground_stiffness: float = 3000.0
    ground_damping: float = 120.0
    ground_friction: float = 6.0


    terrain: bool = False


    terrain_amplitude: float = 0.03
    terrain_wavelength: float = 1.8
    terrain_extent: float = 14.0


    max_speed: float = 25.0
    max_omega: float = 60.0
    brake: float = 120.0


    fade_speed: float = 8.0
    fade_omega: float = 30.0


    pick_max_acceleration: float = 16.0


    mj_solver: str = "cg"


@dataclass
class BreakParams:
    enabled: bool = False


    rupture_stress: float = 5.0e7
    safety_factor: float = 1.0

    max_bend_angle: float | None = None


    mode: str = "detach"
    linear_stiffness: float = 5.0e4
    twist_stiffness_scale: float = 3.0
    broken_color: tuple = (0.30, 0.16, 0.10)


    hysteresis_frames: int = 3


    free_fall: bool = True
    broken_drag_scale: float = 0.2


    break_friction: float = 0.2


    break_friction_ramp_max: float = 5.0
    break_friction_ramp_time: float = 1.5


    freeze_at_bottom: bool = True
    freeze_friction: float = 60.0
    freeze_fall_speed: float = 0.25
    freeze_timeout: float = 3.0


    broken_ang_drag_scale: float = 0.8
    limp_damping_ratio: float = 0.012


@dataclass
class RenderParams:
    wood_tint: tuple = (1.0, 1.0, 1.0)


@dataclass
class FoliageParams:
    enabled: bool = False
    leaves_per_terminal: int = 3
    min_order_for_leaves: int = 3


    leaf_length: float = 0.085
    leaf_width: float = 0.050
    leaf_color: tuple = (0.18, 0.42, 0.12)
    leaf_mass: float = 0.002
    physics: bool = False
    petiole_stiffness: float = 0.02
    petiole_damping: float = 0.002

    def set_density(self, d: float) -> None:
        self.enabled = d > 0.0
        if d <= 0.0:
            return
        self.leaves_per_terminal = max(1, int(round(2 + d * 3)))
        self.min_order_for_leaves = 2 if d > 1.0 else 3


        self.leaf_length = 0.065 + d * 0.033
        self.leaf_width = 0.038 + d * 0.020


@dataclass
class FruitParams:
    enabled: bool = False
    min_order: int = 2
    prob_per_spur: float = 0.35
    max_count: int = 60


    radius: tuple = (0.026, 0.035)


    mass: float = 0.16


    stem_length: float = 0.025


    tether_stiffness: float = 550.0
    tether_damping_ratio: float = 0.9


    detach_force: tuple = (14.0, 23.0)
    detach_hysteresis: int = 6


    tension_hysteresis: int = 12
    tension_factor: float = 1.2


    detach_drag: float = 0.0


    branch_reaction: float = 1.0
    reaction_max: float = 10.0


    joint: str = "slide"


    colors: tuple = ((0.78, 0.10, 0.07), (0.85, 0.30, 0.05),
                     (0.65, 0.62, 0.12))


@dataclass
class RobotParams:
    enabled: bool = False
    position: tuple = (2.4, 0.0)
    yaw: float = 3.14159265


    drive_speed: float = 1.1
    turn_speed: float = 1.5
    drive_kd: float = 4000.0
    turn_kd: float = 800.0


    drive_effort: float = 900.0
    turn_effort: float = 350.0
    arm_kp: float = 1500.0
    arm_kd: float = 120.0


    camera: bool = True
    camera_width: int = 192
    camera_height: int = 144


    camera_fov: float = 75.0


    camera_range: float = 2.0
    camera_every: int = 12


@dataclass
class TreeConfig:
    lsystem: LSystemParams = field(default_factory=lambda: preset("tb"))
    physics: PhysicsParams = field(default_factory=PhysicsParams)
    breaking: BreakParams = field(default_factory=BreakParams)
    render: RenderParams = field(default_factory=RenderParams)
    foliage: FoliageParams = field(default_factory=FoliageParams)
    fruit: FruitParams = field(default_factory=FruitParams)
    robot: RobotParams = field(default_factory=RobotParams)


    deformable: bool = False
    seed: int = 0
    device: str = "cuda"

    @classmethod
    def rigid(cls, preset_name: str = "tb", n: int = 5) -> "TreeConfig":
        c = cls(lsystem=preset(preset_name))
        if preset_name != "apple":
            c.lsystem.n = n
        c.deformable = False
        c.physics.model = StiffnessModel.RIGID
        return c

    @classmethod
    def compliant(cls, preset_name: str = "tb", n: int = 5) -> "TreeConfig":
        c = cls(lsystem=preset(preset_name))
        if preset_name != "apple":
            c.lsystem.n = n
        c.deformable = True
        c.physics.model = StiffnessModel.BEAM
        return c
