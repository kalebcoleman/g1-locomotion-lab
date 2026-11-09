"""Lane-independent environment: G1 (full collision model), Newton/MJWarp, fallen-pose resets, success tracking.
Lanes subclass `G1RecoveryBaseEnvCfg` and only override `rewards`, `curriculum` (and optionally extra events)."""
from __future__ import annotations
import os
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass, replace
from isaaclab_assets import G1_CFG
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from . import common as K

HELDOUT_NPZ = os.environ.get("G1R_HELDOUT", os.path.join(os.path.dirname(__file__), "..", "heldout_poses.npz"))


@configclass
class G1RecoveryBaseEnvCfg(G1FlatEnvCfg):
    """Matched across lanes: robot model/actuators, Newton MJWarp physics, 50 Hz control, 10 s episodes, observations,
    start-pose distribution, success definition. No pushes, no velocity command (zero), no torso-contact termination."""
    fixed_poses_path: str | None = None

    def __post_init__(self):
        super().__post_init__()
        # --- robot: same G1 joints/actuators/limits, but the FULL collision model (the minimal one has only feet+torso boxes)
        self.scene.robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        # --- physics: same Newton/MJWarp backend and step rates as the walking task; more contact/constraint room for floor contact
        nm = self.sim.physics.newton_mjwarp
        nm.solver_cfg.njmax = 400; nm.solver_cfg.nconmax = 80
        self.sim.physics.default = nm
        self.episode_length_s = K.EPISODE_S
        # --- commands: zero velocity command, kept only so the observation vector is identical to the walking policy's (123-d)
        cm = self.commands.base_velocity
        cm.heading_command = False; cm.rel_standing_envs = 0.0; cm.resampling_time_range = (1e6, 1e6); cm.debug_vis = False
        cm.ranges.lin_vel_x = (0.0, 0.0); cm.ranges.lin_vel_y = (0.0, 0.0); cm.ranges.ang_vel_z = (0.0, 0.0)
        # --- events: keep the walking task's startup randomisation (friction, torso mass); drop pushes / push-style resets
        ev = self.events
        ev.reset_base = None; ev.reset_robot_joints = None; ev.push_robot = None; ev.base_external_force_torque = None
        ev.reset_fallen = EventTerm(func=K.reset_fallen, mode="reset", params={"fixed_poses": self.fixed_poses_path})
        # --- terminations: time-out only (+ a never-terminating metrics hook); falling is the starting condition
        self.terminations.base_contact = None
        self.terminations.track = DoneTerm(func=K.track_recovery)
        self.curriculum.terrain_levels = None
        self.rewards = None   # lanes set their own
