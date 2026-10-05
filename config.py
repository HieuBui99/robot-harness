"""OMX configuration. ROS names can be overridden with an OMX_CONFIG JSON file."""

import json
import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class RobotConfig:
    urdf_path: str = str(ROOT / "assets" / "omx_f.urdf")
    base_link: str = "link0"
    end_effector_link: str = "end_effector_link"
    joint_names: list[str] = field(default_factory=lambda: [f"joint{i}" for i in range(1, 6)])
    gripper_joint: str = "gripper_joint_1"
    joint_states_topic: str = "/joint_states"
    arm_trajectory_topic: str = "/arm_controller/joint_trajectory"
    gripper_action: str = "/gripper_controller/gripper_cmd"
    camera_topics: dict[str, str] = field(default_factory=lambda: {
        "top": "/top_camera/image_raw/compressed",
        "wrist": "/wrist_camera/image_raw/compressed",
    })
    # These are radians of the revolute gripper joint, from OMX's SRDF.
    gripper_closed: float = 0.0
    gripper_open: float = 1.0
    gripper_max_effort: float = 1.0
    joint_limit_overrides: dict[str, list[float]] = field(default_factory=dict)
    max_joint_step_rad: float = math.radians(20)
    min_cartesian_step_mm: float = 2.0  # MCP translation commands only.
    max_cartesian_step_mm: float = 50.0
    max_rotation_step_deg: float = 15.0
    max_velocity_rad_s: float = 0.35
    max_acceleration_rad_s2: float = 0.7
    min_duration_s: float = 0.5
    min_tool_z_m: float = 0.01
    state_max_age_s: float = 2.0
    camera_max_age_s: float = 2.0
    observation_timeout_s: float = 3.0
    action_timeout_s: float = 15.0
    joint_tolerance_rad: float = 0.03  # Legacy configuration field; unused.
    # Legacy arm tracking settings are accepted for existing JSON configs, but unused.
    tool_position_tolerance_mm: float = 1.0
    tool_angle_tolerance_deg: float = 1.0
    arm_feedback_correction_attempts: int = 3
    arm_feedback_settle_s: float = 1.0
    max_arm_tracking_bias_rad: float = math.radians(2)
    ik_position_tolerance_m: float = 0.0005
    ik_angle_tolerance_rad: float = math.radians(0.5)
    image_max_size: int = 768
    # Arm joint1..joint5 in radians, saved from live /joint_states on 2026-10-05.
    initial_joint_positions_rad: list[float] = field(default_factory=lambda: [
        0.026077673393849032,
        -0.6841554313972029,
        0.8574952604278665,
        1.3314953238845293,
        0.0644271930909901,
    ])
    # Arm joint1..joint5 saved from live /joint_states on 2026-10-05; None disables it.
    post_grasp_joint_positions_rad: list[float] | None = field(default_factory=lambda: [
        0.8605632220036377,
        -0.049087385212547296,
        0.1994175024249265,
        1.3238254199451012,
        0.0644271930909901,
    ])
    presets: dict[str, list[float]] = field(default_factory=lambda: {
        "init": [0.0] * 5,
        "home": [0.0, -1.57, 1.57, 1.57, 0.0],
    })

    def __post_init__(self):
        if self.joint_names != [f"joint{i}" for i in range(1, 6)]:
            raise ValueError("This OMX model expects joint1 through joint5 in URDF order")
        positive = ("max_joint_step_rad", "min_cartesian_step_mm", "max_cartesian_step_mm", "max_rotation_step_deg",
                    "max_velocity_rad_s", "max_acceleration_rad_s2", "min_duration_s",
                    "state_max_age_s", "camera_max_age_s", "observation_timeout_s",
                    "action_timeout_s", "joint_tolerance_rad", "tool_position_tolerance_mm",
                    "tool_angle_tolerance_deg", "ik_position_tolerance_m",
                    "ik_angle_tolerance_rad", "image_max_size", "gripper_max_effort",
                    "arm_feedback_settle_s", "max_arm_tracking_bias_rad")
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_cartesian_step_mm > self.max_cartesian_step_mm:
            raise ValueError("min_cartesian_step_mm must not exceed max_cartesian_step_mm")
        if not isinstance(self.image_max_size, int):
            raise ValueError("image_max_size must be an integer")
        if (isinstance(self.arm_feedback_correction_attempts, bool) or
                not isinstance(self.arm_feedback_correction_attempts, int) or
                not 0 <= self.arm_feedback_correction_attempts <= 10):
            raise ValueError("arm_feedback_correction_attempts must be an integer from 0 to 10")
        for name in ("gripper_closed", "gripper_open", "min_tool_z_m"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.gripper_closed == self.gripper_open:
            raise ValueError("Gripper open and closed positions must differ")
        if not self.camera_topics or any(not name or not topic for name, topic in self.camera_topics.items()):
            raise ValueError("At least one named camera topic is required")
        for name in ("initial_joint_positions_rad", "post_grasp_joint_positions_rad"):
            positions = getattr(self, name)
            if name == "post_grasp_joint_positions_rad" and positions is None:
                continue
            if (not isinstance(positions, list) or len(positions) != len(self.joint_names) or
                    any(isinstance(q, bool) or not isinstance(q, (int, float)) or
                        not math.isfinite(q) for q in positions)):
                raise ValueError(f"{name} must contain five finite numbers in radians")

    @classmethod
    def load(cls, path: str | None = None) -> "RobotConfig":
        path = path or os.environ.get("OMX_CONFIG")
        if not path:
            return cls()
        config_path = Path(path).expanduser().resolve()
        data = json.loads(config_path.read_text())
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        if "urdf_path" in data:
            urdf = Path(data["urdf_path"]).expanduser()
            data["urdf_path"] = str(urdf if urdf.is_absolute() else config_path.parent / urdf)
        return cls(**data)

    @property
    def robot_description(self) -> str:
        return f"""You control an OMX-F arm with five revolute joints and a gripper.
Kinematics use its URDF, base frame {self.base_link}, tool frame {self.end_effector_link}.
Cartesian coordinates are millimeters: base +X forward, +Y left, +Z up.
Angles exposed to you are degrees; ROS joint values internally are radians.
Tool pitch is joint2+joint3+joint4 (positive tilts down); tool roll is joint5.
This five-DOF arm cannot achieve arbitrary six-DOF poses.
The top camera shows the workspace. The wrist camera shows the gripper vicinity.
Use labeled, current images and measured joint feedback; dry-run images are synthetic.
Inspect get_robot_state before acting. Use small iterative moves and inspect feedback
after each move. Call only one motion tool per model turn. Do not infer calibrated
3D coordinates from pixels. Ask the user when the goal or geometry is uncertain.
MCP move_robot and move_cartesian translations must be at least {self.min_cartesian_step_mm:g} mm
in total requested distance; smaller translations are rejected, not rounded up.
Rotation-only commands are allowed; use get_robot_state to observe without moving.
Gripper openness: 0 percent closed, 100 percent open.
control_gripper accepts action="open" or action="close" only, commanding the
configured full-open/full-closed position. It sends without joint feedback,
goal acceptance, stall, or action-result checks, then returns camera images.
The response confirms sending only. Use the images to inspect the grasp.
The controller checks joint limits, speed, motion increments and tool height along
the joint trajectory for ordinary motion tools.
go_to_initial_position returns the arm to initial_joint_positions_rad from config,
saved from measured joints. go_to_post_grasp_position uses post_grasp_joint_positions_rad.
These two tools preserve the gripper, allow large moves, and check joint limits and
speed only, without FK/IK or tool-height checks. Use them with a clear swept area.
The controller has no obstacle or self-collision planner. Avoid obstacles.
Arm commands are JointTrajectory messages on {self.arm_trajectory_topic}.
Arm calls wait for the scheduled trajectory end and fresh measured joint positions.
Joint/tool tracking errors are not checked; success does not confirm target attainment.
Use stop_robot to request an arm hold and cancel gripper goals; it is a software stop,
not a hardware E-stop.
"""
