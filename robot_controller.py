"""Shared OMX controller for MCP tools, diagnostics, and keyboard control."""

import io
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image

from config import RobotConfig
from kinematics import KinematicsModel, finite_number
from ros_transport import DryRunTransport, Ros2Transport


@dataclass
class MoveResult:
    ok: bool
    msg: str
    robot_state: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    completed_at: float = 0.0
    completed_stamp_ns: int = 0

    def to_json(self) -> dict:
        return {"status": "success" if self.ok else "error", "message": self.msg,
                "robot_state": self.robot_state, "warnings": self.warnings}


class MotionFeedbackError(RuntimeError):
    """A failed command with measured state retained for tracking diagnostics."""

    def __init__(self, message: str, robot_state: dict):
        super().__init__(message)
        self.robot_state = robot_state


class RobotController:
    def __init__(self, config: RobotConfig | None = None, *, dry_run: bool = False,
                 read_only: bool = False, transport=None):
        self.config = config or RobotConfig.load()
        self.kinematics = KinematicsModel(self.config)
        self.transport = transport or (DryRunTransport(self.config) if dry_run else Ros2Transport(self.config))
        self.read_only = read_only
        self.motion_lock = threading.Lock()
        self.command_fault = False
        self._initial_joint_positions_rad = list(self.config.initial_joint_positions_rad)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.disconnect()

    def _state(self, measured: dict, *, include_tool_pose: bool = True) -> dict:
        joints = [measured[n] for n in self.config.joint_names]
        grip = measured[self.config.gripper_joint]
        self.kinematics.validate_joint(self.config.gripper_joint, grip)
        pct = (grip - self.config.gripper_closed) / (self.config.gripper_open - self.config.gripper_closed) * 100
        state = {
            "dry_run": self.transport.dry_run,
            "joints_rad": measured,
            "joints_deg": {n: math.degrees(measured[n]) for n in self.config.joint_names},
            "gripper_openness_pct": pct,
            "joint_limits_rad": self.kinematics.limits,
            "command_fault": self.command_fault,
        }
        if include_tool_pose:
            state["tool_pose"] = self.kinematics.forward_kinematics(joints)
        return state

    def get_current_robot_state(self) -> MoveResult:
        return MoveResult(True, "Measured robot state", self._state(self.transport.get_joints()))

    def _check_motion(self):
        if self.read_only:
            raise ValueError("Controller is read-only")
        if self.command_fault:
            raise ValueError("Motion or feedback failed; restart the controller before commanding more motion")

    def _arm_duration(self, start, target):
        velocities = [min(self.config.max_velocity_rad_s, self.kinematics.velocities[n])
                      for n in self.config.joint_names]
        delta = max(abs(b - a) for a, b in zip(start, target))
        # Quintic interpolation: maximum normalized speed 1.875 and acceleration 5.774.
        return max(self.config.min_duration_s,
                   max(1.875 * abs(b - a) / v for a, b, v in zip(start, target, velocities)),
                   math.sqrt(5.774 * delta / self.config.max_acceleration_rad_s2))

    def _execute_arm(self, start: list[float], target: list[float], *, allow_large: bool = False,
                     tool_target: dict | None = None) -> MoveResult:
        self._check_motion()
        self.kinematics.validate_path(start, target)
        delta = max(abs(b - a) for a, b in zip(start, target))
        if not allow_large and delta > self.config.max_joint_step_rad:
            raise ValueError("Joint change exceeds max_joint_step_rad; use smaller moves")
        duration = self._arm_duration(start, target)
        start_pose = self.kinematics.forward_kinematics(start)
        if tool_target is not None:
            command_pose = self.kinematics.forward_kinematics(target)
            if math.dist(list(start_pose["position_mm"].values()),
                         list(command_pose["position_mm"].values())) > self.config.max_cartesian_step_mm:
                raise ValueError("Command exceeds max_cartesian_step_mm")
            if max(abs(command_pose[n] - start_pose[n]) for n in ("pitch_deg", "roll_deg")) > self.config.max_rotation_step_deg:
                raise ValueError("Command exceeds max_rotation_step_deg")
        try:
            self.transport.move_arm(start, target, duration)
            completed = time.monotonic()
            stamp = self.transport.now_ns()
            measured = self.transport.get_joints(after=completed)
            self._check_motion()
            state = self._state(measured)
            message = "Arm trajectory elapsed; fresh feedback received"
            if tool_target is not None:
                changes = [state["tool_pose"]["position_mm"][axis] - start_pose["position_mm"][axis]
                           for axis in ("x", "y", "z")]
                message += f"; measured tool delta XYZ {[round(v, 2) for v in changes]} mm"
            return MoveResult(True, message, state, completed_at=completed, completed_stamp_ns=stamp)
        except BaseException as exc:
            self.command_fault = True
            state = {}
            if isinstance(exc, (ValueError, RuntimeError, TimeoutError)):
                try:
                    measured = self.transport.get_joints()
                    state = self._state(measured)
                    actual_point = state["tool_pose"]["position_mm"]
                    start_point = start_pose["position_mm"]
                    requested_point = (tool_target or self.kinematics.forward_kinematics(target))["position_mm"]
                    state["motion_diagnostics"] = {
                        "start_xyz_mm": start_point,
                        "requested_xyz_mm": requested_point,
                        "requested_delta_xyz_mm": {axis: requested_point[axis] - start_point[axis] for axis in "xyz"},
                        "measured_delta_xyz_mm": {axis: actual_point[axis] - start_point[axis] for axis in "xyz"},
                        "commanded_joints_deg": {name: math.degrees(q)
                                                 for name, q in zip(self.config.joint_names, target)},
                        "joint_errors_deg": {name: math.degrees(q - measured[name])
                                             for name, q in zip(self.config.joint_names, target)},
                        "trajectory_duration_s": duration,
                    }
                except (ValueError, RuntimeError, TimeoutError):
                    # Stale/missing feedback must not be presented as measured state.
                    pass
            self.transport.stop()
            if isinstance(exc, (ValueError, RuntimeError, TimeoutError)):
                raise MotionFeedbackError(str(exc), state) from exc
            raise

    def _guarded(self, operation) -> MoveResult:
        if not self.motion_lock.acquire(blocking=False):
            return MoveResult(False, "Another motion is in progress")
        try:
            self._check_motion()
            return operation()
        except (ValueError, RuntimeError, TimeoutError) as exc:
            return MoveResult(False, str(exc), getattr(exc, "robot_state", {}))
        finally:
            self.motion_lock.release()

    def set_joints_absolute(self, positions_deg: dict[str, float]) -> MoveResult:
        def operation():
            if not positions_deg or set(positions_deg) - set(self.config.joint_names):
                raise ValueError("Supply one or more of joint1 through joint5 in degrees")
            current = self.transport.get_joints()
            start = [current[n] for n in self.config.joint_names]
            target = [math.radians(finite_number(positions_deg[n], n)) if n in positions_deg else current[n]
                      for n in self.config.joint_names]
            return self._execute_arm(start, target)
        return self._guarded(operation)

    def _execute_joint_target(self, positions_rad: list[float]) -> MoveResult:
        """Execute a complete arm target using joint limits only, without FK/IK."""
        self._check_motion()
        measured = self.transport.get_joints()
        start = [measured[n] for n in self.config.joint_names]
        if len(positions_rad) != len(self.config.joint_names):
            raise ValueError("Expected five joint positions in radians")
        target = [finite_number(q, n) for n, q in zip(self.config.joint_names, positions_rad)]
        self.kinematics.validate(start)  # Numeric URDF bounds; no transforms.
        self.kinematics.validate(target)
        duration = self._arm_duration(start, target)
        try:
            self.transport.move_arm(start, target, duration)
            completed = time.monotonic()
            stamp = self.transport.now_ns()
            measured = self.transport.get_joints(after=completed)
            self._check_motion()
            return MoveResult(True, "Arm trajectory elapsed; fresh feedback received",
                              self._state(measured, include_tool_pose=False),
                              completed_at=completed, completed_stamp_ns=stamp)
        except BaseException as exc:
            self.command_fault = True
            state = {}
            if isinstance(exc, (ValueError, RuntimeError, TimeoutError)):
                try:
                    state = self._state(self.transport.get_joints(), include_tool_pose=False)
                except (ValueError, RuntimeError, TimeoutError):
                    pass
            self.transport.stop()
            if isinstance(exc, (ValueError, RuntimeError, TimeoutError)):
                raise MotionFeedbackError(str(exc), state) from exc
            raise

    def go_to_initial_position(self) -> MoveResult:
        """Return arm joints to the saved initial target from configuration."""
        return self._guarded(lambda: self._execute_joint_target(self._initial_joint_positions_rad))

    def go_to_post_grasp_position(self) -> MoveResult:
        """Move the arm to the configured post-grasp joints, preserving the gripper."""
        def operation():
            target = self.config.post_grasp_joint_positions_rad
            if target is None:
                raise ValueError("Set post_grasp_joint_positions_rad in OMX_CONFIG before using this tool")
            return self._execute_joint_target(target)
        return self._guarded(operation)

    def move_cartesian(self, x_mm: float, y_mm: float, z_mm: float,
                       pitch_deg: float | None = None, roll_deg: float | None = None, *,
                       enforce_minimum_step: bool = False) -> MoveResult:
        def operation():
            measured = self.transport.get_joints()
            seed = [measured[n] for n in self.config.joint_names]
            pose = self.kinematics.forward_kinematics(seed)
            target = [finite_number(v, name) for v, name in zip((x_mm, y_mm, z_mm), ("x", "y", "z"))]
            old = list(pose["position_mm"].values())
            if math.dist(old, target) > self.config.max_cartesian_step_mm:
                raise ValueError("Cartesian change exceeds max_cartesian_step_mm; use smaller moves")
            pitch = pose["pitch_deg"] if pitch_deg is None else finite_number(pitch_deg, "pitch_deg")
            roll = pose["roll_deg"] if roll_deg is None else finite_number(roll_deg, "roll_deg")
            if enforce_minimum_step:
                self._check_minimum_translation(math.dist(old, target),
                                                has_rotation=pitch != pose["pitch_deg"] or roll != pose["roll_deg"])
            if max(abs(pitch - pose["pitch_deg"]), abs(roll - pose["roll_deg"])) > self.config.max_rotation_step_deg:
                raise ValueError("Orientation change exceeds max_rotation_step_deg")
            joints = self.kinematics.inverse_kinematics(target, seed, math.radians(pitch), math.radians(roll))
            return self._execute_arm(seed, joints, tool_target={
                "position_mm": dict(zip(("x", "y", "z"), target)), "pitch_deg": pitch, "roll_deg": roll,
            })
        return self._guarded(operation)

    def _check_minimum_translation(self, distance_mm: float, *, has_rotation: bool) -> None:
        if distance_mm == 0 and has_rotation:
            return
        if distance_mm < self.config.min_cartesian_step_mm:
            raise ValueError(f"Requested translation {distance_mm:g} mm is below min_cartesian_step_mm "
                             f"({self.config.min_cartesian_step_mm:g} mm); use a larger move "
                             "or get_robot_state to observe without moving")

    def execute_intuitive_move(self, move_gripper_up_mm: float = 0.0,
                               move_gripper_forward_mm: float = 0.0,
                               tilt_gripper_down_angle: float = 0.0,
                               rotate_gripper_counterclockwise_angle: float = 0.0,
                               rotate_robot_left_angle: float = 0.0, *,
                               enforce_minimum_step: bool = False) -> MoveResult:
        """Original SO101 controls, with OMX tool-tip IK and base rotation."""
        def operation():
            up, forward, tilt, roll, yaw = [finite_number(v, n) for v, n in zip(
                (move_gripper_up_mm, move_gripper_forward_mm, tilt_gripper_down_angle,
                 rotate_gripper_counterclockwise_angle, rotate_robot_left_angle),
                ("up", "forward", "tilt", "roll", "yaw"))]
            if enforce_minimum_step:
                self._check_minimum_translation(math.hypot(up, forward), has_rotation=bool(tilt or roll or yaw))
            if math.hypot(up, forward) > self.config.max_cartesian_step_mm:
                raise ValueError("Cartesian change exceeds max_cartesian_step_mm")
            if max(abs(tilt), abs(roll), abs(yaw)) > self.config.max_rotation_step_deg:
                raise ValueError("Rotation exceeds max_rotation_step_deg")
            measured = self.transport.get_joints()
            start = [measured[n] for n in self.config.joint_names]
            rotated = list(start)
            rotated[0] += math.radians(yaw)
            pose = self.kinematics.forward_kinematics(rotated)
            point = pose["position_mm"]
            target = [point["x"] + forward * math.cos(rotated[0]),
                      point["y"] + forward * math.sin(rotated[0]), point["z"] + up]
            if up or forward or tilt or roll:
                rotated = self.kinematics.inverse_kinematics(
                    target, rotated, sum(start[1:4]) + math.radians(tilt), start[4] + math.radians(roll))
            return self._execute_arm(start, rotated, tool_target={
                "position_mm": dict(zip(("x", "y", "z"), target)),
                "pitch_deg": math.degrees(sum(start[1:4])) + tilt,
                "roll_deg": math.degrees(start[4]) + roll,
            } if up or forward or tilt or roll else None)
        return self._guarded(operation)

    def control_gripper(self, action: Literal["open", "close"]) -> MoveResult:
        """Send a calibrated open/close command without waiting for feedback."""
        def operation():
            if action not in ("open", "close"):
                raise ValueError("Gripper action must be 'open' or 'close'")
            q = self.config.gripper_open if action == "open" else self.config.gripper_closed
            self.transport.move_gripper(q)
            return MoveResult(True, f"Gripper {action} command sent",
                              completed_at=time.monotonic(), completed_stamp_ns=self.transport.now_ns())
        return self._guarded(operation)

    def apply_named_preset(self, name: str) -> MoveResult:
        def operation():
            if name not in self.config.presets:
                raise ValueError(f"Unknown preset; available: {list(self.config.presets)}")
            measured = self.transport.get_joints()
            start = [measured[n] for n in self.config.joint_names]
            # Presets are deliberate complete trajectories, still speed/path checked.
            return self._execute_arm(start, self.config.presets[name], allow_large=True)
        return self._guarded(operation)

    def get_camera_images(self, after: float = 0.0, stamp_after_ns: int = 0):
        frames = self.transport.get_frames(after=after, stamp_after_ns=stamp_after_ns)
        pictures = []
        for name in self.config.camera_topics:
            frame = frames[name]
            with Image.open(io.BytesIO(frame.data)) as image:
                image = image.convert("RGB")
                image.thumbnail((self.config.image_max_size, self.config.image_max_size))
                encoded = io.BytesIO()
                image.save(encoded, format="JPEG", quality=85)
            pictures.append(({
                "camera": name, "topic": self.config.camera_topics[name], "frame_id": frame.frame_id,
                "stamp_ns": frame.stamp_ns, "age_s": round(time.monotonic() - frame.received_at, 3),
                "source": frame.source, "timestamp_available": bool(frame.stamp_ns),
            }, encoded.getvalue()))
        return pictures

    def stop(self) -> MoveResult:
        self.command_fault = True
        self.transport.stop()
        return MoveResult(True, "Arm hold and gripper cancellation requested; restart the controller to resume motion")

    def disconnect(self) -> None:
        self.transport.close()
