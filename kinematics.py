"""URDF FK and bounded damped least-squares IK using pytorch_kinematics."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytorch_kinematics as pk
import torch

from config import RobotConfig


def finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


class KinematicsModel:
    def __init__(self, config: RobotConfig):
        self.config = config
        urdf = Path(config.urdf_path).read_text()
        root = ET.fromstring(urdf)
        # Only link/joint geometry is needed. Strip rendering/inertia elements
        # so permissive vendor collision tags do not pollute MCP stderr.
        for link in root.findall("link"):
            for child in list(link):
                link.remove(child)
        self.chain = pk.build_serial_chain_from_urdf(
            ET.tostring(root, encoding="unicode"), config.end_effector_link, config.base_link,
        ).to(dtype=torch.float64, device="cpu")
        if self.chain.get_joint_parameter_names() != config.joint_names:
            raise ValueError("URDF chain does not match the configured OMX arm joints")
        elements = {j.attrib["name"]: j for j in root.findall("joint")}
        self.limits = {}
        self.velocities = {}
        for name in [*config.joint_names, config.gripper_joint]:
            limit = elements[name].find("limit")
            bounds = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
            if name in config.joint_limit_overrides:
                requested = config.joint_limit_overrides[name]
                if len(requested) != 2:
                    raise ValueError(f"Invalid limit override for {name}")
                lower, upper = [finite_number(x, name) for x in requested]
                if not bounds[0] <= lower < upper <= bounds[1]:
                    raise ValueError(f"Limit override for {name} must narrow the URDF limits")
                bounds = (lower, upper)
            self.limits[name] = bounds
            self.velocities[name] = float(limit.attrib["velocity"])
        if set(config.joint_limit_overrides) - set(self.limits):
            raise ValueError("Unknown joint in joint_limit_overrides")
        self.lower = torch.tensor([self.limits[n][0] for n in config.joint_names], dtype=torch.float64)
        self.upper = torch.tensor([self.limits[n][1] for n in config.joint_names], dtype=torch.float64)
        for name, pose in config.presets.items():
            self.validate(pose)
        for q in (config.gripper_closed, config.gripper_open):
            self.validate_joint(config.gripper_joint, q)

    def validate_joint(self, name: str, q: float) -> None:
        q = finite_number(q, name)
        if name not in self.limits:
            raise ValueError(f"Unknown joint: {name}")
        low, high = self.limits[name]
        if not low <= q <= high:
            raise ValueError(f"{name}={q:.4f} rad outside [{low:.4f}, {high:.4f}]")

    def validate(self, joints: list[float]) -> None:
        if len(joints) != len(self.config.joint_names):
            raise ValueError("Expected five joint positions")
        for name, q in zip(self.config.joint_names, joints):
            self.validate_joint(name, q)

    def forward_kinematics(self, joints: list[float]) -> dict:
        self.validate(joints)
        with torch.no_grad():
            transform = self.chain.forward_kinematics(joints).get_matrix()[0]
            quaternion = pk.matrix_to_quaternion(transform[:3, :3]).tolist()
        return {
            "frame": self.config.base_link,
            "tool_frame": self.config.end_effector_link,
            "position_mm": dict(zip(("x", "y", "z"), (transform[:3, 3] * 1000).tolist())),
            "quaternion_wxyz": quaternion,
            "pitch_deg": math.degrees(sum(joints[1:4])),
            "roll_deg": math.degrees(joints[4]),
            "base_yaw_deg": math.degrees(joints[0]),
        }

    def inverse_kinematics(self, target_mm: list[float], seed: list[float],
                           pitch_rad: float | None = None, roll_rad: float | None = None) -> list[float]:
        """Solve position, optionally tool pitch and roll; do not impose an unattainable yaw.

        Jacobian rows 0:3 come from pytorch_kinematics. The two additional rows
        follow OMX's three parallel pitch axes and its final roll joint.
        """
        self.validate(seed)
        if len(target_mm) != 3:
            raise ValueError("Expected XYZ target")
        target = torch.tensor([finite_number(v, "target_mm") / 1000 for v in target_mm], dtype=torch.float64)
        if target[2] < self.config.min_tool_z_m:
            raise ValueError("Tool target is below min_tool_z_m")
        if pitch_rad is not None:
            finite_number(pitch_rad, "pitch_rad")
        if roll_rad is not None:
            finite_number(roll_rad, "roll_rad")
        # Seed locally to avoid a different elbow branch or whole revolutions.
        candidates = [seed, [seed[0], seed[1] - 0.15, seed[2] + 0.3, seed[3] - 0.15, seed[4]],
                      [seed[0], seed[1] + 0.15, seed[2] - 0.3, seed[3] + 0.15, seed[4]]]
        best_error = float("inf")
        solutions = []
        with torch.no_grad():
            for initial in candidates:
                q = torch.tensor(initial, dtype=torch.float64).clamp(self.lower, self.upper)
                for _ in range(180):
                    position = self.chain.forward_kinematics(q).get_matrix()[0, :3, 3]
                    error = target - position
                    position_error = error.norm().item()
                    best_error = min(best_error, position_error)
                    rows = self.chain.jacobian(q)[0, :3, :]
                    angle_errors = []
                    for angle, actual, row in (
                        (pitch_rad, q[1:4].sum(), [0., 1., 1., 1., 0.]),
                        (roll_rad, q[4], [0., 0., 0., 0., 1.]),
                    ):
                        if angle is not None:
                            angle_error = angle - actual
                            angle_errors.append(abs(angle_error.item()))
                            error = torch.cat((error, (angle_error * 0.15).reshape(1)))
                            rows = torch.cat((rows, torch.tensor([row], dtype=torch.float64) * 0.15))
                    if (position_error <= self.config.ik_position_tolerance_m and
                            all(e <= self.config.ik_angle_tolerance_rad for e in angle_errors)):
                        solutions.append(q.tolist())
                        break
                    damping = 0.005
                    step = rows.T @ torch.linalg.solve(
                        rows @ rows.T + damping ** 2 * torch.eye(rows.shape[0], dtype=torch.float64), error,
                    )
                    q = (q + step.clamp(-0.1, 0.1)).clamp(self.lower, self.upper)
        if not solutions:
            raise ValueError(f"IK did not converge within joint limits (best position error {best_error * 1000:.2f} mm)")
        return min(solutions, key=lambda sol: sum((a - b) ** 2 for a, b in zip(sol, seed)))

    def validate_path(self, start: list[float], target: list[float]) -> None:
        self.validate(start)
        self.validate(target)
        # Quintic joint trajectories have the same geometric path as linear interpolation.
        count = max(20, math.ceil(max(abs(b - a) for a, b in zip(start, target)) / 0.01))
        points = torch.linspace(0, 1, count + 1, dtype=torch.float64)[:, None]
        batch = torch.tensor(start, dtype=torch.float64) + points * (
            torch.tensor(target, dtype=torch.float64) - torch.tensor(start, dtype=torch.float64))
        with torch.no_grad():
            heights = self.chain.forward_kinematics(batch).get_matrix()[:, 2, 3]
        if heights.min().item() < self.config.min_tool_z_m:
            raise ValueError("Joint trajectory passes below min_tool_z_m")
