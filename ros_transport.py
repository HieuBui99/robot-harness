"""ROS 2 trajectory-topic/gripper-action transport and hardware-free simulation."""

import io
import math
import threading
import time
from dataclasses import dataclass

from PIL import Image, ImageDraw

from config import RobotConfig


@dataclass(frozen=True)
class CameraFrame:
    data: bytes
    received_at: float
    stamp_ns: int
    frame_id: str
    source: str = "ros2"


def trajectory_samples(start: list[float], target: list[float], duration: float):
    """Quintic interpolation, zero velocity and acceleration at both endpoints."""
    count = max(2, math.ceil(duration / 0.05))
    for index in range(count + 1):
        u = index / count
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        ds = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / duration
        dds = (60 * u - 180 * u ** 2 + 120 * u ** 3) / duration ** 2
        delta = [b - a for a, b in zip(start, target)]
        yield (duration * u, [a + d * s for a, d in zip(start, delta)],
               [d * ds for d in delta], [d * dds for d in delta])


class DryRunTransport:
    """Explicit simulation: FK is real, feedback/images are synthetic."""
    dry_run = True

    def __init__(self, config: RobotConfig):
        self.config = config
        self.positions = dict(zip(config.joint_names, config.presets["init"]))
        self.positions[config.gripper_joint] = config.gripper_closed
        self.lock = threading.Lock()
        self.closed = False

    def get_joints(self, after: float = 0.0) -> dict[str, float]:
        with self.lock:
            if self.closed:
                raise RuntimeError("Transport is closed")
            return dict(self.positions)

    def move_arm(self, start: list[float], target: list[float], duration: float) -> None:
        with self.lock:
            self.positions.update(zip(self.config.joint_names, target))

    def move_gripper(self, position: float) -> None:
        with self.lock:
            self.positions[self.config.gripper_joint] = position

    def now_ns(self) -> int:
        return time.time_ns()

    def get_frames(self, after: float = 0.0, stamp_after_ns: int = 0) -> dict[str, CameraFrame]:
        result = {}
        for name in self.config.camera_topics:
            picture = Image.new("RGB", (640, 360), (35, 40, 48))
            ImageDraw.Draw(picture).text((20, 20), f"DRY RUN / {name}\nSynthetic image. No robot or camera connected.", fill="white")
            buffer = io.BytesIO()
            picture.save(buffer, format="JPEG")
            result[name] = CameraFrame(buffer.getvalue(), time.monotonic(), self.now_ns(), name, "synthetic")
        return result

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class Ros2Transport:
    dry_run = False

    def __init__(self, config: RobotConfig):
        # ROS must come from the sourced distribution, not pip.
        import rclpy
        from control_msgs.action import GripperCommand
        from rclpy.action import ActionClient
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, JointState
        from trajectory_msgs.msg import JointTrajectory

        self.config = config
        self.condition = threading.Condition()
        self.positions: dict[str, tuple[float, float]] = {}
        self.frames: dict[str, CameraFrame] = {}
        # ClientGoalHandle defines equality but has no hash in ROS Jazzy.
        self.active = {}
        self.arm_motion_active = False
        self.stop_epoch = 0
        self.fault = False
        self.closed = False
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node("omx_robot_harness", context=self.context)
        self.node.create_subscription(JointState, config.joint_states_topic, self._on_joints, qos_profile_sensor_data)
        for name, topic in config.camera_topics.items():
            self.node.create_subscription(CompressedImage, topic,
                                          lambda msg, name=name: self._on_camera(name, msg), qos_profile_sensor_data)
        self.arm_publisher = self.node.create_publisher(JointTrajectory, config.arm_trajectory_topic, 10)
        self.gripper = ActionClient(self.node, GripperCommand, config.gripper_action)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, name="omx-ros-spin", daemon=True)
        self.thread.start()

    def _on_joints(self, msg):
        now = time.monotonic()
        with self.condition:
            for name, value in zip(msg.name, msg.position):
                if math.isfinite(value):
                    self.positions[name] = (float(value), now)
                else:
                    self.positions.pop(name, None)
            self.condition.notify_all()

    def _on_camera(self, name, msg):
        frame = CameraFrame(bytes(msg.data), time.monotonic(),
                            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec,
                            msg.header.frame_id)
        with self.condition:
            self.frames[name] = frame
            self.condition.notify_all()

    def now_ns(self) -> int:
        return self.node.get_clock().now().nanoseconds

    def get_joints(self, after: float = 0.0) -> dict[str, float]:
        names = [*self.config.joint_names, self.config.gripper_joint]
        deadline = time.monotonic() + self.config.observation_timeout_s
        with self.condition:
            while True:
                now = time.monotonic()
                missing = [name for name in names if name not in self.positions or
                           now - self.positions[name][1] > self.config.state_max_age_s or
                           self.positions[name][1] <= after]
                if not missing:
                    return {name: self.positions[name][0] for name in names}
                if self.closed or now >= deadline:
                    raise RuntimeError(f"Missing/stale joint feedback on {self.config.joint_states_topic}: {missing}")
                self.condition.wait(min(0.05, deadline - now))

    def get_frames(self, after: float = 0.0, stamp_after_ns: int = 0) -> dict[str, CameraFrame]:
        deadline = time.monotonic() + self.config.observation_timeout_s
        with self.condition:
            while True:
                now = time.monotonic()
                missing = [name for name in self.config.camera_topics if name not in self.frames or
                           now - self.frames[name].received_at > self.config.camera_max_age_s or
                           self.frames[name].received_at <= after or
                           (self.frames[name].stamp_ns and self.frames[name].stamp_ns <= stamp_after_ns)]
                if not missing:
                    return dict(self.frames)
                if self.closed or now >= deadline:
                    raise RuntimeError(f"Missing/stale camera frames: {missing}")
                self.condition.wait(min(0.05, deadline - now))

    def _execute_gripper(self, goal):
        """Send only; callbacks retain handles for software stop, not validation."""
        with self.condition:
            self._check_available()
            epoch = self.stop_epoch
            accepted = self.gripper.send_goal_async(goal)

        def track_or_cancel(future):
            if future.exception() is not None:
                return
            handle = future.result()
            if not handle.accepted:
                return
            with self.condition:
                cancel = self.fault or self.closed or self.stop_epoch != epoch
                if not cancel:
                    self.active[id(handle)] = handle
            if cancel:
                handle.cancel_goal_async()
                return

            def forget_result(_):
                with self.condition:
                    self.active.pop(id(handle), None)

            handle.get_result_async().add_done_callback(forget_result)

        accepted.add_done_callback(track_or_cancel)

    def move_arm(self, start: list[float], target: list[float], duration: float) -> None:
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        trajectory = JointTrajectory()
        trajectory.joint_names = self.config.joint_names
        for t, positions, velocities, accelerations in trajectory_samples(start, target, duration):
            ns = round(t * 1_000_000_000)
            trajectory.points.append(JointTrajectoryPoint(
                positions=positions, velocities=velocities, accelerations=accelerations,
                time_from_start=Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000),
            ))
        with self.condition:
            deadline = time.monotonic() + self.config.observation_timeout_s
            while not self.arm_publisher.get_subscription_count():
                self._check_available()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"No subscriber on {self.config.arm_trajectory_topic}; check robot bringup")
                self.condition.wait(min(0.05, remaining))
            self._check_available()
            # Serialize publish with stop so a late command cannot replace a hold.
            self.arm_motion_active = True
            published_at = time.monotonic()
            self.arm_publisher.publish(trajectory)
        try:
            self._wait_for_arm_feedback(published_at, duration)
        except BaseException:
            self.stop()
            raise

    def _check_available(self):
        if self.fault or self.closed:
            raise RuntimeError("Transport fault/closed; restart the controller before commanding motion")

    def _wait_for_arm_feedback(self, published_at: float, duration: float):
        """Wait for the scheduled end and three fresh samples, without tracking checks."""
        finish_at = published_at + duration
        deadline = finish_at + self.config.action_timeout_s
        last_sample = 0.0
        fresh_samples = 0
        with self.condition:
            while True:
                self._check_available()
                now = time.monotonic()
                missing = [name for name in self.config.joint_names if name not in self.positions or
                           now - self.positions[name][1] > self.config.state_max_age_s]
                if missing:
                    raise RuntimeError(f"Missing/stale arm feedback during trajectory: {missing}")
                sample_stamp = min(self.positions[name][1] for name in self.config.joint_names)
                if sample_stamp > finish_at and sample_stamp > last_sample:
                    last_sample = sample_stamp
                    fresh_samples += 1
                    if fresh_samples >= 3:
                        self.arm_motion_active = False
                        return
                if now >= deadline:
                    raise TimeoutError("Fresh arm feedback not received after trajectory before timeout")
                self.condition.wait(min(0.05, deadline - now))

    def move_gripper(self, position: float) -> None:
        from control_msgs.action import GripperCommand
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = self.config.gripper_max_effort
        self._execute_gripper(goal)

    def stop(self) -> None:
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        hold_error = None
        with self.condition:
            self.fault = True
            self.stop_epoch += 1
            active = list(self.active.values())
            if self.arm_motion_active:
                now = time.monotonic()
                missing = [name for name in self.config.joint_names if name not in self.positions or
                           now - self.positions[name][1] > self.config.state_max_age_s]
                if missing:
                    hold_error = f"Arm hold not sent: missing/stale measured positions for {missing}"
                else:
                    # A valid single-point trajectory replaces the running command.
                    hold = JointTrajectory(joint_names=self.config.joint_names, points=[JointTrajectoryPoint(
                        positions=[self.positions[name][0] for name in self.config.joint_names],
                        velocities=[0.0] * len(self.config.joint_names),
                        time_from_start=Duration(nanosec=100_000_000),
                    )])
                    self.arm_publisher.publish(hold)
                    self.arm_motion_active = False
            self.condition.notify_all()
        for handle in active:
            handle.cancel_goal_async()
        if hold_error:
            raise RuntimeError(hold_error)

    def close(self) -> None:
        with self.condition:
            if self.closed:
                return
            self.closed = True
            self.condition.notify_all()
        try:
            self.stop()
        finally:
            self.executor.shutdown(timeout_sec=2.0)
            self.thread.join(timeout=2.0)
            self.node.destroy_publisher(self.arm_publisher)
            self.gripper.destroy()
            self.node.destroy_node()
            self.context.try_shutdown()
