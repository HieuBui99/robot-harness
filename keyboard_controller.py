"""Terminal keyboard controller; works in a local terminal or over SSH."""

import argparse
import os
import queue
import select
import sys
import termios
import threading
import tty
from datetime import datetime
from pathlib import Path

from config import RobotConfig
from kinematics import finite_number
from robot_controller import RobotController

HELP = """OMX keyboard controls
  W/S       tool forward/backward ({step_mm:g} mm)
  Up/Down   tool up/down ({step_mm:g} mm in base +Z/-Z)
  Left/Right base yaw left/right (2 degrees)
  R/F       tool pitch up/down (2 degrees)
  A/D       tool roll CCW/CW (2 degrees)
  Q/E       gripper fully open/fully close
  1/2       init/home presets (full trajectories)
  C         save labeled top/wrist snapshots
  P         print measured robot state
  Space     cancel motion (restart controller to resume)
  Esc or X  cancel motion and exit
Commands are ignored while a command is running. Startup/shutdown do not move.
"""

ARROW_KEYS = {f"\x1b{prefix}{direction}": f"\x1b[{direction}"
              for prefix in ("[", "O") for direction in "ABCD"}


def read_key(stream=sys.stdin):
    # TextIO can prefetch an entire escape sequence, hiding its remaining bytes
    # from select(). Read the terminal fd directly so arrow keys stay intact.
    fd = stream.fileno()
    first = os.read(fd, 1).decode("ascii", errors="replace")
    if first != "\x1b":
        return first.lower()
    sequence = first
    while select.select([fd], [], [], 0.04)[0]:
        sequence += os.read(fd, 1).decode("ascii", errors="replace")
        if sequence in ARROW_KEYS:
            return ARROW_KEYS[sequence]
        if len(sequence) >= 8:
            break
    return sequence


class KeyboardController:
    def __init__(self, robot: RobotController, snapshots_dir="camera_snapshots", *, step_mm=2.):
        self.robot = robot
        self.step_mm = finite_number(step_mm, "step_mm")
        if not 0 < self.step_mm <= robot.config.max_cartesian_step_mm:
            raise ValueError("step_mm must be positive and no larger than max_cartesian_step_mm")
        self.snapshots_dir = Path(snapshots_dir)
        self.commands = queue.Queue(maxsize=1)
        self.busy = threading.Event()
        self.running = threading.Event()
        self.key_mappings = {
            "w": {"move_gripper_forward_mm": self.step_mm}, "s": {"move_gripper_forward_mm": -self.step_mm},
            "\x1b[A": {"move_gripper_up_mm": self.step_mm}, "\x1b[B": {"move_gripper_up_mm": -self.step_mm},
            "\x1b[D": {"rotate_robot_left_angle": 2.}, "\x1b[C": {"rotate_robot_left_angle": -2.},
            "r": {"tilt_gripper_down_angle": -2.}, "f": {"tilt_gripper_down_angle": 2.},
            "a": {"rotate_gripper_counterclockwise_angle": 2.}, "d": {"rotate_gripper_counterclockwise_angle": -2.},
        }

    def take_camera_snapshot(self):
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for metadata, image in self.robot.get_camera_images():
            path = self.snapshots_dir / f"{metadata['camera']}_{timestamp}.jpg"
            path.write_bytes(image)
            print(f"Saved {path}")

    def execute(self, key):
        key = ARROW_KEYS.get(key, key)
        if key in self.key_mappings:
            print(f"Command: {self.key_mappings[key]}", flush=True)
            result = self.robot.execute_intuitive_move(**self.key_mappings[key])
        elif key in {"q", "e"}:
            result = self.robot.control_gripper("open" if key == "q" else "close")
        elif key in {"1", "2"}:
            result = self.robot.apply_named_preset("init" if key == "1" else "home")
        elif key == "c":
            self.take_camera_snapshot()
            return
        elif key == "p":
            result = self.robot.get_current_robot_state()
        else:
            return
        print(result.to_json())

    def _worker(self):
        while self.running.is_set():
            try:
                key = self.commands.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self.running.is_set():
                    self.execute(key)
            except Exception as exc:
                print(f"Command failed: {exc}")
            finally:
                self.busy.clear()

    def run(self):
        if not sys.stdin.isatty():
            raise RuntimeError("Keyboard controller needs a terminal; use check_positions.py for diagnostics")
        print(HELP.format(step_mm=self.step_mm))
        print(self.robot.get_current_robot_state().to_json())
        settings = termios.tcgetattr(sys.stdin)
        self.running.set()
        worker = threading.Thread(target=self._worker, name="omx-keyboard-worker", daemon=True)
        worker.start()
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running.is_set():
                key = read_key()
                if key in {"\x1b", "x", ""}:
                    break
                if key == " ":
                    print(self.robot.stop().msg)
                elif not self.busy.is_set():
                    self.busy.set()
                    self.commands.put_nowait(key)
        finally:
            self.running.clear()
            self.robot.stop()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            worker.join(timeout=self.robot.config.observation_timeout_s + 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--snapshots-dir", default="camera_snapshots")
    parser.add_argument("--step-mm", type=float, default=2., help="Translation per key press in mm (default: 2)")
    args = parser.parse_args()
    try:
        with RobotController(RobotConfig.load(args.config), dry_run=args.dry_run) as robot:
            KeyboardController(robot, args.snapshots_dir, step_mm=args.step_mm).run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
