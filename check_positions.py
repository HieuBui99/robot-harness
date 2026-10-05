"""Read-only diagnostics for joints, URDF tool pose, and camera topics."""

import argparse
import json

from config import RobotConfig
from robot_controller import RobotController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--joints-only", action="store_true")
    args = parser.parse_args()
    with RobotController(RobotConfig.load(args.config), dry_run=args.dry_run, read_only=True) as robot:
        print(json.dumps(robot.get_current_robot_state().to_json(), indent=2))
        if not args.joints_only:
            images = robot.get_camera_images()
            print(json.dumps({"cameras": [metadata for metadata, _ in images]}, indent=2))


if __name__ == "__main__":
    main()
