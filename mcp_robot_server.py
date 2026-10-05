"""OMX robot MCP server. Startup and shutdown never move the robot."""

import argparse
import asyncio
import base64
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent

from config import RobotConfig
from robot_controller import RobotController

_robot = None
_robot_lock = threading.Lock()
_dry_run = os.environ.get("OMX_DRY_RUN", "0") == "1"


def get_robot() -> RobotController:
    global _robot
    with _robot_lock:
        if _robot is None:
            _robot = RobotController(dry_run=_dry_run)
        return _robot


@asynccontextmanager
async def lifespan(server):
    global _robot
    try:
        yield {}
    finally:
        if _robot is not None:
            await asyncio.to_thread(_robot.disconnect)
            _robot = None


mcp = FastMCP("OMX robot controller", host=os.environ.get("MCP_HOST", "127.0.0.1"),
              port=int(os.environ.get("MCP_PORT", "3001")), lifespan=lifespan)


def text_block(value) -> TextContent:
    return TextContent(type="text", text=json.dumps(value, allow_nan=False))


def observation(operation=None, *, check_cameras_before=True, fault_on_camera_error=True) -> CallToolResult:
    """Return real image blocks, each immediately preceded by camera metadata."""
    robot = get_robot()
    # For vision-driven tools, check both cameras before any motion.
    if operation is not None and check_cameras_before:
        try:
            robot.get_camera_images()
        except Exception as exc:
            return CallToolResult(content=[text_block({"status": "error", "message": str(exc)})], isError=True)
    result = operation(robot) if operation else robot.get_current_robot_state()
    content = [text_block(result.to_json())]
    try:
        for metadata, data in robot.get_camera_images(result.completed_at, result.completed_stamp_ns):
            content.append(text_block(metadata))
            content.append(ImageContent(type="image", data=base64.b64encode(data).decode(), mimeType="image/jpeg"))
    except Exception as exc:
        content.append(text_block({"status": "error", "message": f"Camera observation failed: {exc}"}))
        if result.completed_at and fault_on_camera_error:
            robot.command_fault = True
        return CallToolResult(content=content, isError=True)
    return CallToolResult(content=content, isError=not result.ok)


async def run_observation(operation=None, **options) -> CallToolResult:
    try:
        return await asyncio.to_thread(observation, operation, **options)
    except Exception as exc:
        return CallToolResult(content=[text_block({"status": "error", "message": str(exc)})], isError=True)


@mcp.tool()
def get_initial_instructions() -> str:
    """Read the OMX robot description, units, frames, and operation instructions first."""
    return RobotConfig.load().robot_description


@mcp.tool()
async def get_robot_state() -> CallToolResult:
    """Get measured joints, URDF tool pose, and labeled top and wrist images."""
    return await run_observation()


@mcp.tool()
async def move_robot(move_gripper_up_mm: float = 0.0, move_gripper_forward_mm: float = 0.0,
                     tilt_gripper_down_angle: float = 0.0,
                     rotate_gripper_counterclockwise_angle: float = 0.0,
                     rotate_robot_left_angle: float = 0.0) -> CallToolResult:
    """Relative intuitive motion: up/forward in mm, pitch/roll/base yaw in degrees.

    Forward follows the arm's base yaw. Positive pitch tilts down; positive yaw
    rotates the robot left. Returns measured state and fresh post-motion images.
    Nonzero translations must meet configured min_cartesian_step_mm (default 2 mm).
    Rotation-only moves are allowed; use get_robot_state instead of a zero move.
    """
    return await run_observation(lambda robot: robot.execute_intuitive_move(
        move_gripper_up_mm, move_gripper_forward_mm, tilt_gripper_down_angle,
        rotate_gripper_counterclockwise_angle, rotate_robot_left_angle, enforce_minimum_step=True))


@mcp.tool()
async def move_cartesian(x_mm: float, y_mm: float, z_mm: float,
                         pitch_deg: float | None = None, roll_deg: float | None = None) -> CallToolResult:
    """Move the tool to absolute XYZ in the base frame, preserving pitch/roll if omitted.

    Small moves only. Optional pitch/roll are absolute degrees; yaw follows arm
    geometry. The path is a smooth joint trajectory, not a straight Cartesian line.
    Nonzero translations must meet configured min_cartesian_step_mm (default 2 mm).
    Orientation-only moves are allowed; use get_robot_state instead of a zero move.
    """
    return await run_observation(lambda robot: robot.move_cartesian(
        x_mm, y_mm, z_mm, pitch_deg, roll_deg, enforce_minimum_step=True))


@mcp.tool()
async def set_joint_positions(positions_deg: dict[str, float]) -> CallToolResult:
    """Set a subset of joint1..joint5 to absolute degrees; small changes only."""
    return await run_observation(lambda robot: robot.set_joints_absolute(positions_deg))


@mcp.tool()
async def go_to_initial_position() -> CallToolResult:
    """Return arm joints to initial_joint_positions_rad saved from /joint_states.

    Preserves the gripper. Direct joint trajectory without FK/IK or tool-height
    checks; large moves are allowed. Returns measured joints and fresh images.
    """
    return await run_observation(lambda robot: robot.go_to_initial_position())


@mcp.tool()
async def go_to_post_grasp_position() -> CallToolResult:
    """Move arm to configured post_grasp_joint_positions_rad (joint1..joint5).

    Errors if unconfigured. Preserves the gripper. Direct trajectory without
    FK/IK or tool-height checks; returns measured joints and fresh images.
    """
    return await run_observation(lambda robot: robot.go_to_post_grasp_position())


@mcp.tool()
async def control_gripper(action: Literal["open", "close"]) -> CallToolResult:
    """Command fully open or fully closed; partial opening is unavailable.

    Sends without checking joint feedback, goal acceptance, stalls, or the action
    result, then returns camera images. Does not confirm that the gripper reached
    the target or grasped an object. Camera errors do not latch a command fault.
    """
    return await run_observation(lambda robot: robot.control_gripper(action),
                                 check_cameras_before=False, fault_on_camera_error=False)


@mcp.tool()
async def apply_named_preset(name: str) -> CallToolResult:
    """Move to a configured preset (init/home), with speed and tool-height checks.

    Presets can traverse a large distance; only use when the full swept area is clear.
    """
    return await run_observation(lambda robot: robot.apply_named_preset(name))


@mcp.tool()
async def stop_robot() -> CallToolResult:
    """Request a measured-position arm hold and cancel gripper goals. Software stop."""
    try:
        result = await asyncio.to_thread(get_robot().stop)
        return CallToolResult(content=[text_block(result.to_json())])
    except Exception as exc:
        return CallToolResult(content=[TextContent(type="text", text=str(exc))], isError=True)


def main():
    global _dry_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--dry-run", action="store_true", default=_dry_run)
    parser.add_argument("--config", help="OMX JSON configuration file")
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "3001")))
    args = parser.parse_args()
    if args.config:
        os.environ["OMX_CONFIG"] = args.config
    _dry_run = args.dry_run
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
