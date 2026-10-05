# OMX robot control with MCP

This adapts the setup in `../robot_MCP` for the OMX-F arm: a ROS 2 robot
controller, keyboard teleoperation, an MCP server, and an interactive VLM agent.
Like the local `omx_f_teleop.py`, arm commands publish `JointTrajectory` messages
on `/arm_controller/joint_trajectory`; the gripper uses its separate action.
The existing local OpenAI-compatible and native Gemini adapters are retained.
There is no LeRobot/serial-servo dependency. Kinematics use the supplied OMX URDF
with `pytorch_kinematics`.

The agent sends tool schemas and the top/wrist images to your model, executes a
function call through MCP, waits for the trajectory duration and fresh feedback, then
sends fresh images back for the next decision.

## Setup

Use the Python version belonging to your ROS distribution (Python 3.12 for the
installed ROS 2 Jazzy). Source ROS before starting the application:

```bash
cd /home/hieubhm2/workspace/robot-harness
source /opt/ros/jazzy/setup.bash
# Replace with the workspace where open_manipulator was built.
# The repository's Docker setup uses /root/ros2_ws.
source /path/to/ros_ws/install/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
# Kinematics run on CPU; install this first to avoid large CUDA downloads.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For zsh, source `setup.zsh` instead of `setup.bash`. `rclpy`, `sensor_msgs`,
`control_msgs`, `trajectory_msgs`, `builtin_interfaces`, and `action_msgs` come
from ROS, not pip. The application connects to an already running OMX bringup;
it does not start robot hardware or cameras.

Use the standard OMX-F bringup, which starts both the arm trajectory controller
and the gripper action controller:

```bash
ros2 launch open_manipulator_bringup omx_f.launch.py \
  port_name:=/dev/ttyACM0 init_position:=false
```

Change `port_name` to your robot's serial device. `init_position:=false` disables
the launch file's automatic initialization trajectory. The follower-AI launch
does not start the separate gripper controller required by this application.

If the cameras are not already running, start them in another sourced terminal:

```bash
ros2 launch open_manipulator_bringup camera_usb_cam_dual.launch.py \
  camera1_device:=/dev/video0 camera2_device:=/dev/video2
```

Those defaults name the cameras `top_camera` and `wrist_camera`; change the video
devices to match your cameras.

Defaults match the local OMX controller configuration:

| Interface | ROS name |
| --- | --- |
| Joint feedback | `/joint_states` |
| Arm command topic (`JointTrajectory`) | `/arm_controller/joint_trajectory` |
| Gripper action (`GripperCommand`) | `/gripper_controller/gripper_cmd` |
| Top camera (`CompressedImage`) | `/top_camera/image_raw/compressed` |
| Wrist camera (`CompressedImage`) | `/wrist_camera/image_raw/compressed` |

Copy `omx_config.example.json` to `omx_config.json` if your names, limits, or
gripper calibration differ. Pass `--config omx_config.json` or set `OMX_CONFIG`.
The gripper defaults are the local SRDF's revolute joint positions: closed=0 rad,
open=1 rad. Only `gripper_joint_1` is commanded; the hardware handles its mimic
joint. An optional `urdf_path` can point to your original generated URDF file.

## Check the setup

```bash
python check_positions.py --dry-run
python keyboard_controller.py --dry-run
```

Dry-run uses real URDF kinematics, synthetic joint feedback, and clearly labeled
synthetic camera images. It never creates ROS nodes or commands hardware.

With OMX bringup and both cameras running, inspect feedback without commanding
motion:

```bash
python check_positions.py
```

The result includes all joints, the gripper's openness, the tool pose, and camera
topic/frame/timestamp information. Use `--joints-only` to diagnose joint feedback
independently of cameras.

## Keyboard controller

```bash
python keyboard_controller.py
```

Up increases the tool's Z coordinate in the robot base frame while preserving
pitch and roll. The default translation is only 2 mm per press. For a larger
step, use `python keyboard_controller.py --step-mm 5`. Motion results print the
measured XYZ displacement in millimeters; Up should give a positive Z change.
Both normal and application-mode terminal arrow sequences are supported.
Failed arm motions retain fresh measured state and `motion_diagnostics`: starting
XYZ, requested/measured displacement, commanded joint angles, and joint errors
(commanded minus measured, in degrees). This distinguishes no movement from a
partial move and identifies which joints missed their targets. If feedback is
unavailable, the error leaves the measured state empty.
Arm commands do not check measured joint/tool tracking errors and do not send
feedback correction retries. Each call sends one trajectory, waits for its
scheduled end and fresh joint feedback, and reports the measured displacement.
Success confirms that this wait finished; it does not confirm that the requested
target was reached. Missing/stale feedback and stop requests still fail the call.
The legacy arm tracking tolerance/correction fields remain accepted in existing
JSON configs but no longer affect arm commands. `joint_tolerance_rad` is also
accepted for compatibility and no longer affects gripper commands.

It runs in a terminal, including over SSH, and keeps the original key mappings:

| Key | Command |
| --- | --- |
| W / S | Tool forward / backward, 2 mm |
| Up / Down | Tool up / down, 2 mm |
| Left / Right | Base yaw left / right, 2 degrees |
| R / F | Tool pitch up / down, 2 degrees |
| A / D | Tool roll counterclockwise / clockwise, 2 degrees |
| Q / E | Fully open / fully close gripper |
| 1 / 2 | Local OMX `init` / `home` presets |
| C | Save both camera snapshots |
| P | Print measured state |
| Space | Cancel motion; restart the controller to resume |
| Esc / X | Cancel motion and exit |

Presets are copied from OMX's SRDF and can cover a large distance. Commands are
dropped while another command is running, so key repeats do not queue a series
of movements. Startup and shutdown never return the arm to a preset.

## Local VLM agent

Use a model/server supporting **both image input and function calling** through
OpenAI-compatible Chat Completions. The server's model ID must be explicit.

```bash
cp .env.example .env
# Edit LLM_MODEL and OPENAI_BASE_URL for your local server.
python agent.py --dry-run
```

Or pass the settings directly:

```bash
python agent.py --dry-run --provider openai --model YOUR_MODEL_ID \
  --base-url http://127.0.0.1:8000/v1 --api-key local-unused
```

For the real robot, remove `--dry-run`. The agent launches its own MCP subprocess
over stdio by default. `--show-images` opens an optional Tk camera viewer;
`--prompt 'Inspect the robot and describe what you see'` runs a single task.
`--max-steps` bounds the number of model turns for a task (default 20).

Interactive commands: `/state`, `/stop`, `/reset` (clear conversation), `/quit`.
`/reset` does not clear controller faults; restart the MCP server after a fault
or software stop. Use the keyboard controller in a separate mode, not
concurrently with an agent commanding the same physical robot.

## Gemini / Gemini Robotics ER

Native Gemini uses `google-genai`, retaining the model's original function calls
and thought signatures. Camera images are sent as native inline image parts.

```bash
# Set GEMINI_API_KEY in .env, then:
python agent.py --dry-run --provider gemini --model gemini-robotics-er-2-preview
```

The [Gemini Robotics documentation](https://ai.google.dev/gemini-api/docs/robotics-overview)
lists `gemini-robotics-er-2-preview` as a vision/reasoning model with function
calling. It orchestrates these robot tools; this app supplies the OMX execution
layer. Google's [pricing page](https://ai.google.dev/gemini-api/docs/pricing)
currently lists a free tier for that model; availability and quotas depend on
your account and region. You can select another available Gemini model with
vision and function-calling support using `--model`.

## Standalone MCP server

```bash
python mcp_robot_server.py --transport streamable-http --port 3001 --dry-run
python agent.py --provider openai --model YOUR_MODEL_ID \
  --mcp-url http://127.0.0.1:3001/mcp
```

Configure dry-run and robot config on the standalone server. For SSE use
`--transport sse` on both commands and URL `http://127.0.0.1:3001/sse`.
The server binds to localhost by default. Stdio clients can use:

```json
{
  "mcpServers": {
    "omx": {
      "command": "/home/hieubhm2/workspace/robot-harness/.venv/bin/python",
      "args": ["/home/hieubhm2/workspace/robot-harness/mcp_robot_server.py"],
      "env": {"OMX_CONFIG": "/home/hieubhm2/workspace/robot-harness/omx_config.json"}
    }
  }
}
```

Launch MCP clients from the sourced ROS environment. Omit `OMX_CONFIG` when
using the defaults. Available tools are `get_initial_instructions`,
`get_robot_state`, `move_robot`, `move_cartesian`, `set_joint_positions`,
`go_to_initial_position`, `go_to_post_grasp_position`, `control_gripper`,
`apply_named_preset`, and `stop_robot`.

`control_gripper` accepts `{"action": "open"}` or `{"action": "close"}` to
command the configured full-open or full-closed position. Measured openness is
still reported as a percentage by `get_robot_state`. The gripper tool sends the
command without checking joint feedback, goal acceptance, stall flags, or the
action result, then returns camera images received after sending. Object contact
does not latch a command fault. The response confirms only that the command was
sent; use the images to inspect the grasp. Camera errors do not latch a command
fault. Read-only mode and software
stop still block new commands, and stop cancels accepted gripper goals.

`go_to_initial_position` returns to `initial_joint_positions_rad` from config.
The defaults and example config contain the live `/joint_states` arm pose saved
on 2026-10-05. The saved pose persists across server restarts; edit those five
radian values and restart the server to change it. This is separate from the
SRDF `init` preset.

`go_to_post_grasp_position` uses `post_grasp_joint_positions_rad` in your
`OMX_CONFIG` JSON file. The defaults and example config contain the live
`/joint_states` post-grasp arm pose saved on 2026-10-05. Edit the five radian
values in `joint1`, `joint2`, `joint3`, `joint4`, `joint5` order and restart the
server to change it. Set it to `null` to disable the tool; it then returns an
error without moving.

Both tools move only the arm, preserving the gripper so a grasped object stays
held. They use direct, smooth joint trajectories with joint-limit, velocity,
acceleration, stop, and fresh-feedback checks. They allow large joint changes
and do not run FK/IK or tool-height checks. Their results include measured joints
and fresh camera images, with no calculated tool pose. Keep the swept area clear.
