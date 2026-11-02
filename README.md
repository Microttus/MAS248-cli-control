# MAS248-cli-control

This project shows how the MAS248 Robotics project at the University of Agder can be solved
without ROS, using the [`ur_rtde`](https://sdurobotics.gitlab.io/ur_rtde/) Python driver to control a
Universal Robots arm, together with OpenCV, NumPy and matplotlib.

The robot "draws" an image: the image is converted to contours, the contours become a 3D
path (pen down at `z = 0`, pen up at `z = z_lift`), and the path is sent to the robot as a
blended `moveL` sequence.

## Files

| File | Purpose |
|------|---------|
| `icecube_proc.py` | Image → contours → scaled `Nx3` trajectory (metres). Usable as a CLI or imported as a library. |
| `ur5_run_trajectory.py` | Loads a trajectory (from `.npy` or directly from an image) and executes it on the robot. |
| `using_trajectory.py` | Minimal example of using `run_pipeline` as a library. |
| `Bold_R.png`, `icecube.png`, `bold_r_path.npy` | Example image inputs and a pre-generated trajectory. |

## Installation

```shell
pip install numpy opencv-python matplotlib ur_rtde
```

## Workflow

1. **Generate a path** from an image and inspect it (`icecube_proc.py`).
2. **Dry-run** it against the robot to check offsets and waypoints (`--dry-run`).
3. **Run it** on the robot (URSim or a real robot with a high `--z-offset` first).

```shell
# 1. Image -> trajectory, show a 3D plot and save it
python icecube_proc.py icecube.png --plot -v --save-trajectory out/path.npy --x-span 0.21 --y-span 0.29

# 2. Inspect without moving the robot (saves preview_path_xyz.npy)
python ur5_run_trajectory.py <robot_ip> --traj out/path.npy --z-offset 0.015 --dry-run -v

# 3. Draw
python ur5_run_trajectory.py <robot_ip> --traj out/path.npy --z-offset 0.015 -v
```

You can also skip step 1 and let the robot script build the path itself:

```shell
python ur5_run_trajectory.py <robot_ip> --image icecube.png --z-offset 0.015 -v
```

> **Safety:** Check that the drawing area is free of obstacles and that the coordinate frames
> are correct. Keep a hand on the e-stop. Test in URSim, or with a large `--z-offset`
> (e.g. `0.05`), before putting a pen on paper. Ctrl-C stops the robot.

### What the robot script does

1. Loads/builds the trajectory and adds `--base-offset` and `--z-offset`.
2. Connects to the robot and checks that it is `RUNNING` and in `NORMAL` safety mode.
3. Moves (joint move) to `--init-joints-deg`. The tool orientation at this pose is used for the
   whole drawing, so this pose should have the tool pointing at the table.
4. Drops waypoints that are closer together than `--min-step`, then builds the `moveL` path.
5. Checks that every waypoint is reachable (inverse kinematics and safety limits). If not, it
   exits with code 7 and lists the offending waypoints.
6. Asks for confirmation and executes the path.

## `icecube_proc.py`

```
python icecube_proc.py <image> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `image` | – | Input image path (required). |
| `--show-steps` | off | Show OpenCV windows for grayscale, threshold, edges and contours. |
| `--plot` | off | Show the resulting 3D trajectory in matplotlib. |
| `--save-overlay PATH` | – | Save the contour overlay image (PNG). |
| `--save-trajectory PATH` | – | Save the trajectory as an `Nx3` `.npy` file (metres, x/y/z). |
| `--x-span` | `0.210` | Width (m) the drawing is scaled to along X. |
| `--y-span` | `0.290` | Height (m) the drawing is scaled to along Y. |
| `--epsilon-px` | `0.0` | Simplify contours with `approxPolyDP`; tolerance in pixels. `0` disables. |
| `--z-lift` | `0.03` | Pen-up height in metres (pen-down is `z = 0`). |
| `--decimate N` | `1` | Keep every N-th point on each contour (endpoints are kept). `1` disables. |
| `-v`, `-vv` | – | Increase log verbosity (info, debug). |

## `ur5_run_trajectory.py`

```
python ur5_run_trajectory.py <robot_ip> (--traj FILE.npy | --image IMAGE) [options]
```

### Required

| Argument | Description |
|----------|-------------|
| `robot_ip` | IP address of the robot (or URSim). |
| `--traj FILE` | `Nx3` `.npy` trajectory in metres. Mutually exclusive with `--image`. |
| `--image FILE` | Build the trajectory from an image using `icecube_proc`. Mutually exclusive with `--traj`. Only `--x-span`/`--y-span` are forwarded; for `--epsilon-px`, `--decimate` or `--z-lift` generate a `.npy` with `icecube_proc.py` and use `--traj`. |

### Path placement

| Option | Default | Description |
|--------|---------|-------------|
| `--x-span` | `0.210` | X size (m) of the drawing when using `--image`. |
| `--y-span` | `0.290` | Y size (m) of the drawing when using `--image`. |
| `--z-offset` | `0.0` | Calibration offset (m) added to all Z values. Raise it to hover above the table, lower it to press the pen. |
| `--base-offset X0 Y0 Z0` | `0 -0.6 0` | Offset (m) added to every point to place the drawing in the robot base frame. |

### Motion

| Option | Default | Description |
|--------|---------|-------------|
| `--vel` | `0.15` | Linear velocity (m/s). |
| `--acc` | `0.25` | Linear acceleration (m/s²). |
| `--blend` | `0.001` | Blend radius (m) for pen-down waypoints. Clamped to half of `--min-step`. Pen up/down waypoints and the endpoints are never blended. |
| `--min-step` | `0.003` | Drop waypoints closer than this (m) to the previous one. Overlapping blends make the controller reject the path. Z-change waypoints are always kept. |
| `--init-joints-deg J1..J6` | `-60.27 -134.74 -112.74 -124.0 -8.24 -192.42` | Start pose in degrees, tool pointing at the table. Its tool orientation is used for the whole drawing. **Calibrate this for your robot/table.** |
| `--init-pose X Y Z RX RY RZ` | – | Start pose as a TCP pose in the base frame (m, rotation vector in rad) instead of joint angles. Uses a joint move to the IK solution closest to the current joints, so start from a sensible configuration. Its orientation is used for the whole drawing, and with `--dry-run` it is the orientation tested by `--check-points`. Read the values from the pendant's Move tab (TCP) or print them with `rtde_r.getActualTCPPose()`. |
| `--no-init` | off | Skip the move to the start pose and use the current pose and orientation. |

### Safety and confirmation

| Option | Default | Description |
|--------|---------|-------------|
| `--dry-run` | off | Do not move the robot. Saves the processed path as `preview_path_xyz.npy` and exits. |
| `--check-points` | off | Test every waypoint (inverse kinematics and safety limits) and print a summary: X/Y/Z ranges, distance from the TCP to the first waypoint, and any unreachable waypoints. Does not draw. Exits with code 7 if any point fails. Add `--dry-run` to also skip the move to the start pose (the current tool orientation is then tested). |
| `-y`, `--yes` | off | Skip the "press Enter" confirmations before moving. |

### Connection

| Option | Default | Description |
|--------|---------|-------------|
| `--ext-urcap` | off | Use the *ExternalControl* URCap instead of uploading the control script. Needs the ExternalControl program running on the pendant with this computer's IP as host. Not needed on a UR5e with default settings. |
| `--urcap-port` | `50002` | Port used by the ExternalControl URCap. Must match the port set in the URCap on the pendant (e.g. `50001`). |
| `--rtde-hz` | `-1` | RTDE frequency (Hz). `-1` lets the library choose. |
| `--no-stop` | off | Do not call `stopScript()` at exit (debugging). |
| `-v`, `-vv` | – | Increase log verbosity (info, debug). |

Example with the URCap:

```shell
python ur5_run_trajectory.py <robot_ip> --traj out/path.npy --ext-urcap --urcap-port 50001 -v
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `2` | Could not load/build the trajectory, or it is empty. |
| `3` | RTDE receive connection failed. |
| `4` | RTDE control connection failed. |
| `5` | Robot not ready (needs mode `RUNNING`/7 and safety `NORMAL`/1). Power on, release brakes, clear protective stops. |
| `6` | `moveL` failed. The error output shows the current TCP, the first waypoint and the robot state. |
| `8` | `moveJ_IK` to `--init-pose` failed. |
| `7` | Pre-flight check found unreachable waypoints or waypoints outside the safety limits. |
| `130` | Interrupted with Ctrl-C. |

## Troubleshooting

- **"RTDE control script is not running!"**: the script on the robot stopped, usually because of a
  runtime error (unreachable waypoint, singularity, joint limit). Check the pendant's *Log* tab for the
  exact message, and adjust `--base-offset`, `--x-span`/`--y-span` or `--init-joints-deg`.
- **Robot stops after the initial move**: run with `-v` and look at the error message and exit code above.
- **Robot not ready (exit 5)**: power on the arm, release the brakes and clear any protective stop.
- **Cannot connect with `--ext-urcap`**: start the ExternalControl program on the pendant and make sure
  its host IP and port match `robot_ip` and `--urcap-port`.

## Using it as a library

```python
from icecube_proc import run_pipeline

out = run_pipeline("Bold_R.png", x_span=0.3, y_span=0.3, do_plot=True)
traj = out.trajectory_xyz   # (N, 3) array in metres
```
