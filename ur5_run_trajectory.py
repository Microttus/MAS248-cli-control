#!/usr/bin/env python3
"""Execute a 3D (x,y,z) Cartesian trajectory on a UR5 using ur_rtde.

Typical usage:
  # Directly from an image using the earlier module
  python3 ur5_run_trajectory.py 192.168.0.10 --image icecube.png --blend 0.01

  # From a saved Nx3 .npy file
  python3 ur5_run_trajectory.py 192.168.0.10 --traj out/path.npy --z-offset 0.015

Safety:
- Verify collision-free space and correct coordinate frames before running.
- Prefer URSim or a high Z-offset for first tests; use --dry-run to inspect.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

try:
    # Optional import; only needed if using --image
    from icecube_proc import run_pipeline  # noqa: F401
except Exception:
    run_pipeline = None  # type: ignore[assignment]

# ur_rtde
import rtde_control  # type: ignore
import rtde_receive  # type: ignore


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UR5 trajectory executor (moveL path + blend).")
    p.add_argument("robot_ip", type=str, help="Robot IP (or URSim IP).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--traj", type=str, help="Path to Nx3 .npy trajectory (meters).")
    src.add_argument("--image", type=str, help="Image file to build trajectory via icecube_proc.")
    p.add_argument("--x-span", type=float, default=0.210, help="X span if --image is used.")
    p.add_argument("--y-span", type=float, default=0.290, help="Y span if --image is used.")
    p.add_argument("--z-offset", type=float, default=0.0,
                   help="Calibration Z-offset (meters) added to all points.")
    p.add_argument("--base-offset", type=float, nargs=3, metavar=("X0", "Y0", "Z0"),
                   default=[0.000, 0.400, 0.000],
                   help="Base frame offset added to path (meters).")
    p.add_argument("--vel", type=float, default=0.15, help="Linear velocity (m/s).")
    p.add_argument("--acc", type=float, default=0.25, help="Linear acceleration (m/s^2).")
    p.add_argument("--blend", type=float, default=0.005, help="Blend radius per waypoint (m).")
    p.add_argument("--ext-urcap", action="store_true",
                   help="Use ExternalControl URCap flag for control interface.")
    p.add_argument("--rtde-hz", type=float, default=-1.0,
                   help="RTDE communication frequency (Hz). -1 lets library decide.")
    p.add_argument("--urcap-port", type=int, default=50002,
                   help="URCap socket port used by ExternalControl (default 50002).")
    p.add_argument("--no-stop", action="store_true",
                   help="Do not call stopScript() at the end (debug).")
    p.add_argument("--dry-run", action="store_true", help="Do not move robot; only log path.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    return p.parse_args(argv)


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _load_or_build_traj(args: argparse.Namespace) -> np.ndarray:
    """Returns (N, 3) float32 in meters (x, y, z)."""
    if args.traj:
        traj = np.load(args.traj).astype(np.float32)
        if traj.ndim != 2 or traj.shape[1] != 3:
            raise ValueError(f"Expected Nx3 trajectory, got {traj.shape}")
        print(f"Received path! First, second and last points: 0:{traj[0]}, 1:{traj[1]}, -1:{traj[-1]}")
        return traj

    if run_pipeline is None:
        raise RuntimeError("icecube_proc not available; cannot use --image.")
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    out = run_pipeline(args.image, show_steps=False, do_plot=False,
                       x_span=args.x_span, y_span=args.y_span)
    traj = out.trajectory_xyz.astype(np.float32)
    if traj.ndim != 2 or traj.shape[1] != 3:
        raise ValueError(f"Pipeline produced invalid shape: {traj.shape}")
    return traj


def _apply_offsets(traj_xyz: np.ndarray,
                   base_offset: Sequence[float],
                   z_offset: float) -> np.ndarray:
    """Adds base XYZ offset and calibration Z offset."""
    out = traj_xyz.copy()
    out[:, 0] += float(base_offset[0])
    out[:, 1] += float(base_offset[1])
    out[:, 2] += float(base_offset[2] + z_offset)
    return out


def _build_movel_path(
    xyz_path: np.ndarray,
    tcp_rvec: Sequence[float],
    vel: float,
    acc: float,
    blend: float,
) -> List[List[float]]:
    """Form a moveL path where each entry is [x,y,z,rx,ry,rz, v, a, blend]."""
    x, y, z = xyz_path[:, 0], xyz_path[:, 1], xyz_path[:, 2]
    rx, ry, rz = float(tcp_rvec[0]), float(tcp_rvec[1]), float(tcp_rvec[2])

    path: List[List[float]] = []
    for xi, yi, zi in zip(x, y, z):
        path.append([round(float(xi), 3), round(float(yi), 3), round(float(zi), 3), round(rx, 3), round(ry, 3), round(rz,3), vel, acc, blend])
    return path


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    # Load or compute path
    try:
        traj_xyz = _load_or_build_traj(args)
    except Exception as e:
        logging.error("Failed to load/build trajectory: %s", e)
        return 2

    if traj_xyz.size == 0:
        logging.error("Empty trajectory.")
        return 2

    # Offsets (base frame & calibration Z)
    traj_xyz = _apply_offsets(traj_xyz, args.base_offset, args.z_offset)

    # RTDE flags
    flags = 0
    if args.ext_urcap:
        # In Python API: RTDEControlInterface.FLAG_USE_EXT_UR_CAP
        flags = rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP  # type: ignore[attr-defined]

    # Then Control (RTDE). URCap port only matters with ExternalControl.
    try:
        rtde_c = rtde_control.RTDEControlInterface(
            args.robot_ip,
            args.rtde_hz,
            flags,
            args.urcap_port
        )
    except Exception as e:
        logging.error("RTDEControl connect failed to %s, URCap %d: %s",
                      args.robot_ip, args.urcap_port, e)
        return 4

    # Connect Receive first for diagnostics on 30004
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    except Exception as e:
        logging.error("RTDEReceive connect failed to %s port: %s", args.robot_ip, e)
        return 3

    # Graceful shutdown
    stop = {"value": False}

    def _sigint_handler(_sig, _frm):
        stop["value"] = True
    signal.signal(signal.SIGINT, _sigint_handler)


    # Use current TCP orientation; keep orientation consistent across path.
    tcp_pose = rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
    tcp_rvec = tcp_pose[3:6]

    logging.info("TCP Pose: %s", tcp_pose)

    path = _build_movel_path(traj_xyz, tcp_rvec, args.vel, args.acc, args.blend)

    # --- Preflight diagnostics ---
    robot_mode = rtde_r.getRobotMode()
    safety_mode = rtde_r.getSafetyMode()
    speed_scaling = rtde_r.getSpeedScaling()  # 0.0..1.0 (speed slider)
    prog_running = rtde_c.isProgramRunning()

    logging.info(
        "Waypoints: %d (vel=%.3f m/s, acc=%.3f m/s^2, blend=%.3f m)",
        len(path), args.vel, args.acc, args.blend
    )
    logging.info("RobotMode=%s SafetyMode=%s SpeedScaling=%.2f ProgramRunning=%s",
                 robot_mode, safety_mode, speed_scaling, prog_running)
    logging.debug("First waypoint: %s", path[0])

    if args.dry_run:
        np.save("preview_path_xyz.npy", traj_xyz)
        logging.warning("Dry-run enabled. Saved preview_path_xyz.npy and NOT moving the robot.")
        return 0

    # Moving to initial joint position
    init_pos_deg = np.array([-60.27, -134.74, -112.74, -124.0, -8.24, -192.42])
    init_pos = np.deg2rad(init_pos_deg)
    logging.info("Moving to initial position: %s", init_pos)
    rtde_c.moveJ(init_pos)

    logging.info("Executing path")
    rtde_c.moveL(path)
    rtde_c.stopScript()

    return 0


if __name__ == "__main__":
    sys.exit(main())
