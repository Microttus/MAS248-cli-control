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
                   default=[0.000, -0.600, 0.000],
                   help="Base frame offset added to path (meters).")
    p.add_argument("--vel", type=float, default=0.15, help="Linear velocity (m/s).")
    p.add_argument("--acc", type=float, default=0.25, help="Linear acceleration (m/s^2).")
    p.add_argument("--blend", type=float, default=0.001,
                   help="Blend radius for pen-down waypoints (m); clamped to half of --min-step. "
                        "Pen up/down waypoints are never blended.")
    p.add_argument("--min-step", type=float, default=0.003,
                   help="Drop waypoints closer than this to the previous one (m). "
                        "Blends overlapping between points make the controller reject the path.")
    p.add_argument("--init-joints-deg", type=float, nargs=6, metavar="J",
                   default=[-60.27, -134.74, -112.74, -124.0, -8.24, -192.42],
                   help="Start pose (joint angles in degrees) with the tool pointing at the "
                        "table. Its tool orientation is used for the whole drawing. "
                        "CALIBRATE THIS for your robot/table.")
    p.add_argument("--init-pose", type=float, nargs=6, metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                   help="Start pose as a TCP pose in the base frame (m, rotation vector in rad) "
                        "instead of --init-joints-deg. The robot does a joint move to it using the "
                        "inverse-kinematics solution closest to the current joints, so start from a "
                        "sensible configuration. Its orientation is used for the whole drawing.")
    p.add_argument("--no-init", action="store_true",
                   help="Skip the move to the start pose; use the current pose/orientation.")
    p.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation before moving.")
    p.add_argument("--ext-urcap", action="store_true",
                   help="Use the ExternalControl URCap instead of uploading the control script. "
                        "Not needed on a UR5e with default settings.")
    p.add_argument("--rtde-hz", type=float, default=-1.0,
                   help="RTDE communication frequency (Hz). -1 lets library decide.")
    p.add_argument("--urcap-port", type=int, default=50002,
                   help="URCap socket port used by ExternalControl (default 50002).")
    p.add_argument("--no-stop", action="store_true",
                   help="Do not call stopScript() at the end (debug).")
    p.add_argument("--dry-run", action="store_true", help="Do not move robot; only log path.")
    p.add_argument("--step-by-step", action="store_true",
                   help="Debug: send each waypoint as its own non-blended moveL and stop at the first "
                        "one that fails, printing its index and xyz. Slower than the blended path.")
    p.add_argument("--check-points", action="store_true",
                   help="Test every waypoint (inverse kinematics + safety limits) without drawing, "
                        "print a summary and exit. Combine with --dry-run to also skip the move to "
                        "the start pose (then the current tool orientation is tested).")
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


def _prune_close_points(xyz: np.ndarray, min_step: float) -> np.ndarray:
    """Drops points closer than min_step to the last kept one.

    Points where Z changes (pen up/down) are always kept, as are the endpoints.
    """
    z = xyz[:, 2]
    keep = [0]
    for i in range(1, len(xyz) - 1):
        z_change = z[i] != z[i - 1] or z[i] != z[i + 1]
        if z_change or np.linalg.norm(xyz[i] - xyz[keep[-1]]) >= min_step:
            keep.append(i)
    keep.append(len(xyz) - 1)
    return xyz[keep]


def _build_movel_path(
    xyz_path: np.ndarray,
    tcp_rvec: Sequence[float],
    vel: float,
    acc: float,
    blend: float,
) -> List[List[float]]:
    """Form a moveL path where each entry is [x,y,z,rx,ry,rz, v, a, blend].

    Waypoints where the pen goes up/down and the final waypoint get blend 0 so
    the robot actually reaches them instead of cutting the corner.
    """
    z = xyz_path[:, 2]
    n = len(xyz_path)
    rx, ry, rz = (float(r) for r in tcp_rvec)

    path: List[List[float]] = []
    for i, (xi, yi, zi) in enumerate(xyz_path):
        exact = i in (0, n - 1) or z[i] != z[i - 1] or z[i] != z[i + 1]
        path.append([float(xi), float(yi), float(zi), rx, ry, rz,
                     vel, acc, 0.0 if exact else blend])
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

    # RTDE flags: by default ur_rtde uploads its own control script to the robot.
    flags = rtde_control.RTDEControlInterface.FLAG_UPLOAD_SCRIPT  # type: ignore[attr-defined]
    if args.ext_urcap:
        flags = rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP  # type: ignore[attr-defined]

    # Connect Receive first for diagnostics on 30004
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    except Exception as e:
        logging.error("RTDEReceive connect failed to %s: %s", args.robot_ip, e)
        return 3

    robot_mode = rtde_r.getRobotMode()
    safety_mode = rtde_r.getSafetyMode()
    logging.info("RobotMode=%s SafetyMode=%s SpeedScaling=%.2f",
                 robot_mode, safety_mode, rtde_r.getSpeedScaling())
    if robot_mode != 7 or safety_mode != 1:
        logging.error("Robot not ready (mode=%s, safety=%s): need RUNNING(7) / NORMAL(1). "
                      "Power on, release brakes and clear any protective stop.",
                      robot_mode, safety_mode)
        return 5

    try:
        rtde_c = rtde_control.RTDEControlInterface(
            args.robot_ip, args.rtde_hz, flags, args.urcap_port)
    except Exception as e:
        logging.error("RTDEControl connect failed to %s (URCap port %d): %s",
                      args.robot_ip, args.urcap_port, e)
        return 4

    stop = {"value": False}

    def _sigint_handler(_sig, _frm):
        # Halt the robot right away; the main thread then unwinds via `finally`.
        stop["value"] = True
        try:
            rtde_c.stopL(2.0)
        except Exception:
            pass
    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        if not args.no_init:
            if args.init_pose:
                logging.info("Moving to initial TCP pose: %s", args.init_pose)
            else:
                init_pos = np.deg2rad(args.init_joints_deg)
                logging.info("Moving to initial position: %s", init_pos)
            if not args.yes and not args.dry_run:
                input("Robot will move to the start pose. Press Enter to continue (Ctrl-C aborts)...")
            if not args.dry_run:
                if args.init_pose:
                    if not rtde_c.moveJ_IK(list(args.init_pose)):
                        logging.error("moveJ_IK to --init-pose failed (unreachable or unsafe pose).")
                        return 8
                else:
                    rtde_c.moveJ(init_pos)

        # Orientation must be read AFTER reaching the start pose.
        tcp_pose = rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
        if args.dry_run and args.init_pose and not args.no_init:
            tcp_pose = list(args.init_pose)  # nothing moved: use the requested start pose
        logging.info("TCP Pose: %s", tcp_pose)

        min_step = args.min_step
        blend = min(args.blend, min_step / 2)
        before = len(traj_xyz)
        traj_xyz = _prune_close_points(traj_xyz, min_step)
        path = _build_movel_path(traj_xyz, tcp_pose[3:6], args.vel, args.acc, blend)
        logging.info("Waypoints: %d (pruned from %d; vel=%.3f m/s, acc=%.3f m/s^2, blend=%.4f m)",
                     len(path), before, args.vel, args.acc, blend)
        logging.debug("First waypoint: %s", path[0])

        if args.dry_run:
            np.save("preview_path_xyz.npy", traj_xyz)
            logging.warning("Dry-run enabled. Saved preview_path_xyz.npy and NOT moving the robot.")
            if not args.check_points:
                return 0

        # Pre-flight: find waypoints the robot cannot reach (these kill the control script).
        q_ref = rtde_r.getActualQ()
        bad = []
        for i, wp in enumerate(path):
            pose = wp[:6]
            try:
                ik_ok = len(rtde_c.getInverseKinematics(pose, q_ref)) == 6
                safe = rtde_c.isPoseWithinSafetyLimits(pose)
            except Exception:
                ik_ok, safe = False, False
            if not (ik_ok and safe):
                bad.append(i)
        if args.check_points:
            xyz = np.array([wp[:3] for wp in path])
            dist = float(np.linalg.norm(xyz[0] - np.array(tcp_pose[:3])))
            print(f"Checked {len(path)} waypoints with tool orientation {np.round(tcp_pose[3:6], 3)}")
            print(f"  X range: {xyz[:, 0].min():.3f} .. {xyz[:, 0].max():.3f} m")
            print(f"  Y range: {xyz[:, 1].min():.3f} .. {xyz[:, 1].max():.3f} m")
            print(f"  Z range: {xyz[:, 2].min():.3f} .. {xyz[:, 2].max():.3f} m")
            print(f"  First waypoint {np.round(xyz[0], 3)} is {dist:.3f} m from the current TCP")
            if bad:
                print(f"  UNREACHABLE / outside safety limits: {len(bad)}/{len(path)} "
                      f"(first indices {bad[:10]}, first bad xyz {np.round(xyz[bad[0]], 3)})")
                return 7
            print("  All waypoints reachable and within safety limits.")
            return 0
        if bad:
            logging.error("%d/%d waypoints unreachable or outside safety limits, e.g. idx %s "
                          "(xyz %s). Adjust --base-offset / --x-span / --y-span / --init-joints-deg.",
                          len(bad), len(path), bad[:5], np.round(path[bad[0]][:3], 3))
            return 7

        if not args.yes:
            input(f"About to draw {len(path)} waypoints. Hand on the e-stop. Press Enter to start...")
        if stop["value"]:
            return 130

        logging.warning("Executing path: %d waypoints, first=%s, TCP now=%s",
                        len(path), np.round(path[0][:3], 3), np.round(tcp_pose[:3], 3))
        if args.step_by_step:
            for i, wp in enumerate(path):
                if stop["value"]:
                    return 130
                ok = rtde_c.moveL(wp[:6], wp[6], wp[7])
                if not ok or not rtde_c.isProgramRunning():
                    prev = np.round(path[i - 1][:3], 3) if i else np.round(tcp_pose[:3], 3)
                    logging.error("Waypoint %d/%d failed: xyz %s (from %s), safety mode=%s, "
                                  "program running=%s. Read the pendant Log tab for the cause.",
                                  i, len(path), np.round(wp[:3], 3), prev,
                                  rtde_r.getSafetyMode(), rtde_c.isProgramRunning())
                    return 6
                if i % 50 == 0:
                    logging.warning("Waypoint %d/%d done", i, len(path))
            return 0

        ok = rtde_c.moveL(path)
        if stop["value"]:
            return 130
        if not ok:
            first = np.array(path[0][:3])
            cur = np.array(tcp_pose[:3])
            logging.error(
                "moveL(path) was rejected/failed. Current TCP=%s, first waypoint=%s "
                "(distance %.3f m). Check reachability, protective/safety stop "
                "(safety mode=%s, program running=%s).",
                np.round(cur, 3), np.round(first, 3), np.linalg.norm(first - cur),
                rtde_r.getSafetyMode(), rtde_c.isProgramRunning())
            return 6
        return 0
    finally:
        if not args.no_stop:
            rtde_c.stopScript()


if __name__ == "__main__":
    sys.exit(main())
