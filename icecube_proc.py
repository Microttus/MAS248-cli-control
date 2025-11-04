#!/usr/bin/env python3
"""Image → contours → 3D path utility.

Can be used as a library (import functions) or as a CLI script:

  python3 icecube_proc.py path/to/icecube.png --plot
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

try:
    # Only import when needed; matplotlib is optional for library use.
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


# -------------------------- Data Structures -------------------------- #

@dataclass(frozen=True)
class PipelineOutputs:
    """Artifacts from the pipeline."""
    image_bgr: np.ndarray
    image_gray: np.ndarray
    image_thresh: np.ndarray
    image_edges: np.ndarray
    contours: List[np.ndarray]
    contour_overlay_bgr: np.ndarray
    trajectory_xyz: np.ndarray  # shape: (N, 3)


# -------------------------- Core Functions -------------------------- #

def load_image(path: str | Path) -> np.ndarray:
    """Loads an image in BGR format.

    Args:
      path: Path to the image.

    Returns:
      Image array (H, W, 3) in BGR.

    Raises:
      FileNotFoundError: If the file cannot be read.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def preprocess(
    img_bgr: np.ndarray,
    blur_ksize: Tuple[int, int] = (5, 5),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blur → grayscale → adaptive threshold → edges.

    Args:
      img_bgr: Input image in BGR.
      blur_ksize: Gaussian blur kernel size.

    Returns:
      (gray, thresh, edges)
    """
    blur = cv2.GaussianBlur(img_bgr, blur_ksize, cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        11, 5
    )
    edges = cv2.Canny(thresh, 30, 200)
    return gray, thresh, edges


def find_contours(edges: np.ndarray) -> List[np.ndarray]:
    """Finds contours (RETR_TREE, CHAIN_APPROX_SIMPLE) from edges.

    Handles OpenCV version differences in return signature.

    Args:
      edges: Binary edge map.

    Returns:
      List of contours, each of shape (M_i, 1, 2).
    """
    found = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(found) == 3:
        _, contours, _ = found  # OpenCV 3.x
    else:
        contours, _ = found     # OpenCV 4.x
    return contours


def draw_contours_overlay(img_bgr: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
    """Renders all contours in green on a blank canvas matching input size."""
    overlay = np.zeros_like(img_bgr)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def contours_to_trajectory(
    contours: Iterable[np.ndarray],
    z_lift: float = 0.2,
) -> np.ndarray:
    """Flattens contours into a sequence of (x, y, z) points.

    Adds a small z-lift after each contour to emulate 'pen up'.

    Args:
      contours: Iterable of contour arrays.
      z_lift: Z value inserted after each contour’s last point.

    Returns:
      (N, 3) float32 trajectory.
    """
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []

    for cont in contours:
        # cont shape: (M, 1, 2) → (M, 2)
        pts = cont.reshape(-1, 2)
        # Note: original script swapped axes; keep that behavior:
        # x <- column index (pts[:, 0]) becomes y in image coords; preserve naming as x
        # y <- row index (pts[:, 1]) becomes x in image coords; preserve naming as y
        # To stay consistent with the original output:
        #   x_coord.extend(pixel_pose[:,1])
        #   y_coord.extend(pixel_pose[:,0])
        xs.extend(pts[:, 1].astype(float))
        ys.extend(pts[:, 0].astype(float))
        zs.extend([0.0] * len(pts))

        # Add final lifted point
        xs.append(float(pts[-1, 1]))
        ys.append(float(pts[-1, 0]))
        zs.append(z_lift)

    if not xs:
        return np.zeros((0, 3), dtype=np.float32)

    traj = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=1).astype(np.float32)
    return traj


def normalize_xy_to_workspace(
    traj_xyz: np.ndarray,
    x_span: float = 0.210,
    y_span: float = 0.290,
) -> np.ndarray:
    """Normalizes X/Y independently to [0, span] ranges; Z unchanged.

    Args:
      traj_xyz: (N, 3) points.
      x_span: Target span for X.
      y_span: Target span for Y.

    Returns:
      (N, 3) float32 normalized trajectory.
    """
    if traj_xyz.size == 0:
        return traj_xyz

    out = traj_xyz.copy()
    x = out[:, 0]
    y = out[:, 1]

    # Avoid division-by-zero if flat
    x_range = float(np.ptp(x)) or 1.0
    y_range = float(np.ptp(y)) or 1.0

    out[:, 0] = (x - x.min()) / x_range * x_span
    out[:, 1] = (y - y.min()) / y_range * y_span
    return out.astype(np.float32)


def plot_trajectory(traj_xyz: np.ndarray) -> None:
    """Plots the 3D trajectory with matplotlib (if available)."""
    if plt is None:
        raise RuntimeError("matplotlib is not available. Install it or omit --plot.")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[arg-type]
    ax.plot(traj_xyz[:, 0], traj_xyz[:, 1], traj_xyz[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def maybe_imshow(title: str, img: np.ndarray, enabled: bool) -> None:
    """Conditionally show an image using OpenCV."""
    if not enabled:
        return
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_pipeline(
    image_path: str | Path,
    show_steps: bool = False,
    do_plot: bool = False,
    x_span: float = 0.210,
    y_span: float = 0.290,
    epsilon_px: float | None = None,
    decimate: int = 1
) -> PipelineOutputs:
    """High-level convenience wrapper for the full flow."""
    img = load_image(image_path)
    gray, thresh, edges = preprocess(img)
    contours = find_contours(edges)
    contours = simplify_contours(contours, epsilon_px=epsilon_px)
    overlay = draw_contours_overlay(img, contours)

    traj = contours_to_trajectory(contours)
    if decimate and decimate > 1:
        before = len(traj)
        traj = decimate_trajectory_keep_lifts(traj, stride=decimate)
        logging.info("Decimated trajectory: %d → %d points (stride=%d)", before, len(traj), decimate)

    traj_scaled = normalize_xy_to_workspace(traj, x_span=x_span, y_span=y_span)

    logging.info("Contours found: %d", len(contours))
    logging.debug("Image size: (h=%d, w=%d, c=%d)", *img.shape)

    # Optional visual steps
    maybe_imshow("grayscale", gray, show_steps)
    maybe_imshow("threshold", thresh, show_steps)
    maybe_imshow("edges", edges, show_steps)
    maybe_imshow("contours", overlay, show_steps)

    if do_plot and traj_scaled.size:
        plot_trajectory(traj_scaled)

    return PipelineOutputs(
        image_bgr=img,
        image_gray=gray,
        image_thresh=thresh,
        image_edges=edges,
        contours=contours,
        contour_overlay_bgr=overlay,
        trajectory_xyz=traj_scaled,
    )

def simplify_contours(contours: List[np.ndarray], epsilon_px: float | None = None) -> List[np.ndarray]:
    if not epsilon_px or epsilon_px <= 0:
        return contours
    out: List[np.ndarray] = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        eps = min(float(epsilon_px), max(1e-6, 0.25 * peri))
        c2 = cv2.approxPolyDP(c, eps, True)
        out.append(c2 if len(c2) >= 2 else c)
    return out


def decimate_trajectory_keep_lifts(traj_xyz: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1 or traj_xyz.size == 0:
        return traj_xyz
    kept: List[np.ndarray] = []
    i, n = 0, len(traj_xyz)
    while i < n:
        s = i
        while i < n and np.isclose(traj_xyz[i, 2], 0.0, atol=1e-12):
            i += 1
        e = i  # draw segment [s:e)
        if e > s:
            seg = traj_xyz[s:e]
            idx = [0] + [k for k in range(1, len(seg) - 1) if k % stride == 0]
            if len(seg) > 1:
                idx.append(len(seg) - 1)
            kept.append(seg[np.unique(idx)])
        if i < n and traj_xyz[i, 2] > 0:
            kept.append(traj_xyz[i:i+1])  # keep lift
            i += 1
    return np.vstack(kept).astype(np.float32)


# -------------------------- CLI -------------------------- #

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract contours and build a 3D trajectory.")
    p.add_argument("image", type=str, help="Path to input image (e.g. icecube.png)")
    p.add_argument("--show-steps", action="store_true", help="Show OpenCV windows for each step.")
    p.add_argument("--plot", action="store_true", help="Plot 3D trajectory with matplotlib.")
    p.add_argument("--save-overlay", type=str, default="",
                   help="Optional path to save contour overlay image (PNG).")
    p.add_argument("--save-trajectory", type=str, default="",
                   help="Optional path to save trajectory as .npy (Nx3).")
    p.add_argument("--x-span", type=float, default=0.210, help="Target span for X normalization.")
    p.add_argument("--y-span", type=float, default=0.290, help="Target span for Y normalization.")
    p.add_argument("--epsilon-px", type=float, default=0.0,
                   help="Simplify contours with approxPolyDP; epsilon in pixels (0=off).")
    p.add_argument("--decimate", type=int, default=1,
                   help="Keep every Nth point on draw segments; keep lifts/endpoints (1=off).")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    return p.parse_args(argv)


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    outputs = run_pipeline(
        image_path=args.image,
        show_steps=args.show_steps,
        do_plot=args.plot,
        x_span=args.x_span,
        y_span=args.y_span,
        epsilon_px=args.epsilon_px if args.epsilon_px > 0 else None,
        decimate=max(1, args.decimate),
    )

    if args.save_overlay:
        out_p = Path(args.save_overlay)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), outputs.contour_overlay_bgr)
        logging.info("Saved overlay: %s", out_p)

    if args.save_trajectory:
        out_p = Path(args.save_trajectory)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_p), outputs.trajectory_xyz)
        logging.info("Saved trajectory: %s (shape=%s)", out_p, outputs.trajectory_xyz.shape)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
