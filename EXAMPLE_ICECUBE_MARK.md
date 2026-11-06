# Example: drawing the IceCube mark

A worked example of drawing `icecube_mark.png` with the UR5e, starting from the tool-down pose
`(-0.390, -0.169, -0.151, 1.461, -2.530, 2.530)`. Every option is explained in the main
[README](README.md).

## Which image file

| File | Usable? |
|------|---------|
| `icecube_mark.png` | **Yes.** 1024×1024, fully opaque, so OpenCV reads it directly. |
| `icecube-mark-mono-navy.svg` | **Not directly.** `icecube_proc.py` loads images with `cv2.imread`, which cannot read SVG. Rasterise it to PNG first (see below). |

To use the SVG, convert it to a PNG with a white background:

```shell
rsvg-convert -w 1024 -h 1024 -b white icecube-mark-mono-navy.svg -o icecube_mark_navy.png
# or: inkscape icecube-mark-mono-navy.svg -w 1024 -h 1024 -b white -o icecube_mark_navy.png
```

and use `icecube_mark_navy.png` in place of `icecube_mark.png` below.

## 1. Build the path and look at it

```shell
python icecube_proc.py icecube_mark.png --plot -v \
    --save-trajectory out/mark_path.npy --save-overlay out/mark_overlay.png \
    --x-span 0.21 --y-span 0.29
```

- `--x-span 0.21 --y-span 0.29` scales the drawing to 21 × 29 cm.
- `--save-overlay` writes the detected contours so you can check what will be drawn.
- `--plot` shows the 3D path, with the pen-up moves lifted by `--z-lift` (3 cm by default).

## 2. Pick the seed joints

The start pose has several joint solutions, and some fold the elbow or wrists below the pen. The
seed picks the good one.

1. On the pendant, jog the robot to a pose with the **tool pointing down at the start position
   and the elbow and wrists clearly above the pen**.
2. Read the six joint angles (degrees) from the Move tab.
3. Use them as `--seed-joints-deg`.

The command below uses `J1 … J6` as placeholders. **Replace them with your own values.**

## 3. Check without moving the pen

```shell
python ur5_run_trajectory.py <robot_ip> --traj out/mark_path.npy \
    --init-pose -0.390 -0.169 -0.151 1.461 -2.530 2.530 \
    --seed-joints-deg J1 J2 J3 J4 J5 J6 \
    --z-offset 0.015 --dry-run --check-points -v
```

What it does:

- `--init-pose` is the start pose. Its orientation is used for the whole drawing. (This is also
  the default, so the option can be left out.)
- `--seed-joints-deg` makes the IK solve the start pose on your chosen branch.
- `--dry-run` computes everything but does not move the robot.
- `--check-points` tests every waypoint (IK, safety limits and the height guard), prints the X/Y/Z
  ranges and the **lowest joint height relative to the tool tip**, and exits with code `7` if
  anything fails.

Expected end of the output:

```
All waypoints reachable, within safety limits and with joints above the tip.
```

If it prints `JOINTS BELOW TIP`, choose another seed or move the drawing (`--base-offset`). Exit
code `9` means the start pose itself is bad with this seed.

## 4. Draw

Start with a large `--z-offset` (for example `0.05`, pen 5 cm above the paper) to watch the whole
motion in the air, then reduce it to the real calibration value.

```shell
python ur5_run_trajectory.py <robot_ip> --traj out/mark_path.npy \
    --init-pose -0.390 -0.169 -0.151 1.461 -2.530 2.530 \
    --seed-joints-deg J1 J2 J3 J4 J5 J6 \
    --base-offset 0.000 -0.600 0.000 \
    --z-offset 0.015 \
    --vel 0.15 --acc 0.25 --blend 0.001 \
    -v
```

| Option | Value | Meaning |
|--------|-------|---------|
| `--init-pose` | `-0.390 -0.169 -0.151 1.461 -2.530 2.530` | Start TCP pose (m, rotation vector in rad) with the correct pen orientation. |
| `--seed-joints-deg` | `J1 … J6` | The robot moves here first, then to the IK solution of `--init-pose`. |
| `--base-offset` | `0 -0.6 0` | Moves the drawing's origin in the base frame (this is the default). Check that the drawing area lies where the paper is. |
| `--z-offset` | `0.015` | Calibration height added to every point (m). |
| `--vel` / `--acc` | `0.15` / `0.25` | Pen speed (m/s) and acceleration (m/s²). |
| `--blend` | `0.001` | Corner rounding for pen-down points (m). Pen up/down points are never blended. |

The script asks for Enter twice: before the move to the start pose, and before drawing. Keep a
hand on the e-stop. Ctrl-C stops the robot.

To build the path in one go, replace `--traj out/mark_path.npy` with
`--image icecube_mark.png --x-span 0.21 --y-span 0.29`.

## If something goes wrong

| Symptom | What to try |
|---------|-------------|
| Exit `9` | The IK solution puts the elbow or a wrist below the tip. Use a better seed. |
| Exit `8` | The start pose is unreachable from the seed. Check the pose and the seed. |
| Exit `7` | Unreachable waypoint, or a joint dipped below the tip. Adjust `--base-offset`, the spans or the seed. |
| Fails partway through | Add `--step-by-step` to see the exact waypoint that fails, and read the pendant Log tab. |
