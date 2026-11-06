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

The example below uses this seed, read from the pendant:

| Joint | Base | Shoulder | Elbow | Wrist 1 | Wrist 2 | Wrist 3 |
|-------|------|----------|-------|---------|---------|---------|
| Degrees | -11.55 | -10.59 | 58.55 | -47.88 | -71.22 | -180.20 |
| Radians | -0.2016 | -0.1848 | 1.0219 | -0.8357 | -1.2430 | -3.1451 |

`--seed-joints-deg` takes **degrees**, so pass the first row as it is. The radian row is for
reference, since the pendant shows degrees while `ur_rtde` (`getActualQ()`, `moveJ`) uses radians.
Passing the radian values to this flag would be read as degrees, and would send the robot to a
completely different pose. **Use your own values if your table or setup differs.**

At this seed, forward kinematics (UR5e) puts the elbow about 39 cm and wrists 1-2 about 10 cm above
the tool tip height of the start pose, but wrist 3 and the flange are only about 1 mm above it. If
the height guard reports `wrist3` or `flange`, that is why. Check the pen mounting, or lower
`--joint-clearance` slightly, rather than changing the other joints.

## 3. Check without moving the pen

```shell
python ur5_run_trajectory.py <robot_ip> --traj out/mark_path.npy \
    --init-pose -0.390 -0.169 -0.151 1.461 -2.530 2.530 \
    --seed-joints-deg -11.55 -10.59 58.55 -47.88 -71.22 -180.20 \
    --z-offset -0.251 --dry-run --check-points -v
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

### Z height

The start pose only gives the tool orientation. It does not set the drawing height. The path is in
absolute base-frame coordinates:

```
pen-down z = 0.00 + base-offset z + z-offset
pen-up   z = 0.03 + base-offset z + z-offset
```

The start pose is at z = -0.151 m, about 10 cm above the paper, so the paper is at about
**z = -0.251 m**. That is the `--z-offset` used below. A positive value such as `0.015` would
leave the pen about 26 cm above the paper. To calibrate, jog the pen tip until it just touches the
paper, read the TCP z on the pendant and use that value.

For the first run, add 3-5 cm (for example `--z-offset -0.21`) to watch the whole motion in the
air, then lower it to the measured value.

The drawing is placed at x 0…0.21 m, y -0.60…-0.31 m by the default `--base-offset`. The start
pose is at x -0.39, y -0.17, so the first move crosses about half a metre sideways. Adjust
`--base-offset` so the drawing lies on the paper.

```shell
python ur5_run_trajectory.py <robot_ip> --traj out/mark_path.npy \
    --init-pose -0.390 -0.169 -0.151 1.461 -2.530 2.530 \
    --seed-joints-deg -11.55 -10.59 58.55 -47.88 -71.22 -180.20 \
    --base-offset 0.000 -0.600 0.000 \
    --z-offset -0.251 \
    --vel 0.15 --acc 0.25 --blend 0.001 \
    -v
```

| Option | Value | Meaning |
|--------|-------|---------|
| `--init-pose` | `-0.390 -0.169 -0.151 1.461 -2.530 2.530` | Start TCP pose (m, rotation vector in rad) with the correct pen orientation. |
| `--seed-joints-deg` | `-11.55 -10.59 58.55 -47.88 -71.22 -180.20` | The robot moves here first, then to the IK solution of `--init-pose`. |
| `--base-offset` | `0 -0.6 0` | Moves the drawing's origin in the base frame (this is the default). Check that the drawing area lies where the paper is. |
| `--z-offset` | `-0.251` | Height of the paper in the base frame (m), added to every point. See below. |
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
