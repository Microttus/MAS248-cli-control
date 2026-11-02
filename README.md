# MAS248-cli-control
This project aims to show how the MAS248 Robotics project at the University of Agder can be solved purely without the use of ROS by taking advantage of the UR_RTDE driver for python control of an Unvíversial Robots Arm and other python libearies such as cv2 and matplotlib

Comand to launch:

```shell
python ur5_run_trajectory.py <ip> --traj <path_file> --z-offset 0.015 --dry-run -v   # inspect first
python ur5_run_trajectory.py <ip> --traj <path_file> --z-offset 0.015 -v
```

```shell
python icecube_proc.py <img> --plot -v --save-trajectory <name> --x-span 0.4 --y-span 0.25
```