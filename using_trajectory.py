import numpy as np
from icecube_proc import run_pipeline

out = run_pipeline("icecube.png", x_span=0.3, y_span=0.3, show_steps=False, do_plot=True)
traj: np.ndarray = out.trajectory_xyz  # (N, 3)
