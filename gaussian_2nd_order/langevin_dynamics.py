#!/usr/bin/env python3

"""
Overdamped Langevin dynamics simulation
Author: Anže Hubman, Theory department, National Institute of Chemistry, Slovenia
"""

import argparse
import yaml
import gauss
import numpy as np
import matplotlib.pyplot as plt

# parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_steps", type=int, help="No. of integration step", default=10000000)
parser.add_argument("--dt", type=float, help="Integration timestep [ps]", default=1e-2)
parser.add_argument("--q_0", type=float, help="Initial position on q", default=0.0)
args = parser.parse_args()

# save arguments
with open(f"sim_params.yaml", "w") as f:
    yaml.dump(vars(args), f, default_flow_style=False)

# generate trajectory
traj_u, traj_w = gauss.langevin_dynamics(n_step=args.n_steps, dt=args.dt, q=args.q_0)

# save
np.save(f"traj_w.npy", traj_w)
np.save(f"traj_u.npy", traj_u)

# visualise
fig, ax = plt.subplots(figsize=(5,2.5))
ax.hist(traj_w, bins=50, density=True, range=(0, 2*np.pi))
ax.set_xlim([0, 2*np.pi])
ax.set_xlabel(r"$q$ [rad.]")
ax.set_ylabel(r"$P(q)$")
plt.tight_layout()
plt.show()



