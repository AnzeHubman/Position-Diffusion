""" Second Kramers-Moyal coefficient to estimate D(q) """

import argparse
import yaml
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt

# --- parameters ---
parser = argparse.ArgumentParser()
parser.add_argument("--n_blocks", type=int, help="No. of blocks for error analysis", default=4)
parser.add_argument("--tau", type=float, help="Lag-time [ps]", required=True)
parser.add_argument("--dt", type=float, help="Integration step [ps]", default=1e-2)
parser.add_argument("--n_propagators", type=int, help="No. of propagators", default=24)
args = parser.parse_args()

# --- load trajectory ---
traj_w = np.load('traj_w.npy')
traj_u = np.load('traj_u.npy')
traj_w_blocks = np.array_split(traj_w, args.n_blocks)
traj_u_blocks = np.array_split(traj_u, args.n_blocks)

# --- estimation ---
lag = int(args.tau / args.dt)
h = 2 * np.pi / args.n_propagators
D_blocks = np.zeros((args.n_blocks, args.n_propagators))

for block in range(args.n_blocks):
    
    n_points = traj_w_blocks[block].shape[0] - lag

    # pre-compute array sizes for efficiency
    counts = np.zeros(args.n_propagators, dtype=int)
    array_indexes = np.zeros(args.n_propagators, dtype=int)

    for i in range(n_points):
        j = int(traj_w_blocks[block][i] / h)
        counts[j] += 1

    # collect displacements
    displ = np.zeros((args.n_propagators, np.max(counts)))

    for i in tqdm.tqdm(range(n_points), desc=f'Analyzing block {block + 1}'):
        j = int(traj_w_blocks[block][i] / h)
        k = array_indexes[j]
        displ[j, k] = traj_u_blocks[block][i + lag] - traj_u_blocks[block][i]
        array_indexes[j] += 1

    # estimate D(q)
    for i in range(args.n_propagators):
        D_blocks[block, i] = np.var(displ[i, 0:counts[i]]) / (2 * args.tau)

np.savez_compressed(f"results_tau_{args.tau}.npy", D=D_blocks)

for i in range(args.n_blocks):
    plt.plot(D_blocks[i])
plt.show()
        
        

