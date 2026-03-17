#!/usr/bin/env python3

"""
Inferring position-dependent diffusion coefficient using Fokker-Planck equation
Author: Anže Hubman, Theory department, National Institute of Chemistry, Slovenia
"""

import argparse
import yaml
import fpe
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt

# --- parameters ---
parser = argparse.ArgumentParser()
# general
parser.add_argument("--dt_langevin", type=float, help="Timestep for Langevin dynamics [ps]", default=1e-2)
parser.add_argument("--dt_fpe", type=float, help="Timestep for FPE [ps]", default=1e-3)
parser.add_argument("--tau", type=float, help="Lag-time [ps]", required=True)
parser.add_argument("--D_0", type=float, help="Initial guess for D(x)", default=0.2)
parser.add_argument("--n_blocks", type=int, help="No. of blocks for uncertainty estimation", default=4)
parser.add_argument("--n_propagators", type=int, help="No. of propagators", default=24)
parser.add_argument("--n_grid_points", type=int, help="Fine discretisation", default=200)
# Monte Carlo
parser.add_argument("--max_mc_step", type=float, help="Abs. range of MC step", default=5e-3)
parser.add_argument("--n_mc_steps", type=int, help="No. of optimization steps", default=5000)
parser.add_argument("--T_0", type=float, help="Initial MC temperature", default=1e-4)
parser.add_argument("--T_update_freq", type=int, help="Temperature re-adjustment freq.", default=100)
parser.add_argument("--T_hyperparam", type=float, help="Controls acceptance ratio", default=5e-3)
args = parser.parse_args()

# --- free energy, grids ---
x = np.linspace(0, 2 * np.pi, args.n_grid_points, endpoint=False)
x_coarse = (np.arange(args.n_propagators) + 0.5) * (2 * np.pi / args.n_propagators)
F = -np.cos(2 * x)

# --- lags ---
lag_langevin = int(args.tau / args.dt_langevin)
lag_fpe = int(args.tau / args.dt_fpe)
print("--> Parameters defined.")

# --- load x(t), partition into blocks ---
traj = np.load('traj.npy')
traj_blocks = np.array_split(traj, args.n_blocks)
print("--> Trajectory divided into blocks.")

# --- fitting ---
print("--> Starting fitting procedure.")
D_blocks = np.zeros((args.n_blocks, args.n_propagators))
loss_evolution = np.zeros((args.n_blocks, args.n_mc_steps))
acceptance_ratio = []

for block in range(args.n_blocks):

    # initialize
    temperature = args.T_0
    dL_track = []
    acr_block = []
    n_accepted = 0
    D_old = np.full(args.n_propagators, args.D_0)
    D_old_spline = fpe.construct_spline(D_coarse=D_old,
                                        n_propagators=args.n_propagators,
                                        x=x)

    # empirical propagators
    P_emp = fpe.empirical_propagators(traj=traj_blocks[block],
                                      lag=lag_langevin,
                                      n_propagators=args.n_propagators,
                                      n_bin=args.n_grid_points)

    # initial conditions
    P_0 = fpe.empirical_propagators(traj=traj_blocks[block],
                                    lag=0,
                                    n_propagators=args.n_propagators,
                                    n_bin=args.n_grid_points)
    # evaluate
    P_old = fpe.run_fpe(P_0=P_0,
                        F=F,
                        D=D_old_spline,
                        x=x,
                        n_steps=lag_fpe,
                        dt=args.dt_fpe)
    
    kl_old = fpe.kl_divergence(p=P_old,
                               q=P_emp,
                               x=x)
    
    loss_old = fpe.loss(kl_array=kl_old)

    # MC
    for step in tqdm.tqdm(range(args.n_mc_steps), desc=f"Analyzing block {block + 1}"):
        
        # bookkeeping
        loss_evolution[block, step] = loss_old

        # propose perturbation
        D_pro = fpe.perturb(y=D_old, max_mc_step=args.max_mc_step)
        
        D_pro_spline = fpe.construct_spline(D_coarse=D_pro,
                                            n_propagators=args.n_propagators,
                                            x=x)
        
        # run FPE
        P_pro = fpe.run_fpe(P_0=P_0,
                            F=F,
                            D=D_pro_spline,
                            x=x,
                            n_steps=lag_fpe,
                            dt=args.dt_fpe)

        # evaluate
        kl_pro = fpe.kl_divergence(p=P_pro, q=P_emp, x=x)
        loss_pro = fpe.loss(kl_array=kl_pro)

        # accept/reject
        dL = loss_pro - loss_old
        dL_track.append(dL)
        accept = fpe.metropolis(dL=dL, D=D_pro, temperature=temperature)

        if accept == True:
            D_old = D_pro
            loss_old = loss_pro
            n_accepted += 1

        # adjust T
        if len(dL_track) == args.T_update_freq:
            temperature = args.T_hyperparam * np.std(dL_track)
            dL_track = []
            acr_block.append(n_accepted / args.T_update_freq)
            n_accepted = 0
            
    D_blocks[block] = D_old
    acceptance_ratio.append(acr_block)

# --- save ---
np.savez_compressed(f"results_tau_{args.tau}.npy",
                    D=D_blocks,
                    loss=loss_evolution,
                    acr=acceptance_ratio)

# parameters
with open(f"fit_params_tau_{args.tau}.yaml", "w") as f:
    yaml.dump(vars(args), f, default_flow_style=False)

for i in range(args.n_blocks):
    plt.plot(D_blocks[i])

plt.tight_layout()
plt.show()
