#!/usr/bin/env python3

"""
Monte Carlo fitting of the diffusion profile
Author: Anže Hubman, Theory department, National Institute of Chemistry, Slovenia
"""

import argparse
import yaml
import gauss
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt

# --- parameters ---
parser = argparse.ArgumentParser()
parser.add_argument("--n_blocks", type=int, help="No. of blocks for error analysis", default=4)
parser.add_argument("--biased", type=str, help="Use biased MC moves [yes/no]", default="no")
parser.add_argument("--n_propagators", type=int, help="No. of empirical propagators", default=24)
parser.add_argument("--n_mc_steps", type=int, help="No. of MC moves", default=100000)
parser.add_argument("--max_mc_step", type=float, help="Max. size of MC step", default=1e-3)
parser.add_argument("--dt", type=float, help="Timestep [ps]", default=1e-2)
parser.add_argument("--T_0", type=float, help="Initial MC temperature", default=1e-4)
parser.add_argument("--D_0", type=float, help="Initial D(q)", default=0.2)
parser.add_argument("--tau", type=float, help="Lag-time [ps]", required=True)
parser.add_argument("--T_update_freq", type=int, help="Temperature adjustment freq.", default=100)
parser.add_argument("--T_hyperparam", type=float, help="Controls acceptance ratio", default=1.0)
parser.add_argument("--alpha", type=float, help="Regularization strength", default=0.0)
args = parser.parse_args()

# --- precompute ---
# q-axis
w = 2 * np.pi / args.n_propagators
q = (np.arange(args.n_propagators) + 0.5) * w

# gradient of F(q)
dF = 2 * np.sin(2 * q)

# lag in frames
lag = int(args.tau / args.dt)

# --- trajectory ---
traj_w = np.load('traj_w.npy')
traj_u = np.load('traj_u.npy')
traj_w_blocks = np.array_split(traj_w, args.n_blocks)
traj_u_blocks = np.array_split(traj_u, args.n_blocks)

# --- fitting ---
D_blocks = np.zeros((args.n_blocks, args.n_propagators))
loss_evolution = np.zeros((args.n_blocks, args.n_mc_steps))
kl_evolution = np.zeros((args.n_blocks, args.n_mc_steps, args.n_propagators))
acr_evolution = []

for block in range(args.n_blocks):

    # initialize
    temperature = args.T_0
    dL_track = []
    acr_block = []
    n_accepted = 0
    D_old = np.full(args.n_propagators, args.D_0)
    dD_old = gauss.first_derivative(y=D_old)

    # empirical propagators
    mu_emp, var_emp = gauss.propagators_md(traj_u=traj_u_blocks[block],
                                           traj_w=traj_w_blocks[block],
                                           n_pr=args.n_propagators,
                                           lag=lag)

    # evaluate initial guess
    mu_old, var_old = gauss.propagators_model(dF=dF,
                                              D=D_old,
                                              dD=dD_old,
                                              n_pr=args.n_propagators,
                                              lag=lag,
                                              dt=args.dt)   # check this

    kl_old = gauss.kl_divergence(p_mu=mu_old,
                                 q_mu=mu_emp,
                                 p_var=var_old,
                                 q_var=var_emp)

    loss_old = gauss.loss(kl_arr=kl_old) + gauss.L2_penalty(dy=dD_old, alpha=args.alpha)

    weights_old = gauss.weights(kl_arr=kl_old)

    for step in tqdm.tqdm(range(args.n_mc_steps), desc=f"Analyzing block {block + 1}"):
        
        # bookkeeping
        kl_evolution[block, step] = kl_old
        loss_evolution[block, step] = loss_old

        # choose bin to perturb
        if args.biased == 'yes':
            pb = np.random.choice(np.arange(args.n_propagators), size=1, p=weights_old)[0]
        elif args.biased == 'no':
            pb = np.random.randint(0, args.n_propagators)
        else:
            raise ValueError("Invalid choice.")

        # suggest perturbation
        dy = np.random.uniform(-args.max_mc_step, args.max_mc_step)
        D_pro = gauss.perturb(y=D_old, pbin=pb, delta=dy)
        dD_pro = gauss.first_derivative(y=D_pro)

        # evaluate perturbation
        mu_pro, var_pro = gauss.propagators_model(dF=dF,
                                                  D=D_pro,
                                                  dD=dD_pro,
                                                  n_pr=args.n_propagators,
                                                  lag=lag,
                                                  dt=args.dt)   # check this
        
        kl_pro = gauss.kl_divergence(p_mu=mu_pro,
                                     q_mu=mu_emp,
                                     p_var=var_pro,
                                     q_var=var_emp)

        loss_pro = gauss.loss(kl_arr=kl_pro) + gauss.L2_penalty(dy=dD_pro, alpha=args.alpha)

        weights_pro = gauss.weights(kl_arr=kl_pro)

        # accept/reject
        dL = loss_pro - loss_old
        dL_track.append(dL)

        accept = gauss.metropolis(dL=dL, D=D_pro, temperature=temperature)

        if accept == True:
            D_old = D_pro
            dD_old = dD_pro
            loss_old = loss_pro
            kl_old = kl_pro
            weights_old = weights_pro
            n_accepted += 1

        # adjust T
        if len(dL_track) == args.T_update_freq:
            acr_block.append(n_accepted / args.T_update_freq)
            temperature = args.T_hyperparam * np.std(dL_track)
            dL_track = []
            n_accepted = 0
            
    # save final D(q)
    D_blocks[block] = D_old
    acr_evolution.append(acr_block)

# --- save ---
np.savez_compressed(f"results_tau_{args.tau}.npy",
                    D=D_blocks,
                    loss=loss_evolution,
                    acr=acr_evolution,
                    kl=kl_evolution)

with open(f"fit_params_tau_{args.tau}.yaml", "w") as f:
    yaml.dump(vars(args), f, default_flow_style=False)
    
for i in range(args.n_blocks):
    plt.plot(D_blocks[i])
plt.show()
