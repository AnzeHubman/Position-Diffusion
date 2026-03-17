""" Module for working with Fokker-Planck equations """

import numpy as np
import numba as nb
from scipy.interpolate import CubicSpline

@nb.jit(nopython=True)
def langevin_dynamics(n_step, dt, q_start):
    """ Langevin dynamics with Euler-Maruyama on known F(Q) and D(Q) """
    q = q_start
    traj = np.zeros(n_step)

    for t in range(n_step):
        # save
        traj[t] = q

        # draw random number
        R = np.random.normal()

        # forces 
        f = -2.0*np.sin(2.0*q)
        D = 0.2 + 0.1*np.sin(q)
        dD = 0.1*np.cos(q)

        # propagate
        q += (dD + D * f) * dt + R * np.sqrt(2.0 * D * dt)

        # wrap
        q = q % (2 * np.pi)

    return traj

@nb.jit(nopython=True)
def omega(m, n, F, D, dx, beta=1.0):
    """ transition frequency between m an n (Bicout & Szabo, JCP, 1998) """
    A = (D[n] + D[m]) / (2 * dx**2)
    return A * np.exp(-0.5 * beta * (F[m] - F[n]))

@nb.jit(nopython=True)
def solve_FPE(P, dt, F, D, dx):
    """ solves FPE (Bicout & Szabo, JCP, 1998) """
    N = P.shape[0]
    dP = np.zeros(N)
    
    # notation : n = i; n - 1 = j; n + 1 = k
    for i in range(N):
        # PBC
        j = (i - 1) % N
        k = (i + 1) % N

        # increment
        dP[i] = (omega(m=i, n=k, F=F, D=D, dx=dx) * P[k] + 
                 omega(m=i, n=j, F=F, D=D, dx=dx) * P[j] - 
                 (omega(m=k, n=i, F=F, D=D, dx=dx) +  
                  omega(m=j, n=i, F=F, D=D, dx=dx))* P[i])
                 
    P_next = P + dt * dP
    return P_next

@nb.jit(nopython=True)
def empirical_propagators(traj, lag, n_propagators, n_bin):
    """ estimates propagators as histograms """
    n_frames = traj.shape[0]
    h = 2 * np.pi / n_bin
    s = 2 * np.pi / n_propagators
    P = np.zeros((n_propagators, n_bin))

    for t in range(n_frames - lag):

        # t = 0
        i = int(traj[t] / s)

        # t = tau
        j = int(traj[t + lag] / h)

        if (j >= 0 and j < n_bin):
            P[i, j] += 1.0
            
    # normalize
    P_norm = np.zeros((n_propagators, n_bin))
    for i in range(n_propagators):
        P_norm[i] = P[i] / (np.sum(P[i])*h)
    return P_norm

def construct_spline(D_coarse, n_propagators, x):
    """ represents D(x) with periodic spline """
    
    # coarse grid
    x_coarse = (np.arange(n_propagators) + 0.5) * (2 * np.pi / n_propagators)

    # extension to enforce periodicity
    x_ext = np.concatenate([x_coarse, [x_coarse[0] + 2*np.pi]])
    y_ext = np.concatenate([D_coarse, [D_coarse[0]]])

    # spline
    spline = CubicSpline(x_ext, y_ext, bc_type='periodic')
    D_spline = spline(x)

    return D_spline

@nb.jit(nopython=True)
def run_fpe(P_0, F, D, x, n_steps, dt):
    """ runs a series of FPEs from initial condition """
    n_propagators = P_0.shape[0]
    n_grid_points = x.shape[0]
    dx = x[1] - x[0]
    P = np.zeros((n_propagators, n_grid_points))

    for i in range(n_propagators):
        P_i = P_0[i]
        for t in range(n_steps):
            P_next = solve_FPE(P=P_i, dt=dt, F=F, D=D, dx=dx)
            P_i = P_next
        P[i] = P_i
    return P

@nb.jit(nopython=True)
def kl_divergence(p, q, x):
    """ numerical KL divergence """
    n_propagators, n_grid_points = p.shape
    kl_array = np.zeros(n_propagators)
    dx = x[1] - x[0]

    for i in range(n_propagators):
        for j in range(n_grid_points):
            if (p[i, j] > 0.0) and (q[i, j] > 0.0):
                kl_array[i] += (p[i, j] * np.log(p[i, j] / q[i, j])) * dx
    return  kl_array

def loss(kl_array):
    """ loss as a sum of KL divergences """
    L = np.mean(kl_array)
    return L

def perturb(y, max_mc_step):
    """ perturbs selected bin """
    i = np.random.randint(0, y.shape[0])
    delta = np.random.uniform(-max_mc_step, max_mc_step)
    y_new = y.copy()
    y_new[i] += delta
    return y_new

def metropolis(dL, D, temperature):
    """ accept move according to Metropolis criterion + enforce positive D(x) """
    if (np.min(D) > 0.0):
        if (dL < 0.0):
            accept = True
        else:
            w = np.exp(-dL / temperature)
            v = np.random.uniform(0, 1)
            if (v <= w):
                accept = True
            else:
                accept = False
    else:
        accept = False
    return accept
