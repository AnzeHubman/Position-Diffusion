
""" Helper functions for fitting overdamped Langevin equation """

import numpy as np
import numba as nb

@nb.jit(nopython=True)
def langevin_dynamics(n_step, dt, q):
    """ Langevin dynamics with Euler-Maruyama on known F(Q) and D(Q) """
    traj_u = np.zeros(n_step)
    traj_w = np.zeros(n_step)

    for t in range(n_step):
        traj_u[t] = q
        traj_w[t] = q - 2.0*np.pi*np.divmod(q,2.0*np.pi)[0]
        R = np.random.normal()
        f = -2.0*np.sin(2.0*q)
        D = 0.2 + 0.1*np.sin(q)
        dD = 0.1*np.cos(q)
        q = q + (dD + D*f)*dt + R*np.sqrt(2.0*D*dt)

    return traj_u, traj_w

def propagators_md(traj_u, traj_w, n_pr, lag):
    """ approximates local propagators from simulation data """

    # count no. of displecements per bin
    w = 2.0*np.pi / n_pr
    n_pts = traj_u.shape[0] - lag
    counts = np.zeros(n_pr, dtype=int)
    arrind = np.zeros(n_pr, dtype=int)
    for i in range(n_pts):
        j = int(traj_w[i] / w)
        counts[j] += 1

    # collect displacements
    dim_1 = n_pr
    dim_2 = np.max(counts)
    disp = np.zeros((dim_1, dim_2))
    for i in range(n_pts):
        j = int(traj_w[i] / w)
        k = arrind[j]
        disp[j,k] = traj_u[i + lag] - traj_u[i]
        arrind[j] += 1

    # histograms
    mu = np.zeros(n_pr)
    var = np.zeros(n_pr)

    for i in range(n_pr):
        mu[i] = np.mean(disp[i, 0:counts[i]])
        var[i] = np.var(disp[i, 0:counts[i]])

    return mu, var

def propagators_model(dF, D, dD, n_pr, lag, dt):
    """ 1st order Gaussian approximation for the propagator """
    tau = lag * dt
    mu = np.zeros(n_pr)
    var = np.zeros(n_pr)

    for i in range(n_pr):
        mu[i] = (-D[i] * dF[i] + dD[i]) * tau
        var[i] = 2 * D[i] * tau

    return mu, var

def kl_divergence(p_mu, q_mu, p_var, q_var):
    """ computes KL divergences """
    N = p_mu.shape[0]
    KL = np.zeros(N)

    for i in range(N):
        KL[i] = -0.5 + 0.5 * np.log(q_var[i] / p_var[i]) + ((p_var[i] + (p_mu[i] - q_mu[i])**2)/(2 * q_var[i]))
    return KL

def loss(kl_arr):
    """ loss as a sum of KL divergences """
    L = np.mean(kl_arr)
    return L

def weights(kl_arr):
    """ weights to bias new MC move """
    w = kl_arr / np.sum(kl_arr)
    return w

def perturb(y, pbin, delta):
    """ perturbs selected q-bin """
    N = y.shape[0]
    y_new = np.zeros(N)
    
    for i in range(N):
        
        if (i == pbin):
            y_new[i] = y[i] + delta
        else:
            y_new[i] = y[i]

    return y_new

def first_derivative(y):
    """ first derivative on a periodic domain """
    N = y.shape[0]
    dx = 2 * np.pi / N
    dy = np.zeros(N)
    
    for i in range(N):
        
        if i == N - 1:
            j = 0
        else:
            j = i + 1

        dy[i] = (y[j] - y[i]) / dx
    return dy

def L2_penalty(dy, alpha):
    """ regularization """
    N = dy.shape[0]
    L2 = alpha * (np.sum(dy**2) / N)
    return L2
            
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
