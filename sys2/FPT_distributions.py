"""
  Visualise distributions of first passage times,
  compute mean first passage times and their uncertainties
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', size = 9)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.latex.preamble'] = r"""\usepackage{tgheros}\usepackage{sansmath}\sansmath"""

# load data
fpt_LD = np.load('LD/first_passages_ld.npy')
fpt_MD = np.load('MD/first_passages_MD.npy')

# distributions
fig, ax = plt.subplots(figsize=(3.37, 2.5))

t_min = 0
t_max = 150
num_MD_bins = 50
num_LD_bins = 100

hist, bin_edges = np.histogram(fpt_LD, bins=num_LD_bins, range=(t_min, t_max), density=True)
time_axis = bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

ax.hist(fpt_MD, bins=num_MD_bins, range=(t_min, t_max), label='MD', density=True, alpha=0.6, color='tab:gray')
ax.plot(time_axis, hist, label='Langevin model', lw=2.0, color='tab:red')
ax.tick_params(axis='both', direction='in')
ax.set_xlabel(r'First passage time $\tau$ (ns)')
ax.set_ylabel(r'$P(\tau)$')
ax.set_xticks([0,25,50,75,100,125,150])
ax.text(82, 0.0145, r'$\langle \tau_{\mathrm{MD}} \rangle = 37.7 \pm 0.7$ ns', fontsize=8)
ax.text(82, 0.012, r'$\langle \tau_{\mathrm{Langevin}} \rangle = 41 \pm 3$ ns', color='tab:red', fontsize=8)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('FPT.pdf')
#plt.show()

# estimate uncertainty
num_sets = 50
fpt_LD_split = np.array_split(fpt_LD, num_sets)
fpt_MD_split = np.array_split(fpt_MD, num_sets)

mfpt_LD = np.empty(num_sets)
mfpt_MD = np.empty(num_sets)

for i in range(num_sets):
    mfpt_LD[i] = np.mean(fpt_LD_split[i])
    mfpt_MD[i] = np.mean(fpt_MD_split[i])

print(f'MFPT LD: {np.mean(mfpt_LD)} +/- {np.std(mfpt_LD)}')
print(f'MFPT MD: {np.mean(mfpt_MD)} +/- {np.std(mfpt_MD)}')
