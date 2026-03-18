""" Plots for third system """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

# font and custom colors
plt.rc('font', size = 9)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.latex.preamble'] = r"""\usepackage{tgheros}\usepackage{sansmath}\sansmath"""

# parameters
num_replicas = 10
num_propagators = 36
min_q = -1.2
max_q = 1.2

# true F(q) and D(q)
qt = np.linspace(-1.2,1.2,100)
Ft = 8*(qt**2 - 1.0**2)**2
Dt = 0.02 + 0.01*np.exp(-(qt**2)/(2*0.3**2))

# load data
samples_D1 = np.zeros((num_replicas,num_propagators))
samples_F2 = np.zeros((num_replicas,num_propagators))
samples_D2 = np.zeros((num_replicas,num_propagators))

for i in range(num_replicas):
    samples_D1[i] = np.load(f'D_fit_D/D1_rep_{i+1}.npy')
    samples_F2[i] = np.load(f'F_fit_F_D/F2_rep_{i+1}.npy')
    samples_D2[i] = np.load(f'D_fit_F_D/D2_rep_{i+1}.npy')

# averages and standard deviations
ave_D1 = np.mean(samples_D1,axis=0)
err_D1 = np.std(samples_D1,axis=0)
ave_F2 = np.mean(samples_F2,axis=0)
ave_D2 = np.mean(samples_D2,axis=0)
err_F2 = np.std(samples_F2,axis=0)
err_D2 = np.std(samples_D2,axis=0)

# q-axis
w = (max_q-min_q)/num_propagators
q = (np.arange(num_propagators) + 0.5)*w + min_q

# plots
fig, ax = plt.subplots(2,1,figsize=(3.37, 4.0))

# a) fit D(q)
ax[0].errorbar(q,ave_F2-np.min(ave_F2),yerr=err_F2,marker='o',linestyle='None',ms=3,capsize=2,color='tab:red',elinewidth=1)
ax[0].plot(qt,Ft-np.min(Ft),lw=1.0,color='black')
ax[0].set_xlabel(r'$q$')
ax[0].set_ylabel(r'$\beta F(q)$')
ax[0].set_xlim([-1.2,1.2])
ax[0].set_xticks([-1.2,-0.8,-0.4,0.0,0.4,0.8,1.2])
ax[0].text(-1.18,8.50,r'(a)')
ax[0].tick_params(axis="y",direction="in")
ax[0].tick_params(axis="x",direction="in")
ax[0].set_ylim([-0.5,9.5])

# b) fit F(q) and D(q)
ax[1].errorbar(q,100*ave_D1,yerr=100*err_D1,marker='^',linestyle='None',
               ms=4,capsize=2,color='tab:blue',label='setup 1',elinewidth=1)
ax[1].errorbar(q,100*ave_D2,yerr=100*err_D2,marker='o',linestyle='None',
               ms=3,capsize=2,color='tab:red',label='setup 2',elinewidth=1)
ax[1].set_xlabel(r'$q$')
ax[1].set_ylabel(r'$D(q) \times 10^{2}$ [ps$^{-1}$]')
ax[1].set_xlim([-1.2,1.2])
ax[1].set_ylim([1.9,3.2])
ax[1].plot(qt,100*Dt,lw=1.0,color='black',label='true')
ax[1].legend(fontsize=8)
#ax[1].legend(ncol=3,columnspacing=1.0,frameon=True,loc='upper right',fontsize=8,handletextpad=0.4,handlelength=1.5)
ax[1].set_xticks([-1.2,-0.8,-0.4,0.0,0.4,0.8,1.2])
ax[1].text(-1.18,3.07,r'(b)')
ax[1].tick_params(axis="y",direction="in")
ax[1].tick_params(axis="x",direction="in")

plt.tight_layout()
#plt.show()
plt.savefig('sys3_fig1.pdf')
