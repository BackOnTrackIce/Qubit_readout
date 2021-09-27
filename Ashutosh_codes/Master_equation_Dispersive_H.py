#%%
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import math
import time

from scipy.linalg.matfuncs import tanhm

#%%

def drive_freq_fun(t,args):
    return 0.2*math.tanh(t)
    #return 0.2

def drive_freq_fun_conj(t,args):
    return 0.2*math.tanh(t)
    #return 0.2

def qubit_integrate(Nq,Nr,w_r,w_a,g_coup,psi0,tlist):
    
    # define the operators
    a = tensor(destroy(Nr),qeye(Nq)) # resonator
    sz = tensor(qeye(Nr),sigmaz()) # qubit
    sm = tensor(qeye(Nr),sigmap())
    sp = tensor(qeye(Nr),sigmam())


    # Define the time dependent hamiltonian
    #H0 = h_cross * w_r * a.dag() * a  + h_cross * w_a * sz/2  + h_cross * g_coup *(a.dag() * sp + a * sm) # JC hamiltonian
    H0 = h_cross * (w_r - w_d) * a.dag() * a  + h_cross * (w_a + chi) * sz/2  + h_cross * chi * a.dag() * a * sz 
    H1 = h_cross * a.dag()
    H2 = h_cross * a
    H  = [H0,[H1,drive_freq_fun],[H2,drive_freq_fun_conj]]
    
    # Define the collapse operators
    #c_ops = [(g * a/2),(kappa * b/2)]
    
    # define the list of operators whose expectation values are needed 
    #e_ops = [n_a,n_b]

    c_ops = [kappa * a]
    #c_ops = []
    e_ops = []

    # using mesolver 
    output  = mesolve(H,psi0,tlist,c_ops,e_ops)

    return output

#%%
h_cross = 1
w_r = 5 # frequency of resonator
w_a = 6 # frequency of qubit
g_coup = 0.01 # coupling between resonator and qubit
kappa = 0.1 * 2 * np.pi
Nr = 5  # No. of levels of resonator
Nq = 2   # No. of levels of qubit
w_d = 5  # drive frequency
chi = 0.05 * 2 * np.pi #g_coup**2/(w_a - w_r)


#define the initial state of the system
psi0 = tensor(basis(Nr,0),basis(Nq,0))
psi1 = tensor(basis(Nr,0),basis(Nq,1))
#print(psi0)
#psi0 = tensor(basis(Nr,0),(basis(Nq,0) + basis(Nq,1))/math.sqrt(2) )

tlist = np.linspace(0,20,1000)

# %%
out0 = qubit_integrate(Nq,Nr,w_r,w_a,g_coup,psi0,tlist)
out1 = qubit_integrate(Nq,Nr,w_r,w_a,g_coup,psi1,tlist)
# %%
# Trace over qubit
state0 = out0.states
state1 = out1.states

alpha_g = [i.ptrace(0) for i in state0]
alpha_e = [i.ptrace(0) for i in state1]

# %%
a = destroy(Nr)
Q_g = [expect((a + a.dag())/2,i) for i in alpha_g ]
I_g = [expect(1j*(a.dag() - a)/2,i) for i in alpha_g ]

Q_e = [expect((a + a.dag())/2,i) for i in alpha_e ]
I_e = [expect(1j*(a.dag() - a)/2,i) for i in alpha_e ]

# %%
plt.plot(I_g,Q_g)
plt.plot(I_e,Q_e)
# %%
plt.plot(Q_g,I_g)
plt.plot(Q_e,I_e)

# %%

# %%
