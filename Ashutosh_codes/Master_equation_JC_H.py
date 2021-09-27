#%%
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import math
import time

#%%

def drive_JC(t,args):
    return 0.2*math.tanh(t)*np.exp(-1j*w_d*t)

def drive_JC_conj(t,args):
    return 0.2*math.tanh(t)*np.exp(1j*w_d*t)
    

def Master_Equation(Nq,Nr,w_r,w_a,g_coup,psi0,tlist):
    
    # define the operators
    a = tensor(destroy(Nr),qeye(Nq)) # resonator
    sz = tensor(qeye(Nr),sigmaz()) # qubit
    sm = tensor(qeye(Nr),sigmap())
    sp = tensor(qeye(Nr),sigmam())


    # Define the time dependent hamiltonian
    H0 = h_cross * w_r * a.dag() * a  + h_cross * w_a * sz/2  + h_cross * g_coup *(a.dag() * sm + a * sp) # JC hamiltonian 
    H1 = h_cross * a.dag()
    H2 = h_cross * a
    H  = [H0,[H1,drive_JC],[H2,drive_JC_conj]]

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
g_coup = 0.1 # coupling between resonator and qubit
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

tmax = 20
tlist = np.linspace(0,tmax,1000)

# %%
out0 = Master_Equation(Nq,Nr,w_r,w_a,g_coup,psi0,tlist)
out1 = Master_Equation(Nq,Nr,w_r,w_a,g_coup,psi1,tlist)
# %%
# Trace over qubit
state0 = out0.states
state1 = out1.states

# %%
# Transform frame to frame of drive
a = tensor(destroy(Nr),qeye(Nq)) # resonator
sz = tensor(qeye(Nr),sigmaz()) # qubit
sm = tensor(qeye(Nr),sigmap())
sp = tensor(qeye(Nr),sigmam())


U_exp = [1j * t * w_d * (a.dag() * a + sz ) for t in tlist]
U_matrix = [i.expm() for i in U_exp]

state0_rot = [U_matrix[i] * state0[i] * U_matrix[i].dag()  for i in range(len(state0))]
state1_rot = [U_matrix[i] * state1[i] * U_matrix[i].dag() for i in range(len(state1))]


# %%
alpha_g = [i.ptrace(0) for i in state0_rot]
alpha_e = [i.ptrace(0) for i in state1_rot]


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

# %%
