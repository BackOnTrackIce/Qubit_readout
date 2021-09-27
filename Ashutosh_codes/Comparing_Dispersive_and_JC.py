#%%
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import math
import time

# %%
def drive_JC(t,args):
    return Drive_strength * math.tanh(t) * np.exp(-1j*w_d*t)

def drive_JC_conj(t,args):
    return Drive_strength * math.tanh(t) * np.exp(1j*w_d*t)


def drive_disp(t,args):
    return Drive_strength * math.tanh(t)



def Master_Equation(H0,Nq,Nr,psi0,tlist,flag):
    
    # define the operators
    a = tensor(destroy(Nr),qeye(Nq))

    # Define the time dependent hamiltonian 
    H1 = h_cross * a.dag()
    H2 = h_cross * a
    if flag == "JC":
        H  = [H0,[H1,drive_JC],[H2,drive_JC_conj]]
    else:
        H = [H0,[H1,drive_disp],[H2,drive_disp]]

    c_ops = [kappa * a]
    #c_ops = []
    e_ops = []

    # using mesolver 
    output  = mesolve(H,psi0,tlist,c_ops,e_ops)

    return output


def frame_of_drive(output1, output2, tlist):
    state0 = output1.states
    state1 = output2.states

    a = tensor(destroy(Nr),qeye(Nq)) # resonator
    sz = tensor(qeye(Nr),sigmaz()) # qubit

    U_exp = [1j * t * w_d * (a.dag() * a + sz ) for t in tlist]
    
    U_matrix = [i.expm() for i in U_exp]

    state0_rot = [U_matrix[i] * state0[i] * U_matrix[i].dag()  for i in range(len(state0))]
    state1_rot = [U_matrix[i] * state1[i] * U_matrix[i].dag() for i in range(len(state1))]

    return state0_rot, state1_rot


def partial_trace_over_qubit(state0, state1):
    alpha_g = [i.ptrace(0) for i in state0]
    alpha_e = [i.ptrace(0) for i in state1]
    return alpha_g, alpha_e


def phase_plot_quantities(alpha_g, alpha_e):
    a = destroy(Nr)
    Q_g = [expect((a + a.dag())/2,i) for i in alpha_g ]
    I_g = [expect(1j*(a.dag() - a)/2,i) for i in alpha_g ]

    Q_e = [expect((a + a.dag())/2,i) for i in alpha_e ]
    I_e = [expect(1j*(a.dag() - a)/2,i) for i in alpha_e ]

    return Q_g, I_g, Q_e, I_e

# %%
h_cross = 1
w_r = 5 # frequency of resonator
w_a = 6 # frequency of qubit
g_coup = 0.1 # coupling between resonator and qubit
kappa = 0.1 * 2 * np.pi
Nr = 5  # No. of levels of resonator
Nq = 2   # No. of levels of qubit
w_d = 5  # drive frequency
chi = g_coup**2/(w_a - w_r)
Drive_strength = 0.2

a = tensor(destroy(Nr),qeye(Nq)) # resonator
sz = tensor(qeye(Nr),sigmaz()) # qubit
sm = tensor(qeye(Nr),sigmap())
sp = tensor(qeye(Nr),sigmam())

H0_JC = h_cross * w_r * a.dag() * a  + h_cross * w_a * sz/2  + h_cross * g_coup *(a.dag() * sm + a * sp)

H0_Disp = h_cross * (w_r - w_d) * a.dag() * a  + h_cross * (w_a + chi) * sz/2  + h_cross * chi * a.dag() * a * sz 

#define the initial state of the system
psi0 = tensor(basis(Nr,0),basis(Nq,0))
psi1 = tensor(basis(Nr,0),basis(Nq,1))
#print(psi0)
#psi0 = tensor(basis(Nr,0),(basis(Nq,0) + basis(Nq,1))/math.sqrt(2) )

tmax = 20
tlist = np.linspace(0,tmax,1000)

# %%
out0_JC = Master_Equation(H0_JC,Nq,Nr,psi0,tlist,"JC")
out1_JC = Master_Equation(H0_JC,Nq,Nr,psi1,tlist,"JC")
# %%
out0_Disp = Master_Equation(H0_Disp,Nq,Nr,psi0,tlist,"Disp")
out1_Disp = Master_Equation(H0_Disp,Nq,Nr,psi1,tlist,"Disp")

#%%

state0_JC, state1_JC = frame_of_drive(out0_JC,out1_JC,tlist)
State0_Res_JC, State1_Res_JC = partial_trace_over_qubit(state0_JC, state1_JC)

Q_g_JC, I_g_JC, Q_e_JC, I_e_JC = phase_plot_quantities(State0_Res_JC, State1_Res_JC)

#%%
#state0_Disp, state1_Disp = frame_of_drive(out0_Disp,out1_Disp,tlist)
state0_Disp = out0_Disp.states
state1_Disp = out1_Disp.states

State0_Res_Disp, State1_Res_Disp = partial_trace_over_qubit(state0_Disp, state1_Disp)

Q_g_Disp, I_g_Disp, Q_e_Disp, I_e_Disp = phase_plot_quantities(State0_Res_Disp, State1_Res_Disp)

# %%
plt.plot(Q_g_JC, I_g_JC, label = "JC")
plt.plot(Q_e_JC, I_e_JC, label = "JC")

plt.plot(Q_g_Disp, I_g_Disp, label = "Dispersive")
plt.plot(Q_e_Disp, I_e_Disp, label = "Dispersive")

plt.legend()
# %%
