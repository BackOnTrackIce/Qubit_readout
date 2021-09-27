#%%
from os import times
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode

#%%
def drive_amp(t):
    #return np.cos(t)
    return 1

#%%
def ode_to_solve(t,w):
    alpha_e, alpha_g = w
    #Delta_r, chi, k = p

    f = [-1j*drive_amp(t) -1j*(Delta_r + chi)*alpha_e - k*alpha_e/2, -1j*drive_amp(t) -1j*(Delta_r - chi)*alpha_g - k*alpha_g/2 ]

    return f

# %%
# Initial conditions
Delta_r = 10 * 2 * np.pi
chi = 2 * np.pi * 5
k = 2 * np.pi * 1

alpha_e = 1*0
alpha_g = 1*0

abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 10.0
numpoints = 25000

dt = stoptime/numpoints
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

p = [Delta_r, chi, k]
w0 = [alpha_e, alpha_g]
#%%

ode_sol = complex_ode(ode_to_solve)
ode_sol.set_initial_value(w0,0.0)

sol1 = []
sol2 = []

for i in t:
    temp = ode_sol.integrate(i + dt)    
    sol1.append(temp[0])
    sol2.append(temp[1])

# %%
plt.plot(np.real(sol1),np.imag(sol1))
plt.plot(np.real(sol2),np.imag(sol2))

# %%

# %%
